"""Learn a shared auxiliary budget, then a matched person-dependent split of that budget."""
import numpy as np
import torch
import torch.nn.functional as F

from scripts.emotic.fit_d9_task10_probe import LR, PROJECTION_SEED, STD_FLOOR, apply_rule, mix, project


STEPS = 300
CAP = 0.5
FEATURE_NAMES = [f'task10_{view}_projection_{i}' for view in ('full', 'person', 'face') for i in range(4)]
FEATURE_NAMES += [f'frozen9_{view}_to_full_{stat}' for view in ('person', 'face')
                  for stat in ('mass', 'concentration', 'key_alignment', 'value_alignment')]


def _person_features(data, stats=None):
    projected = project(data)
    values = np.column_stack((projected['content'], projected['relation']))
    reliable = data['reliable']
    if values.shape != (len(reliable), len(FEATURE_NAMES)) or not np.isfinite(values[reliable]).all():
        raise ValueError('Invalid available Task10/Frozen9 features')
    if stats is None:
        if not np.any(reliable) or np.all(reliable):
            raise ValueError('Both Face availability groups are required for fitting both budgets')
        stats = dict(mean=values[reliable].mean(0).tolist(), sd=values[reliable].std(0).tolist())
    sd = np.asarray(stats['sd'])
    result = np.zeros_like(values, dtype=np.float32)
    result[reliable] = np.divide(values[reliable] - stats['mean'], sd,
                                 out=np.zeros_like(values[reliable]), where=sd > STD_FLOOR)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite standardized auxiliary input')
    return torch.from_numpy(result), stats


def _weights(budgets, person_logit, reliable):
    b = budgets[reliable.long()]
    share = torch.where(reliable, torch.sigmoid(person_logit), 1.)
    return torch.stack((1. - b, b * share, b * (1. - share)), dim=1)


def _optimize(params, weights, z, y):
    opt = torch.optim.Adam(params, lr=LR)
    for _ in range(STEPS):
        loss = F.binary_cross_entropy_with_logits(mix(weights(), z), y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite fusion loss')
        opt.zero_grad()
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError('Nonfinite fusion parameter gradient')
        opt.step()
    with torch.no_grad():
        loss = F.binary_cross_entropy_with_logits(mix(weights(), z), y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite final fusion loss')
        return float(loss)


def fit_task(data):
    """Full-batch BCE on all TRAIN people/classes in each of two consecutive phases."""
    x, stats = _person_features(data)
    r = torch.from_numpy(data['reliable'])
    z = torch.as_tensor(data['branch_logits'], dtype=torch.float32).detach()
    y = torch.as_tensor(data['targets'], dtype=torch.float32).detach()
    if (y.ndim != 2 or len(x) != len(y) or z.shape != (*y.shape, 3)
            or not torch.isfinite(z).all() or not torch.all((y == 0) | (y == 1))):
        raise ValueError('Invalid branch logits or TRAIN targets')

    budget_logits = torch.zeros(2, requires_grad=True)
    person_logit = torch.zeros((), requires_grad=True)

    def group_weights():
        return _weights(CAP * torch.sigmoid(budget_logits), person_logit, r)

    group_loss = _optimize([budget_logits, person_logit], group_weights, z, y)
    budgets = CAP * torch.sigmoid(budget_logits.detach())
    center = person_logit.detach()
    coef = torch.zeros(x.shape[1], requires_grad=True)

    def person_weights():
        return _weights(budgets, center + x @ coef, r)

    person_loss = _optimize([coef], person_weights, z, y)
    return dict(budget_by_reliability=budgets.tolist(), budget_logits=budget_logits.detach().tolist(),
                group_person_logit=float(center), group_person_share=float(center.sigmoid()),
                person_coefficients=coef.detach().tolist(), feature_names=FEATURE_NAMES.copy(),
                feature_stats=stats, feature_projection_seed=PROJECTION_SEED,
                fit=dict(cap=CAP, steps_per_phase=STEPS, optimizer='Adam', lr=LR,
                         initial_budget=.25, initial_person_share=.5,
                         people=len(y), reliable_people=int(r.sum()), n_classes=y.shape[1],
                         group_train_bce=group_loss, person_train_bce=person_loss))


@torch.no_grad()
def apply_task(data, model):
    """Use only middle features and Face availability to assign the N x 3 weights."""
    x, _ = _person_features(data, model['feature_stats'])
    r = torch.from_numpy(data['reliable'])
    budgets = torch.tensor(model['budget_by_reliability'], dtype=torch.float32)
    center = torch.tensor(model['group_person_logit'], dtype=torch.float32)
    coef = torch.tensor(model['person_coefficients'], dtype=torch.float32)
    if (model['feature_names'] != FEATURE_NAMES or budgets.shape != (2,) or coef.shape != (x.shape[1],)
            or not torch.all((budgets >= 0) & (budgets <= CAP)) or not torch.isfinite(center)
            or not torch.isfinite(coef).all()):
        raise ValueError('Invalid learned auxiliary budget or allocation')
    return dict(fixed=apply_rule('fixed', data, None, None).numpy(),
                group=_weights(budgets, center, r).numpy(),
                person=_weights(budgets, center + x @ coef, r).numpy())
