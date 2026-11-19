"""Learn personal corrections to a frozen TRAIN-derived group mean."""
import numpy as np
import torch

from scripts.emotic.d9_quality_fit import (
    LR, STEPS, VIEWS, _balanced_bce, _features, _inputs, _mixed, _paired_loss,
    _row_loss, _utility_targets,
)
from scripts.emotic.fit_d9_task10_probe import stats


def _centers(weights):
    a = torch.as_tensor(weights, dtype=torch.float32).detach()
    if (a.shape != (2, 3) or not torch.isfinite(a).all() or a[0, 2] != 0
            or (a[0, :2] <= 0).any() or (a[1] <= 0).any()
            or not torch.allclose(a.sum(-1), torch.ones(2), atol=2e-7, rtol=0)):
        raise ValueError('Expected frozen A weights for unavailable and available Face')
    return a


def _personal_weights(x, r, matrix, a, means=None):
    # Axes of x: person, input version, middle feature.
    u = torch.tanh(x @ matrix.T)
    u = torch.cat([u[..., :2], u[..., 2:] * r[:, None, None]], dim=-1)
    if means is None:
        # Empty evaluation subgroups have no rows to correct.
        selectors = torch.stack((~r, r)).to(u.dtype)
        selectors = selectors / selectors.sum(1, keepdim=True).clamp_min(1.)
        means = torch.einsum('gn,nkv->gkv', selectors, u)
    d = u - means[r.long()]
    base = a[r.long(), None, :]
    w = base * (1. + (d - (d * base).sum(-1, keepdim=True)) / 4.)
    return w, means


def _log_weights(w):
    # Unavailable Face has exactly zero mass. Suppress its logit in odds comparisons.
    return torch.where(w > 0, w.clamp_min(torch.finfo(w.dtype).tiny).log(), -torch.inf)


def fit_dynamic(natural, variants, a_weights_2x3):
    """Fit both heads on identical TRAIN people and augmented versions, with A frozen."""
    valid = np.asarray(natural['reliable'])
    if valid.ndim != 1 or valid.dtype != np.bool_ or not valid.any() or valid.all():
        raise ValueError('Both Face availability groups required on TRAIN')
    a = _centers(a_weights_2x3)
    r = torch.from_numpy(valid)
    content, relation = _features(natural)
    norm = dict(content=stats(content, valid, 8), relation=stats(relation, valid, 4))
    entries = [natural]
    for name in VIEWS:
        entry = variants[name]
        for k in ('sample_ids', 'targets', 'reliable'):
            if k in entry and not np.array_equal(entry[k], natural[k]):
                raise ValueError(f'Unmatched {name} input rows for {k}')
        entries.append(entry)
    x = torch.stack([_inputs(_features(d), valid, norm) for d in entries], dim=1)
    z = torch.tensor(np.stack([d['branch_logits'] for d in entries], axis=1), dtype=torch.float32)
    labels = torch.tensor(natural['targets'], dtype=torch.float32).unsqueeze(1)
    if (labels.ndim != 3 or z.shape != (len(r), 3, labels.shape[-1], 3) or not torch.isfinite(z).all()
            or not torch.all((labels == 0) | (labels == 1))):
        raise ValueError('Invalid paired TRAIN labels or logits')
    usable = torch.stack((torch.ones_like(r), torch.ones_like(r), r), dim=1)
    targets = _utility_targets(z, labels, _log_weights(a)[r.long(), None, :].expand_as(z[:, :, 0, :]), r)
    model = dict(center=a.tolist(), stats=norm, steps=STEPS, lr=LR, paired_coefficient=1., heads={})
    for method in ('B', 'C'):
        matrix = torch.zeros((3, x.shape[-1]), requires_grad=True)
        optimizer = torch.optim.Adam([matrix], lr=LR)
        for _ in range(STEPS):
            w, _ = _personal_weights(x, r, matrix, a)
            bce = _balanced_bce(_row_loss(_mixed(z, w), labels), usable)
            loss = bce + _paired_loss(_log_weights(w), targets) if method == 'C' else bce
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {method}')
            optimizer.zero_grad()
            loss.backward()
            if not torch.isfinite(matrix.grad).all():
                raise FloatingPointError(f'Nonfinite gradient in {method}')
            optimizer.step()
        with torch.no_grad():
            w, means = _personal_weights(x, r, matrix, a)
            for group in (0, 1):
                observed = w[r == bool(group)].mean(0)
                torch.testing.assert_close(observed, a[group].expand_as(observed), atol=2e-6, rtol=0)
            row = _row_loss(_mixed(z, w), labels)
            pair = _paired_loss(_log_weights(w), targets)
            model['heads'][method] = dict(matrix=matrix.tolist(), natural_mean=means[:, :1].tolist(),
                                          train_bce=float(_balanced_bce(row, usable)), natural_bce=float(row[:, 0].mean()),
                                          paired_loss=float(pair), last_gradient_norm=float(matrix.grad.norm()))
    return model


@torch.no_grad()
def apply_dynamic(data, model, center_mode='train'):
    """TRAIN center supports individual inference; cohort center enforces the split's exact mean."""
    if center_mode not in ('train', 'cohort'):
        raise ValueError('Expected train or cohort centering')
    valid = np.asarray(data['reliable'])
    if valid.ndim != 1 or valid.dtype != np.bool_ or not len(valid):
        raise ValueError('Invalid Face availability')
    x = _inputs(_features(data), valid, model['stats']).unsqueeze(1)
    r = torch.from_numpy(valid)
    a = _centers(model['center'])
    result = {}
    for method in ('B', 'C'):
        head = model['heads'][method]
        matrix = torch.tensor(head['matrix'], dtype=x.dtype)
        means = torch.tensor(head['natural_mean'], dtype=x.dtype) if center_mode == 'train' else None
        w, _ = _personal_weights(x, r, matrix, a, means)
        w = w[:, 0]
        if ((w < 0).any() or not torch.isfinite(w).all()
                or not torch.allclose(w.sum(-1), torch.ones(len(w)), atol=3e-7, rtol=0)):
            raise ValueError('Invalid personal weights')
        result[method] = w.numpy()
    return result
