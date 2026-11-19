"""Natural-image evaluation of the three TRAIN-fitted quality readouts."""
import numpy as np
import torch

from multi_lane.track_a.runner import CLASS_ORDER, task_indices
from scripts.emotic.d9_aux_budget_metrics import _class_ap, _stats
from scripts.emotic.d9_direction_permutation import permutation_sources
from scripts.emotic.fit_d9_task10_probe import apply_rule, mix

METHODS = ('fixed', 'A', 'B', 'C')


def checked_weights(weights, valid):
    rows = {m: np.asarray(weights[m], dtype=np.float32) for m in METHODS}
    for method, w in rows.items():
        if (w.shape != (len(valid), 3) or not np.isfinite(w).all()
                or (w < 0).any() or (w > 1).any() or w[~valid, 2].any()
                or not np.allclose(w.sum(1), 1, atol=2e-7, rtol=0)):
            raise ValueError(f'Invalid {method} Full/Person/Face allocation')
    if not np.array_equal(rows['fixed'], apply_rule('fixed', {'reliable': valid}, None, None).numpy()):
        raise ValueError('Fixed proportions differ from the registered readout')
    for mask in (valid, ~valid):
        a = rows['A'][mask]
        if len(a) and np.any(a != a[0]):
            raise ValueError('A must use constant weights within each Face availability group')
    return rows


@torch.no_grad()
def evaluate(data, weights, task, permutations=100):
    """Return current-head mAP in points. mean_C is a same-split, unlabelled diagnostic."""
    names = [CLASS_ORDER[k] for k in task_indices(task)]
    ids, valid = np.asarray(data['sample_ids']), np.asarray(data['reliable'])
    y = np.asarray(data['targets'])
    z = torch.as_tensor(data['branch_logits'], dtype=torch.float32)
    if (ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids)
            or valid.shape != ids.shape or valid.dtype != np.bool_
            or y.shape != (len(ids), len(names)) or not np.isin(y, (0, 1)).all()
            or tuple(z.shape) != (*y.shape, 3) or not torch.isfinite(z).all()
            or not 1 <= permutations <= 100):
        raise ValueError('Invalid current-head data or permutation count')
    rows = checked_weights(weights, valid)
    masks = {'all': np.ones_like(valid), 'reliable': valid, 'unreliable': ~valid}
    rows['mean_C'] = rows['C'].copy()
    for mask in (valid, ~valid):
        if mask.any():
            rows['mean_C'][mask] = rows['C'][mask].mean(0, dtype=np.float64)

    logits = {m: mix(torch.from_numpy(w), z) for m, w in rows.items()}
    fixed_error = None
    if 'fixed_logits' in data:
        archived = torch.as_tensor(data['fixed_logits'], dtype=torch.float32)
        torch.testing.assert_close(logits['fixed'], archived, atol=2e-5, rtol=2e-6)
        fixed_error = (logits['fixed'] - archived).abs().max().item()
        logits['fixed'] = archived
    seeds = list(range(1, permutations + 1))
    available = {g: m for g, m in masks.items() if m.any()}
    ap = {g: {key: _class_ap(value[m], y[m]) for key, value in logits.items()}
          for g, m in available.items()}
    shuffled = {g: [] for g in available}
    for seed in seeds:
        donors = permutation_sources(ids, valid, task + 1, seed)[:, task]
        mixed = mix(torch.from_numpy(rows['C'][donors]), z)
        for g, m in available.items():
            shuffled[g].append(_class_ap(mixed[m], y[m]))

    groups = {}
    stats = {}
    for group, mask in masks.items():
        count = int(mask.sum())
        if not count:
            groups[group] = dict(people=0, positives=[0]*len(names), macro=None, classes={})
            stats[group] = {}
            continue
        ap[group]['shuffle'] = np.mean(shuffled[group], axis=0)
        macro = {k: float(values.mean()) for k, values in ap[group].items()}
        groups[group] = dict(
            people=count, positives=y[mask].sum(0).astype(int).tolist(), macro=macro,
            classes={c: {k: float(values[i]) for k, values in ap[group].items()}
                     for i, c in enumerate(names)},
            C_minus={k: macro['C']-v for k, v in macro.items() if k != 'C'})
        stats[group] = {k: _stats(w[mask]) for k, w in rows.items()}
    result = groups['all']
    return dict(task=task, **result, weight_stats=stats, slices={k: groups[k] for k in ('reliable','unreliable')},
                class_order=names, permutation_seeds=seeds,
                shuffled_task_maps=np.mean(shuffled['all'], axis=1).tolist(),
                fixed_max_abs_logit_error=fixed_error, mean_C_source='same-split unlabelled availability-group means')
