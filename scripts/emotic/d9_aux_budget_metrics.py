"""Score protected Full and matched auxiliary allocations on the current task head."""

import numpy as np
import torch

from multi_lane.track_a.runner import CLASS_ORDER, average_precision, task_indices
from scripts.emotic.d9_direction_permutation import permutation_sources
from scripts.emotic.fit_d9_task10_probe import apply_rule, mix


def _check_weights(weights, valid):
    result = {}
    for method in ('fixed', 'group', 'person'):
        w = np.asarray(weights[method], dtype=np.float32)
        if (w.shape != (len(valid), 3) or not np.isfinite(w).all()
                or (w < 0).any() or (w > 1).any() or w[~valid, 2].any()
                or not np.allclose(w.sum(axis=1), 1., rtol=0, atol=2e-7)):
            raise ValueError(f'Invalid Full/Person/Face weights for {method}')
        result[method] = w
    expected = apply_rule('fixed', {'reliable': valid}, None, None).numpy()
    if not np.array_equal(result['fixed'], expected):
        raise ValueError('Fixed readout does not replay the historical allocation')
    group, person = result['group'], result['person']
    for flag in (False, True):
        rows = group[valid == flag]
        if len(rows) and not np.all(rows == rows[0]):
            raise ValueError('Group readout varies within a reliability group')
    if ((group[:, 0] < .5).any() or not np.array_equal(group[:, 0], person[:, 0])
            or not np.array_equal(group[~valid], person[~valid])):
        raise ValueError('Different learned Full budget or unmatched unreliable allocations')
    return result


def _stats(w):
    w = w.astype(np.float64)
    return dict(mean=w.mean(axis=0).tolist(), std=w.std(axis=0).tolist())


def _class_ap(logits, labels):
    probs = logits.sigmoid().numpy()
    return np.array([100. * average_precision(p, y) for p, y in zip(probs.T, labels.T)])


def evaluate(data, weights, task, permutations=100):
    """Evaluate without fitting. The within-split group means are attribution controls only."""
    indices = task_indices(task)
    names = [CLASS_ORDER[i] for i in indices]
    labels = np.asarray(data['targets'])
    ids = np.asarray(data['sample_ids'])
    valid = np.asarray(data['reliable'])
    branches = torch.as_tensor(data['branch_logits'], dtype=torch.float32)
    if (ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids)
            or valid.shape != ids.shape or valid.dtype != np.bool_
            or labels.shape != (len(ids), len(indices)) or not np.isin(labels, [0, 1]).all()
            or tuple(branches.shape) != (*labels.shape, 3) or not torch.isfinite(branches).all()
            or not 1 <= permutations <= 100):
        raise ValueError('Invalid current-head evaluation data or permutation count')
    methods = _check_weights(weights, valid)
    methods['mean_person'] = methods['person'].copy()
    masks = {'all': np.ones_like(valid), 'reliable': valid, 'unreliable': ~valid}
    stats = {}
    for name, mask in [('reliable', valid), ('unreliable', ~valid)]:
        if mask.any():
            methods['mean_person'][mask] = methods['person'][mask].mean(axis=0, dtype=np.float64)
            stats[name] = {method: _stats(w[mask]) for method, w in methods.items()}
        else:
            stats[name] = {}

    # Use historical sum-of-weighted-logits arithmetic and float32 sigmoid tie behavior.
    logits = {method: mix(torch.from_numpy(w), branches) for method, w in methods.items()}
    fixed_error = None
    if 'fixed_logits' in data:
        archived = torch.as_tensor(data['fixed_logits'], dtype=torch.float32)
        torch.testing.assert_close(logits['fixed'], archived, atol=2e-5, rtol=2e-6)
        fixed_error = (logits['fixed'] - archived).abs().max().item()
        logits['fixed'] = archived
    present = {name: mask for name, mask in masks.items() if mask.any()}
    aps = {name: {method: _class_ap(z[mask], labels[mask]) for method, z in logits.items()}
           for name, mask in present.items()}
    shuffled = {name: [] for name in present}
    for seed in range(1, 1 + permutations):
        # Each donor supplies the same allocation for all of their recipient's current classes.
        donors = permutation_sources(ids, valid, task + 1, seed)[:, task]
        z = mix(torch.from_numpy(methods['person'][donors]), branches)
        for name, mask in present.items():
            shuffled[name].append(_class_ap(z[mask], labels[mask]))
    groups = {}
    for name, mask in masks.items():
        if not mask.any():
            groups[name] = {'people': 0, 'macro': None, 'classes': {}}
            continue
        aps[name]['shuffle'] = np.mean(shuffled[name], axis=0)
        groups[name] = dict(people=int(mask.sum()), positives=labels[mask].sum(axis=0).astype(int).tolist(),
                            macro={m: float(values.mean()) for m, values in aps[name].items()},
                            classes={c: {m: float(values[i]) for m, values in aps[name].items()}
                                     for i, c in enumerate(names)})
    return dict(task=task, people=len(ids), macro=groups['all']['macro'], classes=groups['all']['classes'],
                slices={g: groups[g] for g in ('reliable', 'unreliable')}, weight_stats=stats,
                fixed_max_abs_logit_error=fixed_error, mean_person_source='unlabelled same-split group averages',
                permutation_seeds=list(range(1, 1 + permutations)),
                shuffled_task_maps=np.mean(shuffled['all'], axis=1).tolist())
