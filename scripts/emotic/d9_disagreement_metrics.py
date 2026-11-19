"""Measure label discrimination within disagreement cases and its effect on full-cohort AP."""

import numpy as np
import torch
from scipy.stats import rankdata

from multi_lane.track_a.runner import CLASS_ORDER, average_precision, task_indices
from scripts.emotic.d9_direction_permutation import FORMAL_SEEDS

TRANSFER = .05
METHODS = ('fixed', 'all', 'middle', 'score')


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _auc(labels, scores):
    """Positive-vs-negative concordance, giving half credit for equal scores."""
    pos = int(labels.sum())
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    ranks = rankdata(scores, method='average')
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _candidate_summary(y, mask, middle, score):
    labels = y[mask]
    pos = int(labels.sum())
    neg = len(labels) - pos
    row = dict(n=int(len(labels)), n_positive=pos, n_negative=neg)
    for name, values in [('middle', middle[mask]), ('score', score[mask])]:
        accepts = values >= .5
        pos_accepted = int(accepts[labels].sum())
        neg_accepted = int(accepts[~labels].sum())
        row['auc_' + name] = _auc(labels, values)
        row[name] = dict(accepted_positive=pos_accepted, accepted_negative=neg_accepted,
                         positive_accept_rate=_ratio(pos_accepted, pos),
                         negative_accept_rate=_ratio(neg_accepted, neg))
    return row


def _top50(probabilities, y):
    k = min(50, len(y))
    top = {}
    result = {}
    for method, p in probabilities.items():
        mask = np.zeros(len(y), dtype=bool)
        mask[p.argsort()[::-1][:k]] = True
        top[method] = mask
        positives = int(y[mask].sum())
        result[method] = dict(positives=positives, negatives=k - positives, precision=positives / k)
    added = top['all'] & ~top['fixed']
    rescued = added & y
    false_alarms = added & ~y
    result.update(k=k, rescued_by_all=int(rescued.sum()), false_alarms_by_all=int(false_alarms.sum()))
    for method, suffix in [('middle', ''), ('score', '_score')]:
        result['rescued_retained' + suffix] = int((rescued & top[method]).sum())
        result['false_alarms_prevented' + suffix] = int((false_alarms & ~top[method]).sum())
        # Also count entries unique to this method, since removing one false alarm can admit another.
        entering = top[method] & ~top['fixed']
        result['new_positives_vs_fixed' + suffix] = int((entering & y).sum())
        result['new_negatives_vs_fixed' + suffix] = int((entering & ~y).sum())
    return result


def _shuffle(accepted, groups, task, class_id, seed):
    swapped = accepted.copy()
    for group_id, members in enumerate(groups):
        if len(members) < 2:
            continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, task, class_id, group_id]))
        cycle = rng.permutation(members)
        swapped[cycle] = accepted[np.roll(cycle, 1)]
    return swapped


def _average_present(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def evaluate(data, pred, task, permutations=100):
    labels = np.asarray(data['targets'])
    ids = np.asarray(data['sample_ids'])
    valid = np.asarray(data['reliable'], dtype=bool)
    indices = task_indices(task)
    if not len(ids) or labels.shape != (len(ids), len(indices)) or valid.shape != ids.shape:
        raise ValueError('Invalid evaluation axes')
    if len(np.unique(ids)) != len(ids) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected one row per person and binary labels')
    if not 1 <= permutations <= len(FORMAL_SEEDS):
        raise ValueError('Unsupported permutation count')
    fixed = np.asarray(pred['fixed_logits'], dtype=np.float32)
    candidate = np.asarray(pred['candidate'])
    receiver = np.asarray(pred['receiver'])
    gap = np.asarray(pred['gap'], dtype=np.float32)
    middle = np.asarray(pred['middle_probability'])
    score = np.asarray(pred['score_probability'])
    if any(a.shape != labels.shape for a in (fixed, candidate, receiver, gap, middle, score)):
        raise ValueError('Predictions must match the target axes')
    if not all(np.isfinite(a).all() for a in (fixed, gap, middle, score)):
        raise ValueError('Nonfinite disagreement score')
    if (candidate.dtype != bool or not np.isin(receiver, [1, 2]).all() or
            (candidate & (gap <= 0)).any() or (candidate[~valid] & (receiver[~valid] == 2)).any()):
        raise ValueError('Invalid candidate or unavailable receiver')
    if any(((a < 0) | (a > 1)).any() for a in (middle, score)):
        raise ValueError('Invalid gate probability')
    actions = dict(all=candidate, middle=candidate & (middle >= .5), score=candidate & (score >= .5))
    # Preserve source reference rounding. Apply the same constant-mass logit transfer for every method.
    adjusted = fixed + np.float32(TRANSFER) * gap
    logits = {'fixed': fixed, **{m: np.where(mask, adjusted, fixed) for m, mask in actions.items()}}
    probs = {m: torch.from_numpy(np.ascontiguousarray(v)).sigmoid().numpy() for m, v in logits.items()}
    ordered = np.argsort(ids, kind='stable')
    classes = {}
    shuffle_maps = []
    for c, index in enumerate(indices):
        y = labels[:, c].astype(bool)
        mask = candidate[:, c]
        winner = receiver[:, c]
        q = {m: p[:, c] for m, p in probs.items()}
        ap = {m: float(100 * average_precision(p, y)) for m, p in q.items()}
        oracle = np.where(mask & y, q['all'], q['fixed'])
        ap['oracle_positive'] = float(100 * average_precision(oracle, y))
        groups = [ordered[(mask & (winner == view) & (valid == available))[ordered]]
                  for view, available in ((1, False), (1, True), (2, True))]
        exchanged = []
        for seed in FORMAL_SEEDS[:permutations]:
            swap = _shuffle(actions['middle'][:, c], groups, task, index, seed)
            # Sigmoid is elementwise; using the archived and shifted probabilities keeps tie behavior.
            p = np.where(swap, q['all'], q['fixed'])
            exchanged.append(100 * average_precision(p, y))
        ap['shuffle_mean'] = float(np.mean(exchanged))
        shuffle_maps.append(exchanged)
        subgroups = {'all': mask, 'person': mask & (winner == 1), 'face': mask & (winner == 2)}
        info = {name: _candidate_summary(y, selected, middle[:, c], score[:, c])
                for name, selected in subgroups.items()}
        classes[CLASS_ORDER[index]] = dict(positives=int(y.sum()), candidate=info, ap=ap,
                                           middle_minus_shuffle=ap['middle'] - ap['shuffle_mean'],
                                           top50=_top50(q, y))
    rows = list(classes.values())
    ap = {m: float(np.mean([row['ap'][m] for row in rows])) for m in (*METHODS, 'shuffle_mean', 'oracle_positive')}
    counts = {k: sum(row['top50'][k] for row in rows)
              for k in ('rescued_by_all', 'rescued_retained', 'false_alarms_by_all', 'false_alarms_prevented')}
    summary = dict(ap=ap, top50=counts, middle_minus_shuffle=ap['middle'] - ap['shuffle_mean'])
    for method in ('middle', 'score'):
        aucs = [row['candidate']['all']['auc_' + method] for row in rows]
        summary['auc_' + method] = _average_present(aucs)
        summary['n_auc_' + method] = sum(v is not None for v in aucs)
    return dict(task=task, people=len(ids), classes=classes, permutations=permutations,
                shuffled_task_maps=np.mean(shuffle_maps, axis=0).tolist(), macro=summary)
