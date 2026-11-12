"""Locked TRAIN trajectory weights and identity checks for real-forward replay."""
import numpy as np

TASK_SIZES = (5, 3, 3, 3, 3, 3, 3, 3)


def locked_weights(report, step=99):
    matrices = []
    for task in range(8):
        fit = report[str(task)]['heldout_bce']
        matches = [r for r in fit['trace'] if r['step'] == step]
        if len(matches) != 1:
            raise ValueError('Expected exactly one declared TRAIN trajectory state')
        values = np.asarray(matches[0]['shares'], dtype=np.float32)
        if values.shape != (2,) or not np.isfinite(values).all() or ((values <= 0) | (values >= 1)).any():
            raise ValueError('Invalid saved TRAIN shares')
        a, b = values
        one = np.float32(1)
        matrices.append(np.asarray([[a, one-a, 0], [(one-b)*a, (one-b)*(one-a), b]], dtype=np.float32))
    return matrices


def task_blend(logits, reliable, matrices, stage):
    width = sum(TASK_SIZES[:stage+1])
    if logits.ndim != 3 or logits.shape[1:] != (3, width) or reliable.dtype != np.bool_ or reliable.shape != (len(logits),):
        raise ValueError('Unexpected seen-task logit layout')
    parts, start = [], 0
    for task in range(stage+1):
        stop = start+TASK_SIZES[task]
        w = matrices[task][reliable.astype(np.int8)]
        z = logits[:, :, start:stop]
        parts.append(w[:, 0, None]*z[:, 0]+w[:, 1, None]*z[:, 1]+w[:, 2, None]*z[:, 2])
        start = stop
    return np.concatenate(parts, axis=1)


def align_exact(ids, targets, other_ids, other_targets, other_logits):
    if len(set(ids)) != len(ids) or len(set(other_ids)) != len(other_ids):
        raise ValueError('Duplicate person identifiers')
    lookup = {str(s): i for i, s in enumerate(other_ids)}
    if not set(map(str, ids)).issubset(lookup):
        raise ValueError('Missing replay people')
    order = np.asarray([lookup[str(s)] for s in ids], dtype=np.int64)
    if not np.array_equal(targets, other_targets[order]):
        raise ValueError('Replay labels changed')
    return other_logits[order]
