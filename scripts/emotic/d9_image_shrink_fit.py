"""Choose image-weight shrinkage using only cross-fitted, chronologically available TRAIN predictions."""
import argparse
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, average_precision, task_indices
from scripts.emotic.d9_a_eight_eval import read
from scripts.emotic.d9_image_gate_readout import (
    METHODS, fit_gate, fused, load_stage, probabilities, save_weights, standardized_features,
)
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_fill_image_source import assert_train_roles

CANDIDATES = ('full', 'full_relation')
GRID = tuple(i / 20 for i in range(21))
FOLDS = 3
FOLD_SALT = 'd9-image-shrink-seed1-v1:'


def assign_folds(image_ids):
    return np.asarray([int.from_bytes(sha256((FOLD_SALT + str(i)).encode()).digest(), 'big') % FOLDS
                       for i in image_ids], dtype=np.int64)


def subset_images(stage, chosen):
    """Keep a set of complete images, preserving the native person order within the evaluation fold."""
    ix = torch.as_tensor(np.flatnonzero(chosen))
    if not len(ix):
        raise ValueError('TRAIN fold has no images')
    remap = torch.full((len(chosen),), -1, dtype=torch.long)
    remap[ix] = torch.arange(len(ix))
    new_index = remap[stage.inverse]
    keep = new_index >= 0
    return replace(stage, image_ids=stage.image_ids[chosen], full=stage.full[ix],
                   relation=stage.relation[ix], usable=stage.usable[ix], counts=stage.counts[chosen],
                   reliable_counts=stage.reliable_counts[chosen], inverse=new_index[keep],
                   observed=stage.observed[ix], reliable=stage.reliable[keep], logits=stage.logits[keep],
                   targets=stage.targets[keep])


def fold_weights(stage, held, normalization, previous):
    train = subset_images(stage, ~held)
    test = subset_images(stage, held)
    train_x, normalization = standardized_features(train, normalization)
    held_x, _ = standardized_features(test, normalization)
    gates, weights = {}, {}
    for name in METHODS:
        gates[name] = fit_gate(train, train_x[name], previous.get(name), 300)
        matrix = torch.tensor(gates[name]['matrix']).reshape(held_x[name].shape[1], 3)
        weights[name] = probabilities(torch.tensor(gates[name]['bias']), matrix, held_x[name], test.usable)
    return test, weights, normalization, gates


def fold_curve(stage, mean_weights, image_weights):
    y = stage.targets.numpy()
    pos = np.sum(y, axis=0, dtype=np.int64)
    valid = (pos > 0) & (pos < len(y))
    curve = np.full((len(GRID), len(pos)), np.nan, dtype=np.float64)
    for i, amount in enumerate(GRID):
        weights = (1-amount)*mean_weights + amount*image_weights
        scores = fused(stage.logits, weights, stage.inverse).sigmoid().numpy()
        for c in np.flatnonzero(valid):
            curve[i,c] = 100. * average_precision(scores[:,c], y[:,c])
    return curve


def task_curve(fold_curves):
    """Average independent fold APs per valid class, then give each evaluable class equal weight."""
    stacked = np.stack(fold_curves)
    counts = np.isfinite(stacked[:,0,:]).sum(0)
    keep = counts > 0
    if not keep.any():
        return None
    per_class = np.nansum(stacked[:,:,keep], axis=0)/counts[keep]
    return per_class.mean(axis=1)


def choose(prefix):
    valid = [c for c in prefix if c is not None]
    if not valid:
        return 0., None
    running = np.mean(valid, axis=0)
    # Grid is ascending; argmax returns the first maximum.
    return GRID[int(running.argmax())], running.tolist()


def run(source, image_run, out, smoke):
    gate_path = image_run/'readout'/'fit.json'
    initial = read(gate_path)
    n_tasks = 1 if smoke else len(TASK_SIZES)
    if initial['smoke'] != smoke or len(initial['states']) != n_tasks:
        raise ValueError('Unexpected original image-gate protocol')
    norm = [None] * FOLDS
    prev = [{} for _ in range(FOLDS)]
    history = {m:[] for m in CANDIDATES}
    states = []
    out.mkdir(parents=True,exist_ok=True)
    for t in range(n_tasks):
        stage = load_stage(source, image_run/'features', 'train', t, smoke)
        ref = initial['states'][t]
        if (ref['task'] != t or any(ref[k] != getattr(stage,k) for k in
                                     ('archive_sha256','features_sha256','checkpoint_sha256'))):
            raise ValueError(f'Unexpected TRAIN archive or image role at task {t}')
        assert_train_roles(source, stage.image_ids)
        ids = assign_folds(stage.image_ids)
        all_weights = {m:torch.full((len(ids),3), float('nan')) for m in METHODS}
        curves = {m:[] for m in CANDIDATES}
        folds = []
        for f in range(FOLDS):
            held = ids == f
            test, weights, norm[f], prev[f] = fold_weights(stage,held,norm[f],prev[f])
            pos = test.targets.long().sum(dim=0)
            neg = len(test.targets) - pos
            folds.append(dict(fold=f, fit_images=int((~held).sum()), held_images=len(test.image_ids),
                              held_people=len(test.targets), positives=pos.tolist(), negatives=neg.tolist(),
                              valid_classes=(pos.gt(0) & neg.gt(0)).tolist()))
            for m,w in weights.items():
                all_weights[m][torch.from_numpy(held)] = w
            for m in CANDIDATES:
                curves[m].append(fold_curve(test, weights['constant'], weights[m]))
        if any(not torch.isfinite(w).all() for w in all_weights.values()):
            raise ValueError(f'Missing out-of-fold weight, task {t}')
        weights_path = out/'oof'/f'task{t}.npz'
        save_weights(weights_path, stage, dict(fold=torch.from_numpy(ids), **all_weights))
        amounts, current, cumulative = {}, {}, {}
        for m in CANDIDATES:
            value = task_curve(curves[m])
            history[m].append(value)
            current[m] = None if value is None else value.tolist()
            amounts[m], cumulative[m] = choose(history[m])
        state = dict(task=t, classes=list(task_indices(t)), images=len(ids), people=len(stage.targets),
                     folds=folds, current_curve=current, prefix_curve=cumulative,
                     oof_weights_sha256=sha256_file(weights_path), archive_sha256=stage.archive_sha256,
                     checkpoint_sha256=stage.checkpoint_sha256, features_sha256=stage.features_sha256)
        state['lambda'] = amounts
        states.append(state)
        print(f'task{t} OOF images={len(ids)} lambda={amounts}', flush=True)
    write_verified(out/'fit.json', dict(smoke=smoke, gate_fit_sha256=sha256_file(gate_path),
                   fold_salt=FOLD_SALT, folds=FOLDS, grid=list(GRID), states=states,
                   protocol='TRAIN original-image SHA256 folds across tasks; fold-specific task0 normalization; '
                            'fold-specific incremental Adam/BCE 300 steps per task; omit folds with absent positives '
                            'or negatives for each class; mean AP over valid folds, classes and seen stages; '
                            'first maximum; both masks and all heads share one scalar per method at a stage'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source','image-run','out'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--smoke',action='store_true')
    torch.set_num_threads(2)
    run(**vars(p.parse_args()))
