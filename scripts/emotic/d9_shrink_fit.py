"""Choose a single A/D shrinkage weight by original-image cross-fitting within TRAIN."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_sample_training import image_key
from scripts.emotic.evaluate_d9_relation import write_json
from scripts.emotic.fit_d9_task10_probe import (
    LR, PROJECTION_SEED, STD_FLOOR, apply_rule, build_features, fit_one, load_arrays, mix, scores,
)

TASKS = (0, 3, 6)
STEPS = 300
FOLDS = 3
SALT = 'task10-shrink-v1:1:'
GRID = [i / 100 for i in range(101)]
PROTOCOL = dict(version=1, folds=FOLDS, fold_salt=SALT, steps=STEPS, lr=LR,
                grid=GRID, projection_seed=PROJECTION_SEED, std_floor=STD_FLOOR,
                stratification='Across-task union of original TRAIN images by hash rank, round robin',
                objective='Mean of 3 per-task current-head mAP on pooled out-of-fold rows',
                tie_break='Smallest lambda at maximum unrounded score')


def image_folds(ids_by_task):
    images = {image_key(person) for ids in ids_by_task.values() for person in ids}
    ranked = sorted(images, key=lambda key: (hashlib.sha256((SALT + key).encode()).digest(), key))
    return {image: rank % FOLDS for rank, image in enumerate(ranked)}


def load_train(source, task):
    file = source / 'features' / f'task{task}' / 'train.npz'
    meta = json.loads(file.with_suffix('.json').read_text())
    protocol = meta['protocol']
    if (protocol['task'] != task or protocol['split'] != 'train' or protocol['smoke']
            or protocol['schema'] != 2 or not meta['visibility_mask_applied']
            or sha256_file(file) != meta['features_sha256']):
        raise ValueError(f'Task {task} cache is not the verified formal TRAIN extraction')
    d = load_arrays(file, task)
    ids = d['sample_ids']
    if (len(ids) != meta['people'] or d['reliable'].sum() != meta['reliable_face']
            or hashlib.sha256('\n'.join(ids).encode()).hexdigest() != meta['sample_id_sha256']
            or not d['targets'].any(axis=1).all()):
        raise ValueError(f'Task {task} TRAIN cohort differs')
    return d, meta


def fold_coverage(d):
    valid = d['reliable']
    ids = d['sample_ids']
    return dict(people=len(ids), images=len({image_key(person) for person in ids}),
                reliable_face=int(valid.sum()), positives=d['targets'].sum(0).astype(int).tolist(),
                reliable_positives=d['targets'][valid].sum(0).astype(int).tolist(),
                sample_id_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())


def cross_fit(d, assignment, steps):
    fold = np.array([assignment[image_key(person)] for person in d['sample_ids']], dtype=np.int32)
    a = torch.full((len(fold), 3), torch.nan)
    D = a.clone()
    coverage = []
    for k in range(FOLDS):
        held = fold == k
        if not held.any() or held.all():
            raise ValueError(f'Empty fitting or held-out cohort for fold {k}')
        train = {name: value[~held] for name, value in d.items()}
        query = {name: value[held] for name, value in d.items()}
        x_fit, _, spec = build_features(train)
        center = fit_one(train, None, steps)
        matrix = fit_one(train, x_fit['D'], steps, center)
        model = dict(center=center, matrices={'D': matrix})
        x_eval, _, _ = build_features(query, spec)
        a[held] = apply_rule('A', query, x_eval, model)
        D[held] = apply_rule('D', query, x_eval, model)
        coverage.append({'fold': k, 'fit': fold_coverage(train), 'held': fold_coverage(query)})
    for w in (a, D):
        if not torch.isfinite(w).all() or (w < 0).any() or w[~d['reliable'], 2].count_nonzero():
            raise ValueError('Invalid out-of-fold view weights')
        torch.testing.assert_close(w.sum(1), torch.ones(len(w)), atol=1e-6, rtol=0)
    return fold, a, D, coverage


def interpolate(a, d, fraction):
    if fraction == 0:
        return a
    if fraction == 1:
        return d
    return (1 - fraction) * a + fraction * d


def select(curves):
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 2 or curves.shape[1] != len(GRID) or not np.isfinite(curves).all():
        raise ValueError('Expected complete finite per-task grid results')
    macro = curves.mean(axis=0)
    return int(np.argmax(macro)), macro.tolist()


def artifact_names():
    return ['image_folds.json', *(f'task{t}_oof.npz' for t in TASKS)]


def read_lock(source, out):
    source, out = Path(source), Path(out)
    lock = json.loads((out / 'lambda_lock.json').read_text())
    if lock['protocol'] != PROTOCOL or lock['tasks'] != list(TASKS):
        raise ValueError('Incompatible lambda-lock protocol')
    for t in TASKS:
        p = source / 'features' / f'task{t}' / 'train.npz'
        if (sha256_file(p) != lock['source_hashes'][str(t)]
                or sha256_file(p.with_suffix('.json')) != lock['train_meta_hashes'][str(t)]):
            raise ValueError(f'TRAIN source changed for task{t}')
    for name in artifact_names():
        if sha256_file(out / name) != lock['artifact_hashes'][name]:
            raise ValueError(f'Locked artifact changed: {name}')
    idx, macro = select([lock['task_map_curves'][str(t)] for t in TASKS])
    if idx != lock['lambda_index'] or GRID[idx] != lock['lambda'] or macro != lock['macro_map_curve']:
        raise ValueError('Inconsistent TRAIN lambda selection')
    return lock


def run(source, out):
    torch.set_num_threads(1)
    if (out / 'lambda_lock.json').is_file():
        return read_lock(source, out)
    out.mkdir(parents=True, exist_ok=True)
    source_hashes, meta_hashes, ids, arrays = {}, {}, {}, {}
    for t in TASKS:
        arrays[t], meta = load_train(source, t)
        ids[t] = arrays[t]['sample_ids']
        source_hashes[str(t)] = meta['features_sha256']
        meta_hashes[str(t)] = sha256_file(source / 'features' / f'task{t}' / 'train.json')
    assigned = image_folds(ids)
    write_json(out / 'image_folds.json', assigned)
    oof, curves = {}, {}
    for t, d in arrays.items():
        fold, a, D, coverage = cross_fit(d, assigned, STEPS)
        z = torch.from_numpy(d['branch_logits']).float()
        endpoints = [scores(mix(w, z), d['targets']) for w in (a, D)]
        curve = [scores(mix(interpolate(a, D, x), z), d['targets'])['mAP'] for x in GRID]
        if (curve[0], curve[-1]) != tuple(m['mAP'] for m in endpoints):
            raise ValueError(f'Task {t} interpolation changed an endpoint')
        curves[str(t)] = curve
        fold_curves = []
        for k in range(FOLDS):
            held = fold == k
            fold_curves.append([scores(mix(interpolate(a[held], D[held], x), z[held]), d['targets'][held])['mAP']
                                for x in GRID])
        oof[str(t)] = dict(folds=coverage, A=endpoints[0], D=endpoints[1], held_fold_map_curves=fold_curves)
        dest = out / f'task{t}_oof.npz'
        tmp = dest.with_suffix(f'.{os.getpid()}.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, sample_ids=d['sample_ids'], targets=d['targets'], reliable=d['reliable'],
                                branch_logits=d['branch_logits'], fold=fold, A_weights=a.numpy(), D_weights=D.numpy())
        tmp.replace(dest)
    best, macro = select([curves[str(t)] for t in TASKS])
    lock = dict(protocol=PROTOCOL, tasks=list(TASKS), source_hashes=source_hashes, train_meta_hashes=meta_hashes,
                artifact_hashes={name: sha256_file(out / name) for name in artifact_names()},
                lambda_index=best, task_map_curves=curves, macro_map_curve=macro, oof=oof)
    lock['lambda'] = GRID[best]
    lock['within_0p01_map_of_max'] = [x for x, y in zip(GRID, macro) if y >= macro[best] - .01]
    write_json(out / 'lambda_lock.json', lock)
    return read_lock(source, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    chosen = run(args.source, args.out)
    print(f"Locked shared TRAIN lambda={chosen['lambda']:.2f}, TRAIN OOF macro mAP="
          f"{chosen['macro_map_curve'][chosen['lambda_index']]:.4f}", flush=True)
