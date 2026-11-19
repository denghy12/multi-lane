"""Fit a separate equal-initialized, constant fusion readout at each task's introduction."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, task_indices
from scripts.emotic.d9_sample_training import image_key
from scripts.emotic.d9_shrink_fit import load_train
from scripts.emotic.evaluate_d9_relation import write_json
from scripts.emotic.fit_d9_task10_probe import LR, fit_one, weights

REPLAY = (0, 3, 6)
PROTOCOL = dict(version=1, seed=1, tasks=list(range(len(TASK_SIZES))), steps=300, lr=LR,
                optimizer='Adam', schedule='constant', batch='full current-task fusion TRAIN',
                initialization='zero softmax logits', loss='BCEWithLogits mean over people and current classes',
                groups=['unreliable', 'reliable'], views=['full', 'person', 'face'],
                old_heads='seal at introduction', adjustment_limit=None)


def read(path):
    return json.loads(path.read_text())


def check_cohort(d, meta, cohort, roles, task):
    ids, y, valid = d['sample_ids'], d['targets'], d['reliable']
    images = {image_key(person) for person in ids}
    measured = dict(people=len(ids), images=len(images), sample_id_sha256=meta['sample_id_sha256'],
                    reliable_face=int(valid.sum()), positives=y.sum(0).astype(int).tolist(),
                    reliable_face_positives=y[valid].sum(0).astype(int).tolist(), classes=list(task_indices(task)))
    if any(cohort[k] != v for k, v in measured.items()) or any(roles[i] != 'fusion' for i in images):
        raise ValueError(f'Fitting cohort does not replay the TRAIN holdout allocation at task{task}')
    if not valid.any() or valid.all():
        raise ValueError(f'Empty Face availability group at task{task}')


def group_proportions(center):
    if np.shape(center) != (2, 3) or not np.isfinite(center).all():
        raise ValueError('Invalid fusion parameters')
    return weights(torch.tensor(center, dtype=torch.float32), torch.tensor([False, True])).tolist()


def load_centers(source, prior, out):
    lock = read(out / 'centers_lock.json')
    if (lock['protocol'] != PROTOCOL or set(lock['tasks']) != {str(t) for t in PROTOCOL['tasks']}
            or lock['source_config_sha256'] != sha256_file(source / 'config.json')
            or lock['source_roles_sha256'] != sha256_file(source / 'train_image_roles.json')
            or lock['prior_lock_sha256'] != sha256_file(prior / 'probe' / 'validation_lock.json')):
        raise ValueError('Incompatible set of sealed fusion heads')
    for t in PROTOCOL['tasks']:
        row = lock['tasks'][str(t)]
        sealed = out / f'task{t}_center.json'
        if (sha256_file(sealed) != lock['center_file_sha256'][str(t)] or read(sealed) != row
                or row['group_weights'] != group_proportions(row['center'])
                or row['cohort_sha256'] != sha256_file(source / 'cohorts' / f'task{t}.json')
                or row['checkpoint_sha256'] != sha256_file(source / 'compact_checkpoints' / f'task{t}.pth')):
            raise ValueError(f'Changed sealed fusion head, training cohort or checkpoint at task{t}')
    return lock


def fit(source, prior, features, out):
    torch.set_num_threads(1)
    cfg_path = source / 'config.json'
    cfg = read(cfg_path)
    if (cfg['mode'] != 'sample_holdout' or not cfg['protect_branches'] or cfg['smoke']
            or cfg['image_split'] != 'global_10pct_hash_rank' or cfg['seed'] != 1 or cfg['tasks'] != 8):
        raise ValueError('Expected the protected seed1 90/10 source')
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'centers_lock.json'
    prior_lock = prior / 'probe' / 'validation_lock.json'
    roles_path = source / 'train_image_roles.json'
    roles = read(roles_path)
    old = read(prior_lock)
    if old['protocol']['steps'] != PROTOCOL['steps'] or old['protocol']['LR'] != LR:
        raise ValueError('Prior center fitting budget changed')
    cfg_hash = sha256_file(cfg_path)
    lock = dict(protocol=PROTOCOL, source_config_sha256=cfg_hash, source_roles_sha256=sha256_file(roles_path),
                prior_lock_sha256=sha256_file(prior_lock), tasks={}, center_file_sha256={})
    existing = load_centers(source, prior, out) if path.exists() else None
    for task in PROTOCOL['tasks']:
        root = prior if task in REPLAY else features
        d, meta = load_train(root, task)
        cp = source / 'compact_checkpoints' / f'task{task}.pth'
        cohort_path = source / 'cohorts' / f'task{task}.json'
        cp_hash, cohort_hash = sha256_file(cp), sha256_file(cohort_path)
        proto = meta['protocol']
        if (proto['checkpoint_sha256'] != cp_hash or proto['source_cohort_or_archive_sha256'] != cohort_hash
                or proto['source_config_sha256'] != cfg_hash or proto['batch_size'] != 64):
            raise ValueError(f'Inconsistent TRAIN logits provenance at task{task}')
        cohort = read(cohort_path)['fusion']
        check_cohort(d, meta, cohort, roles, task)
        # Only the current task's labels, branch scores and Face availability enter the optimization.
        center = fit_one(d, None, PROTOCOL['steps']) if existing is None else existing['tasks'][str(task)]['center']
        row = dict(task=task, fit_classes=list(task_indices(task)), fusion_cohort=cohort,
                   checkpoint_sha256=cp_hash, cohort_sha256=cohort_hash, train_feature_sha256=meta['features_sha256'],
                   center=center, group_weights=group_proportions(center))
        if task in REPLAY:
            model = prior / 'probe' / f'task{task}_model.json'
            if (sha256_file(model) != old['artifact_hashes'][model.name]
                    or old['train_val_hash'][str(task)]['train'] != row['train_feature_sha256']
                    or center != read(model)['center']):
                raise ValueError(f'Cannot exactly replay prior A at task{task}')
            row['prior_model_sha256'] = old['artifact_hashes'][model.name]
        sealed = out / f'task{task}_center.json'
        if existing is None:
            write_json(sealed, row)
        elif row != existing['tasks'][str(task)]:
            raise ValueError(f'Different source for existing fit at task{task}')
        lock['tasks'][str(task)] = row
        lock['center_file_sha256'][str(task)] = sha256_file(sealed)
        print(f'task{task} TRAIN {cohort["people"]} people: {row["group_weights"]}', flush=True)
    if existing is None:
        write_json(path, lock)
    return load_centers(source, prior, out)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for option in ('source', 'prior', 'features', 'out'):
        cli.add_argument('--' + option, type=Path, required=True)
    args = cli.parse_args()
    fit(args.source, args.prior, args.features, args.out)
