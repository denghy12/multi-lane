"""Score current-negative, previously labelled TRAIN people on current held-out images."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, set_seed, task_indices
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_matched_extract import collect
from scripts.emotic.d9_matched_natural_fit import checked_data
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_paired_a_extract import ALL_KEYS, load_paired_archive, verify_pairs
from scripts.emotic.d9_sample_training import fusion_source, global_image_roles, image_key
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths


EXPECTED_PEOPLE = (0, 58, 16, 70, 74, 91, 6, 53)
EXPECTED_RELIABLE = (0, 28, 13, 39, 44, 55, 4, 34)


def extra_ids(current, previous):
    """Use only identities visible up to the current task. Each person appears once."""
    current_ids = set(current)
    images = {image_key(i) for i in current_ids}
    return sorted(i for i in set(previous) - current_ids if image_key(i) in images)


def verify_extra(data, ids, flags, classes):
    np.testing.assert_array_equal(data['sample_ids'], ids)
    np.testing.assert_array_equal(data['reliable'], flags)
    labels = data['targets']
    if labels.shape != (len(ids), len(classes)) or (labels != 0).any():
        raise ValueError('Extra TRAIN people must have zero targets on every current class')
    z = data['branch_logits']
    if z.shape != (*labels.shape, 3) or not np.isfinite(z).all():
        raise ValueError('Invalid natural branch logits')
    verify_pairs(data, data)


def save_archive(path, data):
    if path.exists():
        prior = load_paired_archive(path)
        for k in ALL_KEYS:
            np.testing.assert_array_equal(prior[k], data[k])
        return
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **data)
    tmp.replace(path)


def extract(root, out, tasks, workers):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    set_seed(1, True)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    cfg = read(root/'config.json')
    expected = dict(seed=1, tasks=len(TASK_SIZES), epochs=30, batch_size=64, mode='sample_holdout',
                    smoke=False, protect_branches=True, image_split='global_10pct_hash_rank')
    if (any(cfg.get(k) != v for k, v in expected.items()) or
            cfg['equal_updates']['mode'] != 'cycle_90pct_to_full_epoch_size'):
        raise ValueError('Expected original random equal-update TRAIN trajectory')
    roles_path = root/'train_image_roles.json'
    require_digest(roles_path, cfg['equal_updates']['roles_sha256'])
    roles = read(roles_path)
    train = fusion_source(build_joint_source(input_paths(), 'train'))
    if global_image_roles(train, 'sample_holdout', 1) != roles:
        raise ValueError('TRAIN image roles do not match the original split')
    by_id = {str(person): i for i, person in enumerate(train.sample_ids)}
    if len(by_id) != len(train.sample_ids):
        raise ValueError('TRAIN person IDs must be unique')
    seals = read(root/'paired_A'/'centers_lock.json')
    old_people = set()
    old_hashes = {}
    out.mkdir(parents=True, exist_ok=True)
    for task in range(max(tasks) + 1):
        key = str(task)
        data, original, hashes, _ = checked_data(root, task, seals.get(str(task-1)), seals[key], roles)
        current = data['sample_ids'].tolist()
        if task in tasks:
            ids = extra_ids(current, old_people)
            indices = [by_id[i] for i in ids]
            classes = task_indices(task)
            flags = np.asarray([bool(train.face_reliable[i]) for i in indices], dtype=bool)
            if (len(ids), int(flags.sum())) != (EXPECTED_PEOPLE[task], EXPECTED_RELIABLE[task]):
                raise ValueError(f'Unexpected extra population at task{task}: {len(ids)} people, {flags.sum()} reliable')
            if any(roles[image_key(i)] != 'fusion' for i in ids):
                raise ValueError('Extra person outside held TRAIN image role')
            for i in indices:
                labels = train.targets[i]
                if any(c in labels for c in classes) or not any(c in labels for c in range(classes[0])):
                    raise ValueError(f'Extra person fails old-positive/current-negative condition at task{task}')
            path = out/f'task{task}_extra.npz'
            meta = path.with_suffix('.json')
            spec = dict(schema=1, split='train', task=task, classes=list(classes), people=len(ids),
                        reliable_face=int(flags.sum()), images=len({image_key(i) for i in ids}),
                        sample_id_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                        source_archive_sha256=original['archive_sha256'], current_archive_sha256=original['archive_sha256'],
                        history_archive_sha256=dict(old_hashes), input_size=224, reduced_size=112, batch_size=64, **hashes)
            if meta.exists():
                prior = read(meta)
                require_digest(path, prior['archive_sha256'])
                if prior != dict(**spec, archive_sha256=prior['archive_sha256']):
                    raise ValueError(f'Existing extractor provenance differs: {meta}')
                pairs = load_paired_archive(path)
            elif indices:
                model = load_evaluation_model(root, task).requires_grad_(False).eval()
                raw = collect(model, train, indices, task, workers, None)
                pairs = {k: raw[k] for k in ALL_KEYS}
                del model
            else:
                pairs = {k: data[k][:0] for k in ALL_KEYS}
            verify_extra(pairs, np.asarray(ids, dtype=str), flags, classes)
            if not meta.exists():
                save_archive(path, pairs)
                write_verified(meta, dict(**spec, archive_sha256=sha256_file(path)))
            print(f'task{task} same-image TRAIN negatives: {len(ids)} people, {flags.sum()} reliable', flush=True)
        old_people.update(current)
        old_hashes[key] = original['archive_sha256']


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'out'):
        cli.add_argument('--'+name, type=Path, required=True)
    cli.add_argument('--tasks', type=int, nargs='+', choices=range(len(TASK_SIZES)), default=list(range(len(TASK_SIZES))))
    cli.add_argument('--workers', type=int, default=2)
    extract(**vars(cli.parse_args()))
