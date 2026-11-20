"""Compare permanent 90/10 TRAIN image roles without changing the model seed."""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path

import numpy as np

from multi_lane.track_a.r1_crossvit_experiment import read, sha, write
from multi_lane.track_a.runner import CLASS_ORDER, TASK_SIZES, fit_calibration_indices, task_indices
from scripts.emotic.d9_sample_training import describe_cohort, global_image_roles, image_key


STRATEGIES = ('hash2', 'meta', 'online')


def quota(n):
    return (n + 5) // 10


def image_order(images, seed):
    salt = f'd9-split90-hash2-{seed}:'
    return sorted(images, key=lambda k: (hashlib.sha256((salt + k).encode()).digest(), k))


def metadata_holdout(source, people, ranked):
    strata = defaultdict(list)
    for image in ranked:
        rows = people[image]
        reliable = sum(bool(source.face_reliable[i]) for i in rows)
        face_group = 'some'
        if reliable == 0:
            face_group = 'none'
        elif reliable == len(rows):
            face_group = 'all'
        strata[(min(len(rows), 3), face_group)].append(image)
    counts = {k: len(group) // 10 for k, group in strata.items()}
    # Integer largest remainders; equal remainders break on the fixed stratum key.
    order = sorted(strata, key=lambda k: (-(len(strata[k]) % 10), k))
    for k in order[:quota(len(ranked)) - sum(counts.values())]:
        counts[k] += 1
    return {img for k, group in strata.items() for img in group[:counts[k]]}


def assign_task(roles, features, ranked):
    """Append only newly visible image roles, balancing counts for this task."""
    new = [img for img in ranked if img in features and img not in roles]
    if not new:
        return
    count = quota(len(roles) + len(new)) - sum(role == 'fusion' for role in roles.values())
    held = set()
    if count:
        # Columns: eligible people, reliable people, positive people per current
        # class, reliable positive people per current class. Count each person
        # once within each column even when that person has multiple labels.
        target = np.sum(list(features.values()), axis=0) * 0.1
        previous = sum((features[img] for img in features if roles.get(img) == 'fusion'), 0)
        scale = np.maximum(1.0, target)
        residual = (previous - target) / scale
        vectors = np.array([features[img] for img in new]) / scale
        norm_sq = np.einsum('ij,ij->i', vectors, vectors)
        taken = np.zeros(len(new), dtype=bool)
        for _ in range(count):
            # Change in squared (per-column normalized) holdout count error.
            change = norm_sq + 2 * np.einsum('ij,j->i', vectors, residual)
            change[taken] = np.inf
            index = int(change.argmin())
            taken[index] = True
            residual += vectors[index]
            held.add(new[index])
    roles.update((img, 'fusion' if img in held else 'branch') for img in new)


def online_roles(source, keys, ranked):
    roles = {}
    for t in range(len(TASK_SIZES)):
        classes = task_indices(t)
        features = {}
        for row, labels in enumerate(source.targets):
            current = [c in labels for c in classes]
            if not any(current):
                continue
            key = keys[row]
            if key not in features:
                features[key] = np.zeros(2 + 2 * len(classes), dtype=np.int64)
            counts = features[key]
            counts[0] += 1
            counts[2:2+len(classes)] += current
            if source.face_reliable[row]:
                counts[1] += 1
                counts[2+len(classes):] += current
        assign_task(roles, features, ranked)
    # Images without any eligible TRAIN person have no effect on training.
    unused = [img for img in ranked if img not in roles]
    remainder = quota(len(ranked)) - sum(role == 'fusion' for role in roles.values())
    roles.update((img, 'fusion' if j < remainder else 'branch') for j, img in enumerate(unused))
    return roles


def split_roles(source, mode, seed):
    if mode == 'baseline':
        return global_image_roles(source, 'sample_holdout', seed)
    keys = [image_key(x) for x in source.sample_ids]
    if len(set(source.sample_ids)) != len(keys):
        raise ValueError('Duplicate TRAIN person identifier')
    ranked = image_order(set(keys), seed)
    if not 0 < quota(len(ranked)) < len(ranked):
        raise ValueError('TRAIN is too small for two image roles')
    if mode == 'online':
        roles = online_roles(source, keys, ranked)
    else:
        if mode == 'meta':
            people = defaultdict(list)
            for i, key in enumerate(keys):
                people[key].append(i)
            held = metadata_holdout(source, people, ranked)
        elif mode == 'hash2':
            held = set(ranked[:quota(len(ranked))])
        else:
            raise ValueError(f'Unknown split strategy {mode!r}')
        roles = {k: 'fusion' if k in held else 'branch' for k in ranked}
    if len(roles) != len(ranked) or sum(v == 'fusion' for v in roles.values()) != quota(len(ranked)):
        raise ValueError('Incomplete image roles or incorrect holdout size')
    return {k: roles[k] for k in sorted(roles)}


def coverage(source, roles):
    keys = [image_key(id_) for id_ in source.sample_ids]
    tasks = {}
    for t in range(len(TASK_SIZES)):
        classes = task_indices(t)
        eligible, _ = fit_calibration_indices(source, classes, 0.)
        groups = {}
        for role in ('branch', 'fusion'):
            rows = [i for i in eligible if roles[keys[i]] == role]
            counts = describe_cohort(source, rows, classes)
            counts['positive_images'] = [len({keys[i] for i in rows if c in source.targets[i]}) for c in classes]
            counts['reliable_positive_images'] = [
                len({keys[i] for i in rows if source.face_reliable[i] and c in source.targets[i]}) for c in classes]
            groups[role] = counts
        tasks[str(t)] = groups
    return dict(original_images=dict(Counter(roles.values())), classes=CLASS_ORDER, tasks=tasks)


def publish(path, payload):
    if path.exists():
        if read(path) != payload:
            raise ValueError(f'Conflicting existing audit at {path}')
    else:
        write(path, payload)


def audit(out, seed):
    from multi_lane.track_a.d9_joint_data import build_joint_source
    from scripts.emotic.run_r1_frozen_cross_view import input_paths
    source = build_joint_source(input_paths(), 'train')
    summary = {}
    original_held = None
    for mode in ('baseline', *STRATEGIES):
        roles = split_roles(source, mode, seed)
        spec = coverage(source, roles)
        path = out/mode/'train_image_roles.json'
        publish(path, roles)
        spec['classes'] = list(CLASS_ORDER)
        spec['roles_sha256'] = sha(path)
        covpath = out/mode/'coverage.json'
        publish(covpath, spec)
        held = {img for img, role in roles.items() if role == 'fusion'}
        if original_held is None:
            original_held = held
        summary[mode] = dict(counts=spec['original_images'], roles_sha256=sha(path), coverage_sha256=sha(covpath),
                             held_overlap_with_baseline=len(original_held & held))
        print(mode, summary[mode], flush=True)
    publish(out/'audit.json', dict(seed=seed, data='train', splits=summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    audit(args.out, args.seed)
