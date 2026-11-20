"""Fit the two matched losses before and after adding eligible people from the same images."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_matched_natural_fit import checked_data
from scripts.emotic.d9_natural_rank_fit import fit_arm
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_paired_a_extract import ALL_KEYS, load_paired_archive, verify_pairs
from scripts.emotic.d9_quality_fit import LR, STEPS
from scripts.emotic.d9_sample_training import image_key


EXPECTED_EXTRA = (0, 58, 16, 70, 74, 91, 6, 53)
EXPECTED_RELIABLE = (0, 28, 13, 39, 44, 55, 4, 34)


def join_population(original, extra, previous_people):
    """Only old positives on the current TRAIN images can supply the additional negatives."""
    ids = original['sample_ids'].tolist()
    extra_ids = extra['sample_ids'].tolist()
    images = {image_key(person) for person in ids}
    eligible = {person for person in previous_people if image_key(person) in images}.difference(ids)
    if len(extra_ids) != len(set(extra_ids)) or set(extra_ids) != eligible:
        raise ValueError('Extra roster differs from previously seen people on current held TRAIN images')
    if extra['sample_ids'].shape != (len(eligible),):
        raise ValueError('Expected one identifier per additional person')
    for field in ('targets', 'reliable', 'branch_logits', 'person_branch_logits', 'face_branch_logits'):
        x, ref = extra[field], original[field]
        if x.shape != (len(eligible), *ref.shape[1:]) or not np.isfinite(x).all():
            raise ValueError(f'Malformed extra TRAIN field {field}')
        if field.endswith('logits') and x.dtype != ref.dtype:
            raise ValueError(f'Precision changed for extra field {field}')
    if extra['reliable'].dtype != np.bool_ or extra['targets'].any():
        raise ValueError('Expected reliable Face flags and explicit all-negative current targets')
    if any(previous_people[pid] != flag for pid, flag in zip(extra_ids, extra['reliable'])):
        raise ValueError('Face availability changed for an additional person')
    verify_pairs(extra, extra)
    return {k: np.concatenate([original[k], extra[k]], axis=0) for k in ALL_KEYS}


def fit(root, historical, extra, out, smoke=False):
    torch.set_num_threads(1)
    previous_lock = historical/'centers_lock.json'
    previous = read(previous_lock)
    source_lock = root/'paired_A'/'centers_lock.json'
    require_digest(source_lock, previous['source_centers_sha256'])
    require_digest(root/'config.json', previous['config_sha256'])
    roles_path = root/'train_image_roles.json'
    require_digest(roles_path, previous['roles_sha256'])
    roles, seals = read(roles_path), read(source_lock)
    p = previous['protocol']
    expected = dict(split='train', init='equal_available_views', steps=STEPS, lr=LR, seed=1,
                    optimizer='Adam', rank_coefficient=1., temperature=1., scale='raw fused logits')
    task_ids = {str(t) for t in range(len(TASK_SIZES))}
    if (previous['smoke'] or any(p.get(k) != v for k, v in expected.items())
            or set(previous['tasks']) != task_ids or set(seals) != task_ids):
        raise ValueError('Expected sealed full eight-task BCE and ranking reference')
    rows, previous_people, history_hashes = {}, {}, {}
    for task in range(2 if smoke else len(TASK_SIZES)):
        key = str(task)
        data, source, hashes, cohort = checked_data(root, task, seals.get(str(task-1)), seals[key], roles)
        old = previous['tasks'][key]
        identity = dict(task=task, archive_sha256=source['archive_sha256'], source_center_sha256=seals[key],
                        train_ids_sha256=cohort['sample_id_sha256'], people=cohort['people'],
                        reliable_face=cohort['reliable_face'], fit_classes=cohort['classes'], **hashes)
        if any(old[k] != v for k,v in identity.items()):
            raise ValueError(f'Task {task} TRAIN source differs from the sealed experiment')
        extra_path = extra/f'task{task}_extra.npz'
        meta_path = extra_path.with_suffix('.json')
        meta = read(meta_path)
        require_digest(extra_path, meta['archive_sha256'])
        added = load_paired_archive(extra_path)
        extra_ids = added['sample_ids'].tolist()
        id_hash = hashlib.sha256('\n'.join(extra_ids).encode()).hexdigest()
        expected_meta = dict(schema=1, task=task, split='train', classes=cohort['classes'],
                             current_archive_sha256=identity['archive_sha256'], history_archive_sha256=history_hashes,
                             sample_id_sha256=id_hash, people=len(extra_ids),
                             images=len({image_key(pid) for pid in extra_ids}),
                             reliable_face=int(added['reliable'].sum()), **hashes)
        if (any(meta[k] != v for k,v in expected_meta.items()) or len(extra_ids) != EXPECTED_EXTRA[task]
                or meta['reliable_face'] != EXPECTED_RELIABLE[task]):
            raise ValueError(f'Task {task} additional TRAIN roster or extraction provenance differs')
        expanded = join_population(data, added, previous_people)
        row = dict(**identity, extra_meta_sha256=sha256_file(meta_path), extra_archive_sha256=meta['archive_sha256'],
                   extra_people=len(extra_ids), extra_images=meta['images'], extra_reliable_face=meta['reliable_face'],
                   expanded_people=len(expanded['targets']), expanded_reliable_face=int(expanded['reliable'].sum()))
        for use_rank, control, candidate in [(False,'paired','expanded_bce'), (True,'rank','expanded_rank')]:
            row[control], counts = fit_arm(data, use_rank)
            if row[control] != old[control] or counts != old['class_counts']:
                raise ValueError(f'Task {task} {control} failed to replay the historical fit')
            row[candidate], expanded_counts = fit_arm(expanded, use_rank)
            if task == 0 and (row[candidate] != row[control] or expanded_counts != counts):
                raise ValueError('Task 0 should reproduce the corresponding control exactly')
            row.update(class_counts=counts, expanded_class_counts=expanded_counts)
        rows[key] = row
        previous_people.update(zip(data['sample_ids'].tolist(), data['reliable'].tolist()))
        history_hashes[key] = identity['archive_sha256']
        print(f"task{task} TRAIN {row['people']} + {row['extra_people']} same-image negatives", flush=True)

    protocol = dict(p, comparison='original current positive people vs. same held images plus old-class people',
                    rank='natural input, within each arm use all same-class positive-negative pairs from both groups',
                    variants=['natural','person_112','reliable_face_112'])
    out.mkdir(parents=True, exist_ok=True)
    write_verified(out/'centers_lock.json', dict(tasks=rows, protocol=protocol, smoke=smoke,
                   config_sha256=previous['config_sha256'], roles_sha256=previous['roles_sha256'],
                   source_centers_sha256=previous['source_centers_sha256'],
                   historical_centers_sha256=sha256_file(previous_lock)))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for option in ('root', 'historical', 'extra', 'out'):
        cli.add_argument('--'+option, required=True, type=Path)
    cli.add_argument('--smoke', action='store_true')
    fit(**vars(cli.parse_args()))
