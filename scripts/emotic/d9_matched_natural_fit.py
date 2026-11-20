"""Fit natural-only and quality-paired task A from a single equal-update TRAIN trajectory."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, task_indices
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_a_eight_fit import check_cohort, group_proportions
from scripts.emotic.d9_equal_updates_paired import provenance
from scripts.emotic.d9_matched_run import fit_average
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_paired_a_extract import load_paired_archive, verify_pairs
from scripts.emotic.d9_quality_fit import LR, STEPS, _fit_one
from scripts.emotic.fit_d9_task10_probe import fit_one


def fit_natural(data):
    """Run the paired objective with exactly one contributing version for every person."""
    valid = torch.from_numpy(data['reliable'])
    logits = torch.from_numpy(data['branch_logits']).float().unsqueeze(1)
    labels = torch.from_numpy(data['targets']).float().unsqueeze(1)
    usable = torch.ones((len(valid), 1), dtype=torch.bool)
    no_features = torch.empty(len(valid), 1, 0)
    params, diag, _ = _fit_one(no_features, logits, labels, valid, usable, 'A', [])
    center = params['bias']
    learned = group_proportions(center)
    # The historical natural-only solver applies the same BCE with a different sum order.
    historical = group_proportions(fit_one(data, None, STEPS))
    np.testing.assert_allclose(learned, historical, rtol=0, atol=1e-5)
    difference = float(np.abs(np.subtract(learned, historical)).max())
    return dict(center=center, weights=learned, fit=diag, historical_solver_max_weight_delta=difference)


def checked_data(root, task, previous_sha, center_sha, roles):
    paired = root/'paired_A'
    archived = paired/f'task{task}_center.json'
    require_digest(archived, center_sha)
    original = read(archived)
    hashes = provenance(root, task)
    classes = list(task_indices(task))
    expected = dict(split='train', task=task, smoke=False, init='equal_available_views', classes=classes,
                    optimizer='adam', steps=STEPS, lr=LR, previous_center_sha256=previous_sha,
                    variants=['natural', 'person_112', 'reliable_face_112'], **hashes)
    cohort = read(root/'cohorts'/f'task{task}.json')['fusion']
    if (original['protocol'] != expected or original['fusion_cohort'] != cohort
            or original['checkpoint_sha256'] != hashes['checkpoint_sha256']):
        raise ValueError(f'Invalid paired A provenance at task {task}')
    path = paired/f'task{task}_held.npz'
    require_digest(path, original['archive_sha256'])
    data = load_paired_archive(path)
    ids, y, valid, logits = (data[k] for k in ('sample_ids', 'targets', 'reliable', 'branch_logits'))
    if (ids.shape != (len(ids),) or len(np.unique(ids)) != len(ids)
            or valid.dtype != np.bool_ or valid.shape != ids.shape or y.shape != (len(ids), TASK_SIZES[task])
            or not np.isin(y, [0, 1]).all() or not y.any(axis=1).all()
            or logits.shape != (*y.shape, 3) or not np.isfinite(logits).all()):
        raise ValueError(f'Invalid original TRAIN subjects, labels or logits at task {task}')
    id_hash = hashlib.sha256('\n'.join(ids.tolist()).encode()).hexdigest()
    check_cohort(data, dict(sample_id_sha256=id_hash), cohort, roles, task)
    verify_pairs(data, data)
    return data, original, hashes, cohort


def fit(root, out):
    torch.set_num_threads(1)
    config = read(root/'config.json')
    expected = dict(mode='sample_holdout', seed=1, epochs=30, tasks=len(TASK_SIZES), batch_size=64,
                    smoke=False, protect_branches=True, image_split='global_10pct_hash_rank')
    if (any(config.get(k) != v for k, v in expected.items())
            or config.get('equal_updates', {}).get('mode') != 'cycle_90pct_to_full_epoch_size'):
        raise ValueError('Expected equal-step formal 90% protected Frozen9')
    roles_path = root/'train_image_roles.json'
    roles = read(roles_path)
    require_digest(roles_path, config['equal_updates']['roles_sha256'])
    source_seal = root/'paired_A'/'centers_lock.json'
    original_seals = read(source_seal)
    if set(original_seals) != {str(t) for t in range(len(TASK_SIZES))}:
        raise ValueError('Incomplete paired TRAIN center chain')
    tasks = {}
    for t in range(len(TASK_SIZES)):
        data, original, hashes, cohort = checked_data(root, t, original_seals.get(str(t-1)), original_seals[str(t)], roles)
        variants = {v: {'branch_logits': data[v+'_branch_logits']} for v in ('person', 'face')}
        natural = fit_natural(data)
        result = fit_average(data, variants)
        np.testing.assert_array_equal(result['params']['bias'], original['center'])
        np.testing.assert_array_equal(result['weights'], original['group_weights'])
        paired = dict(center=result['params']['bias'], weights=result['weights'], fit=result['fit'])
        tasks[str(t)] = dict(task=t, natural=natural, paired=paired, fit_classes=cohort['classes'],
                             people=cohort['people'], reliable_face=cohort['reliable_face'],
                             train_ids_sha256=cohort['sample_id_sha256'], archive_sha256=original['archive_sha256'],
                             source_center_sha256=original_seals[str(t)], **hashes)
        print(f'task{t} TRAIN people={cohort["people"]}, paired replayed', flush=True)
    protocol = dict(split='train', seed=1, tasks=list(range(len(TASK_SIZES))), steps=STEPS, lr=LR,
                    optimizer='Adam', init='equal_available_views', loss='per-person per-version mean BCE',
                    variants=dict(natural=['natural'], paired=['natural', 'person_112', 'reliable_face_112']))
    out.mkdir(parents=True, exist_ok=True)
    write_verified(out/'centers_lock.json', dict(protocol=protocol, tasks=tasks,
                   config_sha256=sha256_file(root/'config.json'), roles_sha256=sha256_file(roles_path),
                   source_centers_sha256=sha256_file(source_seal)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'out'):
        parser.add_argument('--' + key, required=True, type=Path)
    fit(**vars(parser.parse_args()))
