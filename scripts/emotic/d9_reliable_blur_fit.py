"""Fit the two reliable-Face blur factors on the original paired TRAIN archives."""
import argparse
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_a_eight_fit import group_proportions
from scripts.emotic.d9_matched_natural_fit import checked_data
from scripts.emotic.d9_matched_run import fit_average
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_quality_fit import LR, STEPS


ARMS = ('natural', 'person', 'face', 'paired')


def variants_for(data, arm):
    """Keep N/P for missing faces. For reliable faces substitute N into each disabled slot."""
    if arm not in ARMS:
        raise ValueError(f'Unknown arm: {arm}')
    reliable = data['reliable']
    natural = data['branch_logits']
    variants = {}
    for view in ('person', 'face'):
        z = data[view + '_branch_logits']
        if arm not in (view, 'paired'):
            z = z.copy()
            z[reliable] = natural[reliable]
        variants[view] = {'branch_logits': z}
    return variants


def fit_arm(data, arm, paired, natural):
    result = fit_average(data, variants_for(data, arm))
    center = result['params']['bias']
    weights = result['weights']
    # TEST deployment uses the same missing-Face row for every arm. The independent
    # bias row's gradient also replays on TRAIN when the reliable slots are changed.
    np.testing.assert_array_equal(center[0], paired['center'][0])
    np.testing.assert_array_equal(weights[0], paired['weights'][0])
    if arm == 'paired':
        np.testing.assert_array_equal(center, paired['center'])
        np.testing.assert_array_equal(weights, paired['weights'])
    if arm == 'natural':
        # Three identical natural slots and the historical single natural slot use
        # slightly different summation orders. Use the sealed natural row for evaluation.
        np.testing.assert_allclose(weights[1], natural['weights'][1], atol=1e-5, rtol=0)
        center[1] = natural['center'][1]
        weights[1] = natural['weights'][1]
    center[0] = paired['center'][0]
    weights[0] = paired['weights'][0]
    np.testing.assert_array_equal(group_proportions(center), weights)
    return dict(center=center, weights=weights, fit=result['fit'])


def fit(root, historical, out, smoke=False):
    torch.set_num_threads(1)
    config_path = root/'config.json'
    cfg = read(config_path)
    expected = dict(mode='sample_holdout', seed=1, epochs=30, tasks=len(TASK_SIZES), batch_size=64,
                    smoke=False, protect_branches=True, image_split='global_10pct_hash_rank')
    if (any(cfg.get(k) != v for k, v in expected.items())
            or cfg.get('equal_updates', {}).get('mode') != 'cycle_90pct_to_full_epoch_size'):
        raise ValueError('Expected the original protected, matched-update 90% trajectory')
    roles_path = root/'train_image_roles.json'
    require_digest(roles_path, cfg['equal_updates']['roles_sha256'])
    roles = read(roles_path)
    paired_lock = root/'paired_A'/'centers_lock.json'
    seals = read(paired_lock)
    historical_path = historical/'centers_lock.json'
    old = read(historical_path)
    keys = {str(t) for t in range(len(TASK_SIZES))}
    if (set(seals) != keys or set(old['tasks']) != keys or old['protocol']['split'] != 'train'
            or old['protocol']['init'] != 'equal_available_views'
            or old['protocol']['steps'] != STEPS or old['protocol']['lr'] != LR):
        raise ValueError('Missing or incompatible TRAIN endpoint fits')
    require_digest(config_path, old['config_sha256'])
    require_digest(roles_path, old['roles_sha256'])
    require_digest(paired_lock, old['source_centers_sha256'])
    tasks = {}
    for t in range(1 if smoke else len(TASK_SIZES)):
        key = str(t)
        data, original, hashes, cohort = checked_data(root, t, seals.get(str(t-1)), seals[key], roles)
        ref = old['tasks'][key]
        identity = dict(task=t, source_center_sha256=seals[key], archive_sha256=original['archive_sha256'],
                        train_ids_sha256=cohort['sample_id_sha256'], people=cohort['people'],
                        reliable_face=cohort['reliable_face'], fit_classes=cohort['classes'], **hashes)
        if any(ref[k] != v for k, v in identity.items()):
            raise ValueError(f'Task {t} has a different historical TRAIN roster or checkpoint')
        np.testing.assert_array_equal(ref['paired']['center'], original['center'])
        np.testing.assert_array_equal(ref['paired']['weights'], original['group_weights'])
        fitted = {m:fit_arm(data, m, ref['paired'], ref['natural']) for m in ARMS}
        tasks[key] = dict(**identity, **fitted)
        print(f"task{t} TRAIN {cohort['people']} people, {cohort['reliable_face']} reliable", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_verified(out/'centers_lock.json', dict(tasks=tasks, smoke=smoke, config_sha256=old['config_sha256'],
                   roles_sha256=old['roles_sha256'], source_centers_sha256=old['source_centers_sha256'],
                   historical_centers_sha256=sha256_file(historical_path),
                   protocol=dict(split='train', seed=1, optimizer='Adam', steps=STEPS, lr=LR,
                                 init='equal_available_views', missing_face='sealed original paired row',
                                 reliable_slots=['natural', 'person_112_or_natural', 'face_112_or_natural'],
                                 loss='mean BCE over classes, then valid versions, then people')))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'historical', 'out'):
        cli.add_argument('--'+name, required=True, type=Path)
    cli.add_argument('--smoke', action='store_true')
    fit(**vars(cli.parse_args()))
