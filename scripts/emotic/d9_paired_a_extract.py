"""Collect paired held-out TRAIN logits for eight frozen task heads."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import set_seed
from scripts.emotic.d9_a_eight_fit import REPLAY, check_cohort, load_centers
from scripts.emotic.d9_quality_extract import collect
from scripts.emotic.d9_sample_training import fusion_source, global_image_roles
from scripts.emotic.d9_shrink_fit import load_train
from scripts.emotic.evaluate_d9_relation import write_json
from scripts.emotic.extract_d9_task10_probe import read_json, train_loader
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths


NATURAL_KEYS = ('sample_ids', 'targets', 'reliable', 'branch_logits')
VIEWS = ('person', 'face')
ALL_KEYS = (*NATURAL_KEYS, 'person_branch_logits', 'face_branch_logits')


def verify_pairs(data, reference):
    for key in NATURAL_KEYS:
        np.testing.assert_array_equal(data[key], reference[key], err_msg=f'Natural TRAIN replay: {key}')
    logits = data['branch_logits']
    unavailable = ~data['reliable']
    for view in VIEWS:
        altered = data[f'{view}_branch_logits']
        if altered.shape != logits.shape or not np.isfinite(altered).all():
            raise ValueError(f'Malformed {view} paired logits')
        np.testing.assert_array_equal(altered[..., 0], logits[..., 0], err_msg='Full changed')
        np.testing.assert_array_equal(altered[unavailable, :, 2], np.zeros_like(logits[unavailable, :, 2]))
    np.testing.assert_array_equal(data['face_branch_logits'][unavailable], logits[unavailable],
                                  err_msg='Unavailable Face changed the readout')


def load_paired_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in ALL_KEYS}


def extract(source, prior, natural, matched, out, tasks, workers, smoke):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    set_seed(1, True)
    cfg = read_json(source / 'config.json')
    expected = dict(seed=1, tasks=8, mode='sample_holdout', image_split='global_10pct_hash_rank',
                    protect_branches=True, smoke=False, epochs=30, batch_size=64)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected source model configuration')
    lock = load_centers(source, prior, natural)
    roles = read_json(source / 'train_image_roles.json')
    train = build_joint_source(input_paths(), 'train')
    if roles != global_image_roles(train, 'sample_holdout', 1):
        raise ValueError('90/10 TRAIN role assignment changed')
    train = fusion_source(train)
    out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        row = lock['tasks'][str(task)]
        reference, ref_meta = load_train(prior if task in REPLAY else natural, task)
        proto = ref_meta['protocol']
        if (ref_meta['features_sha256'] != row['train_feature_sha256'] or proto['batch_size'] != 64
                or proto['source_config_sha256'] != lock['source_config_sha256']
                or proto['checkpoint_sha256'] != row['checkpoint_sha256']
                or proto['source_cohort_or_archive_sha256'] != row['cohort_sha256']):
            raise ValueError(f'Natural TRAIN reference disagrees with source task {task}')
        check_cohort(reference, ref_meta, row['fusion_cohort'], roles, task)
        limit = min(64, len(reference['sample_ids'])) if smoke else len(reference['sample_ids'])
        selected = {k: reference[k][:limit] for k in NATURAL_KEYS}
        spec = dict(schema=1, split='train', task=task, seed=1, smoke=smoke, group='held',
                    checkpoint_sha256=row['checkpoint_sha256'], config_sha256=lock['source_config_sha256'],
                    cohort_sha256=row['cohort_sha256'], roles_sha256=lock['source_roles_sha256'],
                    native_holdout_sha256=ref_meta['features_sha256'], input_size=224, reduced_size=112,
                    batch_size=64, people=limit, reliable_face=int(selected['reliable'].sum()),
                    sample_id_sha256=hashlib.sha256('\n'.join(selected['sample_ids']).encode()).hexdigest())
        if task in REPLAY:
            src = matched / f'task{task}_held.npz'
            historical = read_json(src.with_suffix('.json'))
            checksum = sha256_file(src)
            expected_match = dict(**spec)
            expected_match.update(smoke=False, people=len(reference['sample_ids']),
                                  reliable_face=ref_meta['reliable_face'], sample_id_sha256=ref_meta['sample_id_sha256'],
                                  archive_sha256=checksum, original_holdout_replayed=True)
            if any(historical.get(k) != v for k, v in expected_match.items()):
                raise ValueError(f'Matched historical TRAIN identity disagrees for task {task}')
            spec['matched_archive_sha256'] = checksum
        dest = out / f'task{task}_held.npz'
        meta_path = dest.with_suffix('.json')
        if meta_path.exists():
            actual = read_json(meta_path)
            if any(actual.get(k) != v for k, v in spec.items()):
                raise ValueError(f'Conflicting existing paired cache: {meta_path}')
            if dest.exists() and actual['archive_sha256'] == sha256_file(dest):
                verify_pairs(load_paired_archive(dest), selected)
                print(f'task{task}: verified existing output, {limit} people', flush=True)
                continue
        if task in REPLAY:
            pairs = {k: a[:limit] for k, a in load_paired_archive(src).items()}
        else:
            loader = train_loader(source, train, roles, task, workers, 64)
            model = load_evaluation_model(source, task).requires_grad_(False).eval()
            pairs = dict(selected)
            for view, variants in collect(model, loader, reference, VIEWS, smoke):
                for k in ('sample_ids', 'reliable', 'targets'):
                    np.testing.assert_array_equal(variants[k], selected[k])
                pairs[f'{view}_branch_logits'] = variants['branch_logits']
            del model
        verify_pairs(pairs, selected)
        tmp = dest.with_suffix(f'.{os.getpid()}.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, **pairs)
        tmp.replace(dest)
        write_json(meta_path, dict(**spec, archive_sha256=sha256_file(dest)))
        print(f'task{task}: {"reused matched pairs" if task in REPLAY else "fresh paired forwards"}, '
              f'{limit} TRAIN holdouts ({spec["reliable_face"]} reliable)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'prior', 'natural', 'matched', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--tasks', type=int, nargs='+', choices=range(8), default=list(range(8)))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    extract(**vars(parser.parse_args()))
