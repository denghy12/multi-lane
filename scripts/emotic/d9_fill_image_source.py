"""Verified score interface for full-TRAIN and historical equal-update image gates."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit import VIEWS
from multi_lane.track_a.runner import TASK_SIZES, set_seed, task_indices
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_sample_training import fusion_source, global_image_roles
from scripts.emotic.evaluate_d9_relation import write_json, stage_loader
from scripts.emotic.extract_d9_task10_probe import train_loader
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths


def source_mode(source):
    cfg = read(source / 'config.json')
    if cfg.get('seed') != 1 or cfg.get('smoke') or not cfg.get('protect_branches'):
        raise ValueError('Expected formal protected seed1 source')
    mode = cfg['mode']
    if mode == 'sample_holdout':
        if cfg.get('image_split') != 'global_10pct_hash_rank' or 'equal_updates' not in cfg:
            raise ValueError('Only the historical equal-update 90% source is accepted')
    elif mode != 'sample_full':
        raise ValueError('Expected full TRAIN or equal-update 90% branches')
    return mode


def native_scores(source, features, split, task):
    mode = source_mode(source)
    if split != 'train':
        path = source / 'evaluation' / split / f'task{task}.npz'
        meta = read(path.with_suffix('.json'))
        digest = meta['scores_sha256']
    elif mode == 'sample_holdout':
        path = source / 'paired_A' / f'task{task}_held.npz'
        meta = read(source / 'paired_A' / f'task{task}_center.json')
        digest = meta['archive_sha256']
    else:
        path = features.parent / 'native_train' / f'task{task}.npz'
        meta = read(path.with_suffix('.json'))
        digest = meta['archive_sha256']
        for name, ref in [('config', source / 'config.json'),
                          ('cohort', source / 'cohorts' / f'task{task}.json'),
                          ('roles', source / 'train_image_roles.json')]:
            if meta['protocol'][name + '_sha256'] != sha256_file(ref):
                raise ValueError('Full TRAIN cache source changed')
    require_digest(path, digest)
    checkpoint = meta['protocol']['checkpoint_sha256']
    if sha256_file(source / 'compact_checkpoints' / f'task{task}.pth') != checkpoint:
        raise ValueError('Native score/checkpoint provenance mismatch')
    return path, meta, digest, checkpoint


def assert_train_roles(source, image_ids):
    mode = source_mode(source)
    roles = read(source / 'train_image_roles.json')
    expected = 'both' if mode == 'sample_full' else 'fusion'
    if any(roles.get(str(image)) != expected for image in image_ids):
        raise ValueError('TRAIN source role differs from the declared branch exposure')


@torch.inference_mode()
def extract_train(source, out, task, workers=2, batch_size=64):
    if source_mode(source) != 'sample_full':
        raise ValueError('New native extraction is only needed for full TRAIN')
    # No smoke subsampling: the smoke image readout selects whole images from this
    # verified cache, so later formal runs reuse the same deterministic predictions.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    set_seed(1, True)
    protocol = dict(task=task, split='train', smoke=False, classes=list(task_indices(task)),
                    batch_size=batch_size, source_mode='sample_full',
                    exposure='same-current-task-branch-training-people',
                    checkpoint_sha256=sha256_file(source / 'compact_checkpoints' / f'task{task}.pth'),
                    config_sha256=sha256_file(source / 'config.json'),
                    cohort_sha256=sha256_file(source / 'cohorts' / f'task{task}.json'),
                    roles_sha256=sha256_file(source / 'train_image_roles.json'))
    dest = out / f'task{task}.npz'
    report = dest.with_suffix('.json')
    if report.exists():
        old = read(report)
        if old['protocol'] != protocol:
            raise ValueError('Existing full TRAIN cache protocol differs')
        require_digest(dest, old['archive_sha256'])
        print(f'{dest} verified', flush=True)
        return
    original = build_joint_source(input_paths(), 'train')
    roles = global_image_roles(original, 'sample_full', 1)
    if roles != read(source / 'train_image_roles.json'):
        raise ValueError('Full TRAIN original-image roles changed')
    deterministic = fusion_source(original)
    loader = train_loader(source, deterministic, roles, task, workers, batch_size)
    model = load_evaluation_model(source, task).requires_grad_(False).eval()
    # Exact native-sized evaluation replay validates model restoration independently
    # of the image descriptor path. Evaluation labels never enter fitting.
    validation = build_joint_source(input_paths(), 'val')
    val_path = source / 'evaluation' / 'val' / f'task{task}.npz'
    val_meta = read(val_path.with_suffix('.json'))
    require_digest(val_path, val_meta['scores_sha256'])
    native_batch = val_meta['protocol']['batch_size']
    val_loader = stage_loader(validation, task, False, native_batch, workers)
    images, labels, ids = next(iter(val_loader))
    verified_people = len(ids)
    with torch.autocast('cuda'):
        replay = model(move_model_inputs(images, torch.device('cuda')), all_seen=True)
    with np.load(val_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(np.asarray(ids), archive['sample_ids'][:len(ids)])
        observed = torch.stack([replay['branches'][v].float() for v in VIEWS], dim=-1).cpu().numpy()
        np.testing.assert_array_equal(observed, archive['branch_logits'][:len(ids)])
    del replay, observed
    chunks = {k: [] for k in ('sample_ids', 'targets', 'reliable', 'branch_logits')}
    for images, labels, ids in loader:
        inputs = move_model_inputs(images, torch.device('cuda'))
        with torch.autocast('cuda'):
            output = model(inputs, all_seen=False)
        logits = torch.stack([output['branches'][v].float() for v in VIEWS], dim=-1)
        chunks['sample_ids'].append(np.asarray(ids))
        chunks['targets'].append(labels.numpy())
        chunks['reliable'].append(inputs['evidence_face_reliable'].reshape(-1).bool().cpu().numpy())
        chunks['branch_logits'].append(logits.cpu().numpy())
    arrays = {k: np.concatenate(v) for k, v in chunks.items()}
    expected = [str(original.sample_ids[i]) for i in loader.dataset.indices]
    if arrays['sample_ids'].tolist() != expected:
        raise ValueError('Full TRAIN native rows do not replay source cohort')
    if arrays['branch_logits'].shape != (len(expected), TASK_SIZES[task], 3):
        raise ValueError('Unexpected native current-task logits shape')
    if not all(np.isfinite(arrays[k]).all() for k in ('branch_logits', 'targets')):
        raise ValueError('Nonfinite native scores')
    digest = hashlib.sha256('\n'.join(expected).encode()).hexdigest()
    cohort = read(source / 'cohorts' / f'task{task}.json')['fusion']
    if digest != cohort['sample_id_sha256'] or len(expected) != cohort['people']:
        raise ValueError('Full TRAIN native IDs differ from original fusion cohort')
    out.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(dest)
    write_json(report, dict(protocol=protocol, archive_sha256=sha256_file(dest),
                            checkpoint_sha256=protocol['checkpoint_sha256'],
                            people=len(expected), sample_id_sha256=digest,
                            exact_native_val_replay_people=verified_people))
    print(f'{dest}: {len(expected)} exposed TRAIN people', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--task', type=int, choices=range(8), required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=64)
    extract_train(**vars(p.parse_args()))
