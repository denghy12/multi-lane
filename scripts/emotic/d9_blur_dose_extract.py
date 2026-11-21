"""Replay original held TRAIN people with milder 168-pixel auxiliary crops."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit_experiment import single_thread_worker
from multi_lane.track_a.runner import LabelView, TASK_SIZES, set_seed, task_indices
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_matched_extract import verify_intervention
from scripts.emotic.d9_matched_natural_fit import checked_data
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_paired_a_extract import ALL_KEYS, load_paired_archive, verify_pairs
from scripts.emotic.d9_quality_extract import degrade_view, forward
from scripts.emotic.d9_sample_training import fusion_source, global_image_roles
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths


def mild_view(images, view):
    """Use the original area/bilinear computation at the preregistered 168 size."""
    if view not in ('person', 'face'):
        raise ValueError('Expected Person or Face')
    result = images.copy()
    for name in (view, 'evidence_' + view):
        image = images[name]
        if image.ndim != 4 or image.shape[-2:] != (224, 224):
            raise ValueError('Expected normalized 224-pixel crops')
        reduced = F.interpolate(image.float(), size=(168, 168), mode='area')
        restored = F.interpolate(reduced, size=(224, 224), mode='bilinear', align_corners=False).to(image.dtype)
        if view == 'face':
            usable = images['evidence_face_reliable'].bool().reshape(-1, 1, 1, 1)
            restored = torch.where(usable, restored, image)
        result[name] = restored
    return result


@torch.inference_mode()
def collect_mild(model, source, indices, task, workers, reference):
    dataset = LabelView(source, indices, task_indices(task), include_sample_id=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=workers,
                        worker_init_fn=single_thread_worker, generator=torch.Generator().manual_seed(77191 + task),
                        **({'prefetch_factor': 1} if workers else {}))
    chunks = {key: [] for key in ALL_KEYS}
    offset = 0
    for images, targets, ids in loader:
        rows = slice(offset, offset + len(ids))
        inputs = move_model_inputs(images, torch.device('cuda'))
        base = forward(model, inputs)
        batch = dict(sample_ids=np.asarray(ids), targets=targets.numpy(),
                     reliable=inputs['evidence_face_reliable'].reshape(-1).bool().cpu().numpy(),
                     branch_logits=base['branch_logits'])
        for key, value in batch.items():
            np.testing.assert_array_equal(value, reference[key][rows], err_msg=f'Natural replay task{task} {key}')
        for view in ('person', 'face'):
            if offset == 0:
                strong = forward(model, degrade_view(inputs, view))
                np.testing.assert_array_equal(strong['branch_logits'], reference[view+'_branch_logits'][rows],
                                              err_msg=f'Original 112 replay task{task} {view}')
            changed = forward(model, mild_view(inputs, view))
            verify_intervention(base, changed, view, batch['reliable'])
            batch[view + '_branch_logits'] = changed['branch_logits']
        for key, value in batch.items():
            if key != 'sample_ids' and not np.isfinite(value).all():
                raise ValueError(f'Nonfinite mild input {key}')
            chunks[key].append(value)
        offset += len(ids)
    if offset != len(reference['sample_ids']):
        raise ValueError('Wrong held TRAIN population size')
    data = {key: np.concatenate(value) for key, value in chunks.items()}
    verify_pairs(data, reference)
    return data


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
        raise ValueError('Expected the original equal-update random 90% trajectory')
    roles_path = root/'train_image_roles.json'
    require_digest(roles_path, cfg['equal_updates']['roles_sha256'])
    roles = read(roles_path)
    train = fusion_source(build_joint_source(input_paths(), 'train'))
    if global_image_roles(train, 'sample_holdout', 1) != roles:
        raise ValueError('Original TRAIN image roles changed')
    by_id = {str(pid): index for index, pid in enumerate(train.sample_ids)}
    if len(by_id) != len(train.sample_ids):
        raise ValueError('Repeated TRAIN person ID')
    seals = read(root/'paired_A'/'centers_lock.json')
    out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        data, original, hashes, cohort = checked_data(root, task, seals.get(str(task-1)), seals[str(task)], roles)
        ids = data['sample_ids'].tolist()
        indices = [by_id[pid] for pid in ids]
        path = out/f'task{task}_168.npz'
        meta = path.with_suffix('.json')
        spec = dict(schema=1, split='train', task=task, classes=list(task_indices(task)),
                    people=cohort['people'], reliable_face=cohort['reliable_face'],
                    sample_id_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                    original_archive_sha256=original['archive_sha256'], source_archive_sha256=original['archive_sha256'],
                    current_archive_sha256=original['archive_sha256'], input_size=224, reduced_size=168,
                    batch_size=64, natural_replay='all rows exactly', original_112_replay='first full batch exactly',
                    **hashes)
        if meta.exists():
            saved = read(meta)
            require_digest(path, saved['archive_sha256'])
            if saved != dict(**spec, archive_sha256=saved['archive_sha256']):
                raise ValueError(f'Changed mild input provenance at task{task}')
            verify_pairs(load_paired_archive(path), data)
        else:
            model = load_evaluation_model(root, task).requires_grad_(False).eval()
            pairs = collect_mild(model, train, indices, task, workers, data)
            del model
            if path.exists():
                previous = load_paired_archive(path)
                for key in ALL_KEYS:
                    np.testing.assert_array_equal(previous[key], pairs[key])
            else:
                tmp = path.with_suffix(f'.{os.getpid()}.tmp')
                with tmp.open('wb') as handle:
                    np.savez_compressed(handle, **pairs)
                tmp.replace(path)
            write_verified(meta, dict(**spec, archive_sha256=sha256_file(path)))
        print(f'task{task} 168-pixel paired TRAIN: {len(ids)} people; natural and original-112 replay passed', flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'out'):
        cli.add_argument('--'+name, type=Path, required=True)
    cli.add_argument('--tasks', type=int, nargs='+', choices=range(len(TASK_SIZES)), default=list(range(len(TASK_SIZES))))
    cli.add_argument('--workers', type=int, default=2)
    extract(**vars(cli.parse_args()))
