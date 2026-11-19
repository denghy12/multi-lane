"""Extract frozen 90/10 model outputs on separate, matched official TRAIN cohorts."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit_experiment import single_thread_worker
from multi_lane.track_a.runner import LabelView, set_seed, task_indices
from scripts.emotic.d9_matched_cohort import select_groups
from scripts.emotic.d9_quality_extract import degrade_view, forward, verify_replay
from scripts.emotic.d9_sample_training import describe_cohort, fusion_source, global_image_roles, image_key
from scripts.emotic.d9_tcp_probe import identify_source
from scripts.emotic.evaluate_d9_relation import write_json
from scripts.emotic.extract_d9_task10_probe import read_json
from scripts.emotic.fit_d9_task10_probe import PROJECTION_SEED, load_arrays, project
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths


def verify_design(source, rows, roles):
    if not rows['held'] or len(rows['seen']) != len(rows['held']):
        raise ValueError('Matched groups must be nonempty with equal numbers of people')
    for name, role in (('seen', 'branch'), ('held', 'fusion')):
        ids = [str(source.sample_ids[i]) for i in rows[name]]
        if len(set(ids)) != len(ids) or any(roles.get(image_key(i)) != role for i in ids):
            raise ValueError(f'Nonexclusive TRAIN image roles in {name}')


def verify_intervention(natural, changed, view, reliable):
    for key in ('branch_logits', 'content'):
        full_axis = natural[key][:, :, 0] if key == 'branch_logits' else natural[key][:, 0]
        changed_full = changed[key][:, :, 0] if key == 'branch_logits' else changed[key][:, 0]
        np.testing.assert_array_equal(full_axis, changed_full, err_msg=f'Changed Full output after {view} alteration')
    if view == 'face':
        for key, value in natural.items():
            np.testing.assert_array_equal(changed[key][~reliable], value[~reliable],
                                          err_msg=f'Unavailable Face changes {key}')


def held_reference(prior, source_root, task):
    archive, ident = identify_source(prior, task, 'train', source_mode='sample_holdout')
    spec = read_json(archive.with_suffix('.json'))['protocol']
    cohort_file = source_root / 'cohorts' / f'task{task}.json'
    if (spec['schema'] != 2 or spec['batch_size'] != 64 or
            sha256_file(cohort_file) != spec['source_cohort_or_archive_sha256'] or
            sha256_file(source_root/'compact_checkpoints'/f'task{task}.pth') != ident['checkpoint_sha256'] or
            sha256_file(source_root/'config.json') != ident['config_sha256']):
        raise ValueError('Natural holdout feature source and checkpoint disagree')
    return load_arrays(archive, task), ident


@torch.inference_mode()
def collect(model, source, indices, task, workers, reference):
    dataset = LabelView(source, indices, task_indices(task), include_sample_id=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=workers,
                        worker_init_fn=single_thread_worker, generator=torch.Generator().manual_seed(77191 + task),
                        **({'prefetch_factor': 1} if workers else {}))
    chunks = {}
    offset = 0
    for images, targets, ids in loader:
        inputs = move_model_inputs(images, torch.device('cuda'))
        base = forward(model, inputs)
        batch = dict(sample_ids=np.asarray(ids), targets=targets.numpy(),
                     reliable=inputs['evidence_face_reliable'].reshape(-1).bool().cpu().numpy())
        if reference is not None:
            verify_replay(dict(**base, **batch), reference, slice(offset, offset + len(ids)))
        for name in ('natural', 'person', 'face'):
            changed = base
            if name != 'natural':
                changed = forward(model, degrade_view(inputs, name))
                verify_intervention(base, changed, name, batch['reliable'])
            prefix = '' if name == 'natural' else name + '_'
            batch[prefix + 'branch_logits'] = changed['branch_logits']
            batch[prefix + 'relation'] = changed['relation']
            batch[prefix + 'content_proj'] = project(changed)['content']
        for key, array in batch.items():
            if key != 'sample_ids' and not np.isfinite(array).all():
                raise ValueError(f'Nonfinite paired extraction: {key}')
            chunks.setdefault(key, []).append(array)
        offset += len(ids)
    data = {key: np.concatenate(arrays) for key, arrays in chunks.items()}
    if data['sample_ids'].tolist() != [str(source.sample_ids[i]) for i in indices]:
        raise ValueError('Extraction person order differs from the locked match')
    return data


def extract(source, prior, out, tasks, workers, smoke):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    set_seed(1, True)
    config_path = source/'config.json'
    cfg = read_json(config_path)
    expected = dict(seed=1, tasks=8, epochs=30, batch_size=64, mode='sample_holdout', smoke=False,
                    image_split='global_10pct_hash_rank', protect_branches=True)
    if any(cfg.get(k) != value for k, value in expected.items()):
        raise ValueError('Unexpected branch training protocol')
    train = build_joint_source(input_paths(), 'train')
    roles_path = source/'train_image_roles.json'
    roles = read_json(roles_path)
    if global_image_roles(train, 'sample_holdout', 1) != roles:
        raise ValueError('Cannot replay original TRAIN roles')
    train = fusion_source(train)
    out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        cohort_path = source/'cohorts'/f'task{task}.json'
        cohort = read_json(cohort_path)
        reference, ident = held_reference(prior, source, task)
        rows, matching = select_groups(train, task, roles, cohort)
        verify_design(train, rows, roles)
        if reference['sample_ids'].tolist() != [str(train.sample_ids[i]) for i in rows['held']]:
            raise ValueError('Reference holdout ids must equal the complete cohort in the historical order')
        spec = dict(schema=1, split='train', task=task, seed=1, smoke=smoke,
                    native_holdout_sha256=ident['sha256'], checkpoint_sha256=ident['checkpoint_sha256'],
                    config_sha256=ident['config_sha256'], roles_sha256=sha256_file(roles_path),
                    cohort_sha256=sha256_file(cohort_path), input_size=224, reduced_size=112,
                    projection_seed=PROJECTION_SEED, batch_size=64)
        match_path = out / f'task{task}_match.json'
        manifest = dict(**spec, matching=matching, **{
            name: describe_cohort(train, indices, task_indices(task)) for name, indices in rows.items()})
        if match_path.exists():
            if read_json(match_path) != manifest:
                raise ValueError(f'Existing match design differs: {match_path}')
        else:
            write_json(match_path, manifest)
        spec['match_sha256'] = sha256_file(match_path)
        model = None
        for group in ('held', 'seen'):
            path = out / f'task{task}_{group}.npz'
            meta = path.with_suffix('.json')
            indices = rows[group][:64] if smoke else rows[group]
            protocol = dict(**spec, group=group, people=len(indices), original_holdout_replayed=(group == 'held'))
            if meta.exists():
                m = read_json(meta)
                if any(m.get(k) != v for k, v in protocol.items()):
                    raise ValueError(f'Existing archive has different identity: {path}')
                if path.exists() and sha256_file(path) == m['archive_sha256']:
                    continue
            if model is None:
                model = load_evaluation_model(source, task).requires_grad_(False).eval()
            d = collect(model, train, indices, task, workers, reference if group == 'held' else None)
            temp = path.with_suffix(f'.{os.getpid()}.tmp')
            with temp.open('wb') as f:
                np.savez_compressed(f, **d)
            temp.replace(path)
            ids_digest = hashlib.sha256('\n'.join(d['sample_ids'].tolist()).encode()).hexdigest()
            write_json(meta, dict(**protocol, archive_sha256=sha256_file(path), sample_id_sha256=ids_digest,
                                  reliable_face=int(d['reliable'].sum())))
            print(f'task{task} {group}: {len(indices)} people, {d["reliable"].sum()} reliable Face', flush=True)
        del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'prior', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--tasks', nargs='+', type=int, choices=(0,3,6), default=[0,3,6])
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    extract(**vars(parser.parse_args()))
