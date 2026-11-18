"""Extract current-task CLS and the local effect of Frozen9 messages from fixed 90/10 trajectories."""
import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit import VIEWS
from multi_lane.track_a.r1_crossvit_experiment import single_thread_worker
from multi_lane.track_a.runner import LabelView, TASK_SIZES, fit_calibration_indices, set_seed, task_indices
from scripts.emotic.d9_sample_training import describe_cohort, fusion_source, global_image_roles, task_cohorts
from scripts.emotic.evaluate_d9_relation import stage_loader, write_json
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths


def read_json(path):
    return json.loads(path.read_text())


def train_loader(root, source, roles, task, workers, batch_size):
    classes = task_indices(task)
    eligible, _ = fit_calibration_indices(source, classes, 0.)
    branch, held = task_cohorts(source, eligible, roles, False)
    actual = {key: describe_cohort(source, rows, classes) for key, rows in [('branch', branch), ('fusion', held)]}
    if actual != read_json(root / 'cohorts' / f'task{task}.json'):
        raise ValueError(f'TRAIN cohort differs from original task {task}')
    data = LabelView(source, held, classes, include_sample_id=True)
    return DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=workers,
                      worker_init_fn=single_thread_worker, generator=torch.Generator().manual_seed(77191 + task),
                      **({'prefetch_factor': 1} if workers else {}))


def zero_messages(module, args, output):
    # Preserve the normal attention statistic and remove only the addition to receiver tokens.
    return args[0], output[1]


def check_off_path(model, images, capture, task, all_seen):
    """Independently disable each current Frozen9 exchange and verify local counterfactual CLS."""
    with ExitStack() as hooks:
        for view in ('person', 'face'):
            agent = model.branches[view].cross_agent(task, 8)
            hooks.callback(agent.register_forward_hook(zero_messages).remove)
        native = {}
        model(images, all_seen=all_seen, task10_probe=native)
    for view in ('person', 'face'):
        expected_delta = capture[view]['content'].float() - native[view]['content'].float()
        torch.testing.assert_close(capture[view]['delta'], expected_delta, rtol=0, atol=0)
        if torch.count_nonzero(native[view]['delta']):
            raise ValueError(f'Message-off delta should be exactly zero for {view}')


@torch.inference_mode()
def collect(model, loader, task, all_seen, smoke):
    chunks = {name: [] for name in ('sample_ids', 'targets', 'reliable', 'branch_logits', 'relation', 'content', 'delta')}
    cols = TASK_SIZES[task]
    for batch_number, (images, targets, ids) in enumerate(loader):
        inputs = move_model_inputs(images, torch.device('cuda'))
        capture = {}
        with torch.autocast('cuda'):
            normal = model(inputs, all_seen=all_seen, task10_probe=capture)
            if smoke:
                check_off_path(model, inputs, capture, task, all_seen)
        valid = inputs['evidence_face_reliable'].reshape(-1).bool()
        relation = capture['relation'].float().clone()
        relation[~valid, 4:] = 0
        content = torch.stack([capture[v]['content'].float() for v in VIEWS], dim=1)
        delta = torch.stack([capture[v]['delta'] for v in ('person', 'face')], dim=1)
        content[~valid, 2] = 0
        delta[~valid, 1] = 0
        logits = torch.stack([normal['branches'][v][:, -cols:] for v in VIEWS], dim=-1)
        chunks['sample_ids'].append(np.asarray(ids))
        for key, value in dict(targets=targets[:, -cols:], reliable=valid, branch_logits=logits,
                               relation=relation, content=content, delta=delta).items():
            chunks[key].append(value.cpu().numpy())
        if smoke:
            break
    data = {key: np.concatenate(values) for key, values in chunks.items()}
    if len(set(data['sample_ids'])) != len(data['sample_ids']) or not all(
        np.isfinite(data[k]).all() for k in data if k != 'sample_ids'
    ):
        raise ValueError('Duplicate IDs or nonfinite probe features')
    return data


def verify_archive(path, features, task, smoke):
    report = read_json(path.with_suffix('.json'))
    if sha256_file(path) != report['scores_sha256']:
        raise ValueError(f'Damaged original evaluation archive {path}')
    count = len(features['sample_ids'])
    with np.load(path, allow_pickle=False) as a:
        if not smoke and len(a['sample_ids']) != count:
            raise ValueError('Evaluation cohort size differs')
        for key in ('sample_ids', 'targets', 'reliable', 'branch_logits'):
            ref = a[key][:count]
            if key in ('targets', 'branch_logits'):
                ref = ref[:, -TASK_SIZES[task]:]
            if not np.array_equal(features[key], ref):
                raise ValueError(f'Archived current-head output differs ({path}, {key})')
        # Preserve the reference's exact GPU rounding when measuring AP against native controls.
        for key in ('fixed_logits', 'equal_logits'):
            features[key] = a[key][:count, -TASK_SIZES[task]:].copy()


def extract(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    set_seed(1, True)
    config = read_json(args.source / 'config.json')
    if config['mode'] != 'sample_holdout' or config['smoke'] or not config['protect_branches']:
        raise ValueError('Expected formal sample_holdout checkpoint with protected branches')
    sources = {s: build_joint_source(input_paths(), s) for s in args.splits}
    roles = {}
    if 'train' in sources:
        roles = global_image_roles(sources['train'], 'sample_holdout', config['seed'])
        if roles != read_json(args.source / 'train_image_roles.json'):
            raise ValueError('Original-image role allocation mismatch')
        sources['train'] = fusion_source(sources['train'])
    for task in args.tasks:
        model = load_evaluation_model(args.source, task).requires_grad_(False).eval()
        ckpt = args.source / 'compact_checkpoints' / f'task{task}.pth'
        checkpoint_hash = sha256_file(ckpt)
        folder = args.output / 'features' / f'task{task}'
        folder.mkdir(parents=True, exist_ok=True)
        for split, source in sources.items():
            result = folder / f'{split}.npz'
            archive = args.source / 'evaluation' / split / f'task{task}.npz'
            if split == 'train':
                loader = train_loader(args.source, source, roles, task, args.workers, args.batch_size)
                provenance = sha256_file(args.source / 'cohorts' / f'task{task}.json')
            else:
                proto = read_json(archive.with_suffix('.json'))['protocol']
                if proto['checkpoint_sha256'] != checkpoint_hash or proto['batch_size'] != args.batch_size:
                    raise ValueError('Replay requires the same reference checkpoint and evaluation batch size')
                loader = stage_loader(source, task, False, args.batch_size, args.workers)
                provenance = sha256_file(archive)
            protocol = dict(schema=2, task=task, split=split, smoke=args.smoke, batch_size=args.batch_size,
                            checkpoint_sha256=checkpoint_hash, source_cohort_or_archive_sha256=provenance,
                            source_config_sha256=sha256_file(args.source / 'config.json'))
            report = result.with_suffix('.json')
            if result.is_file() and report.is_file():
                old = read_json(report)
                if old['protocol'] != protocol:
                    raise ValueError(f'Output directory already contains a different protocol: {result}')
                if old['features_sha256'] == sha256_file(result):
                    continue
            features = collect(model, loader, task, split != 'train', args.smoke)
            if split != 'train':
                verify_archive(archive, features, task, args.smoke)
            else:
                expected = [str(source.sample_ids[i]) for i in loader.dataset.indices]
                if args.smoke:
                    expected = expected[:args.batch_size]
                if features['sample_ids'].tolist() != expected:
                    raise ValueError('TRAIN prediction rows do not match held-out people')
            temp = result.with_suffix(f'.{os.getpid()}.tmp')
            with temp.open('wb') as f:
                np.savez(f, **features)
            temp.replace(result)
            write_json(report, dict(protocol=protocol, people=len(features['sample_ids']),
                                    reliable_face=int(features['reliable'].sum()),
                                    features_sha256=sha256_file(result),
                                    sample_id_sha256=hashlib.sha256('\n'.join(features['sample_ids']).encode()).hexdigest(),
                                    visibility_mask_applied=True, exact_eval_replay=(split != 'train'),
                                    independently_verified_off_path=args.smoke))
            print(f'{split} task{task}: {len(features["sample_ids"])} people, wrote {result}', flush=True)
        del model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', type=int, nargs='+', choices=range(len(TASK_SIZES)), default=[0, 3, 6])
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val', 'test'])
    parser.add_argument('--smoke', action='store_true', help='Validate the first native evaluation-sized batch')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    extract(parser.parse_args())


if __name__ == '__main__':
    main()
