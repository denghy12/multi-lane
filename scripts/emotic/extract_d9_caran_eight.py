"""Cache the actual three head inputs and replay archived predictions at each stage."""
import argparse
import hashlib
import os
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit import VIEWS
from multi_lane.track_a.runner import TASK_SIZES, set_seed
from scripts.emotic.d9_sample_training import fusion_source, global_image_roles
from scripts.emotic.evaluate_d9_relation import stage_loader, write_json
from scripts.emotic.extract_d9_task10_probe import read_json, train_loader
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths


def head_input_hook(view, captured):
    def capture(module, args):
        if view in captured or len(args) != 1 or args[0].ndim != 3:
            raise ValueError(f'Expected one [person, lane, CLS] head input for {view}')
        captured[view] = args[0].detach()
    return capture


def pack_head_inputs(captured, reliable, lanes):
    if set(captured) != set(VIEWS):
        raise ValueError('Missing classifier input')
    features = torch.stack([captured[view] for view in VIEWS], dim=2)
    if features.shape[:3] != (len(reliable), lanes, len(VIEWS)):
        raise ValueError(f'Unexpected classifier input layout {features.shape}')
    features[:, :, VIEWS.index('face')].masked_fill_(~reliable[:, None, None], 0.)
    if not torch.isfinite(features).all():
        raise FloatingPointError('Nonfinite available-view CLS')
    return features.float()


def reference_path(source, prior, eight, task, split):
    if split == 'train':
        root = prior if task in (0, 3, 6) else eight
        return root / 'features' / f'task{task}' / 'train.npz'
    return source / 'evaluation' / split / f'task{task}.npz'


@torch.inference_mode()
def extract_batches(model, loader, archive, task, split, smoke, batch_size):
    head_ids = [task] if split == 'train' else list(range(task + 1))
    n = min(batch_size, len(loader.dataset)) if smoke else len(loader.dataset)
    keys = ('sample_ids', 'targets', 'reliable', 'branch_logits')
    with np.load(archive, allow_pickle=False) as f:
        reference = {k: f[k][:n] for k in keys}
        if len(f['sample_ids']) != len(loader.dataset):
            raise ValueError(f'Historical cohort length differs for {archive}')
    if len(set(reference['sample_ids'].tolist())) != n:
        raise ValueError(f'Duplicate subject IDs for {archive}')
    features = None
    count = 0
    capture = {}
    with ExitStack() as hooks:
        for view in VIEWS:
            head = model.branches[view].head
            hooks.callback(head.register_forward_pre_hook(head_input_hook(view, capture)).remove)
        for images, targets, ids in loader:
            capture.clear()
            inputs = move_model_inputs(images, torch.device('cuda'))
            with torch.autocast('cuda'):
                result = model(inputs, all_seen=(split != 'train'))
            valid = inputs['evidence_face_reliable'].reshape(-1).bool()
            cls = pack_head_inputs(capture, valid, len(head_ids)).cpu().numpy()
            if features is None:
                features = np.empty((n, *cls.shape[1:]), dtype=cls.dtype)
            rows = slice(count, count + len(ids))
            scores = torch.stack([result['branches'][v].float() for v in VIEWS], dim=2)
            observed = dict(sample_ids=np.asarray(ids), targets=targets.cpu().numpy(),
                            reliable=valid.cpu().numpy(), branch_logits=scores.cpu().numpy())
            for k in keys:
                if not np.array_equal(observed[k], reference[k][rows]):
                    raise ValueError(f'{archive}: {k} replay failed at row {count}')
            features[rows] = cls
            count += len(ids)
            if smoke:
                break
    if count != n or features is None:
        raise ValueError('Incomplete feature collection')
    return dict(**reference, head_cls=features, head_ids=np.asarray(head_ids), view_order=np.asarray(VIEWS))


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    set_seed(1, True)
    config = read_json(args.source / 'config.json')
    if (config['mode'] != 'sample_holdout' or config['smoke'] or not config['protect_branches']
            or config['seed'] != 1 or config['image_split'] != 'global_10pct_hash_rank'):
        raise ValueError('Expected seed1 protected 90/10 formal branch trajectory')
    cfg_hash = sha256_file(args.source / 'config.json')
    sources = {s: build_joint_source(input_paths(), s) for s in args.splits}
    roles = {}
    if 'train' in sources:
        roles = global_image_roles(sources['train'], 'sample_holdout', config['seed'])
        if roles != read_json(args.source / 'train_image_roles.json'):
            raise ValueError('Current TRAIN image allocation differs from branch training')
        sources['train'] = fusion_source(sources['train'])
    for task in args.tasks:
        model = None
        checkpoint = args.source / 'compact_checkpoints' / f'task{task}.pth'
        cp_hash = sha256_file(checkpoint)
        for split, source in sources.items():
            archive = reference_path(args.source, args.prior, args.eight, task, split)
            reference_meta = read_json(archive.with_suffix('.json'))
            ref_hash = sha256_file(archive)
            old = reference_meta['protocol']
            train = split == 'train'
            if (old['smoke'] or old['split'] != split or old['task'] != task
                    or old['checkpoint_sha256'] != cp_hash or old['batch_size'] != args.batch_size
                    or old['source_config_sha256' if train else 'config_sha256'] != cfg_hash
                    or reference_meta['features_sha256' if train else 'scores_sha256'] != ref_hash):
                raise ValueError(f'Incompatible reference: {archive}')
            protocol = dict(schema=1, task=task, split=split, smoke=args.smoke, batch_size=args.batch_size,
                            source_config_sha256=cfg_hash, checkpoint_sha256=cp_hash, reference_sha256=ref_hash)
            path = args.out / 'features' / f'task{task}' / f'{split}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            meta_path = path.with_suffix('.json')
            if meta_path.exists():
                existing = read_json(meta_path)
                if existing['protocol'] != protocol:
                    raise ValueError(f'Output already has another protocol: {meta_path}')
                if path.is_file() and existing['features_sha256'] == sha256_file(path):
                    print(f'Verified reusable {path}', flush=True)
                    continue
            if train:
                loader = train_loader(args.source, source, roles, task, args.workers, args.batch_size)
            else:
                loader = stage_loader(source, task, False, args.batch_size, args.workers)
            if model is None:
                model = load_evaluation_model(args.source, task).requires_grad_(False).eval()
            arrays = extract_batches(model, loader, archive, task, split, args.smoke, args.batch_size)
            tmp = path.with_suffix(f'.{os.getpid()}.tmp')
            with tmp.open('wb') as handle:
                np.savez(handle, **arrays)
            tmp.replace(path)
            ids = arrays['sample_ids']
            write_json(meta_path, dict(protocol=protocol, feature_origin='pre-hook of each branch head',
                                       features_sha256=sha256_file(path), people=len(ids),
                                       reliable_face=int(arrays['reliable'].sum()),
                                       sample_id_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                                       exact_replay=True, visibility_mask_applied=True))
            print(f'{split} task{task}, CLS {arrays["head_cls"].shape}, all branch logits replayed', flush=True)
            del arrays
        del model


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'prior', 'eight', 'out'):
        cli.add_argument('--'+name, type=Path, required=True)
    cli.add_argument('--tasks', nargs='+', type=int, choices=range(len(TASK_SIZES)), default=list(range(len(TASK_SIZES))))
    cli.add_argument('--splits', nargs='+', choices=('train','val','test'), required=True)
    cli.add_argument('--batch-size', type=int, default=64)
    cli.add_argument('--workers', type=int, default=2)
    cli.add_argument('--smoke', action='store_true')
    run(cli.parse_args())
