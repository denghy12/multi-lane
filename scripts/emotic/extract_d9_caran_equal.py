"""Replay native head inputs from the 13,950-update protected 90% trajectory."""
import argparse
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, set_seed
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_sample_training import fusion_source, global_image_roles
from scripts.emotic.evaluate_d9_relation import stage_loader, write_json
from scripts.emotic.extract_d9_caran_eight import extract_batches
from scripts.emotic.extract_d9_task10_probe import train_loader
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths


def audit_source(root):
    cfg = read(root / 'config.json')
    expected = dict(mode='sample_holdout', seed=1, epochs=30, tasks=8, batch_size=64,
                    smoke=False, protect_branches=True, image_split='global_10pct_hash_rank')
    if (any(cfg.get(k) != v for k, v in expected.items())
            or cfg.get('equal_updates', {}).get('mode') != 'cycle_90pct_to_full_epoch_size'):
        raise ValueError('Expected complete seed1 equal-update protected 90/10 source')
    roles = root / 'train_image_roles.json'
    require_digest(roles, cfg['equal_updates']['roles_sha256'])
    tasks = {}
    for t in range(8):
        path = root / 'history' / f'task{t}.json'
        history = read(path)['epochs']
        expected_steps = cfg['equal_updates']['tasks'][str(t)]['updates']
        if (len(history) != 30 or [r['epoch'] for r in history] != list(range(1, 31))
                or any(r['updates'] != expected_steps or r['skipped'] for r in history)):
            raise ValueError(f'Incomplete branch update budget task{t}')
        tasks[str(t)] = dict(updates=sum(r['updates'] for r in history),
                             skipped=sum(r['skipped'] for r in history),
                             history_sha256=sha256_file(path))
    total = sum(v['updates'] for v in tasks.values())
    if total != 13950:
        raise ValueError(f'Wrong branch update budget: {total}')
    return dict(source_config_sha256=sha256_file(root/'config.json'),
                roles_sha256=sha256_file(roles), total_updates=total, tasks=tasks)


def reference(root, task, split, cfg_hash, cp_hash):
    if split == 'train':
        archive = root/'paired_A'/f'task{task}_held.npz'
        meta = read(root/'paired_A'/f'task{task}_center.json')
        old = meta['protocol']
        if (old['task'] != task or old['split'] != split or old['smoke']
                or old['checkpoint_sha256'] != cp_hash or old['config_sha256'] != cfg_hash
                or old['cohort_sha256'] != sha256_file(root/'cohorts'/f'task{task}.json')
                or meta['fusion_cohort'] != read(root/'cohorts'/f'task{task}.json')['fusion']):
            raise ValueError(f'TRAIN source provenance differs at task{task}')
        require_digest(archive, meta['archive_sha256'])
    else:
        archive = root/'evaluation'/split/f'task{task}.npz'
        meta = read(archive.with_suffix('.json'))
        old = meta['protocol']
        if (old['task'] != task or old['split'] != split or old['smoke']
                or old['checkpoint_sha256'] != cp_hash or old['config_sha256'] != cfg_hash
                or old['batch_size'] != 64):
            raise ValueError(f'Evaluation provenance differs at task{task}/{split}')
        require_digest(archive, meta['scores_sha256'])
    return archive


def run(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    set_seed(1, True)
    audit = audit_source(args.source)
    if 'test' in args.splits:
        seal = read(args.out/'validation_seal.json')
        require_digest(args.out/'val_results.json', seal['val_results_sha256'])
        require_digest(args.out/'caran_fit_lock.json', seal['caran_fit_lock_sha256'])
    sources = {s: build_joint_source(input_paths(), s) for s in args.splits}
    roles = read(args.source/'train_image_roles.json')
    if 'train' in sources:
        if global_image_roles(sources['train'], 'sample_holdout', 1) != roles:
            raise ValueError('TRAIN image allocation differs from source')
        sources['train'] = fusion_source(sources['train'])
    for task in args.tasks:
        started = time.monotonic()
        model = None
        cp_hash = sha256_file(args.source/'compact_checkpoints'/f'task{task}.pth')
        for split, source in sources.items():
            archive = reference(args.source, task, split, audit['source_config_sha256'], cp_hash)
            protocol = dict(schema=1, task=task, split=split, smoke=args.smoke, batch_size=64,
                            source_config_sha256=audit['source_config_sha256'],
                            checkpoint_sha256=cp_hash, reference_sha256=sha256_file(archive))
            path = args.out/'features'/f'task{task}'/f'{split}.npz'
            meta_path = path.with_suffix('.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            if meta_path.exists():
                old = read(meta_path)
                if old['protocol'] != protocol:
                    raise ValueError(f'Conflicting cache: {meta_path}')
                require_digest(path, old['features_sha256'])
                print(f'Reusing verified {path}', flush=True)
                continue
            loader = (train_loader(args.source, source, roles, task, args.workers, 64)
                      if split == 'train' else stage_loader(source, task, False, 64, args.workers))
            if model is None:
                model = load_evaluation_model(args.source, task).requires_grad_(False).eval()
            arrays = extract_batches(model, loader, archive, task, split, args.smoke, 64,
                                     mask_unavailable_scores=(split == 'train'))
            tmp = path.with_suffix(f'.{os.getpid()}.tmp')
            with tmp.open('wb') as handle:
                np.savez(handle, **arrays)
            tmp.replace(path)
            ids = arrays['sample_ids']
            write_json(meta_path, dict(protocol=protocol, feature_origin='native branch head pre-hook',
                features_sha256=sha256_file(path), people=len(ids),
                reliable_face=int(arrays['reliable'].sum()),
                sample_id_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
                exact_replay=True, visibility_mask_applied=True, source_update_audit=audit))
            print(f'{split} task{task}: {arrays["head_cls"].shape}, exact replay; '
                  f'{time.monotonic()-started:.1f}s', flush=True)
            del arrays
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'out'):
        p.add_argument('--'+name, required=True, type=Path)
    p.add_argument('--tasks', nargs='+', type=int, choices=range(8), default=list(range(8)))
    p.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())
