"""Fit held-TRAIN paired A online, or replay immutable per-task metadata-split checkpoints."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, TaskMetrics, fit_calibration_indices, set_seed, summarize_tasks, task_indices
from scripts.emotic.d9_a_eight_eval import GROUPS, METHODS, paired, predictions, read, require_digest, verify_controls
from scripts.emotic.d9_matched_extract import collect
from scripts.emotic.d9_matched_run import fit_average
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_quality_fit import LR, STEPS
from scripts.emotic.d9_split90_roles import split_roles
from scripts.emotic.d9_sample_training import describe_cohort, fusion_source, global_image_roles, task_cohorts
from scripts.emotic.evaluate_d9_relation import metrics_for_stage
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS, comparisons
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths


def provenance(root, task):
    paths = dict(config=root / 'config.json', roles=root / 'train_image_roles.json',
                 cohort=root / 'cohorts' / f'task{task}.json',
                 checkpoint=root / 'compact_checkpoints' / f'task{task}.pth')
    return {name + '_sha256': sha256_file(path) for name, path in paths.items()}


def validate_source(cfg, replay_meta):
    expected_split = 'global_10pct_meta' if replay_meta else 'global_10pct_hash_rank'
    if (cfg['mode'] != 'sample_holdout' or not cfg.get('protect_branches') or 'equal_updates' not in cfg
            or cfg['image_split'] != expected_split or cfg['seed'] != 1
            or cfg['tasks'] != (2 if cfg['smoke'] else len(TASK_SIZES))):
        raise ValueError('Expected the declared protected equal-update seed1 90/10 branch trajectory')
    if replay_meta and (cfg.get('split90') != 'meta' or cfg['epochs'] != (1 if cfg['smoke'] else 30)
                        or cfg['batch_size'] != (4 if cfg['smoke'] else 64)
                        or cfg['equal_updates'].get('mode') != 'cycle_90pct_to_full_epoch_size'):
        raise ValueError('Replay requires the original metadata-stratified branch recipe')


def check_fit_point(root, task, replay_meta):
    if replay_meta:
        return  # The caller loads only the compact checkpoint at this stage.
    if ((root / 'cohorts' / f'task{task+1}.json').exists()
            or (root / 'compact_checkpoints' / f'task{task+1}.pth').exists()):
        raise ValueError('New fusion weights must be fitted before the next task starts')


def fit(root, out, cfg, task, workers, replay_meta=False):
    if task is None or not 0 <= task < cfg['tasks']:
        raise ValueError('Specify a valid --task when fitting paired A')
    path = out / f'task{task}_center.json'
    pair_path = out / f'task{task}_held.npz'
    hashes = provenance(root, task)
    previous = sha256_file(out / f'task{task-1}_center.json') if task else None
    protocol = dict(task=task, split='train', smoke=cfg['smoke'], previous_center_sha256=previous,
                    init='equal_available_views', optimizer='adam', steps=STEPS, lr=LR,
                    classes=list(task_indices(task)), variants=['natural', 'person_112', 'reliable_face_112'], **hashes)
    if replay_meta:
        if hashes['roles_sha256'] != cfg['equal_updates']['roles_sha256']:
            raise ValueError('TRAIN role manifest differs from the one used to train the branches')
        protocol['fit_mode'] = 'replay_meta_task_checkpoint'
    if path.exists():
        saved = read(path)
        if saved['protocol'] != protocol:
            raise ValueError(f'Sealed TRAIN inputs for task {task} changed')
        require_digest(pair_path, saved['archive_sha256'])
        return
    check_fit_point(root, task, replay_meta)

    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    set_seed(cfg['seed'], True)
    torch.set_num_threads(2)
    train = build_joint_source(input_paths(), 'train')
    roles = read(root / 'train_image_roles.json')
    expected_roles = (split_roles(train, 'meta', cfg['seed']) if replay_meta
                      else global_image_roles(train, 'sample_holdout', cfg['seed']))
    if roles != expected_roles:
        raise ValueError('Held TRAIN original-image roles changed')
    classes = task_indices(task)
    rows, _ = fit_calibration_indices(train, classes, 0.)
    branch, held = task_cohorts(train, rows, roles, cfg['smoke'])
    cohorts = {'fusion': describe_cohort(train, held, classes), 'branch': describe_cohort(train, branch, classes)}
    if cohorts != read(root / 'cohorts' / f'task{task}.json'):
        raise ValueError(f'Branch or held TRAIN people differ for task {task}')
    model = load_evaluation_model(root, task).requires_grad_(False)
    data = collect(model, fusion_source(train), held, task, workers, None)
    del model
    if replay_meta:
        labels = [[c in train.targets[i] for c in classes] for i in held]
        np.testing.assert_array_equal(data['targets'], labels, err_msg='Wrong current-task TRAIN labels')
        np.testing.assert_array_equal(data['reliable'], [bool(train.face_reliable[i]) for i in held],
                                      err_msg='Incorrect held-person Face mask')
    torch.set_num_threads(1)
    variants = {v: {'branch_logits': data[v + '_branch_logits']} for v in ('person', 'face')}
    fit_result = fit_average(data, variants)
    arrays = {k: data[k] for k in ('sample_ids', 'targets', 'reliable', 'branch_logits', 'person_branch_logits', 'face_branch_logits')}
    # If interrupted between archive and seal, only accept the same observations on replay.
    if pair_path.exists():
        with np.load(pair_path, allow_pickle=False) as old:
            for key, value in arrays.items():
                np.testing.assert_array_equal(old[key], value)
    else:
        tmp = pair_path.with_suffix(f'.{os.getpid()}.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, **arrays)
        tmp.replace(pair_path)
    write_verified(path, dict(protocol=protocol, checkpoint_sha256=hashes['checkpoint_sha256'],
                              fusion_cohort=cohorts['fusion'], archive_sha256=sha256_file(pair_path),
                              center=fit_result['params']['bias'], group_weights=fit_result['weights'], fit=fit_result['fit']))
    print(f'task {task} paired A sealed from {len(held)} held TRAIN people', fit_result['weights'], flush=True)


def evaluate(root, out, cfg, split, replay_meta=False):
    native = root / ('smoke_evaluation' if cfg['smoke'] else 'evaluation')
    hashes, centers = {}, {}
    for t in range(cfg['tasks']):
        key = str(t)
        path = out / f'task{t}_center.json'
        center = read(path)
        expected = provenance(root, t)
        protocol = center['protocol']
        fit_mode = 'replay_meta_task_checkpoint' if replay_meta else None
        if (protocol.get('fit_mode') != fit_mode or protocol['split'] != 'train'
                or protocol['classes'] != list(task_indices(t))
                or any(protocol[k] != v for k, v in expected.items()) or protocol['task'] != t
                or protocol['previous_center_sha256'] != hashes.get(str(t - 1))
                or center['checkpoint_sha256'] != expected['checkpoint_sha256']):
            raise ValueError(f'Current model or previous TRAIN seal differs for task {t}')
        require_digest(out / f'task{t}_held.npz', center['archive_sha256'])
        centers[key], hashes[key] = center, sha256_file(path)
    lock = out / 'centers_lock.json'
    seal = out / 'validation_seal.json'
    if split == 'val':
        write_verified(lock, hashes)
    elif hashes != read(lock):
        raise ValueError('Centers differ from the Validation center lock')
    validation_hash = None
    if split == 'test':
        validation = read(seal)
        require_digest(lock, validation['centers_sha256'])
        require_digest(out / 'val_results.json', validation['val_results_sha256'])
        validation_hash = validation['native_val_result_sha256']
        require_digest(native / 'val' / 'result.json', validation_hash)
    native_path = native / split / 'result.json'
    source = read(native_path)
    if (source['split'] != split or source['mode'] != cfg['mode'] or source['smoke'] != cfg['smoke']
            or len(source['stages']) != cfg['tasks']):
        raise ValueError('Unexpected or incomplete native evaluation')
    stages = []
    for t, saved in enumerate(source['stages']):
        key = str(t)
        spec = dict(task=t, split=split, smoke=cfg['smoke'], config_sha256=centers[key]['protocol']['config_sha256'],
                    checkpoint_sha256=centers[key]['checkpoint_sha256'], validation_result_sha256=validation_hash)
        if any(saved['protocol'][k] != v for k, v in spec.items()):
            raise ValueError(f'Native evaluation does not match sealed model at task {t}')
        archive = native / split / f'task{t}.npz'
        require_digest(archive, saved['scores_sha256'])
        torch.set_num_threads(1)
        with np.load(archive, allow_pickle=False) as data:
            z, y, valid = predictions(data, centers, t)
        torch.set_num_threads(2)
        metrics = metrics_for_stage(t, y, valid, z)
        verify_controls(metrics, saved['metrics'])
        stages.append(dict(stage=t, people=len(y), reliable_face=int(valid.sum()), scores_sha256=saved['scores_sha256'],
                           metrics=metrics, paired=paired(metrics, t)))
    summary = {m: {g: summarize_tasks([TaskMetrics(**r['metrics'][m][g]) for r in stages])
                   for g in GROUPS if all(g in r['metrics'][m] for r in stages)} for m in METHODS}
    for m in ('fixed', 'equal'):
        if summary[m] != source['summary'][m]:
            raise ValueError(f'Native {m} summary changed')
    dest = out / f'{split}_results.json'
    report = dict(split=split, smoke=cfg['smoke'], centers_sha256=sha256_file(lock), native_result_sha256=sha256_file(native_path),
                  stages=stages, summary=summary, comparisons=comparisons(summary, ['A'], split, cfg['smoke']))
    if split == 'test':
        report['validation_seal_sha256'] = sha256_file(seal)
    write_verified(dest, report)
    if split == 'val':
        write_verified(seal, dict(centers_sha256=sha256_file(lock), val_results_sha256=sha256_file(dest),
                                  native_val_result_sha256=report['native_result_sha256']))
    print(split, {m: {k: summary[m]['all'][k] for k in MAP_KEYS} for m in METHODS}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--root', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--phase', choices=('fit', 'val', 'test'), required=True)
    cli.add_argument('--task', type=int)
    cli.add_argument('--workers', type=int, default=2)
    cli.add_argument('--replay-meta', action='store_true',
                     help='Read frozen metadata-split checkpoints in task order and fit fresh held-TRAIN centers')
    args = cli.parse_args()
    cfg = read(args.root / 'config.json')
    validate_source(cfg, args.replay_meta)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'fit':
        fit(args.root, args.out, cfg, args.task, args.workers, args.replay_meta)
    else:
        if args.task is not None:
            cli.error('--task is used only with --phase fit')
        evaluate(args.root, args.out, cfg, args.phase, args.replay_meta)
