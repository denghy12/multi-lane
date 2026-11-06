"""Repeat the locked Frozen9 Person + Face recipe across seeds."""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import statistics
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from multi_lane.track_a.evaluation_scores import align_evaluation_scores, load_evaluation_scores
from multi_lane.track_a.r1_crossvit_experiment import flags, model_from_config, read, sha, write
from multi_lane.track_a.runner import dataset_view, git_metadata, seen_indices, set_seed
from scripts.emotic.run_r1_frozen_cross_view import check_training, input_paths, load_source, preflight, restore
from scripts.emotic.run_r1_frozen_late_stable import summarize_family
from scripts.emotic.run_r1_patch_receiver_depth import LAYER9, REFERENCE, dispatch


ROOT = Path('output/r1_frozen9_seed_sweep/20261106')
OTHER = Path('../multi-lane-main-r1-frozen9-task10-pool-20261106/output/r1_frozen9_task10_pool/seed1_20261106')
SEEDS = (0, 2, 3, 4)
SEED1_RESULT = LAYER9.parent / 'layer9_six_group_results.json'
FULL_HISTORY = {
    0: Path('multi_lane_dual_view_formal_v0.1/image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816/image_token_layer1_full_person_letterbox_formal_test_seed0_20260903_145816_full_anchor'),
    2: Path('multi_lane_dual_view_seed_confirmation_v0.1/dual_view_locked_formal_seed12_20260903_174938/seed2/dual_view_locked_formal_seed12_20260903_174938_seed2_full_anchor'),
}


@dataclass(frozen=True)
class Job:
    seed: int
    view: str
    gpu: int

    @property
    def name(self):
        return f'seed{self.seed}_{self.view}'

    @property
    def kind(self):
        return 'native' if self.view == 'full' else 'attention'

    def path(self, root):
        return root / f'seed{self.seed}' / self.kind / self.view


JOBS = tuple(Job(seed, view, (i + offset) % 2)
             for i, seed in enumerate(SEEDS)
             for offset, view in enumerate(('person', 'face', 'full')))


def command_for(job, root, smoke):
    command = [sys.executable, '-u', '-m', 'scripts.emotic.run_r1_frozen9_seed_sweep',
               '--root', str(root), '--worker', job.name]
    return command + ['--smoke'] if smoke else command


def native_preflight(train, inputs, destination):
    cfg = read(train / 'config.json')
    set_seed(cfg['seed'], True)
    torch.set_num_threads(2)
    model = model_from_config(cfg, inputs.clip_checkpoint)
    restore(model, train, 1)
    model.restore_task(7)
    model.requires_grad_(False).eval()
    data = dataset_view(load_source(train, inputs, 'val').source, seen_indices(1))
    images, _ = next(iter(DataLoader(Subset(data, range(64)), batch_size=64)))
    with torch.no_grad(), torch.autocast('cuda'):
        logits = model.seen_logits(images.cuda())
    if logits.shape != (64, 26) or not torch.isfinite(logits).all():
        raise ValueError('Invalid native eight-task preflight')
    measured = max(row['cuda_peak_reserved_mib'] for rows in
                   read(train / 'training_history.json').values() for row in rows)
    write(destination, dict(status='passed', eight_lane_shape=list(logits.shape),
        smoke_train_peak_reserved_mib=max(measured, torch.cuda.max_memory_reserved() / 1024**2)))


def full_history_replay(train, archive):
    rows = []
    for task in range(8):
        name = f'task{task}.npz'
        old = load_evaluation_scores(archive / 'test_scores' / name, 64)
        new = load_evaluation_scores(train / 'test_scores' / name, 64)
        aligned = align_evaluation_scores(old, new)[2]
        rows.append(dict(task=task, exact=bool(np.array_equal(old.logits, aligned)),
                         max_abs=float(np.abs(old.logits - aligned).max())))
    return rows


def worker(job, root, smoke):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(job.gpu):
        raise ValueError('Worker must use its assigned GPU')
    inputs = input_paths()
    path = job.path(root)
    train = path / 'train'
    command = [sys.executable, '-u', '-m', 'multi_lane.track_a.runner',
               *flags(SimpleNamespace(output=root / f'seed{job.seed}', **vars(inputs)),
                      job.kind, job.view, smoke, seed=job.seed),
               '--r1-fusion-readout', 'logit']
    if job.kind == 'attention':
        command += ['--r1-frozen-mode', 'patch', '--r1-exchange-layer', '8',
                    '--r1-fixed-loss-scale', '4096']
    if smoke:
        command += ['--r1-smoke-eval-samples', '64']
    env = dict(os.environ, OMP_NUM_THREADS='72', MKL_NUM_THREADS='72', OPENBLAS_NUM_THREADS='1')
    print('TRAIN', ' '.join(command), flush=True)
    subprocess.run(command, env=env, check=True)
    counts = check_training(train, REFERENCE, job.view, 2 if smoke else 8, smoke)
    if smoke:
        if job.view == 'full':
            native_preflight(train, inputs, path / 'preflight.json')
        else:
            preflight(train, inputs, path / 'preflight.json')
    else:
        if job.view == 'full' and job.seed in FULL_HISTORY:
            write(path / 'historical_test_replay.json',
                  dict(rows=full_history_replay(train, inputs.reference_root / FULL_HISTORY[job.seed])))
        subprocess.run([sys.executable, '-u', '-m', 'multi_lane.track_a.r1_crossvit_experiment',
            'val', '--training-root', str(train), '--output', str(path / 'val'),
            '--data-root', str(inputs.data_root), '--clip-checkpoint', str(inputs.clip_checkpoint),
            '--face-manifest-root', str(inputs.face_manifest_root)], check=True)
    write(path / 'completion.json', dict(status='complete', seed=job.seed, training=counts, gpu=job.gpu))


def capacity_limit():
    raw = subprocess.check_output(['nvidia-smi', '-i', '0,1',
        '--query-gpu=index,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
    total = {i: used + free for i, used, free in
             (map(int, line.split(',')) for line in raw.splitlines())}
    queue = read(OTHER / 'queue.json')
    for name, job in queue['running'].items():
        total[job['gpu']] -= queue['budgets_mib'][name]
    return {gpu: value - 512 for gpu, value in total.items()}


def result_tables(root, formal, manifest):
    historical = read(SEED1_RESULT)['results']
    results = {'1': {split: group['patch_person_face'] for split, group in historical.items()}}
    for seed in SEEDS:
        selected = [job for job in JOBS if job.seed == seed]
        if not all(formal.get(job.name) == 'complete' for job in selected):
            continue
        reference = root / f'seed{seed}'
        destination = reference / 'fusion.json'
        summarize_family(root, reference, manifest,
            {('patch', job.view): job.path(root) for job in selected if job.view != 'full'},
            destination, modes=('patch',), directions=('person_face',))
        outcome = {s: v['patch_person_face'] for s, v in read(destination)['results'].items()}
        if any(v['status'] != 'complete' for v in outcome.values()):
            raise ValueError(f'Incomplete scores for seed {seed}')
        results[str(seed)] = outcome
    metrics = {f'{split}_{key}': {seed: table[split]['populations']['all']['summary'][key]
        for seed, table in results.items()}
        for split in ('val', 'test') for key in ('final_mAP', 'average_mAP')}
    rankings = {key: sorted(values, key=lambda seed: (-values[seed], int(seed)))
                for key, values in metrics.items()}
    statistics_by_metric = {key: dict(mean=statistics.mean(v.values()),
        sample_std=statistics.stdev(v.values()) if len(v) > 1 else None)
        for key, v in metrics.items()}
    return dict(results=results, values=metrics, rankings=rankings, statistics=statistics_by_metric,
                validation_preferred_seed=rankings['val_average_mAP'][0],
                seed1_reused=True, completed_seeds=sorted(map(int, results)))


def run(root):
    git = git_metadata(Path.cwd())
    if git['dirty']:
        raise ValueError('Clean experiment worktree required')
    inputs = input_paths()
    logs = Path('logs/r1_frozen9_seed_sweep') / root.name
    sources = [SEED1_RESULT]
    for p in FULL_HISTORY.values():
        sources.extend(inputs.reference_root / p / 'test_scores' / f'task{t}.npz' for t in range(8))
    hashes = {str(p): sha(p) for p in sources}
    root.mkdir(parents=True, exist_ok=False)
    for stage in ('smoke', 'formal'):
        (logs / stage).mkdir(parents=True, exist_ok=False)
    write(root / 'protocol.json', dict(git=git, seeds=SEEDS, seed1_reused=True,
        tasks=8, epochs=30, batch=64, optimizer='Adam', scheduler='cosine', weight_decay=0,
        main_lr=.0125, new_lr=.0004, patch_after=9, patch_cap=.05, patch_amp_scale=4096,
        full_amp='historical_dynamic', jobs=[vars(job) for job in JOBS],
        reliable_weights=[.64,.16,.20], fallback_weights=[.8,.2,0],
        data=str(inputs.data_root), pretrained=str(inputs.clip_checkpoint),
        face_manifest=str(inputs.face_manifest_root), source_hashes=hashes))
    smoke = dispatch(JOBS, root / 'smoke', logs / 'smoke', REFERENCE,
        {job.name: 4200 if job.view == 'full' else 5700 for job in JOBS}, True,
        reserve_peak=True, worker_command=command_for, capacity_limit=capacity_limit)
    passed = [job for job in JOBS if smoke[job.name] == 'complete']
    budgets = {job.name: int(read(job.path(root / 'smoke') / 'preflight.json')['smoke_train_peak_reserved_mib']) + 1024
               for job in passed}
    write(root / 'preflight_gate.json', dict(states=smoke, formal_budgets_mib=budgets))
    formal = dispatch(passed, root, logs / 'formal', REFERENCE, budgets, False,
        reserve_peak=True, worker_command=command_for, capacity_limit=capacity_limit)
    torch.set_num_threads(2)
    if any(sha(Path(p)) != h for p, h in hashes.items()):
        raise ValueError('Historical predictions changed')
    result = result_tables(root, formal, inputs.face_manifest_root)
    write(root / 'frozen9_seed_sweep_results.json', result)
    write(root / 'completion.json', dict(status='complete' if len(result['completed_seeds']) == 5 else 'partial',
                                         smoke=smoke, formal=formal, source_hashes_verified=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--worker', choices=[job.name for job in JOBS])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker is None:
        run(args.root)
    else:
        worker(next(job for job in JOBS if job.name == args.worker), args.root, args.smoke)
