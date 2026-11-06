"""Compare Frozen9 patch and Task10 Pool on the Person and Face R1 paths."""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

from multi_lane.track_a.r1_crossvit_experiment import flags, read, sha, write
from multi_lane.track_a.runner import git_metadata
from scripts.emotic.r1_two_layer_attribution import TASKS
from scripts.emotic.run_r1_frozen_cross_view import check_training, input_paths, preflight
from scripts.emotic.run_r1_frozen_late_stable import summarize_family
from scripts.emotic.run_r1_patch_receiver_depth import LAYER9, REFERENCE, dispatch, source_paths


DEFAULT_ROOT = Path('output/r1_frozen9_task10_pool/seed1_20261106')
POOL_CAP = .025


@dataclass(frozen=True)
class Job:
    arm: str
    view: str
    gpu: int

    @property
    def name(self):
        return f'{self.arm}_{self.view}'

    def path(self, root):
        return root / self.arm / 'attention' / self.view


JOBS = (
    Job('patch_pool', 'face', 0), Job('patch_pool', 'person', 1),
    Job('pool_only', 'person', 0), Job('pool_only', 'face', 1),
)


def worker_command(job, root, smoke):
    command = [sys.executable, '-u', '-m', 'scripts.emotic.run_r1_frozen9_task10_pool',
               '--root', str(root), '--worker', job.name]
    return command + ['--smoke'] if smoke else command


def worker(job, root, smoke):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(job.gpu):
        raise ValueError('Worker must use its scheduled GPU')
    inputs = input_paths()
    path = job.path(root)
    train = path / 'train'
    command = [sys.executable, '-u', '-m', 'multi_lane.track_a.runner',
               *flags(SimpleNamespace(output=root / job.arm, **vars(inputs)), 'attention', job.view, smoke),
               '--r1-frozen-mode', 'patch', '--r1-exchange-layer', '8',
               '--r1-task-pool-layer', '9', '--r1-task-pool-source-layer', '8',
               '--r1-task-pool-cap', str(POOL_CAP), '--r1-fixed-loss-scale', '4096',
               '--r1-fusion-readout', 'logit']
    if job.arm == 'pool_only':
        command.append('--r1-disable-frozen-patch')
    if smoke:
        command += ['--r1-smoke-eval-samples', '64']
    env = dict(os.environ, OMP_NUM_THREADS='72', MKL_NUM_THREADS='72', OPENBLAS_NUM_THREADS='1')
    print('TRAIN', ' '.join(command), flush=True)
    subprocess.run(command, env=env, check=True)
    counts = check_training(train, REFERENCE, job.view, 2 if smoke else 8, smoke)
    if smoke:
        preflight(train, inputs, path / 'preflight.json')
    else:
        subprocess.run([sys.executable, '-u', '-m', 'multi_lane.track_a.r1_crossvit_experiment',
            'val', '--training-root', str(train), '--output', str(path / 'val'),
            '--data-root', str(inputs.data_root), '--clip-checkpoint', str(inputs.clip_checkpoint),
            '--face-manifest-root', str(inputs.face_manifest_root)], check=True)
    write(path / 'completion.json', dict(status='complete', training=counts, gpu=job.gpu))
    if not smoke and job.arm == 'patch_pool':
        result = subprocess.run([sys.executable, '-u', '-m', 'scripts.emotic.r1_two_layer_attribution',
            '--training-root', str(train), '--output', str(path / 'attribution')])
        if result.returncode:
            write(path / 'attribution_failure.json', dict(exit_code=result.returncode))


def run(root):
    git = git_metadata(Path.cwd())
    if git['dirty']:
        raise ValueError('Clean experiment worktree required')
    inputs = input_paths()
    logs = Path('logs/r1_frozen9_task10_pool') / root.name
    hashes = {str(p): sha(p) for p in source_paths(REFERENCE, LAYER9, inputs.face_manifest_root)}
    root.mkdir(parents=True, exist_ok=False)
    for stage in ('smoke', 'formal'):
        (logs / stage).mkdir(parents=True, exist_ok=False)
    write(root / 'protocol.json', dict(git=git, seed=1, tasks=8, epochs=30, batch=64,
        optimizer='Adam', scheduler='cosine', weight_decay=0, main_lr=.0125, new_lr=.0004,
        patch_after=9, patch_cap=.05, task_pool_after=10, pool_sender_block=9, pool_cap=POOL_CAP,
        amp_scale=4096, attribution_tasks=TASKS, jobs=[vars(job) for job in JOBS],
        reliable_weights=[.64, .16, .20], fallback_weights=[.8, .2, 0.],
        data=str(inputs.data_root), pretrained=str(inputs.clip_checkpoint),
        face_manifest=str(inputs.face_manifest_root)))
    write(root / 'source_hashes.json', hashes)
    smoke = dispatch(JOBS, root / 'smoke', logs / 'smoke', REFERENCE,
                     {job.name: 8000 for job in JOBS}, True, worker_command=worker_command)
    passed = [job for job in JOBS if smoke[job.name] == 'complete']
    budgets = {job.name: int(read(job.path(root / 'smoke') / 'preflight.json')['smoke_train_peak_reserved_mib']) + 1024
               for job in passed}
    write(root / 'preflight_gate.json', dict(states=smoke, formal_budgets_mib=budgets))
    formal = dispatch(passed, root, logs / 'formal', REFERENCE, budgets, False,
                      reserve_peak=True, worker_command=worker_command)
    torch.set_num_threads(2)

    def summary(name, person, face, tasks=tuple(range(8))):
        destination = root / 'fusion' / f'{name}.json'
        summarize_family(root, REFERENCE, inputs.face_manifest_root,
            {('patch', 'person'): person, ('patch', 'face'): face}, destination,
            modes=('patch',), directions=('person_face',), tasks=tasks)
        result = {s: d['patch_person_face'] for s, d in read(destination)['results'].items()}
        if any(d['status'] != 'complete' for d in result.values()):
            raise ValueError(f'Missing predictions for {name}')
        return result

    p0, f0 = REFERENCE / 'native/person', REFERENCE / 'native/face'
    p9, f9 = LAYER9 / 'patch/attention/person', LAYER9 / 'patch/attention/face'
    paths = {job.name: job.path(root) for job in JOBS}
    pairs = {
        'R1': (p0, f0), 'historical_face_pool': (p0, REFERENCE / 'pool/face'),
        'frozen9_both': (p9, f9),
        'frozen9_plus_task10_face': (p9, paths['patch_pool_face']),
        'frozen9_plus_task10_both': (paths['patch_pool_person'], paths['patch_pool_face']),
        'task10_pool_only_both': (paths['pool_only_person'], paths['pool_only_face']),
    }
    missing = {paths[job.name] for job in JOBS if formal.get(job.name) != 'complete'}
    results = {}
    for name, (person, face) in pairs.items():
        results[name] = summary(name, person, face) if not ({person, face} & missing) else {'status': 'incomplete'}
    write(root / 'frozen9_task10_pool_results.json', dict(results=results))

    attribution = {}
    for name in ('frozen9_plus_task10_face', 'frozen9_plus_task10_both'):
        person, face = pairs[name]
        new_paths = [path for path in (person, face) if path != p9]
        complete = all((path / 'attribution/completion.json').is_file() for path in new_paths)
        attribution[name] = 'complete' if complete else 'incomplete'
        if complete:
            summary(f'{name}_both_on', person, face, TASKS)
            for mode in ('pool_only', 'patch_only', 'both_off'):
                summary(f'{name}_{mode}', person if person == p9 else person / 'attribution' / mode,
                        face / 'attribution' / mode, TASKS)
    if any(sha(Path(p)) != h for p, h in hashes.items()):
        raise AssertionError('Reference artifacts changed')
    write(root / 'completion.json', dict(status='complete' if not missing else 'partial',
                                         smoke=smoke, formal=formal, attribution=attribution))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--worker', choices=[job.name for job in JOBS])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker is None:
        run(args.root)
    else:
        worker(next(job for job in JOBS if job.name == args.worker), args.root, args.smoke)
