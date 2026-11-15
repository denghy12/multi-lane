"""Paired branch-only refinements of the historical seed1 Frozen9 D model."""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from multi_lane.track_a.r1_crossvit_experiment import flags, read, sha, write
from multi_lane.track_a.runner import git_metadata
from scripts.emotic.run_r1_frozen_cross_view import check_training, input_paths, preflight, score_suffix
from scripts.emotic.run_r1_patch_receiver_depth import REFERENCE, dispatch

ROOT = Path('output/d9_branch_refinement/seed1_20261115')
LOGS = Path('logs/d9_branch_refinement')
ANCHOR = Path('../multi-lane-main-r1-frozen9-edges-20261108/output/r1_frozen9_edges/seed1_20261108')
MODULE = 'scripts.emotic.run_d9_branch_refinement'


@dataclass(frozen=True)
class Branch:
    arm: str
    view: str
    gpu: int

    @property
    def name(self):
        return f'{self.arm}_{self.view}'

    @property
    def kind(self):
        return 'native' if self.view == 'full' else 'attention'

    def path(self, root):
        return root / self.arm / self.kind / self.view


JOBS = (Branch('sync', 'person', 0), Branch('sync', 'face', 1),
        Branch('independent', 'person', 0), Branch('independent', 'face', 1),
        Branch('cap035', 'person', 0), Branch('box', 'full', 0))


def train_flags(job, root, smoke, inputs):
    recipe = flags(SimpleNamespace(output=root / job.arm, **vars(inputs)), job.kind, job.view, smoke)
    recipe += ['--r1-fusion-readout', 'logit']
    if job.kind == 'native':
        recipe += ['--d9-full-box']
    else:
        recipe += ['--r1-frozen-mode', 'patch', '--r1-exchange-layer', '8', '--r1-fixed-loss-scale', '4096']
        if job.arm == 'cap035':
            recipe += ['--r1-frozen-cap', '.035']
        else:
            recipe += ['--r1-evidence-flip', job.arm]
    if smoke:
        recipe += ['--r1-smoke-eval-samples', '64']
    return recipe


def execute(module, args, threads=2):
    command = [sys.executable, '-u', '-m', module, *map(str, args)]
    print('RUN', ' '.join(command), flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1')
    subprocess.run(command, env=env, check=True)


def worker(job, root, smoke):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(job.gpu):
        raise RuntimeError(f'{job.name} was assigned GPU {job.gpu}')
    path = job.path(root)
    train = path / 'train'
    inputs = input_paths()
    execute('multi_lane.track_a.runner', train_flags(job, root, smoke, inputs), threads=72)
    counts = check_training(train, REFERENCE, job.view, 2 if smoke else 8, smoke)
    model_args = ['--training-root', train, '--data-root', inputs.data_root, '--clip-checkpoint', inputs.clip_checkpoint]
    small = ['--smoke'] if smoke else []
    if job.kind == 'attention':
        if smoke:
            preflight(train, inputs, path / 'preflight.json')
        else:
            execute('multi_lane.track_a.r1_crossvit_experiment', [
                'val', *model_args, '--output', path / 'val', '--face-manifest-root', inputs.face_manifest_root])
    else:
        execute('multi_lane.track_a.d9_full_eval', [*model_args, '--output', path / 'val', '--split', 'val',
                                                    '--d9-box-mode', 'true', *small])
        for split, suffix in [('val', 'val/scores'), ('test', 'train/test_scores')]:
            swap = root / 'box_swap/native/full'
            execute('multi_lane.track_a.d9_full_eval', [*model_args, '--output', swap / (split + '_audit'),
                    '--score-dir', swap / suffix, '--split', split, '--d9-box-mode', 'swapped', *small])
    write(path / 'completion.json', dict(status='complete', seed=1, branch=job.name, gpu=job.gpu, training=counts))


def command_for(job, root, smoke):
    return [sys.executable, '-u', '-m', MODULE, '--root', str(root), '--worker', job.name] + (['--smoke'] if smoke else [])


def launch(root):
    logs = LOGS / root.name
    root.mkdir(parents=True, exist_ok=False)
    for phase in ('smoke', 'formal'):
        (logs / phase).mkdir(parents=True, exist_ok=True)
    inputs = input_paths()
    paths = [ANCHOR / 'evaluation/trained/branches' / source / score_suffix(split, t)
             for source in ('full', 'person_all', 'face_all') for split in ('val', 'test') for t in range(8)]
    paths += [inputs.face_manifest_root / 'manifests' / (split + '.jsonl') for split in ('train', 'val', 'test')]
    hashes = {str(p): sha(p) for p in paths}
    write(root / 'protocol.json', dict(git=git_metadata(Path.cwd()), seed=1, tasks=8, epochs=30,
          batch_size=64, cap_D=.05, cap035=.035, box_layer=10, box_selectors=2,
          tasks_per_view=[vars(job) for job in JOBS], reliable_weights=[.64,.16,.20], fallback_weights=[.8,.2,0.],
          clip=str(inputs.clip_checkpoint), data=str(inputs.data_root), anchor=str(ANCHOR), sources=hashes,
          training_flags={job.name:train_flags(job, root, False, inputs) for job in JOBS}))
    write(root / 'stage.json', {'stage':'replay_D'})
    execute('scripts.emotic.summarize_d9_refinement', ['--root', root, '--anchor', ANCHOR,
            '--manifest', inputs.face_manifest_root, '--anchor-only'])
    sizes = {job.name: 6000 for job in JOBS}
    for smoke, phase in [(True, 'smoke'), (False, 'formal')]:
        destination = root / 'smoke' if smoke else root
        write(root / 'stage.json', {'stage': phase})
        outcomes = dispatch(JOBS, destination, logs / phase, REFERENCE, sizes, smoke,
                            reserve_peak=True, worker_command=command_for)
        if any(status != 'complete' for status in outcomes.values()):
            write(root / 'completion.json', {'status':'failed', 'phase':phase, 'outcomes':outcomes})
            raise RuntimeError(f'{phase} incomplete: {outcomes}')
        if smoke:
            for job in JOBS:
                if job.kind == 'attention':
                    probe = read(job.path(destination) / 'preflight.json')
                    sizes[job.name] = max(5500, int(probe['admission_memory_mib']))
            write(root / 'preflight_gate.json', {'states': outcomes, 'formal_budgets_mib': sizes})
    write(root / 'stage.json', {'stage':'summarize'})
    execute('scripts.emotic.summarize_d9_refinement', ['--root', root, '--anchor', ANCHOR, '--manifest',inputs.face_manifest_root])
    if any(sha(Path(p)) != digest for p, digest in hashes.items()):
        raise ValueError('Anchor inputs changed during evaluation')
    write(root / 'completion.json', {'status':'complete', 'anchor_hashes_verified':True})
    write(root / 'stage.json', {'stage':'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--worker', choices=[job.name for job in JOBS])
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(next(job for job in JOBS if job.name == args.worker), args.root, args.smoke)
    else:
        launch(args.root)
