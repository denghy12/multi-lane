"""Train the single Face path with Task Forward Pool and Frozen patch exchange."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

from multi_lane.track_a.r1_crossvit_experiment import flags, read, sha, write
from multi_lane.track_a.runner import git_metadata
from scripts.emotic.r1_two_layer_attribution import TASKS, evaluate_switches
from scripts.emotic.run_r1_frozen_cross_view import check_training, input_paths, preflight
from scripts.emotic.run_r1_frozen_late_stable import free_memory, summarize_family
from scripts.emotic.run_r1_patch_receiver_depth import LAYER9, REFERENCE, source_paths


def train_face(root, inputs, logs, smoke):
    train = root / 'attention/face/train'
    command = [sys.executable, '-u', '-m', 'multi_lane.track_a.runner',
               *flags(SimpleNamespace(output=root, **vars(inputs)), 'attention', 'face', smoke),
               '--r1-frozen-mode', 'patch', '--r1-exchange-layer', '8', '--r1-task-pool-layer', '5',
               '--r1-fixed-loss-scale', '4096', '--r1-fusion-readout', 'logit']
    if smoke:
        command += ['--r1-smoke-eval-samples', '64']
    env = dict(os.environ, OMP_NUM_THREADS='72', MKL_NUM_THREADS='72', OPENBLAS_NUM_THREADS='1')
    with (logs / ('smoke.log' if smoke else 'training.log')).open('x') as handle:
        print('TRAIN', ' '.join(command), file=handle, flush=True)
        subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/r1_face_pool_patch/seed1_20261106'))
    args = parser.parse_args()
    root = args.root
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('This run is assigned to GPU0')
    git = git_metadata(Path.cwd())
    if git['dirty']:
        raise ValueError('Clean experiment worktree required')
    inputs = input_paths()
    logs = Path('logs/r1_face_pool_patch') / root.name
    hashes = {str(p): sha(p) for p in source_paths(REFERENCE, LAYER9, inputs.face_manifest_root)}
    root.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=True)
    write(root / 'protocol.json', dict(git=git, seed=1, tasks=8, epochs=30, batch=64,
        optimizer='Adam', scheduler='cosine', weight_decay=0, main_lr=.0125, new_lr=.0004,
        task_pool_after=6, patch_after=9, pool_cap=None, patch_cap=.05, amp_scale=4096,
        reliable_weights=[.64,.16,.20], fallback_weights=[.8,.2,0],
        data=str(inputs.data_root), pretrained=str(inputs.clip_checkpoint),
        face_manifest=str(inputs.face_manifest_root), source_hashes=hashes))
    if free_memory() < 8000:
        raise RuntimeError('GPU0 needs 8000 MiB free for the real preflight')
    smoke = train_face(root / 'smoke', inputs, logs, smoke=True)
    check_training(smoke, REFERENCE, 'face', 2, True)
    preflight(smoke, inputs, root / 'preflight.json')
    torch.cuda.empty_cache()
    peak = read(root / 'preflight.json')['smoke_train_peak_reserved_mib']
    if free_memory() < peak + 1536:
        raise RuntimeError('GPU0 lacks measured capacity for the full run')

    train = train_face(root, inputs, logs, smoke=False)
    write(root / 'training_checks.json', check_training(train, REFERENCE, 'face', 8, False))
    with (logs / 'validation.log').open('x') as handle:
        subprocess.run([sys.executable, '-u', '-m', 'multi_lane.track_a.r1_crossvit_experiment', 'val',
            '--training-root', str(train), '--output', str(train.parent / 'val'),
            '--data-root', str(inputs.data_root), '--clip-checkpoint', str(inputs.clip_checkpoint),
            '--face-manifest-root', str(inputs.face_manifest_root)],
            stdout=handle, stderr=subprocess.STDOUT, check=True)

    torch.set_num_threads(2)
    def summarize(name, face, tasks=tuple(range(8))):
        summarize_family(root, REFERENCE, inputs.face_manifest_root,
            {('patch', 'face'): face}, root / 'fusion' / (name+'.json'),
            modes=('patch',), directions=('face_only',), tasks=tasks)
    for name, face in (('R1', REFERENCE / 'native/face'), ('pool', REFERENCE / 'pool/face'),
                       ('patch', LAYER9 / 'patch/attention/face'), ('combined', train.parent)):
        summarize(name, face)
    write(root / 'formal_completion.json', dict(status='complete'))
    evaluate_switches(train, root / 'attribution')
    for mode in ('pool_only', 'patch_only', 'both_off'):
        summarize(mode, root / 'attribution' / mode, tasks=TASKS)
    summarize('both_on', train.parent, tasks=TASKS)
    if any(sha(Path(p)) != digest for p, digest in hashes.items()):
        raise AssertionError('Reference artifacts changed')
    write(root / 'completion.json', dict(status='complete', attribution_tasks=TASKS, source_hashes_verified=True))


if __name__ == '__main__':
    main()
