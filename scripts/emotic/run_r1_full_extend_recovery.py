"""Explicit numerical recovery of the same full-TRAIN control from a clean boundary.

Copies an immutable safe prefix into a fresh output. No failed epoch is imported.
Only Person loss-scale policy changes after the boundary; data, Adam, scheduler,
RNG, heads and all branch model tensors are inherited from this SAME trajectory.
"""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from multi_lane.track_a import runner
from multi_lane.track_a.r1_crossvit import R1Receiver
from multi_lane.track_a.r1_crossvit_experiment import read, sha, write
from scripts.emotic import run_r1_fusion_return_train as base
from scripts.emotic.run_r1_single_trajectory_fusion import VIEWS, branch_paths

BOUNDARY_SHA = '15d570f6d928854312d473ccdf0a41de3110ce111fc964581feeadcbcceedfd8'


def prefix_paths(source, task, epoch, smoke=False):
    """Whitelist only accepted complete states and data provenance."""
    total = sum(base.phase_sizes(smoke))
    paths = []
    for t in range(task+1):
        end = epoch if t == task else total
        for e in range(1, end+1):
            paths.append(source/'recovery'/f'task{t}'/f'epoch{e:02d}.pth')
            paths.append(source/'barriers'/f'task{t}'/f'epoch{e:02d}_accepted.json')
            for view, branch in branch_paths(source).items():
                paths.append(branch/'epoch_recovery'/f'task{t}'/f'epoch{e:02d}.pth')
                paths.append(source/'barriers'/f'task{t}'/f'epoch{e:02d}_{view}.json')
        for view, branch in branch_paths(source).items():
            paths.append(source/'exposure'/view/f'task{t}.json')
            if t < task:
                paths.append(branch/'task_recovery'/f'task{t}.pth')
                paths.append(branch/'compact_checkpoints'/f'task{t}.pth')
        if t < task:
            paths.append(source/'checkpoints'/f'task{t}.pth')
    return paths


def validate_boundary(whole, task, epoch):
    if whole['condition'] != 'full_extend' or whole['task'] != task or whole['epoch'] != epoch:
        raise ValueError('Wrong whole trajectory boundary')
    for view in VIEWS:
        state = whole['branches'][view]
        required_training = {'optimizer', 'scheduler', 'scaler', 'history',
            'next_epoch', 'completed_task_updates', 'optimizer_parameter_names'}
        if not required_training.issubset(state.get('training', {})):
            raise ValueError('Incomplete training recovery state')
        if not {'python', 'numpy', 'cpu', 'cuda'}.issubset(state.get('rng', {})):
            raise ValueError('Incomplete RNG recovery state')
        if state['task'] != task or state['training']['next_epoch'] != epoch:
            raise ValueError('Branch boundary mismatch')
        if len(state['training']['history']) != epoch or any(
                r['skipped_optimizer_steps'] for r in state['training']['history']):
            raise ValueError('Unsafe skipped/incomplete boundary')
        for value in state['model'].values():
            if torch.is_floating_point(value) and not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite boundary model')
        for group in state['training']['optimizer']['state'].values():
            if any(torch.is_tensor(v) and torch.is_floating_point(v) and not torch.isfinite(v).all()
                   for v in group.values()):
                raise FloatingPointError('Nonfinite boundary Adam')


def prepare(source, root, diagnostic, person_scale, smoke=False):
    if root.exists():
        raise FileExistsError('New recovery output must not exist')
    task, epoch = (0, 1) if smoke else (3, 13)
    boundary = source/'recovery'/f'task{task}'/f'epoch{epoch:02d}.pth'
    boundary_sha = sha(boundary)
    if not smoke and boundary_sha != BOUNDARY_SHA:
        raise ValueError('Unexpected source boundary SHA')
    whole = torch.load(boundary, map_location='cpu')
    validate_boundary(whole, task, epoch)
    if person_scale not in (32., 4096.):
        raise ValueError('Policy must be an explicitly scanned scale')
    diagnostic_sha = None
    if not smoke:
        if diagnostic is None:
            raise ValueError('Full next-epoch read-only gradient diagnostic required')
        d = read(diagnostic)
        mode = 'amp32' if person_scale == 32. else 'amp4096'
        if (d['status'] != 'complete' or not d['all_expected_batches_scanned']
                or d['source_sha256'] != boundary_sha or d['view'] != 'person'
                or d['mode_nonfinite_batches'][mode] != 0
                or not all(d[k] for k in ('source_sha_unchanged','model_unchanged','optimizer_unchanged'))):
            raise ValueError('Chosen numerical policy did not pass the read-only scan')
        diagnostic_sha = sha(diagnostic)
    expected = prefix_paths(source, task, epoch, smoke)
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing safe prefix: {missing[:5]}')
    root.mkdir(parents=True)
    hashes = {}
    for path in expected:
        relative = path.relative_to(source)
        target = root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        source_sha = sha(path)
        shutil.copy2(path, target)
        if sha(target) != source_sha:
            raise AssertionError(f'Prefix copy differs: {relative}')
        hashes[str(relative)] = source_sha
    new_boundary = root/boundary.relative_to(source)
    write(root/'checkpoints/latest.json', {'path': str(new_boundary), 'task': task, 'epoch': epoch})
    provenance = {'source_root': str(source), 'source_whole_checkpoint': str(boundary),
        'source_sha256': boundary_sha, 'task': task, 'completed_epoch': epoch,
        'source_failed_epoch_not_copied': epoch+1, 'same_training_trajectory': True,
        'person_loss_scale_after_boundary': person_scale, 'source_person_loss_scale': 4096.,
        'unchanged_views': ['full','face'], 'optimizer_moments_and_rng_retained': True,
        'diagnostic_path': str(diagnostic) if diagnostic else None, 'diagnostic_sha256': diagnostic_sha,
        'diagnostic_limit': 'Fixed-parameter scan; resumed effective updates must be checked separately',
        'numerical_recovery_revalidation_not_unmodified_original_run': True,
        'prefix_files': hashes, 'copied_bytes': sum((root/p).stat().st_size for p in hashes),
        'git': runner.git_metadata(Path.cwd()), 'smoke': smoke}
    write(root/'recovery_provenance.json', provenance)
    verify_prefix(root)
    return provenance


def verify_prefix(root):
    manifest = read(root/'recovery_provenance.json')
    source = Path(manifest['source_root'])
    for relative, digest in manifest['prefix_files'].items():
        if sha(source/relative) != digest or sha(root/relative) != digest:
            raise AssertionError(f'Immutable source/copy changed: {relative}')
    return manifest


def permitted_config_delta(saved, current, view):
    allowed = {'git'}
    if view == 'person':
        allowed.update(('r1_fixed_loss_scale','r1_loss_scale_growth_interval'))
    missing = object()
    changes = {k for k in saved.keys() | current.keys() if saved.get(k, missing) != current.get(k, missing)}
    if changes-allowed:
        raise ValueError(f'Recovery recipe changed beyond numeric policy: {sorted(changes-allowed)}')
    return sorted(changes)


def configure_runtime(root):
    manifest = read(root/'recovery_provenance.json')
    scale = float(manifest['person_loss_scale_after_boundary'])
    original_flags = base.branch_flags

    def flags(root, view, smoke=False, condition='full_extend'):
        if condition != 'full_extend':
            raise ValueError('Numerical continuation is only for full_extend')
        values = original_flags(root, view, smoke, condition)
        if view == 'person':
            values[values.index('--r1-fixed-loss-scale')+1] = str(scale)
        return values
    base.branch_flags = flags
    original_load = R1Receiver.load_own

    def load_same_trajectory(self, saved):
        changes = permitted_config_delta(saved['config'], self.r1_config, self.r1_view)
        # Only metadata is adapted in memory; tensors, Adam, RNG and source files
        # are unchanged. Native load_own then performs its normal strict check.
        adapted = dict(saved, config=self.r1_config)
        original_load(self, adapted)
        if changes and self.r1_root:
            write(self.r1_root/f'recovery_config_task{saved["task"]}.json', {
                'allowed_changes': changes, 'source_config': saved['config'],
                'current_config': self.r1_config, 'no_parameter_transplant': True})
    R1Receiver.load_own = load_same_trajectory
    original_train = runner.train_task

    def train_preserving_scaler(model, *args, **kwargs):
        resume = getattr(model, 'r1_resume', None)
        old_fixed = getattr(model, 'r1_fixed_loss_scale', None)
        if resume is None:
            return original_train(model, *args, **kwargs)
        # Existing runner initializes a fixed scale after restoring state. For
        # this continuation preserve its growth counter and all other fields;
        # only Person's explicitly declared scale changes at the boundary.
        original_resume = resume
        model.r1_resume = copy.deepcopy(resume)
        if model.r1_view == 'person':
            model.r1_resume['scaler']['scale'] = scale
        model.r1_fixed_loss_scale = None
        try:
            return original_train(model, *args, **kwargs)
        finally:
            model.r1_resume = original_resume
            model.r1_fixed_loss_scale = old_fixed
    runner.train_task = train_preserving_scaler


def run(root, logs, smoke=False):
    manifest = verify_prefix(root)
    if bool(manifest['smoke']) != smoke:
        raise ValueError('Smoke/full recovery mismatch')
    configure_runtime(root)
    original_popen = subprocess.Popen

    def popen(command, *args, **kwargs):
        command = list(command)
        old = 'scripts.emotic.run_r1_fusion_return_train'
        if old in command:
            command[command.index(old)] = 'scripts.emotic.run_r1_full_extend_recovery'
        return original_popen(command, *args, **kwargs)
    subprocess.Popen = popen
    try:
        base.run_one(root, logs, smoke, 'full_extend', resume=True)
    finally:
        subprocess.Popen = original_popen
        verify_prefix(root)
    write(root/'numerical_recovery_complete.json', {
        'status': 'complete', 'source_prefix_unchanged': True,
        'resume_boundary': [manifest['task'], manifest['completed_epoch']],
        'person_loss_scale': manifest['person_loss_scale_after_boundary'],
        'result': str(root/'result.json')})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare','run','worker','evaluate'))
    p.add_argument('--source', type=Path)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--logs', type=Path, default=Path('logs/r1_full_extend_recovery/seed1_20261113'))
    p.add_argument('--diagnostic', type=Path)
    p.add_argument('--person-loss-scale', type=float, default=32.)
    p.add_argument('--view', choices=VIEWS)
    p.add_argument('--condition', choices=('full_extend',), default='full_extend')
    p.add_argument('--smoke', action='store_true')
    a=p.parse_args()
    if a.action == 'prepare':
        if a.source is None:
            raise ValueError('Explicit source trajectory required')
        prepare(a.source, a.root, a.diagnostic, a.person_loss_scale, a.smoke)
    elif a.action == 'run':
        run(a.root, a.logs, a.smoke)
    elif a.action == 'worker':
        if a.view is None:
            raise ValueError('Worker view required')
        configure_runtime(a.root)
        base.worker(a.root, a.view, a.smoke, 'full_extend')
    else:
        base.evaluate(a.root, a.smoke, 'full_extend')
        verify_prefix(a.root)


if __name__ == '__main__':
    main()
