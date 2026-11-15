"""Run the original Frozen9 D with the benchmark schedules, fixed R1 readout and seeds."""
import argparse
import fcntl
import json
import math
import os
import statistics
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


ROOT = Path('output/d9_benchmark_protocols/20261115')
MODULE = 'scripts.emotic.run_d9_benchmark_protocols'
FUSION = 'scripts.emotic.summarize_d9_benchmark_protocols'
PROTOCOLS = ('joint26', 'b10c4', 'b4c2')
SEEDS = (0, 1, 2)
VIEWS = ('full', 'person', 'face')


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def branch_run(root, protocol, seed, view, smoke):
    from multi_lane.track_a.r1_crossvit_experiment import flags
    from multi_lane.track_a.runner import TASK_PROTOCOLS
    from scripts.emotic.run_r1_frozen_cross_view import input_paths

    paths = input_paths()
    experiment = SimpleNamespace(output=root, task_protocol=protocol, **vars(paths))
    kind = 'native' if view == 'full' else 'attention'
    training = root / view / 'train'
    options = flags(experiment, kind, view, smoke, seed=seed)
    options[options.index('--output-root') + 1] = str(training)
    options += ['--r1-fusion-readout', 'logit']
    if view != 'full':
        options += ['--r1-frozen-mode', 'patch', '--r1-exchange-layer', '8',
                    '--r1-fixed-loss-scale', '4096']
    if smoke:
        options += ['--r1-smoke-eval-samples', '64']
    train = [sys.executable, '-u', '-m', 'multi_lane.track_a.runner', *options]
    print('TRAIN', ' '.join(train), flush=True)
    if not (training / 'seed_summary.json').exists():
        subprocess.run(train, check=True, env=dict(os.environ, OMP_NUM_THREADS='72', MKL_NUM_THREADS='72'))
    result = load(training / 'seed_summary.json')
    sizes = TASK_PROTOCOLS[protocol]
    n = min(2, len(sizes)) if smoke else len(sizes)
    cfg = result['config']
    if (result['status'] != 'complete' or result['seed'] != seed
            or tuple(cfg['task_sizes']) != sizes or cfg['max_tasks'] != n
            or cfg['r1_receiver'] != view or cfg['r1_kind'] != kind):
        raise ValueError(f'Training does not match {protocol} seed {seed} {view}')
    hist = load(training / 'training_history.json')
    split = 'val' if smoke else 'test'
    updates = []
    for task in range(n):
        rows = hist[str(task)]
        expected = 2 if smoke else 30 * math.ceil(result['calibration_counts'][str(task)]['fit'] / 64)
        actual = sum(r['optimizer_steps'] for r in rows)
        if not smoke and len(rows) != 30:
            raise ValueError(f'Incomplete epoch budget at task {task}')
        if actual != expected or any(r['skipped_optimizer_steps'] for r in rows):
            raise ValueError(f'Incomplete update budget at task {task}: {actual}/{expected}')
        if not (training / 'compact_checkpoints' / f'task{task}.pth').is_file():
            raise FileNotFoundError(f'Missing compact checkpoint for task {task}')
        if not (training / f'{split}_scores' / f'task{task}.npz').is_file():
            raise FileNotFoundError(f'Missing {split} score archive for task {task}')
        updates.append(int(actual))
    val = root / view / 'val'
    evaluate = [sys.executable, '-u', '-m', 'multi_lane.track_a.r1_crossvit_experiment', 'val',
                '--training-root', str(training), '--output', str(val),
                '--data-root', str(paths.data_root), '--clip-checkpoint', str(paths.clip_checkpoint),
                '--face-manifest-root', str(paths.face_manifest_root)]
    if smoke:
        evaluate.append('--smoke')
    print('VALIDATION', ' '.join(evaluate), flush=True)
    subprocess.run(evaluate, check=True)
    validation = load(val / 'result.json')
    if validation['status'] != 'complete' or validation['tasks'] != list(range(n)):
        raise ValueError('Incomplete validation')
    save(root / view / 'complete.json', dict(protocol=protocol, seed=seed, view=view, stages=n,
         epochs_per_task=None if smoke else 30, updates=updates, skips=0, smoke=smoke))


def locked_branch(root, protocol, seed, view, smoke):
    directory = root / view
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        done = directory / 'complete.json'
        if done.exists():
            record = load(done)
            if (record['protocol'], record['seed'], record['view'], record['smoke']) != (protocol, seed, view, smoke):
                raise ValueError(f'Completion marker does not match worker: {done}')
            return
        branch_run(root, protocol, seed, view, smoke)


def seed_dir(root, protocol, seed, smoke):
    return root / ('smoke' if smoke else 'formal') / protocol / f'seed{seed}'


def summarize(path, protocol, seed, smoke):
    command = [sys.executable, '-m', FUSION, '--root', str(path), '--protocol', protocol,
               '--seed', str(seed)]
    if smoke:
        command += ['--smoke']
    subprocess.run(command, check=True)


def run_phase(root, logs, smoke, slots, selected_seeds=SEEDS):
    phase = 'smoke' if smoke else 'formal'
    seeds = (1,) if smoke else selected_seeds
    protocols = PROTOCOLS
    if tuple(selected_seeds) == (1,) and not smoke:
        protocols = ('b4c2', 'b10c4', 'joint26')
    groups = [(p, s) for s in seeds for p in protocols]
    pending = []
    running, failed, fused = {}, {}, set()
    for p, s in groups:
        for v in VIEWS:
            path = seed_dir(root, p, s, smoke)
            if not (path / v / 'complete.json').exists():
                pending.append((p, s, v))
    while True:
        for job, (proc, handle) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            handle.close()
            del running[job]
            if code:
                failed['_'.join(map(str, job))] = code
        for p, s in groups:
            if (p, s) in fused:
                continue
            path = seed_dir(root, p, s, smoke)
            if all((path / v / 'complete.json').exists() for v in VIEWS):
                summarize(path, p, s, smoke)
                fused.add((p, s))
        if pending and len(running) < slots:
            gpu = os.environ['CUDA_VISIBLE_DEVICES']
            free = int(subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free',
                                               '--format=csv,noheader,nounits'], text=True).strip())
            # Reserve for jobs that have not reached their first epoch yet.
            starting = sum(6200 for p, s, v in running
                           if not (seed_dir(root, p, s, smoke) / v / 'train/progress.json').exists())
            if free - starting >= 7000:
                p, s, v = pending.pop(0)
                job = (p, s, v)
                path = seed_dir(root, p, s, smoke)
                log = logs / phase / f'{p}_seed{s}_{v}.log'
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open('a', buffering=1)
                command = [sys.executable, '-u', '-m', MODULE, '--root', str(path),
                           '--worker', v, '--protocol', p, '--seed', str(s)]
                if smoke:
                    command += ['--smoke']
                proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                         env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
                running[job] = proc, handle
                print('START', phase, p, s, v, proc.pid, flush=True)
        save(root / f'{phase}_queue.json', dict(phase=phase, pending=pending, failed=failed,
             running={'_'.join(map(str,j)): proc.pid for j,(proc,_) in running.items()},
             completed_groups=sorted(fused)))
        if not running and not pending:
            break
        time.sleep(4)
    if failed or len(fused) != len(groups):
        raise RuntimeError(f'{phase} incomplete; inspect {phase}_queue.json: {failed}')


def summarize_seeds(formal, destination, selected_seeds=SEEDS):
    results = {}
    for protocol in PROTOCOLS:
        seeds = {}
        for seed in selected_seeds:
            data = load(formal / protocol / f'seed{seed}' / 'benchmark_summary.json')
            seeds[seed] = {split: data['splits'][split]['summary'] for split in ('val', 'test')}
        aggregate = {}
        for split in ('val', 'test'):
            aggregate[split] = {}
            metrics = ['final_mAP', 'final_cF1', 'final_oF1']
            if protocol != 'joint26':
                metrics.append('average_mAP')
            for metric in metrics:
                values = [record[split][metric] for record in seeds.values()]
                aggregate[split][metric] = dict(mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values) > 1 else None)
        results[protocol] = dict(seeds=seeds, aggregate=aggregate)
    save(destination, results)


def controller(args):
    from multi_lane.track_a.runner import TASK_PROTOCOLS, git_metadata
    from scripts.emotic.run_r1_frozen_cross_view import input_paths

    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        git = git_metadata(Path.cwd())
        if git['dirty']:
            raise ValueError('Run from the clean experiment worktree')
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Seeds must be unique")
        paths = input_paths()
        manifest = dict(git=git, task_sizes={p: list(TASK_PROTOCOLS[p]) for p in PROTOCOLS},
                seeds=args.seeds, branches=list(VIEWS), pretrained=str(paths.clip_checkpoint),
                data=str(paths.data_root), face_manifest=str(paths.face_manifest_root), gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                epochs=30, batch=64, optimizer='Adam', schedule='cosine', weight_decay=0, learning_rate=.0125,
                adapter_lr=.0004, cross_view_lr=.0004, adapter_scale=.1, patch_layer=9, patch_cap=.05,
                receiver_amp_scale=4096, replay_memory=0, threshold=.5, reliable_weights=[.64,.16,.2],
                fallback_weights=[.8,.2,0], readout='weighted raw logits then sigmoid',
                checkpoint='epoch 30 per view; validation monitoring only')
        protocol_file = root / 'protocol.json'
        if protocol_file.exists() and load(protocol_file) != manifest:
            raise ValueError('Refusing to mix incompatible runs')
        save(protocol_file, manifest)
        logs = Path('logs/d9_benchmark_protocols') / root.name
        run_phase(root, logs, True, args.slots, args.seeds)
        run_phase(root, logs, False, args.slots, args.seeds)
        summarize_seeds(root / 'formal', root / 'benchmark_results.json', args.seeds)
        save(root / 'completion.json', dict(status='complete', seeds=args.seeds, protocols=list(PROTOCOLS)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--worker', choices=VIEWS)
    parser.add_argument('--protocol', choices=PROTOCOLS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument('--slots', type=int, choices=(1,2,3), default=3)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        locked_branch(args.root, args.protocol, args.seed, args.worker, args.smoke)
    else:
        controller(args)
