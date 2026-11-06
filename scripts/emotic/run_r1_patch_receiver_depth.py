"""Run the paired Face and Person Frozen Forward patch depth sweep."""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Optional

import torch

from multi_lane.track_a.r1_crossvit_experiment import flags, read, sha, write
from multi_lane.track_a.runner import git_metadata
from scripts.emotic.run_r1_frozen_cross_view import check_training, input_paths, preflight, score_suffix
from scripts.emotic.run_r1_frozen_late_stable import summarize_family


DEFAULT_ROOT = Path('output/r1_patch_receiver_depth/seed1_20261106')
TWO_LAYER_ROOT = Path('output/r1_patch_two_layer/seed1_20261106')
REFERENCE = Path('../multi-lane-main-r1-crossvit-20261101/output/r1_crossvit/seed1_20261101_02')
LAYER9 = Path('../multi-lane-main-r1-frozen-late-stable-20261105/output/r1_frozen_late_stable/seed1_20261105/layer9')
DEPTH_RESULTS = Path('../multi-lane-main-r1-patch-receiver-depth-20261106') / DEFAULT_ROOT / 'single_layer_patch_person_face_results.json'
VIEWS = ('person', 'face')


@dataclass(frozen=True)
class Job:
    layer: int
    view: str
    gpu: int
    second_layer: Optional[int] = None
    sharing: str = 'shared'

    @property
    def family(self):
        if self.second_layer is None:
            return f'layer{self.layer}'
        return f'dual_{self.layer}_{self.second_layer}_{self.sharing}'

    @property
    def name(self):
        return f'{self.family}_{self.view}'

    def path(self, root):
        return root / self.family / 'patch' / 'attention' / self.view


def worker(root, reference, job, smoke):
    inputs = input_paths()
    base = root / job.family / 'patch'
    path = job.path(root)
    train = path / 'train'
    command = [sys.executable, '-u', '-m', 'multi_lane.track_a.runner',
               *flags(SimpleNamespace(output=base, **vars(inputs)), 'attention', job.view, smoke),
               '--r1-frozen-mode', 'patch', '--r1-exchange-layer', str(job.layer - 1),
               '--r1-fixed-loss-scale', '4096', '--r1-fusion-readout', 'logit']
    if job.second_layer is not None:
        command += ['--r1-exchange-second-layer', str(job.second_layer - 1),
                    '--r1-exchange-sharing', job.sharing]
    if smoke:
        command += ['--r1-smoke-eval-samples', '64']
    env = dict(os.environ, OMP_NUM_THREADS='72', MKL_NUM_THREADS='72', OPENBLAS_NUM_THREADS='1')
    print('TRAIN', ' '.join(command), flush=True)
    subprocess.run(command, env=env, check=True)
    counts = check_training(train, reference, job.view, 2 if smoke else 8, smoke)
    if smoke:
        preflight(train, inputs, path / 'preflight.json')
    else:
        val = [sys.executable, '-u', '-m', 'multi_lane.track_a.r1_crossvit_experiment', 'val',
               '--training-root', str(train), '--output', str(path / 'val'),
               '--data-root', str(inputs.data_root), '--clip-checkpoint', str(inputs.clip_checkpoint),
               '--face-manifest-root', str(inputs.face_manifest_root)]
        subprocess.run(val, check=True)
    write(path / 'completion.json', dict(status='complete', training=counts, gpu=job.gpu))
    if not smoke and job.second_layer is not None:
        attribution = subprocess.run([sys.executable, '-u', '-m', 'scripts.emotic.r1_two_layer_attribution',
                                      '--training-root', str(train), '--output', str(path / 'attribution')])
        if attribution.returncode != 0:
            write(path / 'attribution_failure.json', dict(exit_code=attribution.returncode))


def free_memory():
    raw = subprocess.check_output(['nvidia-smi', '-i', '0,1', '--query-gpu=index,memory.free',
                                   '--format=csv,noheader,nounits'], text=True)
    return {int(gpu): int(free) for gpu, free in (line.split(',') for line in raw.splitlines())}


def dispatch(jobs, root, logs, reference, budgets, smoke, *, reserve_peak=False,
             worker_command=None):
    pending = list(jobs)
    running = {}
    finished = {}
    phase = 'smoke' if smoke else 'formal'
    capacity = free_memory()
    while pending or running:
        for job, (process, log) in list(running.items()):
            code = process.poll()
            if code is not None:
                log.close()
                del running[job]
                finished[job.name] = 'complete' if code == 0 else 'failed'

        free = free_memory()
        for gpu in (0, 1):
            # During CUDA initialization, reserve the job's peak until the first
            # epoch is saved. Afterwards the card's actual free memory applies.
            starting = sum(budgets[job.name] for job in running if job.gpu == gpu
                           and not (job.path(root) / 'train/progress.json').exists())
            job = next((job for job in pending if job.gpu == gpu), None)
            if job is None or free[gpu] - starting < budgets[job.name] + 512:
                continue
            if reserve_peak:
                # Several jobs can reach the eight-lane checkpoint evaluation together.
                peak = sum(budgets[item.name] for item in running if item.gpu == gpu)
                if peak + budgets[job.name] + 512 > capacity[gpu]:
                    continue
            if worker_command is not None:
                command = worker_command(job, root, smoke)
            else:
                command = [sys.executable, '-u', '-m', 'scripts.emotic.run_r1_patch_receiver_depth',
                           'worker', '--root', str(root), '--reference', str(reference),
                           '--layer', str(job.layer), '--view', job.view, '--gpu', str(job.gpu)]
                if smoke:
                    command.append('--smoke')
                if job.second_layer is not None:
                    command += ['--second-layer', str(job.second_layer), '--sharing', job.sharing]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job.gpu), OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
            log = (logs / f'{job.name}.log').open('x')
            try:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            pending.remove(job)
            running[job] = process, log
            print('START', phase, job.name, f'gpu={gpu}', f'pid={process.pid}', flush=True)

        write(root / 'queue.json', dict(phase=phase, memory_free_mib=free, budgets_mib=budgets,
                                       pending=[job.name for job in pending], finished=finished,
                                       running={job.name: dict(gpu=job.gpu, pid=process.pid)
                                                for job, (process, _) in running.items()}))
        if pending or running:
            time.sleep(5)
    return finished


def source_paths(reference, layer9, manifest):
    paths = [reference / family / view / score_suffix(split, task)
             for family, view in (('native', 'full'), ('native', 'person'),
                                  ('native', 'face'), ('pool', 'face'))
             for split in ('val', 'test') for task in range(8)]
    paths += [layer9 / 'patch' / 'attention' / view / score_suffix(split, task)
              for view in VIEWS for split in ('val', 'test') for task in range(8)]
    paths.extend(manifest / 'manifests' / f'{split}.jsonl' for split in ('train', 'val', 'test'))
    paths.append(manifest / 'audit_summary.json')
    return paths


def run(args):
    root = args.root
    logs = Path('logs') / ('r1_patch_two_layer' if args.two_layer else 'r1_patch_receiver_depth') / root.name
    git = git_metadata(Path.cwd())
    if git['dirty']:
        raise ValueError('Clean experiment worktree required')
    inputs = input_paths()
    jobs = [Job(layer, view, gpu) for layer, gpu in ((6, 0), (8, 1), (10, 1), (11, 0)) for view in VIEWS]
    if args.two_layer:
        jobs = [Job(early, view, gpu, late, sharing)
                for early, late in ((8, 9), (9, 10))
                for gpu, sharing, view in ((0, 'shared', 'person'), (0, 'independent', 'face'),
                                           (1, 'independent', 'person'), (1, 'shared', 'face'))]
    for view in VIEWS:
        train = args.layer9 / 'patch' / 'attention' / view / 'train'
        check_training(train, args.reference, view, 8, False)
        cfg = read(train / 'config.json')
        if (cfg['r1_exchange_layer'], cfg['r1_frozen_mode'], cfg['r1_fixed_loss_scale']) != (8, 'patch', 4096):
            raise ValueError('Mismatched layer9 reference')
    hashes = {str(path): sha(path) for path in source_paths(args.reference, args.layer9, inputs.face_manifest_root)}
    if args.two_layer:
        hashes[str(DEPTH_RESULTS)] = sha(DEPTH_RESULTS)
    root.mkdir(parents=True, exist_ok=False)
    for sub in ('smoke', 'formal'):
        (logs / sub).mkdir(parents=True, exist_ok=False)
    write(root / 'protocol.json', dict(git=git, seed=1, tasks=8, epochs=30, batch=64,
                                      main_lr=.0125, new_lr=.0004, optimizer='Adam', scheduler='cosine',
                                      wd=0, cap=.05, heads=4, dim=64, loss_scale=4096, amp=True, tf32=True,
                                      paired_readout='patch_person_face', same_depth_sender=True,
                                      layer9=str(args.layer9), reference=str(args.reference),
                                      jobs=[vars(job) for job in jobs], reliable=[.64, .16, .20],
                                      site_caps=[.025, .025] if args.two_layer else [.05],
                                      attribution_tasks=[0, 2, 7] if args.two_layer else [],
                                      fallback=[.8, .2, 0]))
    write(root / 'source_hashes.json', hashes)
    smoke = dispatch(jobs, root / 'smoke', logs / 'smoke', args.reference,
                     {job.name: 9500 if args.two_layer else 5500 for job in jobs}, smoke=True)
    passed = [job for job in jobs if smoke[job.name] == 'complete']
    budgets = {job.name: int(read(job.path(root / 'smoke') / 'preflight.json')['smoke_train_peak_reserved_mib'] + 1024)
               for job in passed}
    write(root / 'preflight_gate.json', dict(states=smoke, formal_budgets_mib=budgets))
    formal = dispatch(passed, root, logs / 'formal', args.reference, budgets, smoke=False,
                      reserve_peak=args.two_layer)
    torch.set_num_threads(2)
    sources = dict(R1={'person': args.reference / 'native/person', 'face': args.reference / 'native/face'},
                   Face_only_Pool={'person': args.reference / 'native/person', 'face': args.reference / 'pool/face'},
                   layer9={view: args.layer9 / 'patch' / 'attention' / view for view in VIEWS})
    results = read(DEPTH_RESULTS)['results'] if args.two_layer else {}
    attribution_status = {}
    for family in dict.fromkeys(job.family for job in jobs):
        selected = [job for job in jobs if job.family == family]
        if all(formal.get(job.name) == 'complete' for job in selected):
            sources[family] = {job.view: job.path(root) for job in selected}
        else:
            results[family] = dict(status='incomplete', paths={job.name: formal.get(job.name, 'blocked')
                                                              for job in selected})
    for name, paths in sources.items():
        destination = root / 'fusion' / f'{name}.json'
        summarize_family(root, args.reference, inputs.face_manifest_root,
                         {('patch', view): path for view, path in paths.items()},
                         destination, modes=('patch',), directions=('person_face',))
        result = {split: value['patch_person_face'] for split, value in read(destination)['results'].items()}
        if any(value['status'] != 'complete' for value in result.values()):
            raise ValueError(f'Incomplete score sources for {name}')
        results[name] = result
        if args.two_layer and name.startswith('dual_'):
            from scripts.emotic.r1_two_layer_attribution import TASKS, VARIANTS
            complete = all((path / 'attribution/completion.json').is_file() for path in paths.values())
            attribution_status[name] = 'complete' if complete else 'partial'
            if complete:
                for variant in VARIANTS:
                    summarize_family(root, args.reference, inputs.face_manifest_root,
                        {('patch', view): path / 'attribution' / variant for view, path in paths.items()},
                        root / 'attribution' / name / f'{variant}.json',
                        modes=('patch',), directions=('person_face',), tasks=TASKS)
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('A reference artifact changed')
    name = 'two_layer_patch_person_face_results.json' if args.two_layer else 'single_layer_patch_person_face_results.json'
    write(root / name, dict(results=results, source_hashes_verified=True))
    expected = 7
    write(root / 'completion.json', dict(status='complete' if len(sources) == expected else 'partial',
                                         smoke=smoke, formal=formal, attribution=attribution_status))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='run', choices=('run', 'worker'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--two-layer', action='store_true')
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--layer9', type=Path, default=LAYER9)
    parser.add_argument('--layer', type=int, choices=(6, 8, 9, 10, 11))
    parser.add_argument('--second-layer', type=int)
    parser.add_argument('--sharing', choices=('shared', 'independent'), default='shared')
    parser.add_argument('--view', choices=VIEWS)
    parser.add_argument('--gpu', type=int, choices=(0, 1))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.root = args.root or (TWO_LAYER_ROOT if args.two_layer else DEFAULT_ROOT)
    if args.action == 'worker':
        if args.layer is None or args.view is None or os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
            parser.error('worker requires a layer, view and matching CUDA device')
        worker(args.root, args.reference, Job(args.layer, args.view, args.gpu, args.second_layer, args.sharing), args.smoke)
    else:
        run(args)


if __name__ == '__main__':
    main()
