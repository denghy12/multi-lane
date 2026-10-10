"""Run locked protected-expert policies on idle GPUs; never terminate foreign jobs."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime

METHODS = ('baseline', 'frozen_pool', 'fresh_only', 'reuse_old')

def timestamp():
    return datetime.now().astimezone().isoformat()


def idle_gpus(allowed):
    query = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True)
    busy = set(apps.splitlines())
    result = []
    for line in query.splitlines():
        index, uuid, used, free, utilization = [v.strip() for v in line.split(',')]
        if (index in allowed and uuid not in busy and int(used) < 512
                and int(free) >= 18000 and int(utilization) <= 5):
            result.append(index)
    return result


def wait_for_smoke(source: Path, controller_pid: int | None, seconds: int):
    """Require a completed audited prerequisite, without inspecting live metrics."""
    deadline = time.monotonic() + seconds
    while not (source/'control/completed.txt').is_file():
        if (source/'control/failed.json').exists():
            raise RuntimeError('Prerequisite smoke failed; formal training was not started')
        if controller_pid is not None:
            try:
                os.kill(controller_pid, 0)
            except ProcessLookupError as error:
                raise RuntimeError('Smoke controller ended without passing comparison') from error
        if time.monotonic() >= deadline:
            raise TimeoutError('Prerequisite smoke wait expired; formal training was not started')
        time.sleep(15)
    data = json.loads((source/'multiseed_comparison.json').read_text())
    if set(data['methods']) != set(METHODS):
        raise RuntimeError('Prerequisite comparison is incomplete')


def main():
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    batch = os.environ.get('BATCH_ID', 'protected_parax_8task_locked_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    seeds = [int(v) for v in os.environ.get('SEEDS', '0,1,2').split(',')]
    allowed = os.environ.get('GPU_POOL', '0,1,2,3,4,5,6,7').split(',')
    if len(set(seeds)) != len(seeds) or not seeds or any(s not in (0,1,2) for s in seeds):
        raise ValueError('SEEDS must be unique members of 0,1,2')
    output = Path(os.environ.get('OUTPUT_ROOT', './output/emotic_protected_parax_full'))
    logs = Path(os.environ.get('LOG_ROOT', './logs/emotic_protected_parax_full'))
    control = output/batch/'control'
    if (output/batch).exists():
        raise FileExistsError(f'Batch already exists: {output/batch}')
    for key in ('DATA_ROOT', 'CLIP_CHECKPOINT', 'FACE_MANIFEST_ROOT'):
        if not os.environ.get(key):
            raise ValueError(f'Set {key}')
    control.mkdir(parents=True)
    (logs/batch).mkdir(parents=True)
    manifest = {
        'branch': subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        'commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'methods': METHODS, 'seeds': seeds, 'gpu_pool': allowed,
        'training': 'EMOTIC incremental task0-7; one training per method/seed; fixed epoch endpoint',
        'evaluation': 'same weights, validation and held-out test; no test-based selection',
        'position': 'normalized final Task Forward CLS, before unchanged fixed feature fusion',
        'epochs_per_task': int(os.environ.get('EPOCHS','30')),
        'optimizer_updates_per_task': int(os.environ.get('UPDATES_PER_TASK','0')),
        'batch_size': 64, 'workers': 2, 'optimizer': 'Adam reset per task',
        'base_lr': .0125, 'parax_lr': .0004, 'weight_decay': 0,
        'scheduler': 'cosine min0 warmup0', 'amp': 'scale1024 growth1e9',
        'experts': '16 preallocated matrix pairs, two new/task; frozen_pool uses only first two',
        'rank': 32, 'router_hidden': 16, 'zero_task_projection': True,
        'raw_residual_smooth_bound': .02, 'scale': 1,
        'old_state': 'expert/router/projection/access/Selector/Prompt/head frozen and audited',
        'fusion': 'fixed features reliableFace [.64,.16,.20] otherwise [.8,.2,0]',
        'loss': 'legacy_full_zero joint_bce + .1 reliable-view BCE',
        'excluded': 'Image-token Adapter, patch Projector, level embedding, distillation, staged training, dynamic fusion',
        'inputs': {k: os.environ[k] for k in ('DATA_ROOT','CLIP_CHECKPOINT','FACE_MANIFEST_ROOT')},
        'output': str(output/batch), 'logs': str(logs/batch),
        'prerequisite_smoke': os.environ.get('WAIT_FOR_SMOKE'),
    }
    (control/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (control/'queued.txt').write_text(timestamp()+'\n')
    if os.environ.get('WAIT_FOR_SMOKE'):
        source = Path(os.environ['WAIT_FOR_SMOKE'])
        pid = int(os.environ['SMOKE_CONTROLLER_PID']) if os.environ.get('SMOKE_CONTROLLER_PID') else None
        print(timestamp(), 'waiting for audited smoke', str(source), flush=True)
        try:
            wait_for_smoke(source, pid, int(os.environ.get('MAX_WAIT_SECONDS','21600')))
        except Exception as error:
            (control/'prerequisite_failed.json').write_text(json.dumps({'error': str(error)})+'\n')
            raise
        (control/'prerequisite_passed.txt').write_text(timestamp()+'\n')
    (control/'started.txt').write_text(timestamp()+'\n')
    pending = [(seed,method) for seed in seeds for method in METHODS]
    active = {}; results = {}; last_dispatch = time.monotonic()
    while pending or active:
        for gpu, (process, handle, seed, method) in list(active.items()):
            code = process.poll()
            if code is not None:
                handle.close()
                results[f'seed{seed}/{method}'] = code
                (control/f'seed{seed}_{method}.exit_code').write_text(str(code)+'\n')
                print(timestamp(),'finished',seed,method,'gpu',gpu,'exit',code,flush=True)
                del active[gpu]
        available = [g for g in idle_gpus(allowed) if g not in active] if pending else []
        for gpu in available:
            if not pending:
                break
            seed, method = pending.pop(0)
            env = dict(os.environ, FULL_EVALUATION='1', GPU=gpu, SEED=str(seed), METHOD=method,
                       RUN_ID=batch, OUTPUT_ROOT=str(output), LOG_ROOT=str(logs),
                       PYTHON=sys.executable, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            logdir = logs/batch/f'seed{seed}'; logdir.mkdir(exist_ok=True)
            handle = (logdir/f'{method}.launcher.log').open('w')
            process = subprocess.Popen(['bash','scripts/emotic/run_protected_parax_validation.sh'],
                                       env=env, stdout=handle, stderr=subprocess.STDOUT)
            active[gpu] = (process,handle,seed,method)
            (control/f'seed{seed}_{method}.started.json').write_text(json.dumps(
                {'time':timestamp(),'gpu':gpu,'pid':process.pid},indent=2)+'\n')
            print(timestamp(),'started',seed,method,'gpu',gpu,flush=True)
            last_dispatch = time.monotonic()
        if pending and not active and time.monotonic()-last_dispatch > int(os.environ.get('MAX_WAIT_SECONDS','21600')):
            raise TimeoutError('No idle authorized GPU became available; no processes were stopped')
        if pending or active:
            time.sleep(15)
    if any(results.values()):
        (control/'failed.json').write_text(json.dumps(results,indent=2)+'\n')
        raise RuntimeError('One or more arms failed; do not publish a complete comparison')
    with (control/'comparison.log').open('w') as handle:
        subprocess.run([sys.executable,'-m','multi_lane.track_a.protected_parax_compare',
                        str(output/batch),'--full-evaluation','--seeds',*[str(s) for s in seeds]],
                       check=True,stdout=handle,stderr=subprocess.STDOUT)
    (control/'completed.txt').write_text(timestamp()+'\n')
    print(timestamp(),'all complete',flush=True)


if __name__ == '__main__':
    main()
