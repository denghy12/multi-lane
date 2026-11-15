"""Backfill the live seed1 benchmark on GPU0 and on fully idle additional GPUs."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.emotic.run_d9_benchmark_protocols import MODULE, load, save, seed_dir


# The three pre-handoff B4-C2 workers stay with the original controller.
JOBS = [('b10c4', 'person'), ('b10c4', 'face'), ('joint26', 'person'),
        ('joint26', 'face'), ('joint26', 'full'), ('b10c4', 'full')]


def gpu_snapshot():
    def query(fields):
        output = subprocess.check_output(
            ['nvidia-smi', fields, '--format=csv,noheader,nounits'], text=True)
        return [line.split(', ') for line in output.strip().splitlines() if line.strip()]
    cards = query('--query-gpu=index,uuid,memory.free,utilization.gpu')
    busy = {row[0] for row in query('--query-compute-apps=gpu_uuid')}
    return [(int(i), int(free), uuid not in busy and int(util) < 5) for i, uuid, free, util in cards]


def pick(pending, running, cards, idle_since, now):
    occupied = {gpu for gpu, _, _ in running.values()}
    for gpu, free, idle in cards:
        if gpu == 0:
            continue
        if idle and gpu not in occupied:
            idle_since.setdefault(gpu, now)
        else:
            idle_since.pop(gpu, None)
        if gpu in idle_since and now - idle_since[gpu] >= 30 and free >= 7000 and pending:
            return pending[0], gpu
    free = next(free for gpu, free, _ in cards if gpu == 0)
    local = [job[1] for job, (gpu, _, _) in running.items() if gpu == 0]
    # Keep room for the primary controller's three workers and their later evaluation peaks.
    for job in pending:
        lightweight = job[1] == 'full'
        if any((v == 'full') == lightweight for v in local):
            continue
        if free >= (7500 if lightweight else 10500):
            return job, 0
    return None


def dispatch(root):
    manifest = load(root / 'protocol.json')
    if manifest['seeds'] != [1] or str(manifest['gpu']) != '0':
        raise ValueError('Backfill expects the active GPU0 seed1 benchmark')
    running, failed, idle_since = {}, {}, {}
    logs = Path('logs/d9_benchmark_protocols') / root.name / 'backfill'
    logs.mkdir(parents=True, exist_ok=True)
    while True:
        for job, (_, proc, handle) in list(running.items()):
            code = proc.poll()
            if code is not None:
                handle.close()
                del running[job]
                if code:
                    failed[job] = code
        todo = [j for j in JOBS if j not in failed and not
                (seed_dir(root, j[0], 1, False) / j[1] / 'complete.json').exists()]
        queue = load(root / 'formal_queue.json')
        primary = queue['running']
        pending = [j for j in todo if j not in running and f'{j[0]}_1_{j[1]}' not in primary]
        cards = gpu_snapshot()
        slot = pick(pending, running, cards, idle_since, time.monotonic())
        if slot:
            (protocol, view), gpu = slot
            path = seed_dir(root, protocol, 1, False)
            log = (logs / f'{protocol}_seed1_{view}_gpu{gpu}.log').open('a')
            cmd = [sys.executable, '-u', '-m', MODULE, '--root', str(path), '--worker', view,
                   '--protocol', protocol, '--seed', '1']
            proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                    stdout=log, stderr=subprocess.STDOUT)
            running[(protocol, view)] = gpu, proc, log
            print('START', protocol, view, gpu, proc.pid, flush=True)
        save(root / 'backfill_queue.json', dict(
            running={f'{p}_{v}': dict(gpu=g, pid=proc.pid) for (p,v),(g,proc,_) in running.items()},
            failed={f'{p}_{v}': code for (p,v),code in failed.items()}, pending=pending))
        if not todo and not running:
            break
        time.sleep(12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    with (args.root / 'backfill.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dispatch(args.root)
