"""Bounded GPU0 orchestration; no changes to running source trees or old artifacts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path('output/d9_fill_fusion_tables/seed1_20261122')
LOGS = Path('logs/d9_fill_fusion_tables/seed1_20261122')


def save(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def jobs(phase):
    return [
        ('replay_equal90', ['bash', 'scripts/emotic/run_d9_fill_replay.sh', phase, 'equal90', '0']),
        ('replay_full', ['bash', 'scripts/emotic/run_d9_fill_replay.sh', phase, 'full', '0']),
        ('probes_equal90', ['bash', 'scripts/emotic/run_d9_fill_probes.sh', phase, '0']),
        ('image_full', ['bash', 'scripts/emotic/run_d9_fill_image.sh', phase, '0']),
    ]


def run(phase):
    ROOT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    pending, active, finished = jobs(phase), {}, []
    status_file = ROOT / f'{phase}_queue.json'
    with (ROOT / f'.{phase}.queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if phase == 'formal':
            for name, _ in pending:
                if (ROOT / 'smoke' / name / 'exit.status').read_text().strip() != '0':
                    raise RuntimeError(f'{name} has not passed its real smoke')
        while pending or active:
            for name, (process, stream, started) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                finished.append(dict(name=name, returncode=code, seconds=time.time()-started))
                del active[name]
                print('FINISH', name, code, flush=True)
            # Reserve at most two replay models plus one extractor. Smoke runs use two slots.
            slots = 2 if phase == 'smoke' else 3
            if pending and len(active) < slots:
                memory = int(subprocess.check_output([
                    'nvidia-smi', '-i', '0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'
                ], text=True).strip())
                if memory >= 7000:
                    name, command = pending.pop(0)
                    if (ROOT / phase / name / 'exit.status').exists() and (ROOT / phase / name / 'exit.status').read_text().strip() == '0':
                        finished.append(dict(name=name, returncode=0, reused_completed=True))
                    else:
                        stream = (LOGS / f'{phase}_{name}_launcher.log').open('a')
                        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                                   env={**os.environ, 'PYTHONUNBUFFERED': '1'})
                        active[name] = (process, stream, time.time())
                        print('START', phase, name, process.pid, flush=True)
            save(status_file, dict(phase=phase, pending=[n for n, _ in pending],
                                  running={n: dict(pid=p.pid, started_at=t) for n, (p, _, t) in active.items()},
                                  finished=finished, updated_at=time.time()))
            if pending or active:
                time.sleep(10)
        failed = [job for job in finished if job['returncode']]
        save(ROOT / f'{phase}_complete.json', dict(status='failed' if failed else 'complete', jobs=finished))
        if failed:
            raise RuntimeError(f'{phase} failures: {failed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('smoke', 'formal'), required=True)
    run(parser.parse_args().phase)
