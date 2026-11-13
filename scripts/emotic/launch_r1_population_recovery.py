"""Queue the verified recovery and population experiment on GPU0, independently."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/r1_population_recovery_batch/seed1_20261113'))
    args = parser.parse_args()
    if args.root.exists():
        raise FileExistsError('A fresh batch record is required')
    recovery = Path('output/r1_full_extend_recovery/seed1_20261113/recovered')
    population = Path('output/r1_fusion_population_train/seed1_20261113')
    replay = read(Path('output/r1_fusion_population_train/smoke01_20261113/replay_audit.json'))
    smoke = read(Path('output/r1_full_extend_recovery/smoke01_20261113/numerical_recovery_complete.json'))
    if replay['status'] != 'passed' or smoke['status'] != 'complete':
        raise ValueError('Both exact replay and numerical recovery smoke must pass')
    if not (recovery/'recovery_provenance.json').is_file() or population.exists():
        raise ValueError('Recovery must be prepared; population training must be fresh')
    git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Experiment source tree must be clean')
    jobs = [
        ('full_extend_recovery', recovery, Path('logs/r1_full_extend_recovery/seed1_20261113/recovered'),
         'scripts.emotic.run_r1_full_extend_recovery'),
        ('fusion_population', population, Path('logs/r1_fusion_population_train/seed1_20261113'),
         'scripts.emotic.run_r1_fusion_population_train'),
    ]
    state = {'status': 'running', 'git_head': git, 'cuda_visible_devices': '0',
             'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'jobs': []}
    for name, root, logs, module in jobs:
        state['jobs'].append({'name': name, 'root': str(root), 'logs': str(logs), 'status': 'queued',
                             'command': [sys.executable, '-u', '-m', module, 'run', '--root', str(root), '--logs', str(logs)]})
    write(args.root/'status.json', state)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='72', MKL_NUM_THREADS='72')
    for job in state['jobs']:
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != git:
            raise RuntimeError('Source changed while queued')
        logs = Path(job['logs']); logs.mkdir(parents=True, exist_ok=True)
        job.update(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        with (logs/'controller.log').open('x') as handle:
            process = subprocess.Popen(job['command'], env=env, stdout=handle, stderr=subprocess.STDOUT)
            job['pid'] = process.pid
            write(args.root/'status.json', state)
            print('STARTED', job['name'], process.pid, flush=True)
            code = process.wait()
        (logs/'exit.status').write_text(str(code) + '\n')
        job.update(exit_code=code, status='complete' if code == 0 else 'failed',
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(args.root/'status.json', state)
        print('FINISHED', job['name'], code, flush=True)
        # Independent second hypothesis should still execute if recovery fails.
    state['status'] = 'complete' if all(j['exit_code'] == 0 for j in state['jobs']) else 'finished_with_failure'
    write(args.root/'status.json', state)
    if state['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
