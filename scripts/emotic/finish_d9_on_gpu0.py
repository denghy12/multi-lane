"""Finish an in-progress D9 batch after both Face receivers have checkpointed on GPU1."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from multi_lane.track_a.r1_crossvit_experiment import read, sha, write
from scripts.emotic.run_d9_branch_refinement import ANCHOR, JOBS, LOGS, ROOT, execute, input_paths

RUN_FACE = """from scripts.emotic.run_d9_branch_refinement import Branch, ROOT, worker
job = Branch('independent', 'face', 0)
if not (job.path(ROOT) / 'completion.json').is_file():
    worker(job, ROOT, False)
"""


def check_source(root, replay):
    handoff = read(root / 'gpu0_handoff.json')['paths']
    for arm, record in handoff.items():
        checkpoint = Path(record['checkpoint'])
        if checkpoint.resolve() != (replay / checkpoint).resolve() or sha(checkpoint) != record['sha256']:
            raise ValueError('The replay worktree must use the original, verified recovery state')
        git_head = subprocess.check_output(['git','-C',str(replay),'rev-parse','HEAD'], text=True).strip()
        if git_head != record['git_commit']:
            raise ValueError('Resume from the original training commit')
    if set(handoff) != {'sync','independent'}:
        raise ValueError('Expected two checkpointed Face workers')


def completed(root, job):
    path = job.path(root) / 'completion.json'
    if not path.exists():
        return False
    record = read(path)
    if (record['status'], record['branch'], record['seed']) != ('complete', job.name, 1):
        raise ValueError(f'Unexpected completed branch: {path}')
    return True


def gpu0_free():
    return int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                                         '--format=csv,noheader,nounits'], text=True).strip())


def run(replay):
    root = ROOT
    if (root / 'completion.json').is_file() and read(root/'completion.json').get('all_branches_complete'):
        return
    check_source(root, replay)
    logs = LOGS / root.name
    formal = logs / 'formal'
    handoff = read(root / 'gpu0_handoff.json')
    replacement = next(j for j in JOBS if j.name == 'independent_face')
    process = None
    while True:
        free = gpu0_free()
        done = {j.name: completed(root, j) for j in JOBS}
        states = {name: 'complete' if yes else 'running on GPU0' for name, yes in done.items()}
        oldq = read(root / 'queue.json')
        unexpected = [k for k,v in oldq['finished'].items()
                      if v != 'complete' and k not in ('sync_face','independent_face')]
        if unexpected:
            raise RuntimeError(f'Original workers reported a failure: {unexpected}')
        sync_exit = formal / 'sync_face_gpu0.exit'
        if sync_exit.exists() and int(sync_exit.read_text()) != 0:
            raise RuntimeError('Migrated sync Face failed; see sync_face_gpu0.log')
        if process is not None and process.poll() not in (None, 0):
            raise RuntimeError('Migrated independent Face failed; see independent_face_gpu0.log')
        if not done[replacement.name] and process is None:
            states[replacement.name] = 'waiting for GPU0 memory'
            # Both dry-run and complete historical eight-lane traces are accounted for.
            # Up to ~0.4 GiB per live receiver beyond its early-task footprint.
            active_receivers = sum(not done[j.name] for j in JOBS if j.kind == 'attention' and j != replacement)
            needed = 5200 + 400 * active_receivers + (400 if not done['box_full'] else 0)
            if free >= needed:
                with (formal / 'independent_face_gpu0.log').open('a') as log:
                    job_lock = (root / '.worker-locks/independent_face_gpu0.lock').resolve()
                    command = ['flock', str(job_lock), sys.executable, '-u', '-c', RUN_FACE]
                    process = subprocess.Popen(command, cwd=replay, env=dict(
                        os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2',
                        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'), stdout=log, stderr=subprocess.STDOUT)
                states[replacement.name] = 'running on GPU0'
                print('Independent Face resumed on GPU0', process.pid, flush=True)
        write(root / 'gpu0_queue.json', dict(device=0, free_mib=free, paths=states,
              independent_pid=process.pid if process else None, handoff_time=handoff['time']))
        if all(done.values()) and (process is None or process.poll() == 0):
            break
        time.sleep(15)

    # The original dispatcher sees the deliberate GPU1 process exits. Let it finish
    # recording them and release all output files before certifying the new batch.
    old_exit = logs / 'exit.status'
    while not old_exit.exists():
        time.sleep(5)
    old_result = read(root / 'completion.json')
    expected = {n for n,v in old_result.get('outcomes', {}).items() if v != 'complete'}
    if expected != {'sync_face','independent_face'} or old_result.get('phase') != 'formal':
        raise RuntimeError(f'Unexpected old dispatcher result: {old_result}')
    write(root / 'retired_gpu01_dispatch.json', dict(result=old_result, queue=read(root/'queue.json'),
            exit_status=int(old_exit.read_text())))
    write(root / 'stage.json', {'stage':'summarize_gpu0'})
    execute('scripts.emotic.summarize_d9_refinement', ['--root',root,'--anchor',ANCHOR,
            '--manifest',input_paths().face_manifest_root])
    if any(sha(Path(p)) != h for p,h in read(root/'protocol.json')['sources'].items()):
        raise RuntimeError('Archive inputs changed')
    write(root/'queue.json',dict(phase='formal', device=0, pending=[], running={},
                                 finished={j.name:'complete' for j in JOBS}))
    write(root/'completion.json',dict(status='complete', anchor_hashes_verified=True,
                                      resumed_face_on_gpu=0, all_branches_complete=True))
    write(root/'stage.json',{'stage':'complete'})
    old_exit.write_text('0\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-worktree', type=Path, required=True)
    args = parser.parse_args()
    with (ROOT / '.worker-locks/gpu0_migration.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.replay_worktree.resolve())
