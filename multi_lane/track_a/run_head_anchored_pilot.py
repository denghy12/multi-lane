"""GPU0-only queue for four equal-budget task0--2 validation adaptations."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

from .head_anchored_pilot import ARMS
from .run_historical_b32_test_attribution import read, write


def verify(root):
    results = {arm:read(root/arm/'result.json') for arm in ('BASE', *ARMS)}
    assert all(r['status'] == 'complete' and not r['config']['test_loaded']
               and r['rng_isolation_verified'] for r in results.values())
    audits = {a:read(root/a/'audits.json') for a in ARMS}
    for task in ('0', '1', '2'):
        traces = [audits[a][task]['batch_trace'] for a in ARMS]
        assert all(t == traces[0] for t in traces), 'Actual training batches differ'
        assert all(audits[a][task]['initial_current_predictions_exact'] and
                   audits[a][task]['old_student_parameters_unchanged'] and
                   audits[a][task]['source_checkpoint_unchanged'] for a in ARMS)
    rows = []
    for arm, result in results.items():
        row = dict(arm=arm, pilot_final_mAP=result['pilot_final_mAP'],
                   pilot_average_mAP=result['pilot_average_mAP'])
        row.update({f'task{m["task_id"]}_mAP':m['mAP'] for m in result['task_metrics']})
        row.update({f'task{m["task_id"]}_attention_off_mAP':m['mAP'] for m in result['off_task_metrics']})
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (root/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(rows)
    return dict(actual_training_batches_equal=True, teacher_replay_exact=True,
        module_only_off_equals_teacher=True, tasks=3, test_loaded=False,
        full_eight_stage_average=False)


def run(args):
    assert args.run_id and '/' not in args.run_id and args.run_id not in ('.', '..')
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    root = Path('output/historical_b32_head_anchored_pilot')/args.run_id
    logs = Path('logs/historical_b32_head_anchored_pilot')/args.run_id
    if not args.smoke:
        gate = read(args.smoke_root/'completion.json')
        assert gate['status'] == 'complete' and gate['smoke'] and gate['head'] == head
    protocol = dict(head=head, smoke=args.smoke, seed=1, tasks=[0,1,2], split='val',
        test_loaded=False, historical_test_exploration=True, epochs=5, learning_rate=.0004,
        batch_size=64, gpu=0, max_parallel=4, evaluation_scope='samples_intersect_seen_classes',
        full_eight_stage_average=False, status='running')
    root.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists():
        assert read(root/'protocol.json') == protocol
    else: write(root/'protocol.json', protocol)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='72', MKL_NUM_THREADS='72', PYTHONUNBUFFERED='1')
    pending = [a for a in (*ARMS, 'BASE') if not (root/a/'result.json').exists()]
    active = []; records = []; failed = False; last_launch = 0
    try:
        while pending or active:
            for process, record in list(active):
                code = process.poll()
                if code is None: continue
                active.remove((process, record)); record.update(exit=code, finished_unix=time.time())
                failed |= code != 0
                print('FINISH', record['arm'], code, flush=True)
            if failed and not active: raise RuntimeError('Worker failed; recoverable checkpoints retained')
            if pending and not failed and len(active)<4 and time.time()-last_launch>=12:
                free = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip())
                stat = os.statvfs('/dev/shm'); shm = stat.f_bavail*stat.f_frsize/1024**2
                if free>=6000 and shm>=1024:
                    arm = pending.pop(0)
                    command = [sys.executable, '-u', '-m', 'multi_lane.track_a.head_anchored_pilot',
                        '--source-run', str(args.source_run), '--data-root', str(args.data_root),
                        '--clip-checkpoint', str(args.clip_checkpoint), '--face-manifest-root', str(args.face_manifest_root),
                        '--output', str(root/arm), '--arm', arm]
                    if args.smoke: command.append('--smoke')
                    log = logs/(arm+'.log')
                    with log.open('a') as f:
                        process = subprocess.Popen(command, env=env, stdout=f, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                    record = dict(arm=arm, pid=process.pid, command=command, log=str(log), started_unix=time.time())
                    active.append((process, record)); records.append(record); last_launch = time.time()
                    print('START', arm, process.pid, flush=True)
            write(logs/'queue.json', dict(records=records, pending=pending, failed=failed))
            if pending or active: time.sleep(3)
        audit = verify(root)
        write(root/'completion.json', dict(protocol, status='complete', verification=audit))
        print('HEAD_ANCHORED_PILOT_COMPLETE', root, flush=True)
    except BaseException as error:
        write(root/'failure.json', dict(error=repr(error))); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    for key in ('source-run','data-root','clip-checkpoint','face-manifest-root'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--smoke-root', type=Path)
    run(p.parse_args())
