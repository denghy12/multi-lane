"""Memory-aware GPU0 queue for teacher-free three-view learning.

Starts only predeclared same-budget task0 arms. A deterministic, recorded gate
may promote them to task0-2 and add U1/U3 controls; never automatically starts
an eight-task run.
"""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .export_compact_test_scores import _sha256
from .run_historical_b32_test_attribution import read, write

INITIAL_ARMS = ('U0','U2','U4')
CONTROLS = ('U1','U3')
GATE = dict(maximum_full_decline=.10, maximum_fused_decline=.10,
    require_fused_not_below_baseline_both_splits=True,
    require_fused_not_below_unprotected_both_splits=True,
    minimum_gain_one_split=.02, require_nonzero_accepted_auxiliary=True,
    require_effective_nonzero_update=True)


def common_train(args, arm, output, tasks):
    cmd = [sys.executable,'-u','-m','multi_lane.track_a.teacher_free_shared_learning','train',
        '--source-run',str(args.source_run), '--data-root',str(args.data_root),
        '--clip-checkpoint',str(args.clip_checkpoint), '--face-manifest-root',str(args.face_manifest_root),
        '--arm',arm,'--tasks',str(tasks),'--output',str(output)]
    if arm == 'U0' and args.historical_reference_root:
        cmd += ['--historical-reference-root',str(args.historical_reference_root)]
    return cmd


def task0_values(result_root, arm, split):
    row = read(result_root/arm/split/'task0.json')
    return dict(full=float(row['views']['metrics']['full']['mAP']),
        fused=float(row['metrics']['mAP']))


def promotion_decision(root):
    values = {split:{arm:task0_values(root,arm,split) for arm in INITIAL_ARMS}
        for split in ('val','test')}
    reasons = []
    gains = []
    for split, rows in values.items():
        base, unprotected, candidate = rows['U0'], rows['U2'], rows['U4']
        if candidate['full'] < base['full']-GATE['maximum_full_decline']:
            reasons.append(split+': Full fell by more than 0.10 mAP')
        if candidate['fused'] < base['fused']-GATE['maximum_fused_decline']:
            reasons.append(split+': fusion fell by more than 0.10 mAP')
        if candidate['fused'] < base['fused']:
            reasons.append(split+': fusion is below matched U0')
        if candidate['fused'] < unprotected['fused']:
            reasons.append(split+': fusion is below U2')
        gains.append(candidate['fused']-base['fused'])
    history = read(root/'U4/train/training_history.json')['0']
    accepted = any(float(row.get('complementary_accepted_batches',0.)) > 0. for row in history)
    effective = any(float(row.get('complementary_effective_nonzero_update',0.)) > 0. for row in history)
    if not accepted:
        reasons.append('No nonzero auxiliary update was accepted')
    if not effective:
        reasons.append('No accepted auxiliary update changed the actual rounded parameter step')
    if max(gains) < GATE['minimum_gain_one_split']:
        reasons.append('No split reached the predeclared 0.02 mAP fusion screening gain')
    return dict(status='promote' if not reasons else 'stop_after_task0',
        gate=GATE, values=values, reasons=reasons, nonzero_auxiliary_update=accepted,
        effective_nonzero_update=effective,
        test_used_for_development=True, statistical_significance_claim=False,
        next_tasks=[0,1,2] if not reasons else [], automatic_eight_task_run=False)


def available_memory():
    free = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())
    shm = os.statvfs('/dev/shm')
    return free, shm.f_bavail*shm.f_frsize


def enqueue_stage(args, root, arms, tasks):
    result = []
    for arm in arms:
        output = root/arm/'train'
        complete = output/'training_complete.json'
        ready = complete.exists() and read(complete).get('tasks') == list(range(tasks))
        if not ready:
            result.append(dict(name=arm+'_train_'+str(tasks), arm=arm, kind='train', tasks=tasks,
                command=common_train(args,arm,output,tasks), output=str(output)))
        else:
            for split in ('val','test'):
                target = root/arm/(split if tasks == 1 else split+'_task012')
                result.append(evaluation_job(arm,output,target,split,tasks))
    return result


def evaluation_job(arm, training, output, split, tasks):
    return dict(name=arm+'_'+split+'_'+str(tasks),arm=arm,kind='eval',tasks=tasks,
        split=split,output=str(output),
        command=[sys.executable,'-u','-m','multi_lane.track_a.teacher_free_shared_learning','eval',
            '--training-root',str(training),'--split',split,'--workers','2','--output',str(output)])


def output_complete(job):
    filename = 'training_complete.json' if job['kind']=='train' else 'result.json'
    path = Path(job['output'])/filename
    return path.exists() and read(path).get('status') == 'complete' \
        and read(path).get('tasks') == list(range(job['tasks']))


def _run(args):
    from .teacher_free_shared_learning import runtime
    git = runtime()
    root = args.output
    logs = args.logs
    logs.mkdir(parents=True,exist_ok=True)
    protocol = dict(git=git,gpu=0,seed=1,initial_tasks=[0],initial_arms=list(INITIAL_ARMS),
        auto_promote_task012=args.auto_promote_task012,controls_on_promotion=list(CONTROLS),
        automatic_eight_task_run=False,gate=GATE,source_config=str(args.source_run/'config.json'),
        source_config_sha256=_sha256(args.source_run/'config.json'), epochs_per_task=30,
        batch_size=64,threads=72,train_workers=2,eval_workers=2,prefetch_factor=1,
        maximum_jobs=args.maximum_jobs,minimum_train_free_mib=args.minimum_train_free_mib,
        minimum_eval_free_mib=args.minimum_eval_free_mib,
        source_task_parameter_loading=False,teacher_outputs=False,
        evaluation_scope='samples_intersect_seen_classes',test_used_for_development=True)
    if (root/'protocol.json').exists():
        if read(root/'protocol.json') != protocol:
            raise ValueError('Queue restart changed the predeclared protocol')
    else:
        write(root/'protocol.json',protocol)
    previous_gate = root/'task0_promotion_decision.json'
    promotion = read(previous_gate) if previous_gate.exists() else None
    continued = bool(args.auto_promote_task012 and promotion and promotion['status']=='promote')
    pending = enqueue_stage(args,root,INITIAL_ARMS+CONTROLS,3) if continued else enqueue_stage(args,root,INITIAL_ARMS,1)
    active, records, failures = [], [], []
    last = 0.
    gate_finished = continued
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72')
    started = time.time()
    try:
        while pending or active or not gate_finished:
            for process,row in list(active):
                code = process.poll()
                if code is None:
                    continue
                row.update(exit=code,finished_unix=time.time())
                active.remove((process,row))
                if code:
                    failures.append(row)
                    continue
                if not output_complete(row):
                    raise AssertionError(('Child did not save expected completion',row))
                if row['kind']=='train':
                    for split in ('val','test'):
                        output = root/row['arm']/(split if row['tasks']==1 else split+'_task012')
                        job = evaluation_job(row['arm'],Path(row['output']),output,split,row['tasks'])
                        if not output_complete(job):
                            pending.append(job)
            if failures:
                # Finish children already in flight, but do not submit dependent work.
                pending=[]
                gate_finished=True
            elif not pending and not active and not gate_finished:
                promotion = promotion_decision(root)
                write(root/'task0_promotion_decision.json',promotion)
                gate_finished=True
                if args.auto_promote_task012 and promotion['status']=='promote':
                    pending.extend(enqueue_stage(args,root,INITIAL_ARMS+CONTROLS,3))
                    print('TASK0_GATE_PROMOTE',promotion,flush=True)
                else:
                    print('TASK0_GATE_STOP',promotion,flush=True)
            if pending and len(active)<args.maximum_jobs and time.time()-last>=args.launch_gap_seconds:
                job = pending[0]
                if output_complete(job):
                    pending.pop(0)
                    if job['kind']=='train':
                        for split in ('val','test'):
                            output=root/job['arm']/(split if job['tasks']==1 else split+'_task012')
                            pending.append(evaluation_job(job['arm'],Path(job['output']),output,split,job['tasks']))
                    continue
                free, shm = available_memory()
                minimum=args.minimum_train_free_mib if job['kind']=='train' else args.minimum_eval_free_mib
                if free>=minimum and shm>=768*1024**2:
                    pending.pop(0)
                    log=logs/(job['name']+'.log')
                    with log.open('a') as handle:
                        process=subprocess.Popen(job['command'],env=env,stdout=handle,
                            stderr=subprocess.STDOUT,start_new_session=True)
                    row=dict(job,pid=process.pid,log=str(log),started_unix=time.time())
                    records.append(row)
                    active.append((process,row))
                    last=time.time()
                    print('START',job['name'],process.pid,'free_mib='+str(free),flush=True)
            write(logs/'queue.json',dict(status='running',records=records,
                active=[row['name'] for _,row in active],pending=[row['name'] for row in pending],
                failures=[row['name'] for row in failures],gate_finished=gate_finished,
                promotion=promotion))
            if active or pending:
                time.sleep(3)
        if failures:
            raise RuntimeError(('Child jobs failed; completed checkpoints retained',failures))
        write(root/'completion.json',dict(status='complete',git=git,
            initial_task0_validation_and_test_complete=True,
            promotion=promotion,controls_task012_requested=bool(args.auto_promote_task012 and
                promotion and promotion['status']=='promote'),
            elapsed_seconds=time.time()-started,records=records,automatic_eight_task_run=False))
        write(logs/'queue.json',dict(status='complete',records=records,promotion=promotion))
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error),records=records,
            active_children=[row['pid'] for _,row in active],
            checkpoints_retained=True))
        raise


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This queue is already running; do not launch duplicate children') from error
        queue = args.logs/'queue.json'
        if queue.exists():
            previous=read(queue)
            active=set(previous.get('active',[]))
            for row in previous.get('records',[]):
                if row['name'] not in active:
                    continue
                command=Path('/proc')/str(row['pid'])/'cmdline'
                if command.exists() and b'multi_lane.track_a.teacher_free_shared_learning' in command.read_bytes():
                    raise RuntimeError('A child from the previous queue is still running: '+str(row['pid']))
        return _run(args)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','logs','source-run','data-root','clip-checkpoint','face-manifest-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--historical-reference-root',type=Path)
    p.add_argument('--maximum-jobs',type=int,choices=(1,2,3),default=3)
    p.add_argument('--minimum-train-free-mib',type=int,default=6000)
    p.add_argument('--minimum-eval-free-mib',type=int,default=4000)
    p.add_argument('--launch-gap-seconds',type=float,default=8.)
    p.add_argument('--auto-promote-task012',action='store_true')
    return p


if __name__=='__main__':
    run(parser().parse_args())
