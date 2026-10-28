"""Lock TRAIN constants, smoke, then complete GPU0 Val/Test in parallel."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .protected_attention_gate_attribution import TASKS, MODES, calibrate
from .run_historical_b32_test_attribution import read, write
from .teacher_free_shared_learning import runtime


def run(args):
    git=runtime();args.output.mkdir(parents=True,exist_ok=True);args.logs.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    constants=args.output/'train_locked_constants.json'
    calibrate(args.training_root,constants)
    protocol=dict(git=git,seed=1,gpu=0,tasks=list(TASKS),conditions=list(MODES),
        training_root=str(args.training_root),batch_size=64,threads=72,workers=2,
        prefetch_factor=1,worker_threads=1,amp=True,tf32=True,training_updates=0,
        common_feature_ratio_cap=.01,constant_policy='last_five_TRAIN_epoch_means',
        evaluation_scope='samples_intersect_seen_classes',test_used_for_diagnostic=True,
        selected_stage_mean_is_not_eight_task_average=True)
    if (args.output/'protocol.json').exists():
        if read(args.output/'protocol.json')!=protocol:raise ValueError('Protocol changed')
    else:write(args.output/'protocol.json',protocol)
    queue=args.logs/'queue.json'
    if queue.exists():
        for row in read(queue).get('records',[]):
            proc=Path('/proc')/str(row['pid'])/'cmdline'
            if row.get('exit') is None and proc.exists() and b'protected_attention_gate_attribution' in proc.read_bytes():
                raise RuntimeError('A previous diagnostic child is still running')
    records=[];active=[]
    module=[sys.executable,'-u','-m','multi_lane.track_a.protected_attention_gate_attribution','eval']
    common=sum((['--'+name.replace('_','-'),str(getattr(args,name))] for name in
                ('training_root','data_root','clip_checkpoint','face_manifest_root')),[])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72')
    def start(name,split,smoke=False,tail=False):
        output=args.output/name
        command=module+common+['--constants',str(constants),'--split',split,'--workers','2','--output',str(output)]
        if smoke:command+=['--smoke']
        if tail:command+=['--smoke-tail']
        with (args.logs/(name+'.log')).open('a') as handle:
            child=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        row=dict(name=name,pid=child.pid,command=command,started_unix=time.time(),output=str(output))
        records.append(row);write(queue,dict(status='running',records=records))
        print('START',name,child.pid,flush=True)
        return child,row
    try:
        if not (args.output/'smoke_val/result.json').exists():
            child,row=start('smoke_val','val',True)
            row['exit']=child.wait();row['finished_unix']=time.time()
            if row['exit']:raise RuntimeError('Smoke failed; full runs not launched')
        smoke=read(args.output/'smoke_val/result.json')
        assert smoke['status']=='complete' and smoke['smoke'] and smoke['tasks']==list(TASKS)
        if not (args.output/'smoke_tail/result.json').exists():
            child,row=start('smoke_tail','val',True,True)
            row['exit']=child.wait();row['finished_unix']=time.time()
            if row['exit']:raise RuntimeError('Tail smoke failed; full runs not launched')
        tail=read(args.output/'smoke_tail/result.json')
        assert tail['status']=='complete' and tail['smoke_tail']
        for split in ('val','test'):
            if (args.output/split/'result.json').exists():
                assert read(args.output/split/'result.json')['status']=='complete';continue
            while True:
                free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True).strip())
                shm=os.statvfs('/dev/shm')
                if free>=4500 and shm.f_bavail*shm.f_frsize>=768*1024**2:break
                write(queue,dict(status='waiting_for_gpu0_memory',records=records));time.sleep(3)
            active.append(start(split,split));time.sleep(8)
        failures=[]
        for child,row in active:
            row['exit']=child.wait();row['finished_unix']=time.time()
            if row['exit']:failures.append(row)
            write(queue,dict(status='running',records=records))
        if failures:raise RuntimeError(('Diagnostic children failed',failures))
        result={sp:read(args.output/sp/'result.json') for sp in ('val','test')}
        assert all(v['status']=='complete' and v['parameters_and_checkpoints_unchanged'] for v in result.values())
        write(args.output/'completion.json',dict(status='complete',protocol=protocol,
            records=records,training_updates=0,checkpoints_unchanged=True))
        write(queue,dict(status='complete',records=records))
        print('PROTECTED_GATE_ATTRIBUTION_COMPLETE',args.output,flush=True)
    except BaseException as error:
        write(args.output/'failure.json',dict(error=repr(error),records=records));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('training-root','data-root','clip-checkpoint','face-manifest-root','output','logs'):
        p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
