"""GPU0-only memory-aware queue for all five independent three-view controls."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .crossvit_three_view import ARMS
from .protected_joint_experiment import runtime
from .run_historical_b32_test_attribution import read,write


def resources():
    values=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free,memory.total',
        '--format=csv,noheader,nounits'],text=True).strip().split(',')
    shm=os.statvfs('/dev/shm')
    return int(values[0]),int(values[1]),shm.f_bavail*shm.f_frsize/1024**2


def audit(root,smoke):
    tasks=[0,1] if smoke else list(range(8));initial=set();batches={}
    for arm in ARMS:
        cfg=read(root/arm/'train/config.json');initial.add(cfg['initial_common_sha256'])
        audits=read(root/arm/'train/audits.json')
        for task in tasks:
            a=audits[str(task)]
            assert a['old_parameters_exact'] and a['frozen_visual_exact']
            trace=a['batches']
            if task in batches:assert trace==batches[task],('Augmented batch mismatch',arm,task)
            else:batches[task]=trace
    assert len(initial)==1,'Unpaired branch initialization'
    write(root/'paired_audit.json',dict(status='complete',tasks=tasks,arms=list(ARMS),
        common_initial_parameters_exact=True,all_traced_augmented_batches_exact=True,
        old_parameters_and_fixed_TRAIN_predictions_protected=True))


def run(args):
    git=runtime();args.output.mkdir(parents=True,exist_ok=True);args.logs.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks=[0,1] if args.smoke else list(range(8))
    profile=None
    if not args.smoke:
        if args.smoke_root is None:raise ValueError('A completed same-source smoke gate is required')
        gate=read(args.smoke_root/'completion.json')
        if gate['status']!='complete' or gate['git']!=git:raise ValueError('Invalid smoke source')
        profile={a:read(args.smoke_root/a/'train/training_complete.json')['peak_allocated_mib'] for a in ARMS}
        eval_profile=max(read(args.smoke_root/a/'val/result.json')['peak_allocated_mib'] for a in ARMS)
    free,total,shm=resources()
    max_jobs=1 if args.smoke else min(5,max(1,int((total-1536)/(max(profile.values())*1.3+768))))
    if args.maximum_jobs:max_jobs=min(max_jobs,args.maximum_jobs)
    protocol=dict(status='declared',git=git,seed=1,gpu=0,arms=list(ARMS),tasks=tasks,
        smoke=args.smoke,maximum_jobs=max_jobs,maximum_evaluation_jobs=2,
        training_peak_profiles_mib=profile,threads_initialization_training=72,
        threads_evaluation=2,workers=2,prefetch_factor=1,
        source_run=str(args.source_run),output=str(args.output))
    if (args.output/'protocol.json').exists():
        existing=read(args.output/'protocol.json')
        for key in ('git','seed','gpu','arms','tasks','smoke','source_run'):
            if existing[key]!=protocol[key]:raise ValueError('Queue configuration changed')
    write(args.output/'protocol.json',protocol)
    common=sum((['--'+k.replace('_','-'),str(getattr(args,k))] for k in
        ('source_run','data_root','clip_checkpoint','face_manifest_root')),[])
    module=[sys.executable,'-u','-m','multi_lane.track_a.crossvit_experiment']
    pending=[];active=[];records=[];failures=[];completed=set();last=0.
    def ready(job):
        f=Path(job['output'])/('training_complete.json' if job['kind']=='train' else 'result.json')
        return f.exists() and read(f).get('status')=='complete' and read(f).get('tasks')==tasks
    def evaluations(arm):
        for split in ('val',) if args.smoke else ('val','test'):
            output=args.output/arm/split
            pending.append(dict(name=arm+'_'+split,kind='eval',arm=arm,output=str(output),
                command=module+['eval','--training-root',str(args.output/arm/'train'),
                                '--split',split,'--workers','2','--output',str(output)]))
    for arm in ARMS:
        output=args.output/arm/'train'
        cmd=module+['train',*common,'--arm',arm,'--output',str(output)]
        if args.smoke:cmd+=['--smoke','--smoke-updates',str(args.smoke_updates)]
        pending.append(dict(name=arm+'_train',kind='train',arm=arm,output=str(output),command=cmd))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72',
             OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    try:
        while pending or active:
            for p,j in list(active):
                code=p.poll()
                if code is None:continue
                active.remove((p,j));j.update(exit=code,finished_unix=time.time())
                if code or not ready(j):failures.append(j)
                else:
                    completed.add(j['name'])
                    if j['kind']=='train':evaluations(j['arm'])
                print('FINISH',j['name'],code,flush=True)
            if failures:pending=[]
            if pending and len(active)<max_jobs and time.time()-last>=20:
                eligible=[j for j in pending if j['kind']=='train' or sum(x['kind']=='eval' for _,x in active)<2]
                for job in eligible:
                    if ready(job):
                        pending.remove(job);completed.add(job['name'])
                        if job['kind']=='train':evaluations(job['arm'])
                        break
                    free,total,shm=resources()
                    need=6000 if args.smoke else ((profile[job['arm']] if job['kind']=='train' else eval_profile)*1.3+768)
                    if free<need or shm<512:continue
                    pending.remove(job);log=args.logs/(job['name']+'.log')
                    with log.open('a') as out:
                        process=subprocess.Popen(job['command'],env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                    job.update(pid=process.pid,started_unix=time.time(),log=str(log),memory_admission_mib=need)
                    active.append((process,job));records.append(job);last=time.time()
                    print('START',job['name'],process.pid,'slots',max_jobs,flush=True);break
            write(args.logs/'queue.json',dict(status='running',records=records,
                active=[j['name'] for _,j in active],pending=[j['name'] for j in pending],
                completed=sorted(completed),failures=failures,maximum_jobs=max_jobs))
            if pending or active:time.sleep(3)
        if failures:raise RuntimeError('Failed jobs: '+repr(failures))
        audit(args.output,args.smoke)
        rows=[]
        for arm in ARMS:
            row=dict(arm=arm)
            for split in ('val',) if args.smoke else ('val','test'):
                result=read(args.output/arm/split/'result.json')
                for name in ('final_mAP','average_mAP'):row[split+'_'+name]=result['summary'][name]
            rows.append(row)
        with (args.output/'summary.csv').open('w') as handle:
            writer=csv.DictWriter(handle,list(rows[0]));writer.writeheader();writer.writerows(rows)
        write(args.output/'completion.json',dict(status='complete',git=git,arms=list(ARMS),tasks=tasks,records=records))
        write(args.logs/'queue.json',dict(status='complete',records=records))
        print('CROSSVIT_QUEUE_COMPLETE',args.output,flush=True)
    except BaseException as error:
        write(args.output/'failure.json',dict(error=repr(error),records=records));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','data-root','clip-checkpoint','face-manifest-root','output','logs'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke-root',type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-updates',type=int,default=4)
    p.add_argument('--maximum-jobs',type=int,choices=range(1,6))
    run(p.parse_args())
