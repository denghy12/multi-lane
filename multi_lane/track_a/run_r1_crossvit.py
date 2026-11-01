"""GPU0 queue: reconstruct archived R1 before admitting any interaction run."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .r1_crossvit_experiment import (FAMILIES,VIEWS,flags,fusion,read,write,source_runs)
from .evaluation_scores import load_evaluation_scores,align_evaluation_scores
from .runner import git_metadata


def memory():
    value=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip()
    s=os.statvfs('/dev/shm')
    return int(value),s.f_bavail*s.f_frsize/1024**2


def reproduction_gate(args):
    rows=[]
    for view,reference in source_runs(args.reference_root).items():
        actual=args.output/'native'/view/'train'
        a,b=read(actual/'seed_summary.json'),read(reference/'seed_summary.json')
        assert a['completed_epochs']==b['completed_epochs']==240
        assert a['config']['adapter_residual_scale']==b['config']['adapter_residual_scale']==.1
        for task in range(8):
            x=load_evaluation_scores(actual/'test_scores'/f'task{task}.npz',64)
            y=load_evaluation_scores(reference/'test_scores'/f'task{task}.npz',64)
            ids,xz,yz,targets,xp,yp=align_evaluation_scores(x,y)
            delta=float(a['task_metrics'][task]['mAP']-b['task_metrics'][task]['mAP'])
            if abs(delta)>1e-4:raise AssertionError(('R1_source_not_reproduced',view,task,delta))
            rows.append(dict(view=view,task=task,mAP_difference=delta,
                max_abs_logits=float(np.max(np.abs(xz-yz))),max_abs_probabilities=float(np.max(np.abs(xp-yp)))))
    summary=read(args.output/'native/fusion.json')['test']['summaries']['probability']
    if abs(summary['final_mAP']-32.957474843299394)>1e-4 or abs(summary['average_mAP']-40.115401219689225)>1e-4:
        raise AssertionError(('R1_fusion_not_reproduced',summary))
    write(args.output/'reproduction_gate.json',dict(status='verified',seed=1,
        expected_final=32.957474843299394,expected_average=40.115401219689225,
        actual=summary,stage_tolerance_mAP=1e-4,source_parameters_loaded=False,records=rows))


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Only GPU0 authorized')
    git=git_metadata(Path(__file__).resolve().parents[2])
    if git['dirty']:raise ValueError('Require clean source worktree')
    if not args.smoke:
        if args.smoke_root is None:raise ValueError('A same-source smoke is required')
        smoke=read(args.smoke_root/'completion.json')
        if smoke.get('status')!='complete' or not smoke.get('smoke') or smoke['git']!=git:
            raise ValueError('Smoke must complete on exactly this clean source revision')
    args.output.mkdir(parents=True,exist_ok=True);args.logs.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(args.output/'protocol.json',dict(git=git,seed=1,gpu=0,smoke=args.smoke,
        families=list(FAMILIES),views=list(VIEWS),primary='archived_R1_probability',secondary='logit_same_weights',
        supervision='independent_native_BCE_Adapter_ASL',source_commit='acf4634',
        teachers=False,load_pretrained_task_weights=False,maximum_jobs=args.maximum_jobs,
        production_interactions_require_exact_R1_reproduction=True))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72',
        OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    pending=[];active=[];records=[];done=set();failures=[];fused=set();admitted=set();last=0.
    def admit(kind):
        if kind in admitted:return
        admitted.add(kind)
        for view in VIEWS:
            pending.append(dict(name=kind+'_'+view+'_train',kind=kind,view=view,action='train',
                command=[sys.executable,'-u','-m','multi_lane.track_a.runner',*flags(args,kind,view,args.smoke)]))
    for k in FAMILIES if args.smoke else ('native',):admit(k)
    def complete(job):
        p=args.output/job['kind']/job['view']/('train/seed_summary.json' if job['action']=='train' else 'val/result.json')
        if not p.exists():return False
        d=read(p)
        if d.get('status')!='complete':return False
        if job['action']=='train':return len(d['task_metrics'])==(2 if args.smoke else 8)
        return d['tasks']==list(range(2 if args.smoke else 8))
    def after_train(kind,view):
        output=args.output/kind/view/'val'
        pending.append(dict(name=kind+'_'+view+'_val',kind=kind,view=view,action='val',
            command=[sys.executable,'-u','-m','multi_lane.track_a.r1_crossvit_experiment','val',
                '--training-root',str(args.output/kind/view/'train'),'--output',str(output),
                '--data-root',str(args.data_root),'--clip-checkpoint',str(args.clip_checkpoint),
                '--face-manifest-root',str(args.face_manifest_root),*(['--smoke'] if args.smoke else [])]))
    try:
        while pending or active:
            for proc,job in list(active):
                code=proc.poll()
                if code is None:continue
                active.remove((proc,job));job.update(exit=code,finished_unix=time.time())
                if code or not complete(job):failures.append(job)
                else:
                    done.add(job['name'])
                    if job['action']=='train':after_train(job['kind'],job['view'])
                print('FINISH',job['name'],code,flush=True)
            for kind in tuple(admitted):
                needed={kind+'_'+v+'_'+a for v in VIEWS for a in ('train','val')}
                if kind not in fused and needed<=done:
                    fusion(args.output,kind,args.face_manifest_root,args.smoke);fused.add(kind)
                    if kind=='native' and not args.smoke:
                        reproduction_gate(args)
                        for other in ('attention','pool'):admit(other)
            if pending and len(active)<args.maximum_jobs and time.time()-last>=15:
                for job in list(pending):
                    if complete(job):
                        pending.remove(job);done.add(job['name'])
                        if job['action']=='train':after_train(job['kind'],job['view'])
                        break
                    if job['action']=='val' and sum(j['action']=='val' for _,j in active)>=2:continue
                    free,shm=memory();need=3500 if job['kind']=='native' else 5000
                    if free<need or shm<768:continue
                    log=args.logs/(job['name']+'.log')
                    with log.open('a') as f:proc=subprocess.Popen(job['command'],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    pending.remove(job);job.update(pid=proc.pid,started_unix=time.time(),log=str(log))
                    active.append((proc,job));records.append(job);last=time.time()
                    print('START',job['name'],proc.pid,flush=True);break
            write(args.logs/'queue.json',dict(status='running',active=[j['name'] for _,j in active],
                pending=[j['name'] for j in pending],complete=sorted(done),failures=failures,records=records))
            if pending or active:time.sleep(3)
        write(args.output/'completion.json',dict(status='complete' if not failures and len(fused)==3 else 'incomplete',
            git=git,smoke=args.smoke,fused=sorted(fused),failures=failures,records=records))
        success=not failures and len(fused)==3
        write(args.logs/'queue.json',dict(status='complete' if success else 'incomplete',failures=failures,records=records))
        if not success:raise RuntimeError('Experiment incomplete; other independent arms were allowed to finish')
    except BaseException as e:
        write(args.output/'failure.json',dict(error=repr(e),records=records));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-root','data-root','clip-checkpoint','face-manifest-root','output','logs'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--maximum-jobs',type=int,choices=(1,2,3),default=3)
    p.add_argument('--smoke-root',type=Path)
    run(p.parse_args())
