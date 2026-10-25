"""One GPU0 queue: preserve pilot progress, finish eight tasks, add Face attention."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

from .head_anchored_pilot import ARMS
from .historical_b32_modules import training_arguments
from .run_historical_b32_modules import verify_modules
from .run_historical_b32_test_attribution import read, write, MODEL_HEAD


def common(args):
    return ['--source-run',str(args.source_run),'--data-root',str(args.data_root),
        '--clip-checkpoint',str(args.clip_checkpoint),'--face-manifest-root',str(args.face_manifest_root)]


def evaluation_jobs(args,root,arm,kind,trained):
    result=[]
    splits=('val',) if args.smoke and kind!='person_reference' else ('val','test')
    heads=('b0','native') if kind=='joint' else ('b0',) if kind=='person_reference' else ('native',)
    for split in splits:
        for head in heads:
            conditions=['on']
            if head=='b0': conditions+=['face_off','off'] if kind=='joint' else ['off']
            elif kind=='student' and arm!='HEAD_FIXED_NO_ATTENTION': conditions+=['off']
            name=f'{arm}/{head}_{split}'
            flags=['eval',*common(args),'--trained-run',str(trained),'--kind',kind,'--head',head,
                '--split',split,'--conditions',*conditions,'--output',str(root/name)]
            if args.smoke:
                flags+=['--checkpoint-tasks',*(['0','1'] if kind=='joint' else ['7'] if kind=='person_reference' else ['0','2','7'])]
            result.append(dict(name=name,kind='eval',arm=arm,module='multi_lane.track_a.head_anchored_eight',arguments=flags))
    return result


def verify_continuation(root,arm,smoke):
    folder=root/arm/'train'; result=read(folder/'training_complete.json')
    assert result['status']=='complete' and result['smoke']==smoke and result['prefix_preserved_exact']
    assert result['repeated_prefix_updates']==0 and len(result['checkpoints'])==8
    history=read(folder/'training_history.json')
    for task in range(8):
        rows=history[str(task)]
        assert all(r['skipped_optimizer_steps']==0 for r in rows)
        if task<3 or not smoke: assert len(rows)==5
        else: assert sum(r['optimizer_steps'] for r in rows)==2
    return result


def summarize(args,root):
    audits={a:read(root/a/'train/audits.json') for a in ARMS}
    for t in map(str,range(8)):
        trace=[audits[a][t]['batch_trace'] for a in ARMS]
        assert all(x==trace[0] for x in trace), 'Actual adaptation batches differ at task '+t
    rows=[]
    for arm in (*ARMS,'PERSON_REFERENCE','FACE_JOINT'):
        heads=('b0','native') if arm=='FACE_JOINT' else ('b0',) if arm=='PERSON_REFERENCE' else ('native',)
        for head in heads:
            split_results={}
            for split in ('val','test'):
                p=root/arm/(head+'_'+split)/'result.json'
                if p.exists(): split_results[split]=read(p)
            for condition in next(iter(split_results.values()))['summaries']:
                row=dict(method=arm,head=head,condition=condition)
                for split,result in split_results.items():
                    assert result['status']=='complete'
                    if not args.smoke: assert result['full_eight_stage_average']
                    row.update({split+'_'+k:v for k,v in result['summaries'][condition].items()
                                if k in ('final_mAP','average_mAP','final_cF1','final_oF1')})
                rows.append(row)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (root/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
    return dict(actual_adaptation_batches_equal=True,pilot_progress_preserved=True,
                old_person_reference_task7_exact=True,test_used_for_parameter_search=True)


def run(args):
    assert args.run_id and '/' not in args.run_id and args.run_id not in ('.','..')
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert read(args.pilot_root/'completion.json')['status']=='complete'
    assert read(args.source_run/'config.json')['git']['commit']==MODEL_HEAD
    if not args.smoke:
        gate=read(args.smoke_root/'completion.json')
        assert gate['status']=='complete' and gate['smoke'] and gate['head']==head
    root=Path('output/head_anchored_eight_face')/args.run_id
    logs=Path('logs/head_anchored_eight_face')/args.run_id
    root.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    protocol=dict(head=head,status='running',smoke=args.smoke,seed=1,gpu=0,max_parallel=4,
        pilot_root=str(args.pilot_root),new_adaptation_tasks=[3,4,5,6,7],adaptation_epochs=5,
        face_joint_epochs=30,batch_size=64,workers=2,cpu_threads=72,amp=True,tf32=True,
        evaluation_scope='samples_intersect_seen_classes',test_used_for_parameter_search=True)
    if (root/'protocol.json').exists(): assert read(root/'protocol.json')==protocol
    else:write(root/'protocol.json',protocol)
    face_flags=training_arguments(args,root/'FACE_JOINT','ATTENTION_BOUND')+['--cross-view-face-attention']
    pending=[dict(name='FACE_JOINT/train',kind='joint_train',arm='FACE_JOINT',
                  module='multi_lane.track_a.runner',arguments=face_flags)]
    for arm in ARMS:
        flags=['train',*common(args),'--pilot-root',str(args.pilot_root),'--arm',arm,'--output',str(root/arm/'train')]
        if args.smoke:flags.append('--smoke')
        pending.append(dict(name=arm+'/train',kind='student_train',arm=arm,
            module='multi_lane.track_a.head_anchored_eight',arguments=flags))
    pending+=evaluation_jobs(args,root,'PERSON_REFERENCE','person_reference',args.source_run.parent.parent/'ATTENTION_BOUND/train')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72',PYTHONUNBUFFERED='1')
    active=[];records=[];failed=False;last_launch=0
    try:
        while pending or active:
            for process,r in list(active):
                code=process.poll()
                if code is None:continue
                active.remove((process,r));r.update(exit=code,finished_unix=time.time());failed|=code!=0
                print('FINISH',r['name'],code,flush=True)
                if code==0:
                    try:
                        arm=r['arm']
                        if r['kind']=='student_train':
                            verify_continuation(root,arm,args.smoke)
                            pending+=evaluation_jobs(args,root,arm,'student',root/arm/'train')
                        elif r['kind']=='joint_train':
                            audit=verify_modules(root/arm/'train',args.smoke)
                            original=read(args.source_run.parent.parent/'ATTENTION_BOUND/training_verification.json')
                            assert audit['initial_base_sha256']==original['initial_base_sha256']
                            if not args.smoke:
                                assert audit['cpu_rng_at_task_boundaries']==original['cpu_rng_at_task_boundaries']
                            write(root/arm/'training_verification.json',audit)
                            pending+=evaluation_jobs(args,root,arm,'joint',root/arm/'train')
                    except BaseException as error:
                        r['verification_error']=repr(error);failed=True
                        print('VERIFICATION_FAILED',r['name'],repr(error),flush=True)
            if failed and not active:raise RuntimeError('Worker or verification failed; all checkpoints retained')
            if pending and not failed and len(active)<4 and time.time()-last_launch>=12:
                free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True).strip())
                stat=os.statvfs('/dev/shm');shm=stat.f_bavail*stat.f_frsize/1024**2
                if free>=6000 and shm>=1024:
                    job=pending.pop(0);log=logs/(job['name']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                    command=[sys.executable,'-u','-m',job['module'],*job['arguments']]
                    with log.open('a') as f:process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    r=dict(job,pid=process.pid,log=str(log),started_unix=time.time());active.append((process,r));records.append(r)
                    last_launch=time.time();print('START',r['name'],process.pid,flush=True)
            write(logs/'queue.json',dict(records=records,pending=pending,failed=failed))
            if pending or active:time.sleep(3)
        audit=summarize(args,root)
        write(root/'completion.json',dict(protocol,status='complete',verification=audit))
        print('EIGHT_TASK_FACE_ATTENTION_COMPLETE',root,flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    for key in ('source-run','pilot-root','data-root','clip-checkpoint','face-manifest-root'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-root',type=Path)
    run(p.parse_args())
