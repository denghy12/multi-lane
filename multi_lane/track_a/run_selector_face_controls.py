"""GPU0 queue for private Selectors and Face-only training on the verified strong model."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .export_compact_test_scores import _sha256, _torch_load
from .historical_b32_modules import training_arguments
from .historical_b32_search import replace_flag
from .run_historical_b32_modules import verify_modules
from .run_historical_b32_test_attribution import read, write

NAMES=('PRIVATE_SELECTORS','FROZEN_PERSON_FACE','REFERENCE_PERSON','REFERENCE_DUAL')


def common(args):
    return ['--source-run',str(args.source_run),'--data-root',str(args.data_root),
        '--clip-checkpoint',str(args.clip_checkpoint),'--face-manifest-root',str(args.face_manifest_root)]


def eval_jobs(args,root,name):
    private=name=='PRIVATE_SELECTORS';frozen=name=='FROZEN_PERSON_FACE'
    kind='frozen_face' if frozen else 'person_reference' if name=='REFERENCE_PERSON' else 'joint'
    trained=(root/name/'train' if private or frozen else args.person_run if name=='REFERENCE_PERSON' else args.dual_run)
    heads=('b0','native') if private else ('b0',)
    splits=('val',) if args.smoke and (private or frozen) else ('val','test')
    jobs=[]
    for split in splits:
        for head in heads:
            modes=['on'] if head=='native' else ['on','face_off'] if frozen else ['on','off'] if name=='REFERENCE_PERSON' else ['on','face_off','off']
            jobname=name+'/'+head+'_'+split
            flags=['eval',*common(args),'--trained-run',str(trained),'--kind',kind,'--head',head,'--split',split,
                '--conditions',*modes,'--output',str(root/jobname)]
            if args.smoke:flags+=['--checkpoint-tasks',*(['0','1'] if private or frozen else ['7'])]
            jobs.append(dict(name=jobname,kind='eval',arm=name,module='multi_lane.track_a.head_anchored_eight',arguments=flags))
    return jobs


def verify_private(root,smoke,source):
    train=root/'PRIVATE_SELECTORS/train';audit=verify_modules(train,smoke)
    if not smoke:
        assert audit['cpu_rng_at_task_boundaries']==read(source.parent/'training_verification.json')['cpu_rng_at_task_boundaries']
    distances=[]
    for task in range(2 if smoke else 8):
        selectors=_torch_load(train/'compact_checkpoints'/f'task{task}.pth')['model']['selectors']
        assert selectors.shape==(8,3,10,768)
        assert not torch.equal(selectors[task,0],selectors[task,1])
        for a,b in ((0,1),(0,2),(1,2)):
            x,y=selectors[task,a].double().flatten(),selectors[task,b].double().flatten()
            distances.append(dict(task=task,left=a,right=b,cosine=float(x.dot(y)/(x.norm()*y.norm())),
                relative_l2=float((x-y).norm()/x.norm())))
    audit.update(private_selectors_shape=[8,3,10,768],view_selectors_diverged=True)
    write(root/'PRIVATE_SELECTORS/training_verification.json',audit)
    write(root/'PRIVATE_SELECTORS/selector_view_distances.json',distances)


def verify_frozen(root,smoke):
    train=root/'FROZEN_PERSON_FACE/train';result=read(train/'training_complete.json')
    assert result['status']=='complete' and result['protected_base_preserved']
    assert result['config']['smoke']==smoke and len(result['checkpoints'])==(2 if smoke else 8)
    history=read(train/'training_history.json');audits=read(train/'audits.json')
    for task,rows in history.items():
        assert all(row['skipped_optimizer_steps']==0 for row in rows)
        assert len(rows)==(1 if smoke else 5)
        assert audits[task]['protected_parameters_unchanged'] and audits[task]['old_face_unchanged']
        assert audits[task]['teacher_current_predictions_exact']
    assert (result['completed_optimizer_updates']==4 if smoke else result['completed_optimizer_updates']<=2325)


def verify_reference(args,root,name,split):
    folder=root/name/('b0_'+split)
    prior=args.previous_study/('PERSON_REFERENCE' if name=='REFERENCE_PERSON' else 'FACE_JOINT')/('b0_'+split)
    result=read(folder/'result.json');old=read(prior/'result.json')
    for mode,rows in result['task_metrics'].items():
        for metric in rows:
            task=metric['task_id']
            for category in ('scores','view_scores'):
                with np.load(folder/mode/category/f'task{task}.npz') as actual,np.load(prior/mode/category/f'task{task}.npz') as ref:
                    assert actual.files==ref.files
                    assert all(np.array_equal(actual[k],ref[k]) for k in actual.files)
            assert all(metric[k]==old['task_metrics'][mode][task][k] for k in ('mAP','cF1','oF1'))


def sources(args):
    result={}
    for root in (args.source_run,args.person_run,args.dual_run):
        audit=read(root.parent/'training_verification.json')['checkpoint_sha256']
        for task in range(8):
            path=root/'compact_checkpoints'/f'task{task}.pth'
            value=_sha256(path);assert value==audit[str(task)];result[str(path)]=value
    return result


def summarize(args,root):
    records=[]
    for name in NAMES:
        for head in (('b0','native') if name=='PRIVATE_SELECTORS' else ('b0',)):
            by={}
            for split in ('val','test'):
                path=root/name/(head+'_'+split)/'result.json'
                if path.exists():by[split]=read(path)
            for condition in next(iter(by.values()))['summaries']:
                row=dict(method=name,head=head,condition=condition)
                for split,result in by.items():
                    assert result['status']=='complete'
                    if not args.smoke:assert result['full_eight_stage_average']
                    if name=='FROZEN_PERSON_FACE':assert result['frozen_face_prediction_invariants']
                    row.update({split+'_'+k:v for k,v in result['summaries'][condition].items()
                                if k in ('final_mAP','average_mAP','final_cF1','final_oF1')})
                records.append(row)
    keys=list(dict.fromkeys(k for row in records for k in row))
    with (root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(records)


def run(args):
    assert args.run_id and '/' not in args.run_id and args.run_id not in ('.','..')
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert read(args.previous_study/'completion.json')['status']=='complete'
    root=Path('output/selector_face_controls')/args.run_id;logs=Path('logs/selector_face_controls')/args.run_id
    root.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    if not args.smoke:
        gate=read(args.smoke_root/'completion.json')
        assert gate['status']=='complete' and gate['smoke'] and gate['head']==head
    before=sources(args)
    protocol=dict(status='running',head=head,smoke=args.smoke,seed=1,gpu=0,max_parallel=4,
        private_selector_epochs=30,face_only_epochs=5,batch_size=64,workers=2,threads=72,amp=True,tf32=True,
        private_selector_semantics='three_independent_banks_of_ten_per_task',
        remaining_shared_components=['Image-token Adapter','Prefix Prompts'],
        full_summary_gradient_detached=False,evaluation_scope='samples_intersect_seen_classes',
        test_used_for_parameter_search=True,source_checkpoint_sha256=before)
    if (root/'protocol.json').exists():assert read(root/'protocol.json')==protocol
    else:write(root/'protocol.json',protocol)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72',PYTHONUNBUFFERED='1')
    if args.smoke:
        command=[sys.executable,'-u','-m','multi_lane.track_a.selector_face_controls','audit',
            '--dual-run',str(args.dual_run),'--data-root',str(args.data_root),
            '--clip-checkpoint',str(args.clip_checkpoint),'--face-manifest-root',str(args.face_manifest_root),
            '--output',str(root/'initialization_audit.json')]
        with (logs/'initialization.log').open('w') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    else:assert read(args.smoke_root/'initialization_audit.json')['status']=='verified'
    private=training_arguments(args,root/'PRIVATE_SELECTORS','ATTENTION_BOUND')
    replace_flag(private,'selector_mode','view_specific')
    private+=['--cross-view-face-attention','--historical-b32-private-selectors']
    face=['train',*common(args),'--person-run',str(args.person_run),
        '--reference-root',str(args.previous_study/'PERSON_REFERENCE'),'--output',str(root/'FROZEN_PERSON_FACE/train')]
    if args.smoke:face.append('--smoke')
    pending=[dict(name='PRIVATE_SELECTORS/train',kind='private_train',arm='PRIVATE_SELECTORS',
        module='multi_lane.track_a.runner',arguments=private),
        dict(name='FROZEN_PERSON_FACE/train',kind='face_train',arm='FROZEN_PERSON_FACE',
        module='multi_lane.track_a.selector_face_controls',arguments=face)]
    pending+=eval_jobs(args,root,'REFERENCE_PERSON')+eval_jobs(args,root,'REFERENCE_DUAL')
    active=[];records=[];failed=False;last_launch=0
    try:
        while pending or active:
            for process,record in list(active):
                code=process.poll()
                if code is None:continue
                active.remove((process,record));record.update(exit=code,finished_unix=time.time());failed|=code!=0
                print('FINISH',record['name'],code,flush=True)
                if code==0:
                    try:
                        if record['kind']=='private_train':
                            verify_private(root,args.smoke,args.dual_run)
                            pending+=eval_jobs(args,root,record['arm'])
                        elif record['kind']=='face_train':
                            verify_frozen(root,args.smoke);pending+=eval_jobs(args,root,record['arm'])
                        elif record['arm'].startswith('REFERENCE_'):
                            verify_reference(args,root,record['arm'],record['name'].split('_')[-1])
                    except BaseException as error:
                        record['verification_error']=repr(error);failed=True
                        print('VERIFICATION_FAILED',record['name'],repr(error),flush=True)
            if failed and not active:raise RuntimeError('Worker or verification failed; checkpoints retained')
            if pending and not failed and len(active)<4 and time.time()-last_launch>=12:
                free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True).strip())
                stat=os.statvfs('/dev/shm');shm=stat.f_bavail*stat.f_frsize/1024**2
                if free>=6000 and shm>=1024:
                    job=pending.pop(0);log=logs/(job['name']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                    with log.open('a') as f:process=subprocess.Popen([sys.executable,'-u','-m',job['module'],*job['arguments']],
                        env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    record=dict(job,pid=process.pid,log=str(log),started_unix=time.time());active.append((process,record));records.append(record)
                    last_launch=time.time();print('START',record['name'],process.pid,flush=True)
            write(logs/'queue.json',dict(records=records,pending=pending,failed=failed))
            if pending or active:time.sleep(3)
        assert sources(args)==before
        summarize(args,root)
        write(root/'completion.json',dict(protocol,status='complete',source_checkpoints_unchanged=True,
            frozen_full_person_predictions_preserved=True,face_off_returns_strong_person=True))
        print('SELECTOR_FACE_CONTROLS_COMPLETE',root,flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    for key in ('source-run','person-run','dual-run','previous-study','data-root','clip-checkpoint','face-manifest-root'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-root',type=Path)
    run(p.parse_args())
