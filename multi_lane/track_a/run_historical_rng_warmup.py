"""Repair/replay the baseline and diagnose archived warmup/final pairs on GPU0."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time

from .export_compact_test_scores import _sha256
from .head_anchored_eight import runtime
from .run_historical_b32_test_attribution import read,write
from .run_protected_view_forward import common
from .self_trained_warmup_attribution import summarize_attribution


def run(args):
    git=runtime()
    root=Path('output/historical_rng_replay')/args.run_id
    attribution=Path('output/self_trained_warmup_attribution')/args.run_id
    logs=Path('logs/historical_rng_replay')/args.run_id
    logs.mkdir(parents=True,exist_ok=True)
    original_files=[*args.student_root.glob('*/train/compact_checkpoints/task*.pth'),
                    *args.student_root.glob('*/train/warmup_checkpoints/task*.pth'),
                    *args.source_run.glob('recovery_checkpoints/task*_before_reporting.pth')]
    hashes={str(p):_sha256(p) for p in original_files}
    assert len(hashes)==72
    write(root/'protocol.json',dict(git=git,seed=1,gpu=0,
        historical_reporting_rng='two_CPU_int64_draws_after_each_task',
        baseline_fresh_initialization=True,baseline_epochs=30,baseline_tasks=list(range(8)),
        baseline_main_lr=.0125,adapter_lr=.0004,batch_size=64,threads=72,
        optimizer='Adam_reset_each_task',scheduler='cosine',weight_decay=0,amp=True,tf32=True,
        original_checkpoint_sha256=hashes,extra_student_training=False,
        attribution_tasks=[0,2,7],source_student_root=str(args.student_root),
        existing_evaluation_root=str(args.existing_eval_root),evaluation_workers=2,
        evaluation_scope='samples_intersect_seen_classes',maximum_concurrent_jobs=3))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72')
    module=[sys.executable,'-u','-m','multi_lane.track_a.self_trained_view_transfer']
    attr=[sys.executable,'-u','-m','multi_lane.track_a.self_trained_warmup_attribution',
        '--source-root',str(args.student_root),'--existing-eval-root',str(args.existing_eval_root)]
    def blocking(name,command):
        log=logs/(name+'.log')
        with log.open('a') as f:result=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        assert result.returncode==0,(name,str(log),result.returncode)
    smoke=root/'smoke_train'
    if not (smoke/'training_complete.json').exists():
        blocking('baseline_smoke',module+['train',*common(args),'--arm','SELF_BASE','--smoke',
            '--historical-reporting-rng','--output',str(smoke)])
    smoke_cfg=read(smoke/'config.json')
    assert smoke_cfg['smoke'] and smoke_cfg['historical_reporting_rng']
    smoke_audit=read(smoke/'audits.json')
    assert all(smoke_audit[str(t)]['historical_reporting_rng']['int64_draws']==2 for t in (0,1))
    if not (root/'smoke_val/result.json').exists():
        blocking('baseline_smoke_eval',module+['eval','--training-root',str(smoke),
            '--data-root',str(args.data_root),'--clip-checkpoint',str(args.clip_checkpoint),
            '--face-manifest-root',str(args.face_manifest_root),'--split','val','--evaluation-workers','2',
            '--output',str(root/'smoke_val')])
    if not (attribution/'smoke_val/result.json').exists():
        blocking('attribution_smoke',attr+['--split','val','--smoke','--output',str(attribution/'smoke_val')])
    pending=[('baseline_train',module+['train',*common(args),'--arm','SELF_BASE',
        '--historical-reporting-rng','--historical-reference-root',str(args.source_run),
        '--output',str(root/'BASE/train')],root/'BASE/train/training_complete.json')]
    for split in ('val','test'):
        pending.append(('attribution_'+split,attr+['--split',split,'--output',str(attribution/split)],
                       attribution/split/'result.json'))
    records=[];active=[];last=0.;added_evaluations=False;failed=[]
    try:
        while pending or active:
            for process,row in list(active):
                code=process.poll()
                if code is None:continue
                row.update(exit=code,finished_unix=time.time());active.remove((process,row))
                if code:failed.append(row);continue
                assert read(Path(row['completion']))['status']=='complete'
                if row['name']=='baseline_train':
                    audits=read(root/'BASE/train/audits.json')
                    assert all(audits[str(t)]['historical_reference_verification']['status']=='verified' for t in range(8))
                    for split in ('val','test'):
                        pending.append(('baseline_'+split,module+['eval','--training-root',str(root/'BASE/train'),
                            '--data-root',str(args.data_root),'--clip-checkpoint',str(args.clip_checkpoint),
                            '--face-manifest-root',str(args.face_manifest_root),'--split',split,
                            '--evaluation-workers','2','--output',str(root/'BASE'/split)],
                            root/'BASE'/split/'result.json'))
                    added_evaluations=True
            if (attribution/'val/result.json').exists() and (attribution/'test/result.json').exists() \
                    and not (attribution/'completion.json').exists():summarize_attribution(attribution)
            if failed:
                pending=[]
            elif pending and len(active)<3 and time.time()-last>=8:
                free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True).strip())
                shm=os.statvfs('/dev/shm')
                if free<6000 or shm.f_bavail*shm.f_frsize<1024**3:
                    time.sleep(3);continue
                name,command,complete=pending.pop(0)
                with (logs/(name+'.log')).open('a') as f:
                    process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                row=dict(name=name,pid=process.pid,command=command,completion=str(complete),
                    log=str(logs/(name+'.log')),started_unix=time.time())
                records.append(row);active.append((process,row));last=time.time()
                print('START',name,process.pid,flush=True)
            write(logs/'queue.json',dict(status='running',records=records,pending=[x[0] for x in pending],
                  active=[r['name'] for _,r in active],failed=[r['name'] for r in failed]))
            if active or pending:time.sleep(3)
        assert not failed,failed
        assert added_evaluations
        for split in ('val','test'):
            result=read(root/'BASE'/split/'result.json')
            historical=read(args.source_run.parent/('historical_'+split)/'result.json')
            for key in ('final_mAP','average_mAP','final_cF1','final_oF1'):
                assert abs(result['summary'][key]-historical['summary'][key])<1e-8,(split,key)
        assert hashes=={p:_sha256(Path(p)) for p in hashes}
        write(root/'completion.json',dict(status='complete',git=git,
            all_eight_historical_model_and_rng_boundaries_exact=True,
            both_splits_match_historical_metrics=True,original_source_checkpoints_unchanged=True,
            attribution_complete=read(attribution/'completion.json')['status']=='complete'))
        write(logs/'queue.json',dict(status='complete',records=records))
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error),records=records));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    for name in ('source-run','person-run','dual-run','reference-root','data-root',
                 'clip-checkpoint','face-manifest-root','student-root','existing-eval-root'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
