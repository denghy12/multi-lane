"""Preflight, isolated resource profiles, then concurrent full Val/Test on GPU0."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

from .export_compact_test_scores import _sha256
from .protected_view_forward import ARMS
from .run_historical_b32_test_attribution import read, write


def sources(args):
    result={}
    for root in (args.source_run,args.person_run,args.dual_run):
        assert read(root/'seed_summary.json')['status']=='complete'
        verified=read(root.parent/'training_verification.json')['checkpoint_sha256']
        for task in range(8):
            p=root/'compact_checkpoints'/f'task{task}.pth';sha=_sha256(p)
            assert sha==verified[str(task)];result[str(p)]=sha
    return result


def common(args):
    return [item for name in ('source-run','person-run','dual-run','reference-root','data-root','clip-checkpoint','face-manifest-root')
            for item in ('--'+name,str(getattr(args,name.replace('-','_'))))]


def run(args):
    assert args.run_id and '/' not in args.run_id and args.run_id not in ('.','..')
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    root=Path('output/protected_view_forward')/args.run_id;logs=Path('logs/protected_view_forward')/args.run_id
    root.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    source=sources(args)
    protocol=dict(status='running',head=head,seed=1,gpu=0,training_updates=0,tasks=list(range(8)),
        arms={k:list(v) for k,v in ARMS.items()},batch_size=64,workers=2,threads=72,amp=True,tf32=True,
        evaluation_scope='samples_intersect_seen_classes',test_used_for_parameter_search=True,
        source_checkpoint_sha256=source,visual_weights_shared=True,frozen_tokens_cached=False,
        classifier_head='corresponding_stage_B0_shared_post_feature_fusion',source_full_summaries_preserved=True,
        queue_parallelism=2,resource_profiles_serial=True)
    if (root/'protocol.json').exists():assert read(root/'protocol.json')==protocol
    else:write(root/'protocol.json',protocol)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72',PYTHONUNBUFFERED='1')
    records=[]

    def job(action,name,flags):
        p=root/name/'result.json'
        if p.exists():
            value=read(p);assert value['status']=='complete' and value['git']['commit']==head
            print('REUSE_COMPLETED',name,flush=True);return
        path=logs/(name+'.log');path.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u','-m','multi_lane.track_a.protected_view_forward',action,
                 *common(args),*flags,'--output',str(root/name)]
        started=time.time()
        with path.open('a') as f:
            process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            record=dict(name=name,pid=process.pid,arguments=command,started_unix=started,log=str(path))
            records.append(record);write(logs/'queue.json',dict(records=records,status='running'))
            code=process.wait();record.update(exit=code,finished_unix=time.time())
            write(logs/'queue.json',dict(records=records,status='running' if code==0 else 'failed'))
            if code:raise RuntimeError(f'{name} failed: {code}, see {path}')
        value=read(p);assert value['status']=='complete';print('FINISH',name,flush=True)

    try:
        job('preflight','preflight',['--split','val'])
        pre=read(root/'preflight/result.json')
        assert pre['source_predictions_exact'] and pre['R0_exact'] and pre['R1_invalid_exact']
        if args.preflight_only:
            assert sources(args)==source
            write(root/'completion.json',dict(protocol,status='complete',preflight_only=True,source_checkpoints_unchanged=True));return
        # Timing must be isolated; do not benchmark under the Val/Test queue.
        own_pid=os.getpid()
        active=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines()
        assert not [x for x in active if x.strip().isdigit() and int(x)!=own_pid], 'GPU0 is busy: cannot produce isolated timing'
        for arm in ARMS:job('benchmark','benchmark/'+arm,['--arm',arm])
        active=[]
        for split in ('val','test'):
            name='evaluation/'+split;path=logs/(name+'.log');path.parent.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-u','-m','multi_lane.track_a.protected_view_forward','eval',
                     *common(args),'--split',split,'--output',str(root/name)]
            with path.open('a') as f:process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            record=dict(name=name,pid=process.pid,arguments=command,started_unix=time.time(),log=str(path))
            records.append(record);active.append((process,record));print('START',name,process.pid,flush=True)
        while active:
            for process,record in list(active):
                code=process.poll()
                if code is None:continue
                record.update(exit=code,finished_unix=time.time());active.remove((process,record))
                print('FINISH',record['name'],code,flush=True)
            write(logs/'queue.json',dict(records=records,status='running'))
            if active:time.sleep(3)
        assert all(r['exit']==0 for r in records)
        assert sources(args)==source
        results={split:read(root/'evaluation'/split/'result.json') for split in ('val','test')}
        rows=[]
        for arm in ARMS:
            row=dict(arm=arm,Full=ARMS[arm][0],Person=ARMS[arm][1],Face=ARMS[arm][2])
            for split,value in results.items():
                assert value['tasks']==list(range(8)) and value['R0_exact'] and value['R1_invalid_exact']
                row.update({split+'_'+k:v for k,v in value['summaries'][arm].items()
                            if k in ('final_mAP','average_mAP','final_cF1','final_oF1')})
            rows.append(row)
        with (root/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
        write(root/'resource_summary.json',{arm:read(root/'benchmark'/arm/'result.json') for arm in ARMS})
        write(logs/'queue.json',dict(records=records,status='complete'))
        write(root/'completion.json',dict(protocol,status='complete',preflight_only=False,
            source_checkpoints_unchanged=True,all_source_predictions_exact=True,R0_exact=True,R1_invalid_exact=True))
        print('PROTECTED_VIEW_FORWARD_COMPLETE',root,flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    for name in ('source-run','person-run','dual-run','reference-root','data-root','clip-checkpoint','face-manifest-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight-only',action='store_true')
    run(p.parse_args())
