"""Schedule checkpoint drift, TRAIN factorial probes, and full task7 swaps."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

from .historical_b32_test_attribution import ARMS
from .run_historical_b32_test_attribution import read, write, checkpoint_hashes, MODEL_HEAD


def arguments(args, root, arm, mode, split=None):
    output=root/'drift' if arm=='drift' else root/arm/(mode if split is None else split+'/'+mode)
    flags=['--source-root',args.source_root,'--bypass-root',args.bypass_root,
        '--data-root',args.data_root,'--clip-checkpoint',args.clip_checkpoint,
        '--face-manifest-root',args.face_manifest_root,'--mode',mode,
        '--output',output]
    if arm!='drift':flags+=['--arm',arm]
    if split is not None:flags+=['--split',split]
    return list(map(str,flags))


def summarize(args, root):
    rows=[]
    for arm in ARMS:
        for split in ('val','test'):
            anchor=read(root/arm/split/'anchors/evaluation.json')
            reference={v['condition']:v for v in anchor['conditions']}
            assert anchor['anchors_exact']
            base=reference['ASPH']['metrics']['mAP'];none=reference['none']['metrics']['mAP']
            values=anchor['conditions']+read(root/arm/split/'evaluate/evaluation.json')['conditions']
            for value in values:
                row=dict(arm=arm,split=split,task=7,restored=value['condition'],
                    mAP=value['metrics']['mAP'],cF1=value['metrics']['cF1'],oF1=value['metrics']['oF1'],
                    change_vs_none=value['metrics']['mAP']-none,change_vs_base=value['metrics']['mAP']-base)
                row.update({name+'_mAP':value['view_diagnostics']['metrics'][name]['mAP'] for name in ('full','person','face','fused')})
                rows.append(row)
    with (root/'summary.csv').open('x') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert len(rows)==56


def run(args):
    if not args.run_id or '/' in args.run_id or args.run_id in {'.','..'}:raise ValueError('Invalid run ID')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Clean checkout required')
    source=read(args.source_root/'completion.json');assert source['status']=='complete' and source['head']==MODEL_HEAD
    bypass=read(args.bypass_root/'completion.json');assert bypass['status']=='complete' and bypass['source_checkpoints_unchanged']
    root=Path('output/historical_b32_backbone_attribution')/args.run_id
    logs=Path('logs/historical_b32_backbone_attribution')/args.run_id
    if root.exists() or logs.exists():raise FileExistsError(args.run_id)
    root.mkdir(parents=True);logs.mkdir(parents=True)
    before=checkpoint_hashes(args.source_root)
    protocol=dict(status='running',seed=1,gpu=0,max_parallel=4,batch_size=64,eval_workers=2,
        cpu_threads=72,training=False,model_head=MODEL_HEAD,
        evaluator_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_checkpoint_sha256=before,full_evaluation_tasks=[7],probe_tasks=[0,2,7],
        probe_batches=2,probe_batch_cap=64,probe_precision='deterministic_fp32',
        parameter_groups=['Adapter','Selector','Prompt','classifier'],probe_factorial_conditions=16,
        full_conditions=['none','A','S','P','H','ASP','ASPH'],new_modules_disabled=True,
        hybrids_are_conditional_counterfactuals=True,test_used_for_selection=True,
        evaluation_scope='samples_intersect_seen_classes',full_eight_task_average=False,started_unix=time.time())
    write(root/'protocol.json',protocol)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72',PYTHONUNBUFFERED='1')
    module='multi_lane.track_a.historical_b32_backbone_attribution'
    drift_command=[sys.executable,'-u','-m',module,*arguments(args,root,'drift','drift')]
    with (logs/'drift.log').open('x') as f:
        try:
            subprocess.run(drift_command,env=dict(env,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=f,stderr=subprocess.STDOUT,check=True)
        except BaseException as error:
            write(root/'failure.json',dict(stage='drift',error=repr(error)));raise
    pending=[dict(arm=arm,mode='probe',split=None) for arm in ARMS]
    active=[];records=[];failed=False;last=0
    try:
        while pending or active:
            for process,record in list(active):
                code=process.poll()
                if code is None:continue
                active.remove((process,record));record.update(exit=code,finished_unix=time.time());failed|=code!=0
                print('FINISH',record['name'],code,flush=True)
                if code==0:
                    output=root/record['arm']/(record['mode'] if record['split'] is None else record['split']+'/'+record['mode'])
                    try:
                        audit=read(output/'completion.json')
                        assert audit['status']=='complete' and audit['source_checkpoints_unchanged'] and audit['frozen_visual_unchanged']
                        if record['mode']=='probe':
                            assert len(read(output/'probe.json')['records'])==96
                            pending += [dict(arm=record['arm'],mode='anchors',split=split) for split in ('val','test')]
                        elif record['mode']=='anchors':
                            assert read(output/'evaluation.json')['anchors_exact']
                            pending.append(dict(arm=record['arm'],mode='evaluate',split=record['split']))
                    except Exception as error:
                        record['verification_error']=repr(error);failed=True
                        print('VERIFICATION_FAILED',record['name'],repr(error),flush=True)
            if failed and not active:raise RuntimeError('A diagnostic failed; remaining jobs were not dispatched')
            if pending and not failed and len(active)<4 and time.time()-last>=12:
                free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                stat=os.statvfs('/dev/shm');shm=stat.f_bavail*stat.f_frsize/1024**2
                if free>=6000 and shm>=1024:
                    job=pending.pop(0);name=job['arm']+'/'+(job['split']+'/' if job['split'] else '')+job['mode']
                    log=logs/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                    command=[sys.executable,'-u','-m',module,*arguments(args,root,job['arm'],job['mode'],job['split'])]
                    with log.open('x') as f:
                        process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    record=dict(job,name=name,pid=process.pid,gpu=0,command=command,log=str(log),started_unix=time.time())
                    active.append((process,record));records.append(record);last=time.time()
                    print('START',name,process.pid,flush=True)
            write(logs/'queue.json',dict(failed=failed,records=records,pending=pending))
            if pending or active:time.sleep(3)
        assert len(records)==20
        fingerprints=[]
        for arm in ARMS:
            probe=read(root/arm/'probe/probe.json')
            fingerprints.append([(r['task'],r['batch'],r['sample_ids'],r['pixel_sha256'])
                for r in probe['records'] if r['condition']=='none'])
        assert all(value==fingerprints[0] for value in fingerprints)
        assert checkpoint_hashes(args.source_root)==before
        summarize(args,root)
        write(root/'completion.json',dict(protocol,status='complete',jobs=len(records),
            source_checkpoints_unchanged=True,full_evaluations=56,finished_unix=time.time()))
        print('HISTORICAL_B32_BACKBONE_ATTRIBUTION_COMPLETE',root,flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    for name in ('source-root','bypass-root','data-root','clip-checkpoint','face-manifest-root'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
