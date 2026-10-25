"""Validation-selected read-only Attention/head compatibility follow-up."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import asdict
import numpy as np
import torch

from .historical_b32_backbone_attribution import build, load_host, restore_components
from .run_historical_b32_test_attribution import read, write, checkpoint_hashes, MODEL_HEAD, VIEWS
from .export_compact_test_scores import _torch_load, _sha256
from .evaluate_joint26_checkpoint import make_source
from .evaluate_moe_task_ffn import evaluation_loader
from .runner import dataset_view, seen_indices, set_seed, tensor_state_digest, git_metadata
from .strong_full_evaluation import evaluate_fused_and_views

ARMS=('ATTENTION','ATTENTION_BOUND')


def worker(args):
    if args.output.exists():raise FileExistsError(args.output)
    config=read(args.source_root/args.arm/'train/config.json')
    assert args.arm in ARMS and config['git']['commit']==MODEL_HEAD
    set_seed(1,tf32=True);args.output.mkdir(parents=True)
    model=build(config,args.clip_checkpoint)
    load_host(model,args.source_root/args.arm/'train',config,7)
    model.cross_view_runtime_enabled=True
    baseline=_torch_load(args.source_root/'BASE/train/compact_checkpoints/task7.pth')['model']
    files=[args.source_root/arm/'train/compact_checkpoints/task7.pth' for arm in ('BASE',args.arm)]
    hashes={str(p):_sha256(p) for p in files}
    visual=tensor_state_digest(model.visual_encoder.state_dict())
    source=make_source(config,args.data_root,args.face_manifest_root,args.split)
    loader=evaluation_loader(dataset_view(source,seen_indices(7),include_sample_id=True),64,config,workers=2)
    records=[]
    for condition in ('none','H','ASPH'):
        directory=args.output/condition;directory.mkdir()
        with restore_components(model,baseline,condition):
            row,views=evaluate_fused_and_views(model,loader,torch.device('cuda'),7,config['threshold'],True,
                directory/'scores/task7.npz',directory/'view_scores/task7.npz')
        if condition=='none':
            reference=args.source_root/args.arm/('historical_'+args.split)
            with np.load(directory/'scores/task7.npz') as a,np.load(reference/'scores/task7.npz') as b:
                assert all(np.array_equal(a[k],b[k]) for k in ('sample_ids','targets','class_indices','logits','probabilities'))
            with np.load(directory/'view_scores/task7.npz') as a,np.load(reference/'view_scores/task7.npz') as b:
                assert all(np.array_equal(a[name+suffix],b[name+suffix]) for name in VIEWS for suffix in ('_logits','_probabilities'))
        record=dict(condition=condition,metrics=asdict(row),view_diagnostics=views)
        write(directory/'result.json',record);records.append(record)
        print('HEAD_TRANSFER',args.arm,args.split,condition,row.mAP,flush=True)
    assert all(_sha256(Path(p))==value for p,value in hashes.items())
    assert tensor_state_digest(model.visual_encoder.state_dict())==visual
    write(args.output/'completion.json',dict(status='complete',arm=args.arm,split=args.split,task=7,
        conditions=records,original_on_replay_exact=True,source_checkpoints_unchanged=True,
        frozen_visual_unchanged=True,training=False,git=git_metadata(Path(__file__).resolve().parents[2])))


def run(args):
    if not args.run_id or '/' in args.run_id or args.run_id in {'.','..'}:raise ValueError('Invalid ID')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Clean checkout required')
    assert read(args.diagnostic_root/'completion.json')['status']=='complete'
    for arm in ARMS:
        head=read(args.diagnostic_root/arm/'val/evaluate/H/result.json')['metrics']['mAP']
        original=read(args.diagnostic_root/arm/'val/anchors/none/result.json')['metrics']['mAP']
        assert head>original, 'Follow-up eligibility is determined by validation head recovery'
    root=Path('output/historical_b32_head_transfer')/args.run_id
    logs=Path('logs/historical_b32_head_transfer')/args.run_id
    if root.exists() or logs.exists():raise FileExistsError(args.run_id)
    root.mkdir(parents=True);logs.mkdir(parents=True)
    before=checkpoint_hashes(args.source_root)
    protocol=dict(status='running',seed=1,gpu=0,training=False,batch_size=64,workers=2,cpu_threads=72,
        task=7,arms=list(ARMS),conditions=['none','H','ASPH'],new_attention_enabled=True,
        eligibility='validation_head_restoration_improves_both_attention_arms',test_used_for_selection=True,
        source_model_head=MODEL_HEAD,git=git_metadata(Path.cwd()),checkpoint_sha256_before=before)
    write(root/'protocol.json',protocol)
    pending=[(arm,split) for arm in ARMS for split in ('val','test')]
    active=[];records=[];last=0;failed=False
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72',PYTHONUNBUFFERED='1')
    try:
        while pending or active:
            for process,record in list(active):
                code=process.poll()
                if code is None:continue
                active.remove((process,record));record.update(exit=code,finished_unix=time.time());failed|=code!=0
                if code==0:
                    complete=read(root/record['arm']/record['split']/'completion.json')
                    assert complete['original_on_replay_exact'] and complete['source_checkpoints_unchanged']
                print('FINISH',record['arm'],record['split'],code,flush=True)
            if failed and not active:raise RuntimeError('A head transfer job failed')
            if pending and not failed and len(active)<4 and time.time()-last>=12:
                free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                stat=os.statvfs('/dev/shm');shm=stat.f_bavail*stat.f_frsize/1024**2
                if free>=6000 and shm>=1024:
                    arm,split=pending.pop(0)
                    flags=['--arm',arm,'--split',split,'--output',root/arm/split]
                    for name in ('source_root','diagnostic_root','data_root','clip_checkpoint','face_manifest_root'):
                        flags+=['--'+name.replace('_','-'),getattr(args,name)]
                    command=[sys.executable,'-u','-m','multi_lane.track_a.historical_b32_head_transfer',*map(str,flags)]
                    log=logs/(arm+'_'+split+'.log')
                    with log.open('x') as f:process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    record=dict(arm=arm,split=split,pid=process.pid,command=command,started_unix=time.time())
                    active.append((process,record));records.append(record);last=time.time();print('START',arm,split,process.pid,flush=True)
            write(logs/'queue.json',dict(failed=failed,records=records,pending=pending))
            if pending or active:time.sleep(3)
        assert checkpoint_hashes(args.source_root)==before
        rows=[]
        for arm in ARMS:
            for split in ('val','test'):
                result=read(root/arm/split/'completion.json')
                base=read(args.diagnostic_root/arm/split/'anchors/ASPH/result.json')['metrics']['mAP']
                off={c:read(args.diagnostic_root/arm/split/folder/c/'result.json')['metrics']['mAP']
                    for c,folder in (('none','anchors'),('H','evaluate'),('ASPH','anchors'))}
                for value in result['conditions']:
                    c=value['condition'];m=value['metrics']['mAP']
                    rows.append(dict(arm=arm,split=split,restored=c,new_attention_on=m,new_attention_off=off[c],
                        attention_contribution=m-off[c],delta_vs_base=m-base,base_mAP=base))
        with (root/'summary.csv').open('x') as f:
            writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
        write(root/'completion.json',dict(protocol,status='complete',jobs=4,evaluations=12,source_checkpoints_unchanged=True))
        print('HISTORICAL_B32_HEAD_TRANSFER_COMPLETE',root,flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--split',choices=('val','test'))
    p.add_argument('--output',type=Path);p.add_argument('--run-id')
    for name in ('source-root','diagnostic-root','data-root','clip-checkpoint','face-manifest-root'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.arm:
        if args.output is None or args.split is None:p.error('Worker requires output and split')
        worker(args)
    else:run(args)
