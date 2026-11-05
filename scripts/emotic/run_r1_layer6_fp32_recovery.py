"""Repair the layer6 Person path in FP32 without touching the live layer9 run."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from scripts.emotic.run_r1_frozen_late_stable import copy_prefix,summarize_family,free_memory,resume_smoke
from scripts.emotic.run_r1_frozen_cross_view import input_paths,folder,load_source,restore,check_training
from multi_lane.track_a.r1_crossvit_experiment import flags,read,write,model_from_config,single_thread_worker
from multi_lane.track_a.runner import set_seed,dataset_view,seen_indices,evaluate,git_metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('output/r1_layer6_fp32_recovery/seed1_20261105'))
    p.add_argument('--old',type=Path,default=Path('../multi-lane-main-r1-frozen-cross-view-20261105/output/r1_frozen_cross_view/seed1_20261105'))
    p.add_argument('--reference',type=Path,default=Path('../multi-lane-main-r1-crossvit-20261101/output/r1_crossvit/seed1_20261101_02'))
    a=p.parse_args();root=a.root;root.mkdir(parents=True,exist_ok=True);inputs=input_paths()
    logs=Path('logs/r1_layer6_fp32_recovery')/root.name;logs.mkdir(parents=True,exist_ok=True)
    git=git_metadata(Path(__file__).resolve().parents[2])
    if git['dirty']:raise ValueError('Clean source required')
    source=a.old/'image_cls/attention/person/train'
    checkpoint=source/'epoch_recovery/task3/epoch18.pth'
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    protocol=dict(git=git,source=str(checkpoint),source_sha=sha,
        precision='FP32; AMP off; TF32 off; finite-gradient guard before optimizer',
        tasks=8,seed=1,epochs=30,batch=64,main_lr=.0125,adapter_and_CA_lr=.0004,
        earlier_epochs='original valid model, optimizer, scheduler and RNG retained',
        first_remaining_epoch=19,layer=5,late_run_untouched=True,
        uniform_fp32_person_test_and_validation=True,original_full_and_face_predictions_reused=True)
    write(root/'protocol.json',protocol)
    while free_memory()<14500:
        write(root/'admission.json',dict(status='waiting_for_smoke_memory',free_mib=free_memory()))
        time.sleep(5)
    resume_smoke(root/'smoke',source,force_fp32=True)
    peak=read(root/'smoke/smoke_result.json')['peak_reserved_mib']
    needed=int(peak*1.12+1024)
    # Release the smoke allocator before starting an independent training process.
    torch.cuda.empty_cache()
    while free_memory()<needed:
        write(root/'admission.json',dict(status='queued',required_mib=needed,free_mib=free_memory()))
        time.sleep(5)
    base=root/'model';train=folder(base,'image_cls','person');copy_prefix(source,train)
    command=[sys.executable,'-u','-m','multi_lane.track_a.runner',
        *flags(SimpleNamespace(output=base/'image_cls',**vars(inputs)),'attention','person',False),
        '--r1-frozen-mode','image_cls','--r1-exchange-layer','5','--r1-fp32-recovery',
        '--r1-recovery-source',str(source),'--r1-fusion-readout','logit','--no-amp','--no-tf32']
    write(root/'admission.json',dict(status='training',required_mib=needed))
    with (logs/'training.log').open('a') as f:
        result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        write(root/'failure.json',dict(status='failed',exit_code=result.returncode));raise RuntimeError('FP32 recovery failed')
    counts=check_training(train,a.reference,'person',8,False)
    write(root/'training_checks.json',counts)
    set_seed(1,False);config=read(train/'config.json')
    model=model_from_config(config,inputs.clip_checkpoint);torch.set_num_threads(2)
    for split in ('val','test'):
        source_data=load_source(train,inputs,split);rows=[]
        for t in range(8):
            restore(model,train,t);model.requires_grad_(False)
            data=dataset_view(source_data,seen_indices(t),include_sample_id=True)
            loader=DataLoader(data,batch_size=64,num_workers=2,prefetch_factor=1,
                worker_init_fn=single_thread_worker,generator=torch.Generator().manual_seed(77191+t))
            score_dir=Path('val/scores' if split=='val' else 'train/test_scores')
            row=evaluate(model,loader,torch.device('cuda'),t,.5,False,
                score_output_path=root/'fp32_readout/person'/score_dir/f'task{t}.npz')
            rows.append(asdict(row));print('FP32_EVALUATION',split,t,row.mAP,flush=True)
        write(root/f'{split}_metrics.json',dict(status='complete',precision='FP32',metrics=rows))
    paths={(m,v):a.old/m/'attention'/v for m in ('image_cls','patch') for v in ('person','face')}
    paths[('image_cls','person')]=root/'fp32_readout/person'
    summarize_family(root,a.reference,inputs.face_manifest_root,paths,root/'completed_layer6_results_fp32.json')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=sha:raise AssertionError('Source changed')
    write(root/'completion.json',dict(status='complete',source_unchanged=True,protocol=protocol))


if __name__=='__main__':main()
