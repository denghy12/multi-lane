"""Complete the valid layer6 recovery and run layer9's six receiver settings."""
import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from scripts.emotic.run_r1_frozen_cross_view import input_paths, folder, preflight, check_training
from multi_lane.track_a.r1_crossvit_experiment import flags, read, write
from multi_lane.track_a.evaluation_scores import load_evaluation_scores, align_evaluation_scores
from multi_lane.track_a.fuse_face_endpoint_validation import load_face_reliability
from multi_lane.track_a.fuse_fixed_three_view_test import LOCKED_RELIABLE_WEIGHTS, LOCKED_FALLBACK_WEIGHTS
from multi_lane.track_a.runner import compute_metrics, summarize_tasks, git_metadata

MODES = ('image_cls', 'patch')
VIEWS = ('person', 'face')


def free_memory():
    return int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())


def run_graph(jobs, root, logs, concurrency=2, available=free_memory, poll=5, on_finish=None):
    """A failed path blocks only its dependants, never other evaluations."""
    pending=list(jobs);running=[];done={}
    while pending or running:
        for job,process,handle in running[:]:
            code=process.poll()
            if code is None:continue
            handle.close();running.remove((job,process,handle))
            done[job['name']]='complete' if code==0 else 'failed'
            if on_finish is not None:on_finish(job,done[job['name']])
        for job in pending[:]:
            if any(done.get(d) in ('failed','blocked') for d in job.get('depends',())):
                done[job['name']]='blocked';pending.remove(job)
        ready=[j for j in pending if all(done.get(d)=='complete' for d in j.get('depends',()))]
        if ready and len(running)<concurrency:
            job=ready[0];need=job.get('memory_mib',8500)
            # Admission includes start-up reservations before CUDA reports usage.
            usage={}
            if running:
                raw=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid,used_memory',
                    '--format=csv,noheader,nounits'],text=True)
                for line in raw.splitlines():
                    a=[x.strip() for x in line.split(',')]
                    if len(a)==2 and all(x.isdigit() for x in a):usage[int(a[0])]=int(a[1])
            reserve=sum(max(0,j.get('memory_mib',8500)-usage.get(p.pid,0)) for j,p,_ in running)
            if available()-reserve>=need+1024:
                pending.remove(job);handle=(logs/(job['name']+'.log')).open('a')
                process=subprocess.Popen(job['command'],stdout=handle,stderr=subprocess.STDOUT)
                running.append((job,process,handle));print('START',job['name'],process.pid,flush=True)
        write(root/'queue.json',dict(status='running' if pending or running else 'finished',
            pending=[j['name'] for j in pending],running=[dict(name=j['name'],pid=p.pid) for j,p,_ in running],finished=done))
        if pending and not running and not ready:
            raise ValueError('Unresolved dependency graph')
        if pending or running:time.sleep(poll)
    return done


def copy_prefix(old, new):
    for sub in ('compact_checkpoints','test_scores'):
        for t in range(3):
            suffix='.pth' if sub=='compact_checkpoints' else '.npz'
            source=old/sub/f'task{t}{suffix}';target=new/sub/source.name
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=source.read_bytes():
                raise ValueError(f'Existing prefix differs: {target}')
            if not target.exists():shutil.copy2(source,target)


def summarize_family(root, reference, manifest, paths, destination):
    """Only report groups whose two branch score streams really exist."""
    output={}
    for split in ('val','test'):
        relmap=load_face_reliability(manifest,split);output[split]={}
        for mode in MODES:
            for direction in ('face_only','person_only','person_face'):
                metrics={p:[] for p in ('all','reliable','unreliable')};missing=[]
                for t in range(8):
                    suffix=Path('val/scores' if split=='val' else 'train/test_scores')/f'task{t}.npz'
                    full=load_evaluation_scores(reference/'native/full'/suffix,64)
                    ids=full.sample_ids;y=full.targets
                    pp=(paths[(mode,'person')] if direction!='face_only' else reference/'native/person')/suffix
                    rr=(paths[(mode,'face')] if direction!='person_only' else reference/'native/face')/suffix
                    if not pp.exists() or not rr.exists():missing.append(t);continue
                    _,_,p,_,_,_=align_evaluation_scores(full,load_evaluation_scores(pp,64))
                    _,_,r,_,_,_=align_evaluation_scores(full,load_evaluation_scores(rr,64))
                    reliable=np.array([relmap[i] for i in ids],dtype=bool)
                    w=np.where(reliable[:,None],LOCKED_RELIABLE_WEIGHTS,LOCKED_FALLBACK_WEIGHTS)
                    z=w[:,0,None]*full.logits+w[:,1,None]*p+w[:,2,None]*r
                    for pop,mask in [('all',np.ones(len(ids),bool)),('reliable',reliable),('unreliable',~reliable)]:
                        metrics[pop].append(compute_metrics(t,torch.from_numpy(z[mask]).sigmoid(),torch.from_numpy(y[mask]),.5))
                output[split][mode+'_'+direction]=dict(status='complete' if not missing else 'incomplete',
                    missing_tasks=missing,populations={p:dict(summary=summarize_tasks(v) if len(v)==8 else None,
                        metrics=[asdict(x) for x in v]) for p,v in metrics.items()})
    write(destination,dict(results=output,independently_predeclared_paths=True))


def resume_smoke(root, source):
    from torch.utils.data import DataLoader,Subset
    from multi_lane.continual_datasets.continual_datasets import EMOTIC
    from multi_lane.track_a.r1_crossvit_experiment import model_from_config
    from multi_lane.track_a.r1_crossvit import attach_evidence
    from multi_lane.track_a.runner import set_seed,build_transforms,fit_calibration_indices,LabelView,task_indices,train_task
    set_seed(1,True);inputs=input_paths();c=read(source/'config.json')
    model=model_from_config(c,inputs.clip_checkpoint)
    model.r1_fixed_loss_scale=4096.;model.recovery_source=source
    model.r1_root=root;model.r1_config=dict(c,r1_fixed_loss_scale=4096.,
        r1_loss_scale_growth_interval=1000000000,r1_recovery_source=str(source))
    model.restore_completed_task(2);model.activate_task(3)
    loaded=model.r1_resume
    if loaded['completed_task_updates']!=2808:raise ValueError('Wrong recovery boundary')
    transform,_=build_transforms(c['input_normalization'],c['train_crop_scale'],input_mode=c['input_mode'],
        person_transform_mode=c['person_transform_mode'],person_color_jitter_strength=c['person_color_jitter_strength'],
        person_color_jitter_probability=c['person_color_jitter_probability'])
    data=EMOTIC(str(inputs.data_root.parent),train=True,transform=transform,input_mode=c['input_mode'],person_crop_margin=c['person_crop_margin'])
    data=attach_evidence(data,'person',inputs.data_root.parent,inputs.face_manifest_root)
    indices,_=fit_calibration_indices(data,task_indices(3),0)
    view=LabelView(data,indices,task_indices(3));loader=DataLoader(view,batch_size=64,shuffle=True,num_workers=2)
    from multi_lane.track_a.r1_crossvit import restore_rng
    restore_rng(torch.load(source/'epoch_recovery/task3/epoch18.pth',map_location='cpu',weights_only=False)['rng'])
    class TwoBatches:
        def __len__(self):return len(loader)
        def __iter__(self):
            for i,b in enumerate(loader):
                if i>=2:break
                yield b
    validation=DataLoader(Subset(view,range(64)),batch_size=64,num_workers=0)
    history=train_task(model,TwoBatches(),validation,torch.device('cuda'),3,19,.0125,0.,1.,True,
        adapter_learning_rate=.0004,loss_routing='adapter_asl')
    last=history[-1]
    if last['optimizer_steps']!=2 or last['skipped_optimizer_steps']:
        raise ValueError('Continuation smoke failed')
    write(root/'smoke_result.json',dict(status='passed',restored_epoch=18,restored_updates=2808,
        restored_optimizer_and_scheduler=True,loss_scale=4096,new_updates=2,skips=0,
        smoke_only_not_a_complete_epoch=True,peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('run','smoke','resume-smoke'),nargs='?',default='run')
    p.add_argument('--root',type=Path,default=Path('output/r1_frozen_late_stable/seed1_20261105'))
    p.add_argument('--old',type=Path,default=Path('../multi-lane-main-r1-frozen-cross-view-20261105/output/r1_frozen_cross_view/seed1_20261105'))
    p.add_argument('--reference',type=Path,default=Path('../multi-lane-main-r1-crossvit-20261101/output/r1_crossvit/seed1_20261101_02'))
    args=p.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('GPU0 only')
    inputs=input_paths();source=args.old/'image_cls/attention/person/train'
    if args.action=='resume-smoke':return resume_smoke(root,source)
    git=git_metadata(Path(__file__).resolve().parents[2])
    if git['dirty']:raise ValueError('Clean source required')
    logs=Path('logs/r1_frozen_late_stable')/root.name;logs.mkdir(parents=True,exist_ok=True)
    smoke=args.action=='smoke'
    protocol=dict(git=git,seed=1,tasks=2 if smoke else 8,epochs=30,batch=64,main_lr=.0125,new_lr=.0004,
        layer6_recovery='complete task0-2 plus full task3 epoch18; optimizer/scheduler/RNG retained',
        late_layer=8,late_actual_block=9,loss_scale=4096,growth_interval=1000000000,
        old_completed_paths_reused=True,late_six_configs_from_four_paths=True,
        fixed_logit_weights=[.64,.16,.20],fallback=[.8,.2,0.],cap=.05,heads=4,dim=64,smoke=smoke)
    if (root/'protocol.json').exists() and read(root/'protocol.json')!=protocol:raise ValueError('Protocol changed')
    write(root/'protocol.json',protocol)
    oldfiles=list(args.old.glob('*/attention/*/train/compact_checkpoints/*.pth'))+[source/'epoch_recovery/task3/epoch18.pth']
    oldfiles+=list(args.old.glob('*/attention/*/train/test_scores/*.npz'))+list(args.old.glob('*/attention/*/val/scores/*.npz'))
    hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in oldfiles}
    write(root/'source_hashes.json',hashes)
    jobs=[]
    def training(name,base,mode,view,layer,recovery=False):
        command=[sys.executable,'-u','-m','multi_lane.track_a.runner',
            *flags(SimpleNamespace(output=base/mode,**vars(inputs)),'attention',view,smoke),
            '--r1-frozen-mode',mode,'--r1-exchange-layer',str(layer),
            '--r1-fixed-loss-scale','4096','--r1-fusion-readout','logit']
        if recovery:command+=['--r1-recovery-source',str(source)]
        if smoke:command+=['--r1-smoke-eval-samples','64']
        return dict(name=name,command=command,memory_mib=9000)
    if not smoke:
        train=folder(root/'layer6_recovery','image_cls','person')
        copy_prefix(source,train)
        jobs.append(training('recover_layer6_person',root/'layer6_recovery','image_cls','person',5,True))
        command=[sys.executable,'-u','-m','multi_lane.track_a.r1_crossvit_experiment','val',
            '--training-root',str(train),'--output',str(train.parent/'val'),
            '--data-root',str(inputs.data_root),'--clip-checkpoint',str(inputs.clip_checkpoint),
            '--face-manifest-root',str(inputs.face_manifest_root)]
        jobs.append(dict(name='recover_validation',command=command,depends=['recover_layer6_person'],memory_mib=7500))
    for mode in MODES:
        for view in VIEWS:
            name='layer9_'+mode+'_'+view
            jobs.append(training(name,root/'layer9',mode,view,8))
            if not smoke:
                train=folder(root/'layer9',mode,view)
                command=[sys.executable,'-u','-m','multi_lane.track_a.r1_crossvit_experiment','val',
                    '--training-root',str(train),'--output',str(train.parent/'val'),
                    '--data-root',str(inputs.data_root),'--clip-checkpoint',str(inputs.clip_checkpoint),
                    '--face-manifest-root',str(inputs.face_manifest_root)]
                jobs.append(dict(name=name+'_validation',command=command,depends=[name],memory_mib=7500))
    if smoke:
        for j in jobs:j['memory_mib']=10000
    oldpaths={(m,v):args.old/m/'attention'/v for m in MODES for v in VIEWS}
    oldpaths[('image_cls','person')]=root/'layer6_recovery/image_cls/attention/person'
    def early_summary(job,status):
        if job['name']=='recover_validation' and status=='complete':
            torch.set_num_threads(2)
            summarize_family(root,args.reference,inputs.face_manifest_root,oldpaths,root/'completed_layer6_results.json')
    states=run_graph(jobs,root,logs,1 if smoke else 2,on_finish=early_summary)
    training_checks={}
    if not smoke and states.get('recover_layer6_person')=='complete':
        training_checks['layer6_person']=check_training(folder(root/'layer6_recovery','image_cls','person'),args.reference,'person',8,False)
    for mode in MODES:
        for view in VIEWS:
            if states.get('layer9_'+mode+'_'+view)=='complete':
                training_checks['layer9_'+mode+'_'+view]=check_training(folder(root/'layer9',mode,view),args.reference,view,2 if smoke else 8,smoke)
    write(root/'training_checks.json',training_checks)
    if smoke:
        checks={}
        for mode in MODES:
            for view in VIEWS:
                if states['layer9_'+mode+'_'+view]!='complete':continue
                out=root/'preflight'/f'{mode}_{view}.json'
                preflight(folder(root/'layer9',mode,view),inputs,out);checks[mode+'_'+view]=read(out)
        if len(checks)!=4:raise ValueError('Smoke paths failed')
    else:
        torch.set_num_threads(2)
        summarize_family(root,args.reference,inputs.face_manifest_root,oldpaths,root/'completed_layer6_results.json')
        newpaths={(m,v):root/'layer9'/m/'attention'/v for m in MODES for v in VIEWS}
        summarize_family(root,args.reference,inputs.face_manifest_root,newpaths,root/'layer9_six_group_results.json')
    changed=[x for x,h in hashes.items() if hashlib.sha256(Path(x).read_bytes()).hexdigest()!=h]
    if changed:raise AssertionError('Source artifacts changed')
    write(root/'completion.json',dict(status='complete' if all(v=='complete' for v in states.values()) else 'partial',
        jobs=states,source_files_unchanged=True,protocol=protocol))


if __name__=='__main__':main()
