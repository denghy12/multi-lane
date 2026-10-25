"""Independent Selector co-training and Face-only adaptation of a fixed strong model."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _sha256, _torch_load
from .head_anchored_eight import runtime, copy_classifier_head
from .head_anchored_pilot import adaptation_rng, source_train, student_optimizer_groups
from .historical_b32_backbone_attribution import build, method_state
from .historical_b32_modules import validate_config
from .paired_transforms import move_model_inputs
from .run_historical_b32_test_attribution import read, write
from .runner import (compact_model_state_dict, dataset_view, seen_indices, set_seed,
                     tensor_state_digest, train_task)
from .task_recovery import atomic_save, save_boundary

FACE_PREFIX='cross_view_bank.face_tasks.'


def protected_state(model):
    return {k:v for k,v in method_state(model).items() if not k.startswith(FACE_PREFIX)}


def face_state(model, before_task=None):
    return {k:v for k,v in method_state(model).items() if k.startswith(FACE_PREFIX)
            and (before_task is None or int(k.split('.')[2])<before_task)}


def install_strong_task(model, person, b0, task):
    face_before=tensor_state_digest(face_state(model))
    result=model.load_state_dict(person,strict=False)
    assert not result.unexpected_keys and result.missing_keys
    assert all(k.startswith(('visual_encoder.',FACE_PREFIX)) for k in result.missing_keys)
    assert face_before==tensor_state_digest(face_state(model))
    copy_classifier_head(model,b0)
    model.restore_task(task)


def configure_face_only(model, task):
    model.restore_task(task);model.requires_grad_(False)
    assert model.cross_view_bank.face_tasks is not None
    model.cross_view_bank.face_tasks[task].requires_grad_(True)
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    assert names and all(n.startswith(FACE_PREFIX+str(task)+'.') for n in names)
    return names


class FaceSignalLoader:
    """Keep the original sampler, omitting only batches with no gradient signal.

    Entirely frozen Full/Person paths cannot train the Face-only optimizer when
    every Face is unreliable. Such batches never reach backward or Adam.step;
    they are recorded separately from AMP skipped optimizer steps.
    """
    def __init__(self, loader):
        self.loader=loader;self.generator=getattr(loader,'generator',None)
        self.traces=[];self.no_signal_batches=[]

    def __len__(self):return len(self.loader)

    def __iter__(self):
        trace=[];self.traces.append(trace);self.no_signal_batches.append(0)
        for images,targets,ids in self.loader:
            if not bool(images['face_reliable'].bool().any()):
                self.no_signal_batches[-1]+=1;continue
            if len(trace)<2:
                trace.append(dict(sample_ids=list(ids),pixels_sha256=tensor_state_digest(images),
                                  targets_sha256=tensor_state_digest({'y':targets})))
            yield images,targets


def training(args):
    start=time.time();git=runtime();set_seed(1,True)
    person_cfg=read(args.person_run/'config.json');assert validate_config(person_cfg)=='ATTENTION_BOUND'
    assert person_cfg['selector_mode']=='shared' and not person_cfg.get('cross_view_face_attention',False)
    assert read(args.person_run/'seed_summary.json')['status']=='complete'
    b0_cfg=read(args.source_run/'config.json');assert validate_config(b0_cfg)=='BASE'
    template=dict(person_cfg,cross_view_face_attention=True)
    tasks=list(range(2 if args.smoke else 8))
    sources={}
    for name,folder in (('person',args.person_run),('b0',args.source_run)):
        verified=read(folder.parent/'training_verification.json')['checkpoint_sha256']
        for task in tasks:
            p=folder/'compact_checkpoints'/f'task{task}.pth'
            h=_sha256(p);assert h==verified[str(task)];sources[f'{name}/{task}']=dict(path=str(p),sha256=h)
    model=build(template,args.clip_checkpoint)
    cfg=dict(kind='frozen_person_face_only',git=git,seed=1,tasks=tasks,smoke=args.smoke,
        model_template=template,reference_root=str(args.reference_root),person_run=str(args.person_run),
        b0_run=str(args.source_run),sources=sources,epochs_per_task=5,learning_rate=.0004,
        train_batch_size=64,workers=2,optimizer='Adam_reset_per_task',scheduler='cosine',weight_decay=0.,
        amp=True,tf32=True,cpu_threads=72,loss_routing='joint_bce',view_auxiliary_loss_weight=.1,
        adapter_runtime_enabled=True,adapter_frozen=True,adapter_pretraining_objective='ASL',
        head_during_training='B0_corresponding_stage',only_trainable='current_task_face_attention',
        adaptation_rng_seeds={str(t):20261025+1009*t for t in tasks},
        sampler_seeds={str(t):317001+t for t in tasks},
        evaluation_scope='samples_intersect_seen_classes',test_used_for_parameter_search=True)
    if (args.output/'config.json').exists():assert read(args.output/'config.json')==cfg
    else:write(args.output/'config.json',cfg)
    train=source_train(person_cfg,args);val=make_source(person_cfg,args.data_root,args.face_manifest_root,'val')
    history={};audits={};visual_hash=tensor_state_digest(model.visual_encoder.state_dict())
    for task in tasks:
        recovery=args.output/'recovery_checkpoints'/f'task{task}.pth'
        with adaptation_rng(20261025+1009*task):
            if recovery.exists():
                saved=torch.load(recovery,map_location='cpu',weights_only=False)
                assert saved['config']==cfg and saved['task_id']==task
                result=model.load_state_dict(saved['model'],strict=False)
                assert not result.unexpected_keys and all(k.startswith('visual_encoder.') for k in result.missing_keys)
                model.restore_task(task)
                history=saved['provenance']['history'];audits=saved['provenance']['audits']
                print('RESUME_COMPLETED_TASK',task,'repeated_updates=0',flush=True);continue
            person=_torch_load(args.person_run/'compact_checkpoints'/f'task{task}.pth')
            b0=_torch_load(args.source_run/'compact_checkpoints'/f'task{task}.pth')
            assert person['task_id']==b0['task_id']==task
            assert person['source_git']==person_cfg['git'] and b0['source_git']==b0_cfg['git']
            install_strong_task(model,person['model'],b0['model'],task)
            model.eval();before=tensor_state_digest(protected_state(model));old_face=tensor_state_digest(face_state(model,task))
            probe=DataLoader(dataset_view(val,seen_indices(task),True),batch_size=64,shuffle=False,num_workers=0,
                generator=torch.Generator().manual_seed(87001+task))
            images,_,ids=next(iter(probe))
            with torch.no_grad(),torch.cuda.amp.autocast():
                fused,views=model.seen_logits_with_views(move_model_inputs(images,torch.device('cuda')))
            a,b=sum(model._task_sizes[:task]),sum(model._task_sizes[:task+1])
            with np.load(args.reference_root/'b0_val/on/view_scores'/f'task{task}.npz') as ref:
                assert np.array_equal(ids,ref['sample_ids'][:64])
                for name,z in dict(fused=fused,**views).items():
                    assert np.array_equal(z.float().cpu().numpy()[:,a:b],ref[name+'_logits'][:64,a:b]),name
            names=configure_face_only(model,task)
            view=dataset_view(train,range(a,b),True)
            reliable=sum(bool(train.face_reliable[i]) for i in view.indices)
            assert reliable>0,'No reliable Face training instances for this task'
            loader=FaceSignalLoader(DataLoader(view,batch_size=64,shuffle=True,num_workers=2,pin_memory=True,
                drop_last=False,generator=torch.Generator().manual_seed(317001+task)))
            state={}
            history[str(task)]=train_task(model,loader,None,torch.device('cuda'),task,5,.0004,0.,1.,True,
                adapter_learning_rate=.0004,adapter_weight_decay=0.,view_fusion_learning_rate=.0004,
                loss_routing='joint_bce',view_auxiliary_loss_weight=.1,
                optimizer_updates_per_task=2 if args.smoke else None,
                optimizer_group_builder=student_optimizer_groups,recovery_state=state)
            rows=history[str(task)];steps=sum(r['optimizer_steps'] for r in rows)
            assert steps==(2 if args.smoke else 5*len(loader)-sum(loader.no_signal_batches))
            assert all(r['skipped_optimizer_steps']==0 for r in rows)
            assert before==tensor_state_digest(protected_state(model))
            assert old_face==tensor_state_digest(face_state(model,task))
            assert visual_hash==tensor_state_digest(model.visual_encoder.state_dict())
            assert all(n.startswith(FACE_PREFIX+str(task)+'.') for group in state['optimizer_parameter_names'] for n in group)
            audits[str(task)]=dict(protected_parameters_unchanged=True,old_face_unchanged=True,
                teacher_current_predictions_exact=True,trainable_names=names,train_instances=len(view),
                reliable_face_instances=reliable,zero_signal_batches=loader.no_signal_batches,
                actual_batch_trace=loader.traces,optimizer_parameter_names=state['optimizer_parameter_names'])
            compact=compact_model_state_dict(model)
            atomic_save(dict(schema_version=1,task_id=task,source_git=git,model=compact),
                args.output/'compact_checkpoints'/f'task{task}.pth')
            save_boundary(model,compact,task,cfg,state,loader.generator,dict(history=history,audits=audits),recovery)
            write(args.output/'training_history.json',history);write(args.output/'audits.json',audits)
            print('FROZEN_FACE_TASK_SAVED',task,'updates',steps,'protected_parameters_exact=1',flush=True)
    assert all(_sha256(Path(row['path']))==row['sha256'] for row in sources.values())
    checkpoints={str(t):dict(path=str(args.output/'compact_checkpoints'/f'task{t}.pth'),source_git=git,
        sha256=_sha256(args.output/'compact_checkpoints'/f'task{t}.pth')) for t in tasks}
    write(args.output/'training_complete.json',dict(status='complete',config=cfg,checkpoints=checkpoints,
        protected_base_preserved=True,epochs=sum(map(len,history.values())),
        completed_optimizer_updates=sum(r['optimizer_steps'] for rows in history.values() for r in rows),
        zero_signal_batches=sum(sum(row['zero_signal_batches']) for row in audits.values()),
        elapsed_seconds=time.time()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))


def initialization_audit(args):
    runtime();set_seed(1,True)
    source=read(args.dual_run/'config.json');assert validate_config(source)=='ATTENTION_BOUND'
    assert source['selector_mode']=='shared' and source['cross_view_face_attention']
    shared=build(source,args.clip_checkpoint);rng=torch.get_rng_state().clone()
    cuda_rng=[v.clone() for v in torch.cuda.get_rng_state_all()]
    set_seed(1,True)
    private_cfg=dict(source,selector_mode='view_specific',historical_b32_private_selectors=True)
    private=build(private_cfg,args.clip_checkpoint)
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
    a,b=method_state(shared),method_state(private)
    assert b['selectors'].shape==(8,3,10,768)
    assert all(torch.equal(b['selectors'][:,i],a['selectors']) for i in range(3))
    assert all(torch.equal(v,b[k]) for k,v in a.items() if k!='selectors')
    shared.restore_task(0);private.restore_task(0)
    shared.eval();private.eval()
    data=make_source(source,args.data_root,args.face_manifest_root,'val')
    loader=DataLoader(dataset_view(data,seen_indices(0),True),batch_size=64,num_workers=0,
        generator=torch.Generator().manual_seed(87001))
    images,_,_=next(iter(loader));images=move_model_inputs(images,torch.device('cuda'))
    with torch.no_grad(),torch.cuda.amp.autocast():
        x,xv=shared.seen_logits_with_views(images);y,yv=private.seen_logits_with_views(images)
    assert torch.equal(x,y) and all(torch.equal(xv[k],yv[k]) for k in xv)
    collapsed=dict(b,selectors=b['selectors'][:,0])
    initial={k:v for k,v in collapsed.items() if k=='selectors' or k.startswith(('prompts.','head.'))}
    assert tensor_state_digest(initial)==source['initial_selector_prompt_classifier_sha256']
    write(args.output,dict(status='verified',common_parameters_exact=True,
        private_selector_copies_exact=True,global_rng_exact=True,initial_predictions_exact=True,
        shared_selectors_total=a['selectors'].numel(),private_selectors_total=b['selectors'].numel(),
        added_parameters=b['selectors'].numel()-a['selectors'].numel()))


def parser():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    for action in ('train','audit'):
        q=s.add_parser(action)
        for key in ('data-root','clip-checkpoint','face-manifest-root','output'):q.add_argument('--'+key,type=Path,required=True)
        if action=='train':
            for key in ('source-run','person-run','reference-root'):q.add_argument('--'+key,type=Path,required=True)
            q.add_argument('--smoke',action='store_true')
        else:q.add_argument('--dual-run',type=Path,required=True)
    return p


if __name__=='__main__':
    args=parser().parse_args();(training if args.action=='train' else initialization_audit)(args)
