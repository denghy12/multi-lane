"""Fresh independent three-view training, recoverable epochs and paired evaluation."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .crossvit_three_view import ARMS, VIEWS, IndependentThreeView, compact_state, old_state, optimizer_groups
from .historical_b32_backbone_attribution import build as baseline_build, method_state
from .historical_b32_modules import validate_config
from .historical_reporting_rng import replay_reporting_rng
from .export_compact_test_scores import _sha256, _torch_load
from .evaluate_joint26_checkpoint import make_source
from .paired_transforms import move_model_inputs
from .protected_joint_experiment import (AuditedLoader, training_source, runtime,
    preserved_rng, old_probe, compare_old_probe, evaluation_loader)
from .run_historical_b32_test_attribution import read, write
from .runner import (dataset_view, seen_indices, set_seed, tensor_state_digest,
    train_task, summarize_tasks, TaskMetrics, compute_metrics, write_evaluation_scores)
from .task_recovery import save_boundary, restore_rng


def build(base, clip, arm):
    base_model=baseline_build(base,clip)
    model=IndependentThreeView(base_model,arm).float().cuda()
    model.assert_independent()
    return model


def restore(model,path,config=None,rng=False):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if config is not None and saved['config']!=config:
        raise ValueError('Recovery protocol changed')
    missing=model.load_state_dict(saved['model'],strict=False)
    if missing.unexpected_keys or any('.visual_encoder.' not in k for k in missing.missing_keys):
        raise AssertionError(('Missing task state',missing))
    model.restore_task(saved['task_id'])
    if rng:restore_rng(saved['rng'])
    return saved


def train(args):
    started=time.time();git=runtime()
    base=read(args.source_run/'config.json')
    if validate_config(base)!='BASE':raise ValueError('Historical BASE recipe required')
    set_seed(1,True)
    model=build(base,args.clip_checkpoint,args.arm)
    first=model.branches['full']
    initial=tensor_state_digest({k:v for k,v in method_state(first).items()
        if k=='selectors' or k.startswith(('prompts.','head.'))})
    if initial!=base['initial_selector_prompt_classifier_sha256']:
        raise AssertionError('Historical untrained initialization changed')
    template=compact_state(model)
    common=tensor_state_digest({k:v for k,v in template.items() if k.startswith('branches.')})
    cfg=dict(kind='crossvit_independent_three_view',git=git,arm=args.arm,seed=1,
        smoke=args.smoke,smoke_updates=args.smoke_updates if args.smoke else None,
        tasks=[0,1] if args.smoke else list(range(8)),model_template=base,
        data_root=str(args.data_root),clip_checkpoint=str(args.clip_checkpoint),
        face_manifest_root=str(args.face_manifest_root),source_run=str(args.source_run),
        recipe_sha256=_sha256(args.source_run/'config.json'),clip_sha256=_sha256(args.clip_checkpoint),
        initial_common_sha256=common,initial_historical_ASPH_sha256=initial,
        external_task_parameters_loaded=False,teachers=False,independent_frozen_backbones=3,
        independent_view_task_parameters=True,epochs_per_task=30,batch_size=64,
        main_lr=.0125,adapter_lr=.0004,interaction_lr=.0004,optimizer='Adam_reset_per_task',
        scheduler='cosine',weight_decay=0.,loss_routing='adapter_asl',auxiliary_bce=.1,
        workers=2,prefetch_factor=1,threads=72,exchange_layer=5,exchange_dim=64,exchange_heads=4,
        memory='other_frozen_spatial_patches_plus_own_CLS',simultaneous_exchange=True,
        fusion='historical_logit_weights' if args.arm.endswith('_W') else 'valid_view_logit_mean',
        face_mask_before_softmax=True,evaluation_scope='samples_intersect_seen_classes',
        test_used_for_development=True,reporting_rng='historical_two_CPU_int64_draws',
        trainable_method_parameters=sum(v.numel() for v in template.values()),
        interaction_parameters=[sum(p.numel() for p in block.parameters()) for block in model.exchange],
        frozen_parameters=sum(p.numel() for b in model.branches.values() for p in b.visual_encoder.parameters()))
    del template
    path=args.output/'config.json'
    if path.exists() and read(path)!=cfg:raise ValueError('Changed own run configuration')
    write(path,cfg)
    source=training_source(base,args)
    probe_source=make_source(base,args.data_root,args.face_manifest_root,'train')
    visual={v:tensor_state_digest(b.visual_encoder.state_dict()) for v,b in model.branches.items()}
    history,audits={},{}
    for task in cfg['tasks']:
        final=args.output/'recovery_checkpoints'/f'task{task}.pth'
        if final.exists():
            saved=restore(model,final,cfg,True)
            history,audits=saved['provenance']['history'],saved['provenance']['audits']
            print('RESUME_COMPLETED_TASK',args.arm,task,flush=True);continue
        epoch_root=args.output/'epoch_recovery'/f'task{task}'
        paths=sorted(epoch_root.glob('epoch*.pth'),key=lambda p:int(p.stem[5:]))
        resume=None
        if paths:
            saved=restore(model,paths[-1],cfg,True)
            prov=saved['provenance'];history,audits=prov['history'],prov['audits']
            old_before,probe_before=prov['old_sha256'],prov['old_probe']
            prior_batches=prov['batches'];resume=saved['training_state']
            print('RESUME_EPOCH',args.arm,task,resume['next_epoch'],flush=True)
        else:
            model.activate_task(task)
            with preserved_rng():
                old_before=tensor_state_digest(old_state(model,task))
                probe_before=old_probe(model,probe_source,task)
            prior_batches=[]
        model.configure_current(task)
        current=range(sum(model.task_sizes[:task]),sum(model.task_sizes[:task+1]))
        loader=AuditedLoader(DataLoader(dataset_view(source,current,True),batch_size=64,
            shuffle=True,num_workers=2,prefetch_factor=1,pin_memory=True,drop_last=False),prior_batches)
        state={}
        def epoch_callback(next_epoch,rows,training_state):
            with preserved_rng():
                save_boundary(model,compact_state(model),task,cfg,training_state,None,
                    dict(history=dict(history,**{str(task):rows}),audits=audits,
                         old_sha256=old_before,old_probe=probe_before,batches=loader.epochs),
                    epoch_root/f'epoch{next_epoch:02d}.pth')
                write(args.output/'training_history.json',dict(history,**{str(task):rows}))
                write(args.output/'progress.json',dict(status='training',arm=args.arm,task=task,
                    completed_epochs=next_epoch,target_epochs=30,
                    completed_task_updates=training_state['completed_task_updates']))
        rows=train_task(model,loader,None,torch.device('cuda'),task,30,.0125,0.,1.,True,
            adapter_learning_rate=.0004,adapter_weight_decay=0.,view_fusion_learning_rate=.0004,
            view_auxiliary_loss_weight=.1,loss_routing='adapter_asl',
            optimizer_updates_per_task=args.smoke_updates if args.smoke else None,
            optimizer_group_builder=optimizer_groups,recovery_state=state,
            epoch_callback=epoch_callback,resume_training_state=resume)
        budget=args.smoke_updates if args.smoke else 30*len(loader)
        if sum(r['optimizer_steps'] for r in rows)!=budget or any(r['skipped_optimizer_steps'] for r in rows):
            raise AssertionError('Update budget or AMP skip failure')
        if tensor_state_digest(old_state(model,task))!=old_before:raise AssertionError('Old parameter drift')
        for v,b in model.branches.items():
            if tensor_state_digest(b.visual_encoder.state_dict())!=visual[v]:raise AssertionError('ViT drift')
        with preserved_rng():probe_after=old_probe(model,probe_source,task)
        history[str(task)]=rows
        audits[str(task)]=dict(old_parameters_exact=True,frozen_visual_exact=True,
            old_prediction_probe=compare_old_probe(probe_before,probe_after),batches=loader.epochs,
            historical_reporting_rng=replay_reporting_rng(task),optimizer_parameter_names=state['optimizer_parameter_names'])
        save_boundary(model,compact_state(model),task,cfg,state,None,
            dict(history=history,audits=audits),final)
        write(args.output/'audits.json',audits)
        print('CROSSVIT_TASK_SAVED',args.arm,task,flush=True)
    write(args.output/'training_complete.json',dict(status='complete',config=cfg,tasks=cfg['tasks'],
        checkpoints={str(t):_sha256(args.output/'recovery_checkpoints'/f'task{t}.pth') for t in cfg['tasks']},
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        elapsed_seconds=time.time()-started,updates=sum(r['optimizer_steps'] for v in history.values() for r in v)))


def evaluate(args):
    started=time.time();git=runtime();set_seed(1,True)
    cfg=read(args.training_root/'config.json');done=read(args.training_root/'training_complete.json')
    if done['status']!='complete' or cfg!=done['config'] or cfg['git']!=git:raise ValueError('Changed source')
    if cfg['smoke'] and args.split=='test':raise ValueError('No smoke Test')
    model=build(cfg['model_template'],Path(cfg['clip_checkpoint']),cfg['arm'])
    model.requires_grad_(False).eval()
    # Only GPU matmuls run in evaluation; constrain CPU metric/preprocessing contention.
    torch.set_num_threads(2)
    source=make_source(cfg['model_template'],Path(cfg['data_root']),Path(cfg['face_manifest_root']),args.split)
    metrics=[];off=[];audit=[]
    for task in done['tasks']:
        result_path=args.output/f'task{task}.json'
        if result_path.exists():
            record=read(result_path)
            if record['checkpoint_sha256']!=done['checkpoints'][str(task)]:raise ValueError('Changed checkpoint')
            metrics.append(record['metrics'])
            if record['off_metrics'] is not None:off.append(record['off_metrics'])
            audit.append(record['audit']);continue
        path=args.training_root/'recovery_checkpoints'/f'task{task}.pth'
        if _sha256(path)!=done['checkpoints'][str(task)]:raise AssertionError('Checkpoint SHA')
        restore(model,path);model.requires_grad_(False).eval()
        before=tensor_state_digest(compact_state(model))
        view=dataset_view(source,seen_indices(task),True)
        if cfg['smoke']:view=Subset(view,range(min(64,len(view))))
        loader=evaluation_loader(view,task,args.workers)
        do_off=bool(model.exchange) and (task in (0,2,6,7) or cfg['smoke'])
        zs={v:[] for v in ('fused',*VIEWS)};targets=[];ids=[];reliable=[];oz=[];lengths=[]
        with torch.no_grad():
            for x,y,sample_ids in loader:
                x=move_model_inputs(x,torch.device('cuda'))
                model._frozen_cache={} if do_off else None
                with torch.cuda.amp.autocast(enabled=True):
                    model.exchange_enabled=True;z,vs=model.seen_logits_with_views(x)
                    if do_off:
                        model.exchange_enabled=False;no,_=model.seen_logits_with_views(x)
                        oz.append(no.float().cpu())
                model._frozen_cache=None;model.exchange_enabled=True
                for name,value in dict(fused=z,**vs).items():zs[name].append(value.float().cpu())
                targets.append(y.float());ids.extend(str(s) for s in sample_ids)
                reliable.append(x['face_reliable'].cpu().bool());lengths.append(len(y))
        y=torch.cat(targets);mask=torch.cat(reliable)
        arrays=dict(sample_ids=np.asarray(ids),targets=y.numpy(),face_reliable=mask.numpy(),
                    class_indices=np.asarray(seen_indices(task)),batch_lengths=np.asarray(lengths))
        views={}
        for name,parts in zs.items():
            values=torch.cat(parts);prob=torch.sigmoid(values)
            arrays[name+'_logits']=values.numpy();arrays[name+'_probabilities']=prob.numpy()
            views[name]=asdict(compute_metrics(task,prob,y,.5))
        if mask.any():views['face_reliable']=asdict(compute_metrics(task,torch.sigmoid(torch.cat(zs['face'])[mask]),y[mask],.5))
        om=None
        if do_off:
            values=torch.cat(oz);arrays['off_fused_logits']=values.numpy()
            om=asdict(compute_metrics(task,torch.sigmoid(values),y,.5));off.append(om)
        args.output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output/f'task{task}_scores.npz',**arrays)
        unchanged=before==tensor_state_digest(compact_state(model))
        if not unchanged:raise AssertionError('Read-only evaluation mutated parameters')
        a=dict(task=task,samples=len(y),parameters_unchanged=True,
               off_frozen_streams_shared=do_off)
        record=dict(metrics=views['fused'],views=views,off_metrics=om,audit=a,
                    checkpoint_sha256=done['checkpoints'][str(task)])
        write(result_path,record);metrics.append(record['metrics']);audit.append(a)
        write(args.output/'progress.json',dict(status='running',completed_tasks=[m['task_id'] for m in metrics]))
        print('CROSSVIT_EVAL',cfg['arm'],args.split,task,record['metrics']['mAP'],flush=True)
    write(args.output/'result.json',dict(status='complete',arm=cfg['arm'],split=args.split,
        tasks=done['tasks'],seed=1,git=git,smoke=cfg['smoke'],metrics=metrics,
        summary=summarize_tasks([TaskMetrics(**m) for m in metrics]),off_metrics=off,
        off_is_conditional_ablation_not_retrained_baseline=True,audits=audit,
        workers=args.workers,threads=2,evaluation_scope='samples_intersect_seen_classes',
        elapsed_seconds=time.time()-started,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('train','eval'))
    p.add_argument('--arm',choices=ARMS)
    for name in ('source-run','data-root','clip-checkpoint','face-manifest-root','training-root','output'):
        p.add_argument('--'+name,type=Path,required=name=='output')
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-updates',type=int,default=4)
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--workers',choices=(0,2,4),type=int,default=2)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.smoke_updates<=0:raise ValueError('Positive smoke budget required')
    if args.action=='train':
        if args.arm is None or any(getattr(args,k) is None for k in ('source_run','data_root','clip_checkpoint','face_manifest_root')):
            raise ValueError('Missing training paths')
        train(args)
    else:
        if args.training_root is None:raise ValueError('Missing training root')
        evaluate(args)
