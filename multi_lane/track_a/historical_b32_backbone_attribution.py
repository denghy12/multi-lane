"""In-memory component restoration of the historical b32 three-view models."""
import argparse
from contextlib import contextmanager
import csv
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .historical_b32_modules import modes, validate_config
from .historical_b32_test_attribution import ARMS, TASKS
from .export_compact_test_scores import _torch_load, _sha256, restore_compact_model_state
from .model import MultiLaneModel
from .module_stacking import runtime_bypass
from .openai_clip_loader import load_openai_clip_visual
from .runner import (TASK_SIZES, LabelView, dataset_view, seen_indices, set_seed,
                     tensor_state_digest, compact_model_state_dict, git_metadata)
from .evaluate_joint26_checkpoint import make_source
from .evaluate_moe_task_ffn import evaluation_loader
from .strong_full_evaluation import evaluate_fused_and_views
from .gradient_attribution import grouped_batches
from .paired_transforms import move_model_inputs
from .run_historical_b32_test_attribution import read, write, MODEL_HEAD, VIEWS

GROUPS = ('A', 'S', 'P', 'H')
CONDITIONS = ('none', 'A', 'S', 'P', 'H', 'ASP', 'ASPH')
FACTORIAL = tuple(''.join(g for g, enabled in zip(GROUPS, flags) if enabled) or 'none'
                  for flags in itertools.product((False, True), repeat=4))


def component(name):
    if name == 'selectors':return 'S'
    if name.startswith('prompts.'):return 'P'
    if name.startswith('adapter_bank.task_adapters.'):return 'A'
    if name.startswith('head.'):return 'H'
    return None


def method_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith('visual_encoder.')}


@contextmanager
def restore_components(model, reference, condition):
    selected = set() if condition == 'none' else set(condition)
    if not selected <= set(GROUPS) or len(selected) != (0 if condition == 'none' else len(condition)):
        raise ValueError('Invalid component restoration')
    state = method_state(model)
    for key, value in reference.items():
        if key not in state or value.shape != state[key].shape:
            raise ValueError('Reference method state differs: '+key)
        if component(key) is None and not torch.equal(value.cpu(), state[key].cpu()):
            raise ValueError('Unaccounted common state differs: '+key)
    keys = [k for k in reference if component(k) in selected]
    if selected and {component(k) for k in keys} != selected:
        raise ValueError('Requested component is absent')
    before = tensor_state_digest(state)
    untouched = {k:v for k,v in state.items() if k not in keys}
    protected = tensor_state_digest(untouched)
    backup = {k:state[k].detach().clone() for k in keys}
    try:
        with torch.no_grad():
            for k in keys:state[k].copy_(reference[k])
        patched = tensor_state_digest(state)
        yield dict(condition=condition, keys=keys, before_sha256=before, patched_sha256=patched)
        assert tensor_state_digest(state) == patched, 'Forward modified patched parameters'
        assert tensor_state_digest(untouched) == protected, 'Unselected parameters changed'
    finally:
        with torch.no_grad():
            for k in keys:state[k].copy_(backup[k])
        assert tensor_state_digest(method_state(model)) == before, 'Memory restoration failed'


def build(config, clip):
    arm = validate_config(config)
    model = MultiLaneModel(load_openai_clip_visual(clip), TASK_SIZES,
        num_selectors=config['num_selectors'], num_prompts=config['num_prompts'],
        num_prompt_layers=config['num_prompt_layers'], selector_mode=config['selector_mode'],
        prompt_mode=config['prompt_mode'], normalize=config['normalize'],
        adapter_mode=config['adapter_mode'], adapter_bottleneck_dim=config['adapter_bottleneck_dim'],
        adapter_layer_indices=config['adapter_layer_indices'], adapter_residual_scale=config['adapter_residual_scale'],
        adapter_activation=config['adapter_activation'], adapter_task_initialization=config['adapter_task_initialization'],
        view_fusion=config['view_fusion'], view_classifier_mode=config['view_classifier_mode'],
        historical_b32_modules=True, cross_view_layer=5, cross_view_dim=64, cross_view_heads=4,
        cross_view_face_attention=config.get('cross_view_face_attention', False),
        cross_view_scale=.1, **modes(arm)).float().cuda()
    return model


def load_host(model, source, config, task):
    payload = _torch_load(source / 'compact_checkpoints' / f'task{task}.pth')
    restore_compact_model_state(model, payload, task, config['git'])
    runtime_bypass(model, ('B','C'))
    model.requires_grad_(False).eval()
    return payload


def comparison(left, right):
    a = torch.cat([x.detach().double().reshape(-1) for x in left])
    b = torch.cat([x.detach().double().reshape(-1) for x in right])
    na, nb = float(a.norm()), float(b.norm()); difference = a-b
    return dict(elements=a.numel(), source_norm=na, baseline_norm=nb,
        relative_l2=float(difference.norm()/nb) if nb else None,
        cosine=float(a.dot(b)/(na*nb)) if na and nb else None,
        rms_change=float(difference.square().mean().sqrt()), exact=torch.equal(a,b))


def coordinates(state, group, task, current=False):
    result=[];start=sum(TASK_SIZES[:task]);stop=sum(TASK_SIZES[:task+1])
    for key,value in state.items():
        if component(key)!=group:continue
        if group=='A':
            t=int(key.split('.')[2])
            if (t!=task if current else t>task):continue
        elif group=='S':value=value[task:task+1] if current else value[:task+1]
        elif group=='P':value=value[:,task:task+1] if current else value[:,:task+1]
        elif group=='H':value=value[start:stop] if current else value[:stop]
        result.append(value)
    return result


def audit_drift(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    rows=[];inherit=[]
    for arm in ARMS:
        for task in range(8):
            a=_torch_load(args.source_root/arm/'train/compact_checkpoints'/f'task{task}.pth')['model']
            b=_torch_load(args.source_root/'BASE/train/compact_checkpoints'/f'task{task}.pth')['model']
            assert all(torch.equal(v,b[k]) for k,v in a.items() if component(k) is None and k in b)
            for group in GROUPS:
                for current in (False,True):
                    rows.append(dict(arm=arm,task=task,group=group,scope='current' if current else 'seen',
                        **comparison(coordinates(a,group,task,current),coordinates(b,group,task,current))))
            if task<7:
                for group in ('S','P'):
                    inherit.append(dict(arm=arm,next_task=task+1,group=group,
                        mechanism='copy_previous_task_at_activate_task',
                        **comparison(coordinates(a,group,task,True),coordinates(b,group,task,True))))
                for group in ('A','H'):
                    inherit.append(dict(arm=arm,next_task=task+1,group=group,
                        mechanism='existing_future_task_coordinates_before_activation',
                        **comparison(coordinates(a,group,task+1,True),coordinates(b,group,task+1,True))))
    for name,values in (('parameter_drift.csv',rows),('task_initialization_inheritance.csv',inherit)):
        with (args.output/name).open('x') as f:
            writer=csv.DictWriter(f,list(values[0]));writer.writeheader();writer.writerows(values)
    write(args.output/'completion.json',dict(status='complete',parameter_comparisons=len(rows),
        inheritance_comparisons=len(inherit),training=False,source_mutations=0))


@torch.no_grad()
def forward_features(model, images):
    fused,views=model.encode_lanes_with_views(images,all_seen_lanes=True)
    lane_ids=model._lane_ids(True)
    logits,branches=model._lane_logits_with_views(fused,views,images,lane_ids)
    mask=model.task_class_mask[lane_ids]
    combine=lambda z: (z*mask.unsqueeze(0)).sum(1)[:,:model.seen_classes].float().cpu()
    return dict(features={'fused':fused.float().cpu(),**{k:v.float().cpu() for k,v in views.items()}},
        logits={'fused':combine(logits),**{k:combine(v) for k,v in branches.items()}})


def probe(args, model, config):
    source=make_source(config,args.data_root,args.face_manifest_root,'train')
    details=[]
    for task in TASKS:
        view=dataset_view(source,seen_indices(task),include_sample_id=True)
        indices=grouped_batches([source.sample_ids[i] for i in view.indices],64,2,20261025+task)
        batches=[]
        for selected in indices:
            data=LabelView(source,[view.indices[i] for i in selected],seen_indices(task),include_sample_id=True)
            images,targets,ids=next(iter(DataLoader(data,batch_size=len(data),shuffle=False,num_workers=0)))
            batches.append((images,targets,list(ids)))
        load_host(model,args.source_root/args.arm/'train',config,task)
        base=_torch_load(args.source_root/'BASE/train/compact_checkpoints'/f'task{task}.pth')['model']
        reference=[]
        with restore_components(model,base,'ASPH'):
            for images,_,_ in batches:reference.append(forward_features(model,move_model_inputs(images,'cuda')))
        for condition in FACTORIAL:
            with restore_components(model,base,condition):
                for batch,(images,targets,ids) in enumerate(batches):
                    values=forward_features(model,move_model_inputs(images,'cuda'))
                    changes={}
                    for name in VIEWS:
                        f=values['features'][name];b=reference[batch]['features'][name]
                        logits=values['logits'][name];baseline_logits=reference[batch]['logits'][name];y=targets
                        if name=='face':
                            valid=images['face_reliable'].bool()
                            f,b,logits,baseline_logits,y=f[valid],b[valid],logits[valid],baseline_logits[valid],y[valid]
                        if not len(f):
                            changes[name]=dict(samples=0);continue
                        changes[name]=dict(feature_relative_l2=float((f-b).norm()/b.norm().clamp_min(1e-12)),
                            feature_mean_cosine=float(F.cosine_similarity(f,b,dim=-1).mean()),
                            logit_mean_abs_change=float((logits-baseline_logits).abs().mean()),
                            bce=float(F.binary_cross_entropy_with_logits(logits,y)),samples=len(f))
                    details.append(dict(arm=args.arm,task=task,condition=condition,batch=batch,samples=len(ids),
                        sample_ids=ids,pixel_sha256=tensor_state_digest({k:images[k] for k in (*VIEWS[:3],'face_reliable')}),views=changes))
        print('TRAIN_PROBE_COMPLETE',args.arm,task,flush=True)
    write(args.output/'probe.json',dict(status='complete',records=details,precision='deterministic_fp32',
        split='train',transform='evaluation_transform_fixed_pixels',parameter_patterns=16,training=False))


def verify_anchor(args, condition, directory):
    if condition=='ASPH':reference=args.source_root/'BASE'/('historical_'+args.split)
    elif args.split=='test':reference=args.bypass_root/args.arm/'off'
    else:reference=args.source_root/args.arm/'runtime_off'
    with np.load(directory/'scores/task7.npz') as a,np.load(reference/'scores/task7.npz') as b:
        assert all(np.array_equal(a[k],b[k]) for k in ('sample_ids','targets','class_indices','logits','probabilities'))
    with np.load(directory/'view_scores/task7.npz') as a,np.load(reference/'view_scores/task7.npz') as b:
        assert all(np.array_equal(a[name+suffix],b[name+suffix]) for name in VIEWS for suffix in ('_logits','_probabilities'))


def evaluate(args, model, config):
    source=make_source(config,args.data_root,args.face_manifest_root,args.split)
    view=dataset_view(source,seen_indices(7),include_sample_id=True)
    loader=evaluation_loader(view,64,config,workers=2)
    load_host(model,args.source_root/args.arm/'train',config,7)
    base=_torch_load(args.source_root/'BASE/train/compact_checkpoints/task7.pth')['model']
    conditions=('none','ASPH') if args.mode=='anchors' else ('A','S','P','H','ASP')
    records=[]
    for condition in conditions:
        directory=args.output/condition;directory.mkdir()
        with restore_components(model,base,condition) as patch:
            torch.cuda.synchronize();start=time.time()
            row,views=evaluate_fused_and_views(model,loader,torch.device('cuda'),7,config['threshold'],True,
                directory/'scores/task7.npz',directory/'view_scores/task7.npz')
            torch.cuda.synchronize()
            from dataclasses import asdict
            value=dict(condition=condition,metrics=asdict(row),view_diagnostics=views,
                memory_patch=patch,elapsed_seconds=time.time()-start)
        if args.mode=='anchors':verify_anchor(args,condition,directory)
        write(directory/'result.json',value);records.append(value)
        print('EVALUATION_COMPLETE',args.arm,args.split,condition,row.mAP,flush=True)
    write(args.output/'evaluation.json',dict(status='complete',arm=args.arm,split=args.split,
        task=7,conditions=records,anchors_exact=args.mode=='anchors',training=False))


def run_worker(args):
    if args.output.exists():raise FileExistsError(args.output)
    if args.mode=='drift':return audit_drift(args)
    if args.arm not in ARMS:raise ValueError('Unknown arm')
    args.output.mkdir(parents=True)
    if args.mode=='probe':
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
    config=read(args.source_root/args.arm/'train/config.json')
    assert validate_config(config)==args.arm and config['git']['commit']==MODEL_HEAD and not config['git']['dirty']
    assert read(args.source_root/args.arm/'train/seed_summary.json')['status']=='complete'
    set_seed(1,tf32=args.mode!='probe')
    files=[args.source_root/arm/'train/compact_checkpoints'/f'task{t}.pth'
           for arm in ('BASE',args.arm) for t in TASKS]
    before={str(p):_sha256(p) for p in files}
    model=build(config,args.clip_checkpoint);visual=tensor_state_digest(model.visual_encoder.state_dict())
    if args.mode=='probe':probe(args,model,config)
    else:evaluate(args,model,config)
    assert all(_sha256(Path(p))==value for p,value in before.items())
    assert tensor_state_digest(model.visual_encoder.state_dict())==visual
    write(args.output/'completion.json',dict(status='complete',source_checkpoints_unchanged=True,
        frozen_visual_unchanged=True,source_sha256=before,training=False,git=git_metadata(Path(__file__).resolve().parents[2])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('drift','probe','anchors','evaluate'),required=True)
    p.add_argument('--arm',choices=ARMS);p.add_argument('--split',choices=('val','test'))
    for name in ('source-root','bypass-root','data-root','clip-checkpoint','face-manifest-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    run_worker(p.parse_args())
