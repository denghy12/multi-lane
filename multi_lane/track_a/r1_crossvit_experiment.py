"""Archived R1 recipe adapters, complete validation, and fixed-domain comparisons."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from .runner import (TASK_SIZES,TaskMetrics,build_transforms,compute_metrics,dataset_view,
    evaluate,git_metadata,seen_indices,set_seed,summarize_tasks)
from .r1_crossvit import R1Receiver,attach_evidence
from .openai_clip_loader import load_openai_clip_visual
from .evaluation_scores import load_evaluation_scores,align_evaluation_scores
from .fuse_face_endpoint_validation import load_face_reliability
from .fuse_fixed_three_view_test import LOCKED_RELIABLE_WEIGHTS,LOCKED_FALLBACK_WEIGHTS
from .three_view_router import _align_three
from multi_lane.continual_datasets.continual_datasets import EMOTIC

VIEWS=('full','person','face')
FAMILIES=('native','attention','pool')


def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_runs(reference_root):
    # Historical seed1 sources; no learned parameter is loaded from these runs.
    dual=reference_root/'multi_lane_dual_view_seed_confirmation_v0.1/dual_view_locked_formal_seed12_20260903_174938/seed1'
    return dict(full=dual/'dual_view_locked_formal_seed12_20260903_174938_seed1_full_anchor',
        person=dual/'dual_view_locked_formal_seed12_20260903_174938_seed1_person_letterbox',
        face=reference_root/'multi_lane_fixed_three_view_test_v0.1/fixed_three_view_seed012_test_20260909_213052/face_sources/fixed_three_view_seed012_test_20260909_213052_seed1_face')


def flags(args,kind,view,smoke=False):
    src=source_runs(args.reference_root)[view];c=read(src/'config.json')
    if c['seed']!=1 or c['adapter_residual_scale']!=.1 or c['parameter_group_loss_routing']!='adapter_asl':
        raise ValueError('Historical R1 recipe changed')
    result=['--data-root',str(args.data_root),'--clip-checkpoint',str(args.clip_checkpoint),
        '--output-root',str(args.output/kind/view/'train'),
        '--seed','1','--epochs','30','--max-tasks','2' if smoke else '8',
        '--train-batch-size','64','--eval-batch-size','64','--workers','2',
        '--source-learning-rate','0.05','--source-reference-batch-size','256','--weight-decay','0',
        '--temperature','1','--scheduler-mode','cosine','--scheduler-min-lr-ratio','0','--scheduler-warmup-ratio','0',
        '--training-loss-mode','legacy_full_zero','--loss-routing','adapter_asl',
        '--asl-gamma-neg','9.8','--asl-gamma-pos','0','--asl-clip','0.05','--asl-eps','1e-8',
        '--adapter-mode','image_token','--adapter-bottleneck-dim','32','--adapter-layer-indices','1',
        '--adapter-residual-scale','0.1','--adapter-learning-rate','0.0004','--adapter-weight-decay','0',
        '--adapter-task-init','independent','--adapter-activation','relu',
        '--input-normalization','clip','--input-mode',c['input_mode'],
        '--person-crop-margin',str(c['person_crop_margin']),'--person-transform-mode',c['person_transform_mode'],
        '--person-color-jitter-strength',str(c['person_color_jitter_strength']),
        '--person-color-jitter-probability',str(c['person_color_jitter_probability']),
        '--train-crop-scale',*[str(v) for v in c['train_crop_scale']],
        '--no-save-checkpoints','--save-compact-checkpoints','--save-evaluation-scores',
        '--evaluation-score-purpose','validation_search' if smoke else 'fixed_test_fusion',
        '--reporting-split','val' if smoke else 'test',
        '--r1-kind',kind,'--r1-view',view]
    if view=='face':result+=['--face-manifest-root',str(args.face_manifest_root)]
    if kind!='native':result+=['--r1-evidence-manifest-root',str(args.face_manifest_root)]
    if smoke:result+=['--optimizer-updates-per-task','2']
    return result


def model_from_config(c,clip):
    model_class = R1Receiver
    extra = {}
    if c.get('r1_frozen_mode'):
        from .r1_frozen_cross_view import FrozenR1Receiver
        model_class = FrozenR1Receiver
        extra = dict(frozen_cross_mode=c['r1_frozen_mode'], frozen_cross_cap=c['r1_frozen_cap'],
                     frozen_cross_second_layer=c.get('r1_exchange_second_layer'),
                     frozen_cross_sharing=c.get('r1_exchange_sharing', 'shared'))
    return model_class(load_openai_clip_visual(clip),TASK_SIZES,
        num_selectors=c['num_selectors'],num_prompts=c['num_prompts'],num_prompt_layers=c['num_prompt_layers'],
        normalize=c['normalize'],adapter_mode='image_token',adapter_bottleneck_dim=32,
        adapter_layer_indices=(1,),adapter_residual_scale=.1,adapter_activation='relu',
        r1_kind=c['r1_kind'],r1_view=c['r1_receiver'],r1_sender_root=c.get('r1_sender_root'),
        r1_layer=c.get('r1_exchange_layer') if c.get('r1_exchange_layer') is not None else 5,
        r1_query_mode=c.get('r1_query_mode','cls'), **extra).float().cuda()


def single_thread_worker(_):torch.set_num_threads(1)


def validation(args):
    set_seed(1,True)
    train=args.training_root;c=read(train/'config.json');complete=read(train/'seed_summary.json')
    if complete['status']!='complete':raise ValueError('Incomplete training source')
    model=model_from_config(c,args.clip_checkpoint);torch.set_num_threads(2)
    _,transform=build_transforms(c['input_normalization'],c['train_crop_scale'],input_mode=c['input_mode'],
        person_transform_mode=c['person_transform_mode'],person_color_jitter_strength=c['person_color_jitter_strength'],
        person_color_jitter_probability=c['person_color_jitter_probability'])
    parent=args.data_root.parent if args.data_root.name=='EMOTIC' else args.data_root
    source=EMOTIC(str(parent),train=False,eval_splits=('val',),transform=transform,
        input_mode=c['input_mode'],person_crop_margin=c['person_crop_margin'],
        face_manifest_root=args.face_manifest_root if c['input_mode']=='face_crop' else None)
    if c['r1_kind']!='native':source=attach_evidence(source,c['r1_receiver'],parent,args.face_manifest_root)
    rows=[]
    for task in range(c['max_tasks']):
        path=train/'compact_checkpoints'/f'task{task}.pth'
        payload=torch.load(path,map_location='cpu',weights_only=False)
        missing=model.load_state_dict(payload['model'],strict=False)
        if missing.unexpected_keys or any(not n.startswith('visual_encoder.') for n in missing.missing_keys):raise ValueError('Missing state')
        model.restore_task(task);model.requires_grad_(False).eval()
        dataset=dataset_view(source,seen_indices(task),include_sample_id=True)
        if args.smoke:dataset=Subset(dataset,range(min(64,len(dataset))))
        loader=DataLoader(dataset,batch_size=64,shuffle=False,num_workers=2,prefetch_factor=1,
            worker_init_fn=single_thread_worker,generator=torch.Generator().manual_seed(77191+task))
        row=evaluate(model,loader,torch.device('cuda'),task,.5,True,
            score_output_path=args.output/'scores'/f'task{task}.npz')
        rows.append(row);write(args.output/'progress.json',dict(tasks_completed=[r.task_id for r in rows]))
        print('R1_VALIDATION',c['r1_kind'],c['r1_receiver'],task,row.mAP,flush=True)
    write(args.output/'result.json',dict(status='complete',kind=c['r1_kind'],view=c['r1_receiver'],split='val',
        seed=1,tasks=list(range(c['max_tasks'])),metrics=[asdict(r) for r in rows],summary=summarize_tasks(rows),
        evaluation_scope='samples_intersect_seen_classes'))


def fusion(root,kind,manifest,smoke=False):
    tasks=list(range(2 if smoke else 8));record={}
    for split in ('val',) if smoke else ('val','test'):
        reliability=load_face_reliability(manifest,split)
        metrics={m:[] for m in ('probability','logit')};views={v:[] for v in VIEWS}
        for task in tasks:
            arrays=[]
            for view in VIEWS:
                path=root/kind/view/('train/test_scores' if split=='test' else 'val/scores')/f'task{task}.npz'
                arrays.append(load_evaluation_scores(path,64))
            ids,y,f,p,r=_align_three(*arrays)
            _,fz,pz,_y,_,_=align_evaluation_scores(arrays[0],arrays[1])
            _,_,rz,_yy,_,_=align_evaluation_scores(arrays[0],arrays[2])
            mask=np.array([reliability[i] for i in ids],dtype=bool)
            w=np.where(mask[:,None],LOCKED_RELIABLE_WEIGHTS,LOCKED_FALLBACK_WEIGHTS)
            fused=w[:,0,None]*f+w[:,1,None]*p+w[:,2,None]*r
            logits=w[:,0,None]*fz+w[:,1,None]*pz+w[:,2,None]*rz
            for mode,scores in (('probability',fused),('logit',torch.from_numpy(logits).sigmoid().numpy())):
                metrics[mode].append(compute_metrics(task,torch.from_numpy(scores),torch.from_numpy(y),.5))
            for view,scores in zip(VIEWS,(f,p,r)):
                views[view].append(compute_metrics(task,torch.from_numpy(scores),torch.from_numpy(y),.5))
        record[split]=dict(primary='probability',summaries={m:summarize_tasks(rows) for m,rows in metrics.items()},
            metrics={m:[asdict(r) for r in rows] for m,rows in metrics.items()},
            view_summaries={v:summarize_tasks(rows) for v,rows in views.items()})
    write(root/kind/'fusion.json',dict(status='complete',kind=kind,seed=1,tasks=tasks,smoke=smoke,
        rule='R1_0.64_0.16_0.20_else_0.8_0.2_0',probability_primary=True,**record))
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('val','fuse'))
    for name in ('training-root','output','clip-checkpoint','data-root','face-manifest-root'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--kind',choices=FAMILIES);p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.action=='val':validation(a)
    else:fusion(a.output,a.kind,a.face_manifest_root,a.smoke)
