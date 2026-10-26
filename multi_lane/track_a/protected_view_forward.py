"""Preserve complete source functions and fuse selected view FEATURES before the B0 head."""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _sha256, _torch_load
from .head_anchored_eight import copy_classifier_head, runtime
from .historical_b32_backbone_attribution import build, method_state
from .historical_b32_modules import validate_config
from .paired_transforms import move_model_inputs
from .run_historical_b32_test_attribution import read, write
from .runner import TaskMetrics, compute_metrics, dataset_view, git_metadata, seen_indices, set_seed, summarize_tasks, tensor_state_digest

VIEWS=('full','person','face')
ARMS={'R0':('A','A','A'),'R1':('A','A','B'),'R2':('A','B','A'),
      'R3':('A','B','B'),'R4':('B','A','A')}


class ProtectedViewFusion(nn.Module):
    """Inference-only reference implementation, sharing weights, not Frozen Forward tokens.

    Each source computes its OWN summaries and cross-view attention. Selecting
    final features never substitutes A's Full summaries inside B's attention.
    """
    def __init__(self, a, b=None):
        super().__init__();self.a=a;self.b=b
        if b is not None:
            if a._task_sizes!=b._task_sizes or a.view_fusion!=b.view_fusion:
                raise ValueError('Incompatible source architectures')
            if not all(torch.equal(v,b.visual_encoder.state_dict()[k])
                       for k,v in a.visual_encoder.state_dict().items()):
                raise ValueError('Frozen visual weights differ')
            if not all(torch.equal(v,b.head.state_dict()[k]) for k,v in a.head.state_dict().items()):
                raise ValueError('Both sources must have the same B0 head')
            b.visual_encoder=a.visual_encoder;b.head=a.head
        self.requires_grad_(False);self.eval()

    def combine(self, logits):
        masks=self.a.task_class_mask[self.a._lane_ids(True)].to(dtype=logits.dtype)
        return (logits*masks.unsqueeze(0)).sum(dim=1)[:,:self.a.seen_classes]

    def sources(self, images, need_b=True):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Protected view composition is no-grad evaluation only')
        features={};outputs={}
        for key,model in (('A',self.a),('B',self.b)):
            if key=='B' and not need_b:continue
            if model is None:raise ValueError('Source B is required')
            fused,views=model.encode_lanes_with_views(images,all_seen_lanes=True)
            z,zv=model._lane_logits_with_views(fused,views,images,model._lane_ids(True))
            features[key]=views
            outputs[key]={'fused':self.combine(z),**{v:self.combine(zv[v]) for v in VIEWS}}
        return features,outputs

    def compose(self, images, features, arm):
        choices=ARMS[arm]
        selected={v:features[source][v] for v,source in zip(VIEWS,choices)}
        fused,weights=self.a.view_fusion_module(selected,self.a._lane_ids(True),images['face_reliable'])
        z,zv=self.a._lane_logits_with_views(fused,selected,images,self.a._lane_ids(True))
        return {'fused':self.combine(z),**{v:self.combine(zv[v]) for v in VIEWS}},weights

    def forward(self, images, arm='R0'):
        features,_=self.sources(images,need_b=arm!='R0')
        return self.compose(images,features,arm)[0]


def source_configuration(args):
    a=read(args.person_run/'config.json');b=read(args.dual_run/'config.json')
    assert validate_config(a)==validate_config(b)=='ATTENTION_BOUND'
    assert not a.get('cross_view_face_attention',False) and b['cross_view_face_attention']
    assert a['seed']==b['seed']==1 and a['max_tasks']==b['max_tasks']==8
    assert validate_config(read(args.source_run/'config.json'))=='BASE'
    return a,b


def models(args, need_b=True):
    ca,cb=source_configuration(args);a=build(ca,args.clip_checkpoint)
    b=build(cb,args.clip_checkpoint) if need_b else None
    # Defer sharing until the corresponding-stage B0 heads have been installed.
    return a,b


def install(args,a,b,task):
    audit={};head=_torch_load(args.source_run/'compact_checkpoints'/f'task{task}.pth')
    assert head['task_id']==task and head['source_git']==read(args.source_run/'config.json')['git']
    for key,model,root in (('A',a,args.person_run),('B',b,args.dual_run)):
        if model is None:continue
        p=root/'compact_checkpoints'/f'task{task}.pth';payload=_torch_load(p)
        assert payload['task_id']==task and payload['source_git']==read(root/'config.json')['git']
        missing=model.load_state_dict(payload['model'],strict=False)
        assert not missing.unexpected_keys and all(k.startswith('visual_encoder.') for k in missing.missing_keys)
        copy_classifier_head(model,head['model']);model.restore_task(task);model.requires_grad_(False).eval()
        audit[key]=dict(path=str(p),sha256=_sha256(p))
    audit['head']=dict(path=str(args.source_run/'compact_checkpoints'/f'task{task}.pth'),
                       sha256=_sha256(args.source_run/'compact_checkpoints'/f'task{task}.pth'))
    return audit


def references(args,split,task):
    result={}
    for key,arm in (('A','REFERENCE_PERSON'),('B','REFERENCE_DUAL')):
        path=args.reference_root/arm/('b0_'+split)/'on/view_scores'/f'task{task}.npz'
        with np.load(path,allow_pickle=False) as z:result[key]={k:z[k].copy() for k in z.files}
    for key in ('sample_ids','targets','class_indices','face_reliable','batch_lengths'):
        assert np.array_equal(result['A'][key],result['B'][key]),key
    return result


def verify_batch(ref,start,ids,targets,reliable,sources,composed):
    stop=start+len(ids)
    assert np.array_equal(np.asarray(ids),ref['A']['sample_ids'][start:stop])
    assert np.array_equal(targets.numpy(),ref['A']['targets'][start:stop])
    assert np.array_equal(reliable.numpy(),ref['A']['face_reliable'][start:stop])
    for source,outputs in sources.items():
        for view,z in outputs.items():
            actual=z.float().cpu().numpy();expected=ref[source][view+'_logits'][start:stop]
            assert np.array_equal(actual,expected),(source,view,start,float(abs(actual-expected).max()))
            assert np.array_equal(torch.sigmoid(z.float().cpu()).numpy(),ref[source][view+'_probabilities'][start:stop])
    for arm,outputs in composed.items():
        for view,source in zip(VIEWS,ARMS[arm]):
            assert torch.equal(outputs[view],sources[source][view]),(arm,view,'source function changed')
        if arm=='R0':assert torch.equal(outputs['fused'],sources['A']['fused'])
        if arm=='R1':
            invalid=~reliable.to(device=outputs['fused'].device)
            assert torch.equal(outputs['fused'][invalid],sources['A']['fused'][invalid]),'R1 invalid Face changed'


def batch_probabilities(z,batches):
    """Match historical CPU sigmoid BEFORE concatenation, including thread chunks."""
    if sum(batches)!=len(z) or any(n<=0 for n in batches):raise ValueError('Invalid batch lengths')
    rows=[];start=0
    for n in batches:
        rows.append(torch.sigmoid(z[start:start+n].clone()));start+=n
    return torch.cat(rows)


def write_scores(root,task,ids,y,reliable,rows,weights,batches):
    arrays=dict(schema_version=np.asarray(1,dtype=np.int64),task_id=np.asarray(task,dtype=np.int64),
        sample_ids=np.asarray(ids,dtype=np.str_),class_indices=np.arange(y.shape[1],dtype=np.int64),targets=y.numpy(),
        batch_lengths=np.asarray(batches,dtype=np.int64),probability_device=np.asarray('cpu'),
        probability_dtype=np.asarray('float32'),probability_operation=np.asarray('torch.sigmoid'),
        torch_version=np.asarray(str(torch.__version__)),face_reliable=reliable.numpy(),lane_weights=weights.numpy())
    metrics={}
    for view,z in rows.items():
        p=batch_probabilities(z,batches);assert torch.isfinite(z).all() and torch.isfinite(p).all()
        arrays[view+'_logits']=z.numpy();arrays[view+'_probabilities']=p.numpy()
        metrics[view]=asdict(compute_metrics(task,p,y,.5))
    if reliable.any():
        face_p=torch.from_numpy(arrays['face_probabilities'])
        metrics['face_reliable']=asdict(compute_metrics(task,face_p[reliable],y[reliable],.5))
    folder=root/'view_scores';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'task{task}.npz';np.savez_compressed(path,**arrays)
    score={k:v for k,v in arrays.items() if k not in ('face_reliable','lane_weights') and not k.endswith(('_logits','_probabilities'))}
    score.update(logits=arrays['fused_logits'],probabilities=arrays['fused_probabilities'])
    (root/'scores').mkdir(exist_ok=True);np.savez_compressed(root/'scores'/f'task{task}.npz',**score)
    write(root/f'task{task}.json',dict(metrics=metrics['fused'],views=metrics,
        sources=dict(zip(VIEWS,ARMS[root.name]))))
    return metrics['fused']


def reexport(args):
    """Repair only CPU probability export; never repeat or modify GPU forwards."""
    git=git_metadata(Path(__file__).resolve().parents[2]);assert not git['dirty']
    assert torch.get_num_threads()==72
    original=read(args.input_root/'completion.json');assert original['status']=='complete'
    args.output.mkdir(parents=True,exist_ok=False);audit=[];summaries={}
    for split in ('val','test'):
        prior=read(args.input_root/'evaluation'/split/'result.json');results={arm:[] for arm in ARMS}
        for task in range(8):
            ref=references(args,split,task)
            for arm,choices in ARMS.items():
                path=args.input_root/'evaluation'/split/arm/'view_scores'/f'task{task}.npz'
                with np.load(path,allow_pickle=False) as saved:z={k:saved[k].copy() for k in saved.files}
                rows={v:torch.from_numpy(z[v+'_logits']) for v in ('fused',*VIEWS)}
                parent=args.output/'evaluation'/split/arm
                result=write_scores(parent,task,z['sample_ids'].tolist(),torch.from_numpy(z['targets']),
                    torch.from_numpy(z['face_reliable']),rows,torch.from_numpy(z['lane_weights']),z['batch_lengths'].tolist())
                with np.load(parent/'view_scores'/f'task{task}.npz') as actual:
                    assert all(np.array_equal(actual[v+'_logits'],z[v+'_logits']) for v in ('fused',*VIEWS))
                    for view,source in zip(VIEWS,choices):
                        for suffix in ('_logits','_probabilities'):
                            assert np.array_equal(actual[view+suffix],ref[source][view+suffix]),(split,task,arm,view+suffix)
                    if arm=='R0':
                        assert np.array_equal(actual['fused_probabilities'],ref['A']['fused_probabilities'])
                    if arm=='R1':
                        invalid=~actual['face_reliable']
                        assert np.array_equal(actual['fused_probabilities'][invalid],ref['A']['fused_probabilities'][invalid])
                results[arm].append(result)
                audit.append(dict(split=split,task=task,arm=arm,source_forward_npz=str(path),sha256=_sha256(path),
                    logits_unchanged=True,selected_source_probabilities_exact=True))
        summaries[split]={arm:summarize_tasks([TaskMetrics(**r) for r in rows]) for arm,rows in results.items()}
        write(args.output/'evaluation'/split/'result.json',dict(prior,metrics=results,summaries=summaries[split],
            forward_git=prior['git'],export_git=git,batch_sigmoid_export=True))
    rows=[]
    for arm in ARMS:
        row=dict(arm=arm,Full=ARMS[arm][0],Person=ARMS[arm][1],Face=ARMS[arm][2])
        for split,values in summaries.items():
            row.update({split+'_'+k:v for k,v in values[arm].items() if k in ('final_mAP','average_mAP','final_cF1','final_oF1')})
        rows.append(row)
    with (args.output/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(args.output/'resource_summary.json',read(args.input_root/'resource_summary.json'))
    assert all(_sha256(Path(p))==sha for p,sha in original['source_checkpoint_sha256'].items())
    assert all(_sha256(Path(r['source_forward_npz']))==r['sha256'] for r in audit)
    write(args.output/'export_audit.json',dict(status='verified',forward_files=audit,
        original_forward_files_unchanged=True,model_execution=False,training_updates=0,R0_probability_exact=True,R1_invalid_exact=True))
    write(args.output/'completion.json',dict(original,export_head=git['commit'],source_forward_root=str(args.input_root),
        export_only=True,batch_sigmoid_export=True,original_model_forward_preserved=True))
    print('BATCH_EXPORT_REPAIRED',args.output,flush=True)


def evaluate(args):
    git=runtime();set_seed(1,True);start=time.time();a,b=models(args)
    before_visual=tensor_state_digest(a.visual_encoder.state_dict());model=None
    data=make_source(read(args.source_run/'config.json'),args.data_root,args.face_manifest_root,args.split)
    tasks=[0,2,7] if args.action=='preflight' else list(range(8));results={arm:[] for arm in ARMS};audits=[]
    for task in tasks:
        source=install(args,a,b,task)
        if model is None:model=ProtectedViewFusion(a,b)
        protected={key:tensor_state_digest(method_state(m)) for key,m in (('A',a),('B',b))}
        ref=references(args,args.split,task)
        loader=DataLoader(dataset_view(data,seen_indices(task),True),batch_size=64,shuffle=False,num_workers=2,
            pin_memory=False,generator=torch.Generator().manual_seed(87001+task))
        rows={arm:{v:[] for v in ('fused',*VIEWS)} for arm in ARMS};ids=[];labels=[];reliability=[];weights=[];batches=[]
        with torch.no_grad():
            for images,y,batch_ids in loader:
                reliable=images['face_reliable'].bool().cpu()
                x=move_model_inputs(images,torch.device('cuda'))
                with torch.cuda.amp.autocast():
                    features,sources=model.sources(x)
                    outputs={}
                    for arm in ARMS:outputs[arm],w=model.compose(x,features,arm)
                verify_batch(ref,len(ids),batch_ids,y,reliable,sources,outputs)
                for arm in ARMS:
                    for v,z in outputs[arm].items():rows[arm][v].append(z.float().cpu())
                ids.extend(batch_ids);labels.append(y.float());reliability.append(reliable);weights.append(w.float().cpu());batches.append(len(y))
                if args.action=='preflight':break
        assert len(ids)==(64 if args.action=='preflight' else len(ref['A']['sample_ids']))
        for key,m in (('A',a),('B',b)):assert protected[key]==tensor_state_digest(method_state(m))
        for arm in ARMS:
            result=write_scores(args.output/arm,task,ids,torch.cat(labels),torch.cat(reliability),
                {v:torch.cat(z) for v,z in rows[arm].items()},torch.cat(weights),batches)
            results[arm].append(result)
            print('EVALUATE',args.split,'task',task,arm,result['mAP'],flush=True)
        audit=dict(task=task,people=len(ids),sources=source,source_predictions_exact=True,
            selected_views_exact=True,R0_exact=True,R1_invalid_exact=True,parameters_unchanged=True)
        audits.append(audit);write(args.output/f'task{task}_audit.json',audit)
    assert tensor_state_digest(a.visual_encoder.state_dict())==before_visual
    write(args.output/'result.json',dict(status='complete',action=args.action,git=git,split=args.split,seed=1,
        tasks=tasks,source_predictions_exact=True,selected_views_exact=True,R0_exact=True,R1_invalid_exact=True,
        training_updates=0,visual_weights_shared=True,frozen_tokens_cached=False,
        evaluation_scope='samples_intersect_seen_classes',test_used_for_parameter_search=True,
        metrics=results,summaries={arm:summarize_tasks([TaskMetrics(**r) for r in rows]) for arm,rows in results.items()}
            if args.action=='eval' else {},audits=audits,elapsed_seconds=time.time()-start,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))


def benchmark(args):
    git=runtime();set_seed(1,True);a,b=models(args,need_b=args.arm!='R0');source=install(args,a,b,7)
    model=ProtectedViewFusion(a,b)
    data=make_source(read(args.source_run/'config.json'),args.data_root,args.face_manifest_root,'val')
    images,_,_=next(iter(DataLoader(dataset_view(data,seen_indices(7),True),batch_size=64,shuffle=False,num_workers=0,
        generator=torch.Generator().manual_seed(87008))))
    x=move_model_inputs(images,torch.device('cuda'))
    with torch.no_grad(),torch.cuda.amp.autocast():
        for _ in range(3):model(x,args.arm)
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        wall=time.perf_counter();start.record()
        for _ in range(10):model(x,args.arm)
        end.record();torch.cuda.synchronize()
        gpu_ms=start.elapsed_time(end)/10;wall_seconds=(time.perf_counter()-wall)/10
        peak=torch.cuda.max_memory_allocated()/1024**2;reserved=torch.cuda.max_memory_reserved()/1024**2
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                    with_flops=True) as profile:model(x,args.arm)
    flops=sum(e.flops for e in profile.key_averages())
    visual=sum(p.numel() for p in a.visual_encoder.parameters());total=sum(p.numel() for p in model.parameters())
    write(args.output/'result.json',dict(status='complete',git=git,arm=args.arm,task=7,batch_size=64,warmup=3,repeats=10,
        sources=source,training_updates=0,visual_parameters=visual,total_stored_parameters=total,
        stored_nonvisual_parameters=total-visual,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        visual_weights_shared=True,frozen_tokens_cached=False,source_passes=1 if args.arm=='R0' else 2,
        frozen_view_forwards=3 if args.arm=='R0' else 6,gpu_milliseconds_per_batch=gpu_ms,
        wall_seconds_per_batch=wall_seconds,people_per_second=64/(gpu_ms/1000),
        peak_allocated_mib=peak,peak_reserved_mib=reserved,profile_reported_flops_per_batch=flops,
        profiler_scope='covered matrix multiplication and convolution; not all operators',
        baseline_memory_measured_in_separate_process=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('preflight','eval','benchmark','reexport'))
    for name in ('source-run','person-run','dual-run','reference-root','data-root','clip-checkpoint','face-manifest-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=('val','test'),default='val');p.add_argument('--arm',choices=tuple(ARMS),default='R0')
    p.add_argument('--input-root',type=Path)
    args=p.parse_args();(reexport if args.action=='reexport' else benchmark if args.action=='benchmark' else evaluate)(args)
