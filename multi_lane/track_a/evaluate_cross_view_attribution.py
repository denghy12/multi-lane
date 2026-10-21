"""Paired residual-off and within-image Query attribution, without optimization."""
import argparse
from dataclasses import asdict
import json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .cross_view_query_pairs import AttributionDataset, plan_for_source
from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import build_model, _torch_load, _sha256, restore_compact_model_state
from .openai_clip_loader import load_openai_clip_visual
from .paired_transforms import move_model_inputs
from .runner import (compute_metrics, compact_model_state_dict, tensor_state_digest,
                     git_metadata, set_seed, seen_indices, summarize_tasks)


def read(path): return json.loads(path.read_text())
def write(path,value): path.write_text(json.dumps(value,indent=2)+'\n')


def audit_source(source):
    c=read(source/'config.json');s=read(source/'seed_summary.json');h=read(source/'training_history.json')
    assert s['status']=='complete' and s['config']==c and c['git']['dirty'] is False
    assert c['seed']==0 and c['max_tasks']==8 and c['save_compact_checkpoints']
    assert c['adapter_mode']==c['parax_mode']==c['joint_summary_mode']=='disabled'
    assert c['cross_view_mode'] in ('task_full','person_full')
    assert c['cross_view_layer']==5 and c['view_auxiliary_loss_weight']==0
    assert c['eval_batch_size']==64 and c['threshold']==.5
    assert c['view_fusion']=='fixed_three_view' and c['view_classifier_mode']=='shared_post_fusion'
    assert sum(r['optimizer_steps'] for rows in h.values() for r in rows)==13950
    assert sum(r['skipped_optimizer_steps'] for rows in h.values() for r in rows)==0
    assert read(source.parent/'verification.json')['status']=='verified'
    assert read(source/'paired_protocol_audit.json')['frozen_visual_unchanged']
    return c


def compare_archive(arrays, source, split, task, partial=False):
    folder='full_val' if split=='val' else 'full_test'
    with np.load(source.parent/folder/'scores'/f'task{task}.npz') as ref:
        n=len(arrays['sample_ids'])
        assert partial or len(ref['sample_ids'])==n
        for name in ('sample_ids','targets','class_indices','logits','probabilities'):
            value=ref[name] if name=='class_indices' else ref[name][:n]
            assert np.array_equal(arrays[name],value),('Unmodified replay mismatch',task,name)
    with np.load(source.parent/folder/'view_scores'/f'task{task}.npz') as ref:
        for view in ('full','person','face'):
            assert np.array_equal(arrays[view+'_logits'],ref[view+'_logits'][:n]),(task,view)
    return dict(status='bitwise_identical_to_archived',samples=n)


def attribution_inputs(images, device):
    # The historical mover casts every field to float. Preserve this new
    # intervention mask separately while leaving all legacy fields unchanged.
    mask=images.get('cross_view_query_mask')
    if mask is None:return move_model_inputs(images,device)
    if mask.dtype!=torch.bool:raise ValueError('Query intervention mask must be boolean')
    moved=move_model_inputs({k:v for k,v in images.items() if k!='cross_view_query_mask'},device)
    moved['cross_view_query_mask']=mask.to(device,non_blocking=True)
    return moved


def evaluate_condition(model,source,plan,classes,condition,device,batch_size,smoke_batches):
    swap=condition=='query_swap'; model.cross_view_runtime_enabled=condition!='residual_off'
    model.cross_view_bank.reset_statistics()
    loader=DataLoader(AttributionDataset(source,plan,classes,swap),batch_size=batch_size,
                      shuffle=False,num_workers=0,pin_memory=False,drop_last=False)
    ids=[];targets=[];indices=[];logits={name:[] for name in ('fused','full','person','face')}
    identity_checked=False
    with torch.no_grad():
        for batch_index,(images,y,sample_ids,index) in enumerate(loader):
            if smoke_batches and batch_index>=smoke_batches:break
            images=attribution_inputs(images,device)
            with torch.cuda.amp.autocast(): fused,views=model.seen_logits_with_views(images)
            if condition=='on' and model.cross_view_mode=='person_full' and batch_index==0:
                identity=dict(images,cross_view_query_donor=images['person'],
                              cross_view_query_mask=torch.ones(len(y),dtype=torch.bool,device=device))
                with torch.cuda.amp.autocast(): control,control_views=model.seen_logits_with_views(identity)
                assert torch.equal(fused,control) and all(torch.equal(views[v],control_views[v]) for v in views),'Identity Query changed predictions'
                identity_checked=True
                # Identity has duplicate observer calls. Reset, then retain the
                # real first batch only; replay does not modify model state.
                model.cross_view_bank.reset_statistics()
                with torch.cuda.amp.autocast(): fused,views=model.seen_logits_with_views(images)
            for name,value in dict(fused=fused,**views).items():logits[name].append(value.float().cpu())
            ids.extend(sample_ids);targets.append(y.float());indices.extend(index.tolist())
    assert ids
    arrays={name:torch.cat(values).numpy() for name,values in logits.items()}
    target=torch.cat(targets)
    eligible=np.array([plan['rows'][i]['eligible'] for i in indices],dtype=bool)
    result={};probabilities={}
    for name,values in arrays.items():
        probs=torch.sigmoid(torch.from_numpy(values));probabilities[name]=probs.numpy()
        result[name]=asdict(compute_metrics(model.current_task_id,probs,target,.5))
    reliable=np.asarray([source.face_reliable[i] for i in indices],dtype=bool)
    subsets={}
    for label,mask in [('eligible',eligible),('ineligible',~eligible),('face_reliable',reliable),('face_unreliable',~reliable)]:
        subsets[label]=dict(samples=int(mask.sum()),metrics={name:asdict(compute_metrics(
            model.current_task_id,torch.from_numpy(probabilities[name][mask]),target[mask],.5))
            for name in probabilities} if mask.any() else {})
    saved=dict(schema_version=np.asarray(1),task_id=np.asarray(model.current_task_id),
               sample_ids=np.asarray(ids,dtype=np.str_),class_indices=np.asarray(classes),
               targets=target.numpy(),eligible=eligible,face_reliable=reliable,logits=arrays['fused'],
               probabilities=probabilities['fused'])
    for name in ('full','person','face'):
        saved[name+'_logits']=arrays[name];saved[name+'_probabilities']=probabilities[name]
    return saved,dict(metrics=result,subsets=subsets,identity_query_bitwise_checked=identity_checked,
                      cross_view_statistics=model.cross_view_bank.statistics())


def evaluate(args):
    if args.output.exists():raise FileExistsError(args.output)
    if args.split=='test' and (args.smoke_batches or args.checkpoint_tasks!=list(range(8))):
        raise ValueError('Test requires full8 complete evaluation')
    assert args.checkpoint_tasks and len(set(args.checkpoint_tasks))==len(args.checkpoint_tasks)
    assert all(0<=t<8 for t in args.checkpoint_tasks)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    c=audit_source(args.source_run);meta=git_metadata(Path(__file__).resolve().parents[2])
    assert not meta['dirty']
    assert Path(c['data_root']).resolve()==args.data_root.resolve()
    assert _sha256(args.clip_checkpoint)==c['clip_checkpoint_sha256']
    source_files=[args.source_run/n for n in ('config.json','seed_summary.json','training_history.json','cross_view_audit.json','paired_protocol_audit.json')]
    source_files += [args.source_run/'compact_checkpoints'/f'task{t}.pth' for t in args.checkpoint_tasks]
    hashes={str(p.resolve()):_sha256(p) for p in source_files}
    manifest_files=[args.face_manifest_root/'manifests'/f'{s}.jsonl' for s in ('train','val','test')]
    manifests={str(p.resolve()):_sha256(p) for p in manifest_files}
    for split in ('train','val'):
        assert _sha256(args.face_manifest_root/'manifests'/f'{split}.jsonl')==c['face_manifest']['artifact_sha256'][split+'_manifest']
    set_seed(0,tf32=True);device=torch.device('cuda')
    model=build_model(c,load_openai_clip_visual(args.clip_checkpoint)).float().to(device)
    model.eval().requires_grad_(False);visual_before=tensor_state_digest(model.visual_encoder.state_dict())
    source=make_source(c,args.data_root,args.face_manifest_root,args.split)
    assert len(source)==(2397 if args.split=='val' else 5368)
    plan=plan_for_source(source,args.minimum_visibility)
    args.output.mkdir(parents=True);write(args.output/'query_pairs.json',plan)
    modes=['on','residual_off']+(['query_swap'] if c['cross_view_mode']=='person_full' else [])
    for condition in modes:(args.output/condition).mkdir()
    rows={mode:[] for mode in modes};details={mode:{} for mode in modes};replay={};start=time.time()
    for task in args.checkpoint_tasks:
        payload=_torch_load(args.source_run/'compact_checkpoints'/f'task{task}.pth')
        restore_compact_model_state(model,payload,task,c['git']);model.eval().requires_grad_(False)
        before=tensor_state_digest(compact_model_state_dict(model));reference=None
        for condition in modes:
            arrays,extra=evaluate_condition(model,source,plan,seen_indices(task),condition,device,args.batch_size,args.smoke_batches)
            if condition=='on':
                replay[str(task)]=compare_archive(arrays,args.source_run,args.split,task,bool(args.smoke_batches));reference=arrays
            else:
                assert np.array_equal(arrays['sample_ids'],reference['sample_ids'])
                assert np.array_equal(arrays['targets'],reference['targets'])
                for view in ('full','face'):
                    assert np.array_equal(arrays[view+'_logits'],reference[view+'_logits']),(task,condition,view)
                if condition=='query_swap':
                    mask=~arrays['eligible']
                    assert np.array_equal(arrays['logits'][mask],reference['logits'][mask]),'Noneligible Query intervention changed output'
                    assert np.array_equal(arrays['person_logits'][mask],reference['person_logits'][mask])
            np.savez_compressed(args.output/condition/f'task{task}.npz',**arrays)
            rows[condition].append(extra['metrics']['fused']);details[condition][str(task)]=extra
            print(f'split={args.split} model={c["cross_view_mode"]} task={task} condition={condition} mAP={extra["metrics"]["fused"]["mAP"]:.6f} eligible={int(arrays["eligible"].sum())}',flush=True)
        assert tensor_state_digest(compact_model_state_dict(model))==before,'Method weights changed'
        write(args.output/'progress.json',dict(completed_tasks=list(replay),rows=rows))
    assert visual_before==tensor_state_digest(model.visual_encoder.state_dict())
    assert all(_sha256(Path(p))==sha for p,sha in {**hashes,**manifests}.items()),'Source artifacts changed'
    from .runner import TaskMetrics
    summaries={mode:summarize_tasks([TaskMetrics(**r) for r in values]) for mode,values in rows.items()}
    result=dict(status='complete',seed=0,split=args.split,source_run=str(args.source_run.resolve()),
                source_config=c,evaluator_git=meta,checkpoint_tasks=args.checkpoint_tasks,
                training_performed=False,optimizer_updates=0,test_used_for_selection=False,
                smoke=bool(args.smoke_batches),full_cohort=not bool(args.smoke_batches),
                source_cohort_samples=len(source),query_eligible_people=plan['eligible_people'],
                query_eligible_images=plan['eligible_images'],minimum_visibility=args.minimum_visibility,
                modes=modes,task_metrics=rows,summary=summaries,details=details,archived_replay=replay,
                source_sha256=hashes,face_manifest_sha256=manifests,source_unchanged=True,
                method_weights_unchanged=True,frozen_visual_unchanged=True,
                full_and_face_endpoints_bitwise_unchanged=True,
                noneligible_query_swap_bitwise_unchanged=True,elapsed_seconds=time.time()-start)
    write(args.output/'result.json',result);print('CROSS_VIEW_ATTRIBUTION_COMPLETE',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','output','data-root','clip-checkpoint','face-manifest-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=('val','test'),required=True)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--checkpoint-tasks',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--smoke-batches',type=int,default=0)
    p.add_argument('--minimum-visibility',type=float,default=.9)
    a=p.parse_args()
    if a.batch_size!=64 or a.smoke_batches<0 or a.minimum_visibility!=.9:p.error('Locked attribution protocol: batch64, visibility90%, nonnegative smoke')
    evaluate(a)

if __name__=='__main__':main()
