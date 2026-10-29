"""TRAIN-locked directions, full historical scoring and fixed-query interventions."""
import argparse
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _sha256,_torch_load
from .gradient_attribution import grouped_batches
from .historical_b32_backbone_attribution import method_state
from .paired_transforms import move_model_inputs
from .protected_joint_experiment import build,runtime,evaluation_loader,single_thread_worker
from .run_historical_b32_test_attribution import read,write
from .runner import compute_metrics,dataset_view,seen_indices,set_seed,tensor_state_digest

DIAGNOSTIC_TASKS=(0,2,6,7)


def match_direction_magnitude(reference,direction):
    length=direction.float().norm(dim=-1,keepdim=True)
    result=direction.float()*(reference.float().norm(dim=-1,keepdim=True)/length.clamp_min(1e-12))
    return torch.where(length>1e-12,result,reference.float())


def grouped_donors(ids,eligible):
    donor=torch.arange(len(ids),device=eligible.device);mask=torch.zeros_like(eligible,dtype=torch.bool)
    groups={}
    for i,s in enumerate(ids):
        if bool(eligible[i]):groups.setdefault(str(s).rsplit('#person=',1)[0],[]).append(i)
    for members in groups.values():
        if len(members)<2:continue
        for i,j in zip(members,members[1:]+members[:1]):donor[i]=j;mask[i]=True
    return donor,mask


def paired_batches(ids):
    groups={}
    for i,s in enumerate(ids):groups.setdefault(str(s).rsplit('#person=',1)[0],[]).append(i)
    batches=[];batch=[]
    for name in sorted(groups):
        members=groups[name]
        if len(members)<2:continue
        if len(members)>64:raise ValueError('One image exceeds paired batch capacity')
        if len(batch)+len(members)>64:batches.append(batch);batch=[]
        batch+=members
    if batch:batches.append(batch)
    return batches


def load_checkpoint(model,training,cfg,task):
    complete=read(training/'training_complete.json');path=training/'compact_checkpoints'/f'task{task}.pth'
    sha=_sha256(path);assert sha==complete['checkpoints'][str(task)]['sha256']
    saved=_torch_load(path);assert saved['source_git']==cfg['git'] and saved['task_id']==task
    missing=model.load_state_dict(saved['model'],strict=False)
    assert not missing.unexpected_keys and all(k.startswith('visual_encoder.') for k in missing.missing_keys)
    model.restore_task(task);model.requires_grad_(False).eval()
    return path,sha


def calibrate(training,output):
    started=time.time();git=runtime();set_seed(1,True)
    cfg=read(training/'config.json');complete=read(training/'training_complete.json')
    assert cfg['kind']=='protected_private_summary_2x2' and complete['config']==cfg and cfg['git']==git
    assert complete['status']=='complete'
    model=build(cfg['model_template'],Path(cfg['clip_checkpoint']),cfg['arm'])
    source=make_source(cfg['model_template'],Path(cfg['data_root']),Path(cfg['face_manifest_root']),'train')
    source.multi_view_transform.emit_spatial_geometry=True
    records={};visual=tensor_state_digest(model.visual_encoder.state_dict())
    for task in complete['tasks']:
        path,sha=load_checkpoint(model,training,cfg,task)
        before=tensor_state_digest(method_state(model))
        start,stop=sum(model._task_sizes[:task]),sum(model._task_sizes[:task+1])
        view=dataset_view(source,range(start,stop),True)
        groups=grouped_batches([source.sample_ids[i] for i in view.indices],64,1 if cfg['smoke'] else 4,929650+task)
        selected=[i for batch in groups for i in batch]
        loader=evaluation_loader(Subset(view,selected),task,2)
        rows=[];ids=[]
        with torch.no_grad():
            for images,_,names in loader:
                with torch.cuda.amp.autocast(enabled=True):
                    model.seen_logits_with_views(move_model_inputs(images,torch.device('cuda')))
                rows.append(model._native_delta[:,-1].double().cpu());ids.extend(names)
        mean=torch.cat(rows).mean(0)
        assert before==tensor_state_digest(method_state(model)) and sha==_sha256(path)
        records[str(task)]=dict(mean_direction=mean.tolist(),samples=len(ids),sample_ids=list(ids),
            original_images=len({str(s).rsplit('#person=',1)[0] for s in ids}),checkpoint_sha256=sha)
        print('PRIVATE_SUMMARY_TRAIN_LOCK',task,len(ids),flush=True)
    assert visual==tensor_state_digest(model.visual_encoder.state_dict())
    write(output/'result.json',dict(status='complete',tasks=complete['tasks'],git=git,source_git=cfg['git'],
        split='train',evaluation_data_read=False,training_updates=0,parameters_unchanged=True,
        training_config_sha256=_sha256(training/'config.json'),records=records,
        elapsed_seconds=time.time()-started))


def locked_constants(training):
    value=read(training/'summary_calibration/result.json')
    assert value['status']=='complete' and value['split']=='train' and not value['evaluation_data_read']
    assert value['training_config_sha256']==_sha256(training/'config.json')
    complete=read(training/'training_complete.json')
    assert value['tasks']==complete['tasks']
    for task,r in value['records'].items():
        assert r['checkpoint_sha256']==complete['checkpoints'][task]['sha256']
        assert r['checkpoint_sha256']==_sha256(training/'compact_checkpoints'/f'task{task}.pth')
    value['_calibration_file_sha256']=_sha256(training/'summary_calibration/result.json')
    return value


def evaluate_private_modes(model,loader,task,output,constants,workers,smoke):
    model.eval();names=['on','off','zero_query','no_face_memory','no_person_memory']
    if task in DIAGNOSTIC_TASKS:names+=['train_mean_direction','permuted_direction']
    zs={n:[] for n in names};ps={n:[] for n in names};ratios={n:[] for n in names}
    views={v:[] for v in ('full','person','face')};viewp={v:[] for v in views}
    ys=[];rs=[];ids=[];lengths=[];statistics=[]
    with torch.no_grad():
        for cpu_images,y,batch_ids in loader:
            images=move_model_inputs(cpu_images,torch.device('cuda'))
            with torch.cuda.amp.autocast(enabled=True):
                z,vs=model.seen_logits_with_views(images);native=model._native_delta.clone();base=model._interaction_cache['base']
                statistics.append(dict(samples=len(y),lanes={str(t):dict(model.interaction_bank[t].last_metrics) for t in model._interaction_cache['lane_ids']}))
                deltas=dict(on=native,off=torch.zeros_like(native),zero_query=model.interaction_delta(zero_query=True),
                    no_face_memory=model.interaction_delta(no_face=True),no_person_memory=model.interaction_delta(no_person=True))
                if task in DIAGNOSTIC_TASKS:
                    direction=base.new_tensor([constants['records'][str(t)]['mean_direction'] for t in model._interaction_cache['lane_ids']])[None]
                    deltas['train_mean_direction']=match_direction_magnitude(native,direction)
                    donor=torch.arange(len(y),device=base.device)
                    generator=torch.Generator().manual_seed(929800+task*1000+len(ids))
                    reliable=images['face_reliable'].bool();visible=images['spatial_visibility']>0
                    for r in (False,True):
                        for v in (False,True):
                            ix=torch.nonzero((reliable==r)&(visible==v)).flatten()
                            donor[ix]=ix[torch.randperm(len(ix),generator=generator).to(ix.device)]
                    deltas['permuted_direction']=match_direction_magnitude(native,native[donor])
                values={n:model.cached_base_seen_logits(d) for n,d in deltas.items()}
                if not torch.equal(values['on'],z):raise AssertionError('Cached native readout mismatch')
            denom=base.float().square().mean(-1).sqrt().clamp_min(1e-12)
            for name,value in values.items():
                value=value.float().cpu();assert bool(torch.isfinite(value).all())
                zs[name].append(value);ps[name].append(value.sigmoid())
                ratio=deltas[name].float().square().mean(-1).sqrt()/denom
                assert not bool((ratio>.010001).any());ratios[name].append(ratio.cpu())
            for v in views:
                value=vs[v].float().cpu();views[v].append(value);viewp[v].append(value.sigmoid())
            ys.append(y.float());rs.append(cpu_images['face_reliable'].bool());ids.extend(map(str,batch_ids));lengths.append(len(y))
    y=torch.cat(ys);r=torch.cat(rs);result={}
    assert len(ids)==len(set(ids))
    common=dict(sample_ids=np.asarray(ids),targets=y.numpy(),face_reliable=r.numpy(),class_indices=np.arange(y.shape[1]),
        batch_lengths=np.asarray(lengths),probability_device=np.asarray('cpu'),probability_dtype=np.asarray('float32'),
        probability_operation=np.asarray('torch.sigmoid'),torch_version=np.asarray(str(torch.__version__)))
    vm={}
    for v in views:
        common[v+'_logits']=torch.cat(views[v]).numpy();common[v+'_probabilities']=torch.cat(viewp[v]).numpy()
        vm[v]=asdict(compute_metrics(task,torch.cat(viewp[v]),y,.5))
    if r.any():vm['face_reliable']=asdict(compute_metrics(task,torch.cat(viewp['face'])[r],y[r],.5))
    for name in names:
        p=torch.cat(ps[name]);z=torch.cat(zs[name]);ratio=torch.cat(ratios[name]);m=compute_metrics(task,p,y,.5)
        metrics=dict(fused=asdict(m),**vm)
        for label,mask in (('reliable',r),('unreliable',~r)):
            if mask.any():metrics['fused_'+label]=asdict(compute_metrics(task,p[mask],y[mask],.5))
        folder=output/name;(folder/'view_scores').mkdir(parents=True,exist_ok=True);(folder/'scores').mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'view_scores'/f'task{task}.npz',**common,fused_logits=z.numpy(),fused_probabilities=p.numpy(),feature_ratio=ratio.numpy())
        np.savez_compressed(folder/'scores'/f'task{task}.npz',**{k:v for k,v in common.items() if not k.endswith(('_logits','_probabilities'))},logits=z.numpy(),probabilities=p.numpy())
        diagnostic=dict(metrics=metrics,intervention=name,training_updates=0,one_shared_image_forward=True,
            query_held_fixed=name in ('no_face_memory','no_person_memory'),
            feature_ratio_mean=float(ratio.mean()),feature_ratio_max=float(ratio.max()),
            near_cap_fraction=float((ratio>.0095).float().mean()))
        write(folder/f'task{task}.json',dict(metrics=asdict(m),views=diagnostic));result[name]=(m,diagnostic)
    invalid=~r
    if invalid.any():assert torch.equal(torch.cat(zs['on'])[invalid],torch.cat(zs['no_face_memory'])[invalid])
    if not model.interaction_bank[0].learnable:assert torch.equal(torch.cat(zs['on']),torch.cat(zs['zero_query']))
    write(output/'query_profiles'/f'task{task}.json',dict(task=task,batches=statistics,
        train_calibration_sha256=constants['_calibration_file_sha256']))
    if task in DIAGNOSTIC_TASKS or smoke:
        evaluate_people_pairs(model,loader.dataset,ids,task,output,workers)
    return result


def evaluate_people_pairs(model,view,ids,task,output,workers):
    batches=paired_batches(ids)
    if not batches:
        write(output/'person_pairs'/f'task{task}.json',dict(status='no_eligible_multi_person_images',samples=0));return
    options=dict(batch_sampler=batches,num_workers=workers,generator=torch.Generator().manual_seed(929910+task))
    if workers:options.update(prefetch_factor=1,persistent_workers=True,pin_memory=True,multiprocessing_context='spawn',worker_init_fn=single_thread_worker)
    loader=DataLoader(view,**options);names=('native','query_swap','face_swap','person_swap')
    zs={n:[] for n in names};ps={n:[] for n in names};masks={n:[] for n in names};donors={n:[] for n in names}
    ids=[];targets=[];lengths=[]
    with torch.no_grad():
        for cpu_images,y,batch_ids in loader:
            images=move_model_inputs(cpu_images,torch.device('cuda'))
            visible=images['spatial_visibility']>0
            person,pm=grouped_donors(batch_ids,visible)
            face,fm=grouped_donors(batch_ids,visible&images['face_reliable'].bool())
            with torch.cuda.amp.autocast(enabled=True):
                z,_=model.seen_logits_with_views(images)
                values=dict(native=z,
                    query_swap=model.cached_base_seen_logits(model.interaction_delta(query_donor=person)),
                    face_swap=model.cached_base_seen_logits(model.interaction_delta(memory_donors={'face':face})),
                    person_swap=model.cached_base_seen_logits(model.interaction_delta(memory_donors={'person':person})))
            for name,value in values.items():
                value=value.float().cpu();zs[name].append(value);ps[name].append(value.sigmoid())
                masks[name].append((fm if name=='face_swap' else pm).cpu())
                donor=face if name=='face_swap' else person
                donors[name].extend([str(batch_ids[i]) for i in donor.tolist()])
            ids.extend(map(str,batch_ids));targets.append(y.float());lengths.append(len(y))
    y=torch.cat(targets);rows={};folder=output/'person_pairs';folder.mkdir(parents=True,exist_ok=True)
    for name in names:
        mask=torch.cat(masks[name]);p=torch.cat(ps[name]);z=torch.cat(zs[name]);native=torch.cat(ps['native'])
        assert torch.equal(z[~mask],torch.cat(zs['native'])[~mask])
        row=dict(samples=int(mask.sum()))
        if mask.any():row.update(metrics=asdict(compute_metrics(task,p[mask],y[mask],.5)),paired_native=asdict(compute_metrics(task,native[mask],y[mask],.5)))
        rows[name]=row
        np.savez_compressed(folder/f'task{task}_{name}.npz',sample_ids=np.asarray(ids),targets=y.numpy(),
            logits=z.numpy(),probabilities=p.numpy(),eligible=mask.numpy(),donor_ids=np.asarray(donors[name]),batch_lengths=np.asarray(lengths))
    write(folder/f'task{task}.json',dict(status='complete',task=task,samples=len(y),conditions=rows,
        same_image_donors=True,both_people_visible=True,own_h0_and_head_preserved=True,
        paired_batch_native_not_formal_full_cohort_score=True,training_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();calibrate(args.training_root,args.output)
