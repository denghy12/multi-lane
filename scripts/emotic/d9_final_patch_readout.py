"""TRAIN-only fitting and paired eight-stage evaluation of final CLS/patch inputs."""
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.final_patch_fusion import FinalPatchFusion, METHODS, LEARNED, FEATURE_METHODS, WIDTH, HIDDEN, PROJECTION_SEED
from multi_lane.track_a.summary_logit_fusion import fused_logits, image_bce, masked_weights
from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_image_gate_extract import group_key
from scripts.emotic.d9_summary_readout import (load_stage as load_base, subset as subset_base,
    folds, metrics, coverage, save_npz, group_mean_scores, permutation, availability)
from scripts.emotic.d9_summary_eight import canonical_fixed, BENCHMARK, KEYS
from scripts.emotic.d9_feature_bounds import summarize, select_on_val
from scripts.emotic.d9_final_patch_extract import SCHEMA, capture_identity
from scripts.emotic.evaluate_d9_relation import metrics_for_stage


@dataclass
class PatchStage:
    base: object
    cls: torch.Tensor
    patch: torch.Tensor
    roster_ids: np.ndarray
    roster_inverse: np.ndarray
    roster_reliable: torch.Tensor
    feature_sha256: str


def load_stage(cfg,name,features,split,task,smoke):
    base=load_base(Path(cfg['sources'][name]['path']),Path(cfg['cache'])/name/'features',split,task,False,current_only=split=='train')
    path=features/split/f'task{task}.npz';meta=read(path.with_suffix('.json'));spec=meta['protocol']
    expected=dict(schema=SCHEMA,split=split,task=task,smoke=smoke,width=WIDTH,projection_seed=PROJECTION_SEED,
                  capture_code=capture_identity(),patch_dtype='float16',cls_dtype='float32',
                  native_sha256=base.provenance['native_sha256'],checkpoint_sha256=base.provenance['checkpoint_sha256'],
                  config_sha256=base.provenance['config_sha256'])
    if any(spec.get(k)!=v for k,v in expected.items()):raise ValueError('Final CLS/patch provenance changed')
    require_digest(path,meta['archive_sha256'])
    with np.load(path,allow_pickle=False) as z:
        positions={str(k):i for i,k in enumerate(base.ids)}
        if len(set(z['image_ids'].tolist()))!=len(z['image_ids']) or any(str(k) not in positions for k in z['image_ids']):
            raise ValueError('Unknown/duplicate image in patch archive')
        if not smoke:np.testing.assert_array_equal(base.ids,z['image_ids'])
        base=subset_base(base,np.asarray([positions[str(k)] for k in z['image_ids']]))
        for key,value in [('image_ids',base.ids),('counts',base.counts),('reliable_counts',base.reliable_counts)]:
            np.testing.assert_array_equal(z[key],value)
        owner=np.repeat(np.arange(len(base.ids)),base.counts)
        np.testing.assert_array_equal(z['roster_inverse'],owner)
        roster=z['roster_ids'];flags=z['roster_reliable']
        np.testing.assert_array_equal(np.asarray([group_key(p,split) for p in roster]),base.ids[owner])
        if len(set(roster.tolist()))!=len(roster) or len(roster)!=int(base.counts.sum()) or flags.dtype!=np.bool_:
            raise ValueError('Invalid complete person roster')
        mapping={str(k):i for i,k in enumerate(roster)}
        take=np.asarray([mapping[str(k)] for k in base.person_ids])
        np.testing.assert_array_equal(flags[take],base.reliable.numpy());np.testing.assert_array_equal(owner[take],base.inverse.numpy())
        np.testing.assert_array_equal(np.bincount(owner[flags],minlength=len(base.ids)),base.reliable_counts)
        if hashlib.sha256('\n'.join(roster.tolist()).encode()).hexdigest()!=meta['roster_sha256']:
            raise ValueError('Person roster digest changed')
        cls,patch=z['cls'],z['patch']
        if cls.shape!=(len(roster),3,WIDTH) or patch.shape!=(len(roster),3,196,WIDTH):
            raise ValueError('Expected complete 196-patch input, not Selectors or a pooled vector')
        if cls.dtype!=np.float32 or patch.dtype!=np.float16 or not np.isfinite(cls).all() or not np.isfinite(patch).all():
            raise ValueError('Nonfinite or unexpected compressed feature dtype')
        if np.any(cls[~flags,2]) or np.any(patch[~flags,2]):raise ValueError('Unavailable Face feature must be zero')
        return PatchStage(base,torch.from_numpy(cls),torch.from_numpy(patch),roster,owner,torch.from_numpy(flags),meta['archive_sha256'])


def subset(stage,indices):
    indices=np.asarray(indices,dtype=np.int64)
    if not np.array_equal(indices,np.sort(np.unique(indices))):raise ValueError('Use sorted unique image indices')
    remap=np.full(len(stage.base.ids),-1,dtype=np.int64);remap[indices]=np.arange(len(indices))
    take=remap[stage.roster_inverse]>=0
    return PatchStage(subset_base(stage.base,indices),stage.cls[take],stage.patch[take],stage.roster_ids[take],
                      remap[stage.roster_inverse[take]],stage.roster_reliable[take],stage.feature_sha256)


def inputs(stage,device):
    return (stage.cls.to(device),stage.patch.to(device),torch.tensor(stage.base.counts,device=device),
            stage.roster_reliable.to(device))


@torch.no_grad()
def predict(stage,gate,device,*,patch_donors=None,spatial_order=None,diagnose=False):
    gate=gate.to(device).eval();cls,patch,counts,flags=inputs(stage,device)
    if patch_donors is not None:patch=patch[torch.as_tensor(patch_donors,device=device)]
    if spatial_order is not None:patch=torch.stack([patch[:,v,order] for v,order in enumerate(spatial_order)],1)
    desc,attention=gate.describe(cls,patch,counts,flags,return_attention=diagnose)
    scores=gate.scores(desc).cpu()
    z=fused_logits(stage.base.logits,scores,stage.base.inverse,stage.base.reliable)
    details={}
    if diagnose:
        for v,alpha in attention.items():
            if v=='face':alpha=alpha[flags]
            if len(alpha):
                entropy=-(alpha*alpha.clamp_min(1e-30).log()).sum(1)/np.log(alpha.shape[1])
                details[v]=dict(mean_max_weight=float(alpha.max(1).values.mean()),mean_normalized_entropy=float(entropy.mean()))
    return scores,z,details


def fit_one(stage,method,config,device):
    gate=FinalPatchFusion(method,config['amplitude'],WIDTH,HIDDEN,config['seed']).to(device)
    cls,patch,counts,flags=inputs(stage,device)
    logits=stage.base.logits.to(device);targets=stage.base.targets.to(device)
    inverse=stage.base.inverse.to(device);reliable=stage.base.reliable.to(device)
    optimizer=torch.optim.Adam(gate.parameters(),lr=config['lr'],weight_decay=0)
    history=[];first_reader_gradient=None
    for step in range(config['steps']+1):
        gate.train();scores=gate(cls,patch,counts,flags,train_center=True)
        bce=image_bce(fused_logits(logits,scores,inverse,reliable),targets,inverse,len(counts))
        if step%25==0 or step==config['steps']:history.append(float(bce.detach()))
        if step==config['steps']:break
        optimizer.zero_grad(set_to_none=True)
        parameters=list(gate.parameters())
        penalty=sum(p.square().sum() for p in parameters)/sum(p.numel() for p in parameters)
        if first_reader_gradient is None and len(gate.readers):
            gradients=torch.autograd.grad(bce,list(gate.readers.parameters()),retain_graph=True,allow_unused=True)
            if any(g is not None and g.abs().sum()>0 for g in gradients):first_reader_gradient=step
        loss=bce+config['l2']*penalty;loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('Nonfinite fusion optimization')
        optimizer.step()
    with torch.no_grad():
        descriptor,_=gate.describe(cls,patch,counts,flags)
        if gate.method in FEATURE_METHODS:gate.center.copy_(descriptor.mean(0))
        sealed_scores=gate.scores(descriptor)
        torch.testing.assert_close(sealed_scores,scores,rtol=1e-6,atol=1e-7)
    return dict(gate=gate.export(),bce_history=history,parameters=sum(p.numel() for p in gate.parameters()),
                first_reader_gradient_step=first_reader_gradient,center_images=len(counts))


def train(cfg,name,out,tasks,config,smoke,device):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;rows=[]
    for task in tasks:
        stage=load_stage(cfg,name,out/'features','train',task,smoke);base=stage.base
        assignment=folds(base.ids,config['folds'])
        if set(assignment.tolist())!=set(range(config['folds'])):raise ValueError('Empty original-image fold')
        states={};arrays={}
        for method in LEARNED:
            oof=torch.empty_like(base.targets);fold_states=[]
            for fold in range(config['folds']):
                fitting=subset(stage,np.flatnonzero(assignment!=fold));held=subset(stage,np.flatnonzero(assignment==fold))
                result=fit_one(fitting,method,config,device)
                _,z,_=predict(held,FinalPatchFusion.restore(result['gate']),device)
                oof[torch.from_numpy(assignment==fold)[base.inverse]]=z
                fold_states.append(dict(fold=fold,train=coverage(fitting.base),held=coverage(held.base),fit=result,
                                        metrics=metrics(z,held.base.targets)))
                del fitting,held
            result=fit_one(stage,method,config,device)
            _,z,_=predict(stage,FinalPatchFusion.restore(result['gate']),device)
            states[method]=dict(**result,train_metrics=metrics(z,base.targets),out_of_fold=metrics(oof,base.targets),folds=fold_states)
            arrays[method]=oof.numpy()
            print(f'FIT {name}/task{task}/{method}: {result["parameters"]} parameters; BCE {result["bce_history"][-1]:.6f}',flush=True)
        _,neutral,_=predict(stage,FinalPatchFusion('initial'),device)
        path=out/'train'/f'task{task}_oof.npz'
        save_npz(path,image_ids=base.ids,person_ids=base.person_ids,inverse=base.inverse.numpy(),targets=base.targets.numpy(),fold=assignment,**arrays)
        rows.append(dict(task=task,coverage=coverage(base),provenance=base.provenance,features_sha256=stage.feature_sha256,
                         initial=metrics(neutral,base.targets),methods=states,oof_sha256=sha256_file(path)))
    write_verified(out/'fit.json',dict(tasks=tasks,config=config,smoke=smoke,methods=list(LEARNED),stages=rows,
                                     initialization='Full1_aux0_no_prior',width=WIDTH,hidden=HIDDEN))


def patch_donors(stage,rng):
    """Swap entire source-image patch content, retaining recipients' CLS and masks.

    One donor image per recipient, within the three availability groups. Match
    reliable/unreliable people within that image, cycling when roster sizes differ.
    """
    images=permutation(stage.base,rng).numpy();donors=np.arange(len(stage.roster_ids))
    flags=stage.roster_reliable.numpy();cycled=0
    for recipient,donor in enumerate(images):
        own=np.flatnonzero(stage.roster_inverse==recipient)
        if recipient==donor:continue
        other=np.flatnonzero(stage.roster_inverse==donor)
        for valid in (False,True):
            targets=own[flags[own]==valid];source=other[flags[other]==valid]
            if len(targets) and not len(source):raise ValueError('Patch swap changed Face availability')
            if len(targets):
                donors[targets]=source[np.arange(len(targets))%len(source)]
                cycled+=max(0,len(targets)-len(source))
    return donors,dict(swapped_images=int((images!=np.arange(len(images))).sum()),images=len(images),cycled_people=cycled,
                       donor_sha256=hashlib.sha256(donors.tobytes()).hexdigest())


def evaluate(cfg,name,out,split,tasks,config,smoke,device):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    fit=read(out/'fit.json')
    if (fit['tasks']!=tasks or fit['config']!=config or fit['smoke']!=smoke
            or fit['methods']!=list(LEARNED) or fit['initialization']!='Full1_aux0_no_prior'
            or fit['width']!=WIDTH or fit['hidden']!=HIDDEN):raise ValueError('Fitting protocol changed')
    seal=read(out/'validation_seal.json') if split=='test' else None
    if seal:
        for file,key in [('fit.json','fit_sha256'),('val_results.json','val_sha256'),('selection.json','selection_sha256')]:
            require_digest(out/file,seal[key])
    source=Path(cfg['sources'][name]['path']);prior=Path(cfg['cache'])/name/'features'
    native=read(source/'evaluation'/split/'result.json');rows=[];locks={}
    for fitted in fit['stages']:
        task=fitted['task'];stage=load_stage(cfg,name,out/'features',split,task,smoke);base=stage.base
        for key in ('checkpoint_sha256','config_sha256'):
            if fitted['provenance'][key]!=base.provenance[key]:raise ValueError('Branch identity changed')
        outputs={'fixed':canonical_fixed(source,prior,split,task,base)}
        scores,outputs['initial'],_=predict(stage,FinalPatchFusion('initial'),device)
        arrays={'initial_scores':scores.numpy(),'initial_weights':masked_weights(scores[base.inverse],base.reliable).numpy()}
        controls={};stage_locks={};statistics={}
        for method in LEARNED:
            gate=FinalPatchFusion.restore(fitted['methods'][method]['gate'])
            scores,z,attention=predict(stage,gate,device,diagnose=True);weights=masked_weights(scores[base.inverse],base.reliable)
            if not (scores[:,0]==1).all() or scores[:,1:].abs().max()>config['amplitude']+1e-7:
                raise ValueError('Full1/auxiliary score constraint violated')
            if not (weights[:,0,None]>weights[:,1:]).all() or (weights[~base.reliable,2]!=0).any():
                raise ValueError('Full dominance or Face mask violated')
            outputs[method]=z;arrays[method+'_scores']=scores.numpy();arrays[method+'_weights']=weights.numpy()
            statistics[method]={g:dict(mean=weights[mask].mean(0).tolist(),std=weights[mask].std(0,unbiased=False).tolist())
                for g,mask in [('reliable',base.reliable),('unreliable',~base.reliable)] if mask.any()}
            statistics[method]['attention']=attention
            statistics[method]['tanh_saturation_fraction']=float((scores[:,1:].abs()>.95*config['amplitude']).float().mean())
            mean,stage_locks[method]=group_mean_scores(scores,base,None if seal is None else seal['mean_scores'][str(task)][method])
            mean_logits=fused_logits(base.logits,mean,base.inverse,base.reliable)
            arrays[method+'_mean_logits']=mean_logits.numpy()
            repeats=[];rng=np.random.default_rng(config['permutation_seed'])
            for _ in range(config['permutations']):
                shuffled=fused_logits(base.logits,scores[permutation(base,rng)],base.inverse,base.reliable)
                repeats.append(metrics(shuffled,base.targets)['mAP'])
            controls[method]=dict(mean=metrics_for_stage(task,base.targets,base.reliable,{method:mean_logits})[method],
                                 permutation_mAP=repeats,permutation_mean=float(np.mean(repeats)))
            if method in ('cls_mean','cls_regions','cls_patch','cls_patch_full'):
                swapped=[];swap_metadata=[];rng=np.random.default_rng(config['patch_swap_seed'])
                for trial in range(config['patch_swaps']):
                    donors,metadata=patch_donors(stage,rng)
                    _,wrong,_=predict(stage,gate,device,patch_donors=donors)
                    swapped.append(metrics(wrong,base.targets)['mAP']);swap_metadata.append(metadata)
                    if trial==0:arrays[method+'_patch_swapped_logits']=wrong.numpy()
                controls[method]['patch_swap']=dict(mAP=swapped,mean=float(np.mean(swapped)),assignments=swap_metadata)
            if method=='cls_regions':
                rng=np.random.default_rng(config['spatial_seed'])
                order=np.stack([rng.permutation(stage.patch.shape[2]) for _ in range(3)])
                _,wrong,_=predict(stage,gate,device,spatial_order=order)
                arrays[method+'_spatial_shuffled_logits']=wrong.numpy()
                controls[method]['spatial_shuffle']=dict(metrics=metrics(wrong,base.targets),order=order.tolist())
            print(f'{split.upper()} {name}/task{task}/{method} evaluated',flush=True)
        calculated=metrics_for_stage(task,base.targets,base.reliable,outputs)
        if not smoke:
            if base.provenance['native_sha256']!=native['stages'][task]['scores_sha256']:raise ValueError('Native stage seal differs')
            for group,value in calculated['fixed'].items():
                for key,number in value.items():np.testing.assert_allclose(number,native['stages'][task]['metrics']['fixed'][group][key],rtol=0,atol=1e-9)
        arrays.update({m+'_logits':z.numpy() for m,z in outputs.items()})
        path=out/split/f'task{task}.npz'
        save_npz(path,image_ids=base.ids,person_ids=base.person_ids,inverse=base.inverse.numpy(),targets=base.targets.numpy(),
                 reliable=base.reliable.numpy(),counts=base.counts,reliable_counts=base.reliable_counts,**arrays)
        rows.append(dict(task=task,coverage=coverage(base),provenance=base.provenance,features_sha256=stage.feature_sha256,
                         metrics=calculated,controls=controls,weights=statistics,predictions_sha256=sha256_file(path)))
        locks[str(task)]=stage_locks
    summary=summarize(rows,METHODS,smoke);comparisons=None
    if summary:
        comparisons={m:{label:dict(delta={k:summary[m]['all'][k]-ref[k] for k in KEYS},
                           both_at_least_reference=all(summary[m]['all'][k]>=ref[k] for k in KEYS))
                        for label,ref in dict(original_D=BENCHMARK[split],same_branches_fixed=summary['fixed']['all'],
                                              initial=summary['initial']['all'],shared=summary['shared']['all'],
                                              cls=summary['cls']['all'],cls_mean=summary['cls_mean']['all']).items()} for m in METHODS}
    report=dict(split=split,smoke=smoke,stages=rows,summary=summary,comparisons=comparisons,
                fit_sha256=sha256_file(out/'fit.json'),native_result_sha256=sha256_file(source/'evaluation'/split/'result.json'),
                target_comparable=not smoke)
    write_verified(out/f'{split}_results.json',report)
    if split=='val':
        selection=dict(selected=select_on_val(summary,list(LEARNED)) if summary else None,
                       rule=cfg['selection'],val_sha256=sha256_file(out/'val_results.json'))
        write_verified(out/'selection.json',selection)
        write_verified(out/'validation_seal.json',dict(fit_sha256=sha256_file(out/'fit.json'),val_sha256=selection['val_sha256'],
                       selection_sha256=sha256_file(out/'selection.json'),mean_scores=locks))
    return report
