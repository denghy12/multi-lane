"""TRAIN fitting; task-specific logit mixing; complete Val-before-Test controls."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from multi_lane.track_a.final_patch_fusion import FinalPatchFusion
from multi_lane.track_a.task_patch_fusion import (TaskPatchFusion, candidates, LEGACY,
    restore_gate, prediction_key, control_key)
from multi_lane.track_a.summary_logit_fusion import fused_logits, masked_weights
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES
from scripts.emotic.d9_a_eight_eval import read,require_digest
from scripts.emotic.d9_final_patch_readout import inputs,predict,subset,patch_donors,fit_one as old_fit
from scripts.emotic.d9_summary_readout import folds,metrics,coverage,save_npz,group_mean_scores,permutation
from scripts.emotic.d9_task_patch_extract import load_features
from scripts.emotic.d9_summary_eight import canonical_fixed,BENCHMARK,KEYS
from scripts.emotic.d9_feature_bounds import summarize,select_on_val
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.evaluate_d9_relation import metrics_for_stage

METHODS = ('fixed','initial',*(m+'__current' for m in LEGACY),
           *(s['key']+'__own' for s in candidates()))


def fit_one(stage,spec,config,device,*,fast=True):
    if spec['reuse']:
        return old_fit(stage,spec['reader'],config,device)
    gate=TaskPatchFusion(spec).to(device)
    cls,patch,counts,flags=inputs(stage,device)
    from multi_lane.track_a.task_patch_execution import PreparedDescriptor
    prepared=PreparedDescriptor(gate,cls,patch,counts,flags) if fast else None
    base=stage.base;logits=base.logits.to(device);targets=base.targets.to(device)
    inverse=base.inverse.to(device);reliable=base.reliable.to(device)
    people=torch.bincount(base.inverse,minlength=len(counts))[base.inverse].to(device)
    optimizer=torch.optim.Adam(gate.parameters(),lr=config['lr'],weight_decay=0)
    history=[];variation_history=[];first=None
    parameters=list(gate.parameters());number=sum(p.numel() for p in parameters)
    for step in range(config['steps']+1):
        desc=prepared() if prepared is not None else gate.describe(cls,patch,counts,flags)[0]
        scores=gate.scores(desc,train_center=True)
        z=fused_logits(logits,scores,inverse,reliable)
        person_loss=torch.nn.functional.binary_cross_entropy_with_logits(z,targets,reduction='none').mean(1)
        bce=(person_loss/people).sum()/len(counts)
        variation=gate.image_term(desc,True).square().mean()
        if step%25==0 or step==config['steps']:
            history.append(float(bce.detach()));variation_history.append(float(variation.detach()))
        if step==config['steps']:break
        optimizer.zero_grad(set_to_none=True)
        penalty=sum(p.square().sum() for p in parameters)/number
        if first is None and len(gate.readers):
            gradients=torch.autograd.grad(bce,list(gate.readers.parameters()),retain_graph=True,allow_unused=True)
            if any(g is not None and g.abs().sum()>0 for g in gradients):first=step
        loss=bce+config['l2']*penalty+spec['variation']*variation
        loss.backward()
        finite=torch.stack([torch.isfinite(loss),*[torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]]).all()
        if not finite:
            raise FloatingPointError('Nonfinite fusion optimization')
        optimizer.step()
    with torch.no_grad():
        desc=prepared() if prepared is not None else gate.describe(cls,patch,counts,flags)[0]
        if gate.method!='shared':gate.center.copy_(desc.mean(0))
        torch.testing.assert_close(gate.scores(desc),scores,rtol=1e-6,atol=1e-7)
    return dict(gate=gate.export(),parameters=number,bce_history=history,
                variation_history=variation_history,first_reader_gradient_step=first,center_images=len(counts))


def train(cfg,name,out,config,smoke,device,*,tasks=None,finalize=True):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    parent=Path(cfg['parent'])/name
    old=read(parent/'fit.json')
    rows=[]
    for task in (range(8) if tasks is None else tasks):
        stage=load_features(cfg,name,out,'train',task,smoke=smoke);base=stage.base
        assignment=folds(base.ids,config['folds'])
        if set(assignment.tolist())!=set(range(config['folds'])):raise ValueError('Empty TRAIN image fold')
        states={};arrays={}
        for spec in candidates():
            key=spec['key']
            # Each fitted task is an immutable resumable unit under the locked protocol.
            saved=out/'fits'/f'task{task}'/(key+'.json')
            saved.parent.mkdir(parents=True,exist_ok=True)
            cached=read(saved) if saved.exists() else None
            if cached:
                if cached['spec']!=spec or cached['config']!=config or cached['features_sha256']!=stage.feature_sha256 or cached['smoke']!=smoke:
                    raise ValueError('Saved fitting protocol differs')
                result=cached['result'];oof=np.array(cached['oof'],dtype=np.float32)
            elif spec['reuse'] and not smoke:
                result=old['stages'][task]['methods'][key]
                require_digest(parent/'train'/f'task{task}_oof.npz',old['stages'][task]['oof_sha256'])
                with np.load(parent/'train'/f'task{task}_oof.npz',allow_pickle=False) as z:
                    for k,v in [('image_ids',base.ids),('person_ids',base.person_ids),('targets',base.targets.numpy()),('fold',assignment)]:
                        np.testing.assert_array_equal(z[k],v)
                    oof=z[key]
                _,replay,_=predict(stage,restore_gate(result['gate']),device)
                np.testing.assert_allclose(metrics(replay,base.targets)['per_class_ap'],result['train_metrics']['per_class_ap'],rtol=0,atol=1e-9)
            else:
                oof=torch.empty_like(base.targets);fold_states=[]
                for fold in range(config['folds']):
                    fitting=subset(stage,np.flatnonzero(assignment!=fold));held=subset(stage,np.flatnonzero(assignment==fold))
                    fitted=fit_one(fitting,spec,config,device)
                    _,z,_=predict(held,restore_gate(fitted['gate']),device)
                    oof[torch.from_numpy(assignment==fold)[base.inverse]]=z
                    fold_states.append(dict(fold=fold,train=coverage(fitting.base),held=coverage(held.base),
                                            fit=fitted,metrics=metrics(z,held.base.targets)))
                result=fit_one(stage,spec,config,device)
                _,z,_=predict(stage,restore_gate(result['gate']),device)
                result.update(train_metrics=metrics(z,base.targets),out_of_fold=metrics(oof,base.targets),folds=fold_states)
                oof=oof.numpy()
            if not cached:
                write_verified(saved,dict(spec=spec,config=config,smoke=smoke,features_sha256=stage.feature_sha256,
                                          result=result,oof=oof.tolist()))
            states[key]=result;arrays[key]=oof
            print(f'FIT {name}/task{task}/{key}: {result["parameters"]} parameters; BCE {result["bce_history"][-1]:.6f}',flush=True)
        path=out/'train'/f'task{task}_oof.npz'
        save_npz(path,image_ids=base.ids,person_ids=base.person_ids,targets=base.targets.numpy(),
                 inverse=base.inverse.numpy(),fold=assignment,**arrays)
        rows.append(dict(task=task,coverage=coverage(base),provenance=base.provenance,
                         features_sha256=stage.feature_sha256,methods=states,oof_sha256=sha256_file(path)))
        write_verified(out/'train'/f'task{task}_fit.json',rows[-1])
    if finalize:
        finalize_train(cfg,name,out,config,smoke)


def finalize_train(cfg,name,out,config,smoke):
    rows=[read(out/'train'/f'task{t}_fit.json') for t in range(8)]
    for task,row in enumerate(rows):
        if row['task']!=task:raise ValueError('Task fitting order changed')
        require_digest(out/'train'/f'task{task}_oof.npz',row['oof_sha256'])
    write_verified(out/'fit.json',dict(config=config,smoke=smoke,candidates=candidates(),stages=rows,
                                     parent_fit_sha256=sha256_file(Path(cfg['parent'])/name/'fit.json')))


def mix_heads(base,scores,scope):
    """scores: original images x task heads x 3; weights repeat only within a head."""
    if scope=='current':
        if scores.shape[1]!=1:raise ValueError('One current gate expected')
        return fused_logits(base.logits,scores[:,0],base.inverse,base.reliable)
    if scores.shape[1]>len(TASK_SIZES) or sum(TASK_SIZES[:scores.shape[1]])!=base.targets.shape[1]:
        raise ValueError('Need every seen task head exactly once')
    parts=[];start=0
    for head,size in enumerate(TASK_SIZES[:scores.shape[1]]):
        parts.append(fused_logits(base.logits[:,start:start+size],scores[:,head],base.inverse,base.reliable))
        start+=size
    return torch.cat(parts,1)


def evaluate(cfg,name,out,split,tasks,config,smoke,device):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    fit=read(out/'fit.json')
    if fit['config']!=config or fit['smoke']!=smoke or fit['candidates']!=candidates():
        raise ValueError('Fitting protocol changed')
    seal=read(out/'validation_seal.json') if split=='test' else None
    if seal:
        for f,k in [('fit.json','fit_sha256'),('val_results.json','val_sha256'),('selection.json','selection_sha256')]:
            require_digest(out/f,seal[k])
    parent=Path(cfg['parent'])/name
    parent_report=read(parent/f'{split}_results.json')
    rows=[];all_locks={}
    for task in tasks:
        stage=load_features(cfg,name,out,split,task,smoke=smoke);base=stage.base
        outputs={'fixed':canonical_fixed(Path(cfg['sources'][name]['path']),Path(cfg['cache'])/name/'features',split,task,base)}
        _,outputs['initial'],_=predict(stage,FinalPatchFusion('initial'),device)
        scores_bank={};means_bank={};head_outputs={};patch_outputs={};spatial_outputs={}
        details={};locks={}
        for head in range(task+1):
            stage=load_features(cfg,name,out,split,task,head,smoke)
            np.testing.assert_array_equal(stage.base.person_ids,base.person_ids)
            for spec in candidates():
                key=spec['key'];own=key+'__own';gate=restore_gate(fit['stages'][head]['methods'][key]['gate'])
                s,z,attention=predict(stage,gate,device,diagnose=True)
                if not (s[:,0]==1).all() or (s[:,1:]< -spec['lower']-1e-6).any() or (s[:,1:]>.25+1e-6).any():
                    raise ValueError('Full1/score interval violated')
                w=masked_weights(s[base.inverse],base.reliable)
                if not (w[:,0]>.5).all() or (w[~base.reliable,2]!=0).any():raise ValueError('Full dominance/mask broken')
                start=sum(TASK_SIZES[:head]);stop=start+TASK_SIZES[head]
                scores_bank.setdefault(own,[]).append(s)
                head_outputs.setdefault(own,[]).append(z[:,start:stop])
                locked=None if seal is None else seal['mean_scores'][str(task)][own][str(head)]
                mean,lock=group_mean_scores(s,base,locked)
                means_bank.setdefault(own,[]).append(mean);locks.setdefault(own,{})[str(head)]=lock
                stats={g:dict(mean=w[mask].mean(0).tolist(),std=w[mask].std(0,unbiased=False).tolist())
                       for g,mask in [('reliable',base.reliable),('unreliable',~base.reliable)] if mask.any()}
                stats.update(attention=attention,near_lower=float((s[:,1:]<-.95*spec['lower']).float().mean()),
                             near_upper=float((s[:,1:]>.2375).float().mean()))
                details.setdefault(own,{})[str(head)]=stats
                if spec['reader'] in ('cls_regions','cls_patch'):
                    rng=np.random.default_rng(config['patch_swap_seed']);swaps=[]
                    for trial in range(config['patch_swaps']):
                        donors,_=patch_donors(stage,rng)
                        _,wrong,_=predict(stage,gate,device,patch_donors=donors)
                        swaps.append(wrong)
                    patch_outputs.setdefault(own,[]).append(torch.stack(swaps)[:,:,start:stop])
                if spec['reader']=='cls_regions':
                    rng=np.random.default_rng(config['spatial_seed'])
                    order=np.stack([rng.permutation(196) for _ in range(3)])
                    _,wrong,_=predict(stage,gate,device,spatial_order=order)
                    spatial_outputs.setdefault(own,[]).append(wrong[:,start:stop])
                if head==task and key in LEGACY:
                    current=key+'__current';outputs[current]=z
                    scores_bank[current]=[s];means_bank[current]=[mean]
                    details[current]={str(head):stats};locks[current]={str(head):lock}
                    if spec['reader'] in ('cls_regions','cls_patch'):patch_outputs[current]=[torch.stack(swaps)]
                    if spec['reader']=='cls_regions':spatial_outputs[current]=[wrong]
            print(f'{split.upper()} {name}/task{task}/head{head}: all fitted methods read own features',flush=True)
        for key,parts in head_outputs.items():outputs[key]=torch.cat(parts,1)
        arrays={};controls={}
        for method in METHODS[2:]:
            scope='current' if method.endswith('__current') else 'own'
            s=torch.stack(scores_bank[method],1);mean=torch.stack(means_bank[method],1)
            replay=mix_heads(base,s,scope)
            torch.testing.assert_close(replay,outputs[method],rtol=0,atol=0)
            arrays['score__'+method]=s.numpy()
            mean_logits=mix_heads(base,mean,scope)
            arrays[control_key(method,'mean')]=mean_logits.numpy()
            rng=np.random.default_rng(config['permutation_seed']);permuted=[]
            for _ in range(config['permutations']):
                z=mix_heads(base,s[permutation(base,rng)],scope)
                permuted.append(metrics(z,base.targets)['mAP'])
            controls[method]=dict(mean=metrics_for_stage(task,base.targets,base.reliable,{method:mean_logits})[method],
                                 permutation_mAP=permuted,permutation_mean=float(np.mean(permuted)))
            if method in patch_outputs:
                swaps=torch.cat(patch_outputs[method],2)
                vals=[metrics(x,base.targets)['mAP'] for x in swaps]
                arrays[control_key(method,'patch_swap0')]=swaps[0].numpy()
                controls[method]['patch_swap']=dict(mAP=vals,mean=float(np.mean(vals)))
            if method in spatial_outputs:
                wrong=torch.cat(spatial_outputs[method],1)
                arrays[control_key(method,'spatial')]=wrong.numpy()
                controls[method]['spatial_shuffle']=metrics(wrong,base.targets)
            if task==0 and method.endswith('__current'):
                torch.testing.assert_close(outputs[method],outputs[method.replace('__current','__own')],rtol=0,atol=0)
        if not smoke:
            archived=parent/split/f'task{task}.npz'
            require_digest(archived,parent_report['stages'][task]['predictions_sha256'])
            with np.load(archived,allow_pickle=False) as z:
                for k,v in [('image_ids',base.ids),('person_ids',base.person_ids),('targets',base.targets.numpy())]:
                    np.testing.assert_array_equal(z[k],v)
                for method in ('fixed','initial',*LEGACY):
                    key=method if method in ('fixed','initial') else method+'__current'
                    np.testing.assert_array_equal(outputs[key].numpy(),z[method+'_logits'])
        calculated=metrics_for_stage(task,base.targets,base.reliable,outputs)
        for key,value in outputs.items():arrays[prediction_key(key)]=value.numpy()
        if len(arrays)!=len(set(arrays)):raise ValueError('Duplicate archive names')
        path=out/split/f'task{task}.npz'
        save_npz(path,image_ids=base.ids,person_ids=base.person_ids,inverse=base.inverse.numpy(),
                 targets=base.targets.numpy(),reliable=base.reliable.numpy(),counts=base.counts,
                 reliable_counts=base.reliable_counts,**arrays)
        rows.append(dict(task=task,coverage=coverage(base),metrics=calculated,controls=controls,
                         weights=details,predictions_sha256=sha256_file(path),provenance=base.provenance))
        all_locks[str(task)]=locks
    summary=summarize(rows,METHODS,smoke)
    report=dict(split=split,smoke=smoke,stages=rows,summary=summary,
                fit_sha256=sha256_file(out/'fit.json'),target_comparable=not smoke)
    write_verified(out/f'{split}_results.json',report)
    if split=='val':
        eligible=[s['key']+'__own' for s in candidates()]
        selected=select_on_val(summary,eligible) if summary else None
        write_verified(out/'selection.json',dict(selected=selected,eligible=eligible,rule=cfg['selection']))
        write_verified(out/'validation_seal.json',dict(mean_scores=all_locks,
            fit_sha256=sha256_file(out/'fit.json'),val_sha256=sha256_file(out/'val_results.json'),
            selection_sha256=sha256_file(out/'selection.json')))
    return report

