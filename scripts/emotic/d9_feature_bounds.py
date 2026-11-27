"""Four auxiliary-score bounds, two cached summary inputs, two TRAIN sources, eight stages."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from multi_lane.track_a.feature_bound_fusion import FeatureBoundFusion, full_bounds
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, TaskMetrics, summarize_tasks
from multi_lane.track_a.summary_logit_fusion import fused_logits, masked_weights, image_bce
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_summary_eight import BENCHMARK, KEYS, canonical_fixed
from scripts.emotic.d9_summary_readout import (load_stage, subset, folds, metrics, coverage, save_npz,
                                              availability, permutation, group_mean_scores)
from scripts.emotic.d9_summary_run import ROOT, code_record, source_check
from scripts.emotic.evaluate_d9_relation import metrics_for_stage, write_json

CONFIG = ROOT/'scripts/emotic/configs/d9_feature_bounds_20261127.json'


def candidates(cfg):
    return {f'{feature}_a{round(a*100):03d}':dict(feature=feature, amplitude=a)
            for feature in cfg['features'] for a in cfg['amplitudes']}


def check(cfg):
    if (cfg['schema'] != 1 or cfg['seed'] != 1 or cfg['tasks'] != list(range(8))
            or cfg['features'] != ['summary_mean','summary_correspondence']
            or cfg['amplitudes'] != [.1,.25,.5,.75]
            or cfg['initialization'] != 'zero_matrix_no_bias_no_prior'
            or cfg['evaluation'] != 'current_stage_gate_broadcast_all_seen'
            or cfg['selection'] != 'Val average_mAP descending, then final_mAP descending, then declared order'):
        raise ValueError('The declared bounds, initialization and stage protocol are locked')
    if cfg['fit'] != dict(steps=300,matrix_lr=.001,matrix_l2=.001,folds=3,permutations=100,permutation_seed=20261127):
        raise ValueError('Unexpected training or diagnostic budget')
    if set(cfg['sources']) != {'equal90','full'} or any(set(v) != {'path','mode','exposure'} for v in cfg['sources'].values()):
        raise ValueError('Source-specific hyperparameters are not allowed')
    for key in ('output','logs'):
        path=Path(cfg[key])
        if path.is_absolute() or '..' in path.parts or path.parts[0] != key:
            raise ValueError('Use project-relative artifact directories')


def record_protocol(cfg):
    check(cfg)
    prior=Path(cfg['cache']);old=read(prior/'protocol.json');complete=read(prior/'complete.json')
    require_digest(prior/'comparison.json',complete['comparison_sha256'])
    if old['digest'] != complete['digest'] or old['config']['tasks'] != list(range(8)):
        raise ValueError('Incomplete eight-stage cache')
    sources=source_check(cfg)
    if sources != old['sources']:
        raise ValueError('Cache and requested branch identities differ')
    # Only new modules are added. Every old Python dependency must still match the cache producer.
    for path,digest in old['code_sha256'].items(): require_digest(ROOT/path,digest)
    files=code_record()
    digest=hashlib.sha256(json.dumps(dict(config=cfg,code=files),sort_keys=True).encode()).hexdigest()
    return dict(digest=digest,config=cfg,sources=sources,code_sha256=files,
                cache_protocol_sha256=sha256_file(prior/'protocol.json'),
                cache_complete_sha256=sha256_file(prior/'complete.json'),
                head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip())


def cached_stage(cfg,name,split,task,smoke):
    stage=load_stage(Path(cfg['sources'][name]['path']),Path(cfg['cache'])/name/'features',split,task,
                     False,current_only=split=='train')
    if smoke:
        kind=availability(stage)
        indices=np.sort(np.concatenate([np.flatnonzero(kind==k)[:12] for k in (0,1,2)]))
        stage=subset(stage,indices)
    return stage


def initial_scores(stage):
    values=torch.zeros(len(stage.ids),3)
    values[:,0]=1
    return values


def predict(stage,gate):
    with torch.no_grad():
        scores=gate(stage.descriptors[gate.feature])
        return scores,fused_logits(stage.logits,scores,stage.inverse,stage.reliable)


def fit_one(stage,spec,config):
    gate=FeatureBoundFusion(**spec)
    gate.set_center(stage.descriptors[spec['feature']])
    optimizer=torch.optim.Adam([gate.matrix],lr=config['matrix_lr'],weight_decay=0)
    history=[]
    for step in range(config['steps']+1):
        scores=gate(stage.descriptors[gate.feature])
        loss=image_bce(fused_logits(stage.logits,scores,stage.inverse,stage.reliable),stage.targets,
                       stage.inverse,len(stage.ids))
        if step % 25 == 0 or step==config['steps']: history.append(float(loss.detach()))
        if step==config['steps']: break
        optimizer.zero_grad(set_to_none=True)
        total=loss+config['matrix_l2']*gate.matrix.square().mean()
        total.backward()
        if not torch.isfinite(total) or gate.matrix.grad is None or not torch.isfinite(gate.matrix.grad).all():
            raise FloatingPointError('Nonfinite fusion optimization')
        optimizer.step()
    return dict(gate=gate.export(),bce_history=history,parameters=gate.matrix.numel())


def train(cfg,name,out,tasks,fitting,smoke):
    torch.set_num_threads(1);torch.manual_seed(cfg['seed']);rows=[]
    for task in tasks:
        stage=cached_stage(cfg,name,'train',task,smoke);assignment=folds(stage.ids,fitting['folds'])
        if set(assignment.tolist()) != set(range(fitting['folds'])):raise ValueError('Empty image fold')
        states={};arrays={}
        for method,spec in candidates(cfg).items():
            held_logits=torch.empty_like(stage.targets);fold_states=[]
            for fold in range(fitting['folds']):
                fitting_stage=subset(stage,np.flatnonzero(assignment!=fold))
                held=subset(stage,np.flatnonzero(assignment==fold))
                fitted=fit_one(fitting_stage,spec,fitting)
                _,z=predict(held,FeatureBoundFusion.restore(fitted['gate']))
                held_logits[torch.from_numpy(assignment==fold)[stage.inverse]]=z
                fold_states.append(dict(fold=fold,train=coverage(fitting_stage),held=coverage(held),
                                        fit=fitted,metrics=metrics(z,held.targets)))
            final=fit_one(stage,spec,fitting)
            _,z=predict(stage,FeatureBoundFusion.restore(final['gate']))
            states[method]=dict(**final,train_metrics=metrics(z,stage.targets),
                                out_of_fold=metrics(held_logits,stage.targets),folds=fold_states)
            arrays[method]=held_logits.numpy()
        neutral=fused_logits(stage.logits,initial_scores(stage),stage.inverse,stage.reliable)
        path=out/'train'/f'task{task}_oof.npz'
        save_npz(path,image_ids=stage.ids,person_ids=stage.person_ids,inverse=stage.inverse.numpy(),
                 targets=stage.targets.numpy(),fold=assignment,**arrays)
        rows.append(dict(task=task,coverage=coverage(stage),provenance=stage.provenance,
                         initial=metrics(neutral,stage.targets),methods=states,oof_sha256=sha256_file(path)))
        print(f'FIT {name} stage{task}: {len(stage.ids)} images, eight configurations x three folds',flush=True)
    write_verified(out/'fit.json',dict(tasks=tasks,config=fitting,smoke=smoke,stages=rows,
                                     candidates=candidates(cfg),initialization=cfg['initialization']))


def summarize(rows,methods,smoke):
    if not smoke and [r['task'] for r in rows] != list(range(8)):raise ValueError('Need all eight stages')
    for row in rows:
        for m in methods:
            if row['metrics'][m]['all']['seen_classes'] != sum(TASK_SIZES[:row['task']+1]):
                raise ValueError('Current-head-only metrics cannot be used here')
    if smoke:return None
    return {m:{g:summarize_tasks([TaskMetrics(**r['metrics'][m][g]) for r in rows])
               for g in ('all','reliable','unreliable') if all(g in r['metrics'][m] for r in rows)} for m in methods}


def select_on_val(summary,names):
    return max(names,key=lambda m:(summary[m]['all']['average_mAP'],summary[m]['all']['final_mAP']))


def evaluate(cfg,name,out,split,tasks,fitting,smoke):
    torch.set_num_threads(2)
    fit=read(out/'fit.json')
    if (fit['tasks']!=tasks or fit['config']!=fitting or fit['candidates']!=candidates(cfg)
            or fit['smoke']!=smoke or fit['initialization']!=cfg['initialization']):raise ValueError('Fit protocol changed')
    seal=read(out/'validation_seal.json') if split=='test' else None
    if seal:
        for file,key in [('fit.json','fit_sha256'),('val_results.json','val_sha256'),('selection.json','selection_sha256')]:
            require_digest(out/file,seal[key])
    source=Path(cfg['sources'][name]['path']);features=Path(cfg['cache'])/name/'features'
    native=read(source/'evaluation'/split/'result.json')
    rows=[];locks={};methods=['fixed','initial',*candidates(cfg)]
    for fitted in fit['stages']:
        task=fitted['task'];stage=cached_stage(cfg,name,split,task,smoke)
        for key in ('config_sha256','checkpoint_sha256'):
            if fitted['provenance'][key]!=stage.provenance[key]:raise ValueError('Branch identity changed')
        outputs={'fixed':canonical_fixed(source,features,split,task,stage)}
        neutral=initial_scores(stage)
        outputs['initial']=fused_logits(stage.logits,neutral,stage.inverse,stage.reliable)
        arrays={'initial_scores':neutral.numpy(),'initial_weights':masked_weights(neutral[stage.inverse],stage.reliable).numpy()}
        controls={};stage_locks={};statistics={}
        for m,spec in candidates(cfg).items():
            gate=FeatureBoundFusion.restore(fitted['methods'][m]['gate']);scores,z=predict(stage,gate)
            weights=masked_weights(scores[stage.inverse],stage.reliable)
            if not (scores[:,0]==1).all() or scores[:,1:].abs().max()>spec['amplitude']+1e-7:
                raise ValueError('Score constraint violated')
            if not (weights[:,0,None]>weights[:,1:]).all() or (weights[~stage.reliable,2]!=0).any():
                raise ValueError('Full dominance or face mask violated')
            statistics[m]={g:dict(mean=weights[mask].mean(0).tolist(),std=weights[mask].std(0,unbiased=False).tolist())
                           for g,mask in [('reliable',stage.reliable),('unreliable',~stage.reliable)] if mask.any()}
            statistics[m]['tanh_saturation_fraction']=float((scores[:,1:].abs()>.95*spec['amplitude']).float().mean())
            outputs[m]=z;arrays[m+'_scores']=scores.numpy();arrays[m+'_weights']=weights.numpy()
            mean,stage_locks[m]=group_mean_scores(scores,stage,None if seal is None else seal['mean_scores'][str(task)][m])
            mean_logits=fused_logits(stage.logits,mean,stage.inverse,stage.reliable)
            mean_metrics=metrics_for_stage(task,stage.targets,stage.reliable,{m:mean_logits})[m]
            repeats=[];rng=np.random.default_rng(fitting['permutation_seed'])
            for _ in range(fitting['permutations']):
                shuffled=fused_logits(stage.logits,scores[permutation(stage,rng)],stage.inverse,stage.reliable)
                repeats.append(metrics(shuffled,stage.targets)['mAP'])
            controls[m]=dict(mean=mean_metrics,permutation_mAP=repeats,permutation_mean=float(np.mean(repeats)))
            arrays[m+'_mean_logits']=mean_logits.numpy()
        calculated=metrics_for_stage(task,stage.targets,stage.reliable,outputs)
        if not smoke:
            if stage.provenance['native_sha256']!=native['stages'][task]['scores_sha256']:raise ValueError('Native stage seal differs')
            for group,value in calculated['fixed'].items():
                for key,number in value.items():np.testing.assert_allclose(number,native['stages'][task]['metrics']['fixed'][group][key],rtol=0,atol=1e-9)
        arrays.update({m+'_logits':z.numpy() for m,z in outputs.items()})
        path=out/split/f'task{task}.npz'
        save_npz(path,image_ids=stage.ids,person_ids=stage.person_ids,inverse=stage.inverse.numpy(),targets=stage.targets.numpy(),
                 reliable=stage.reliable.numpy(),counts=stage.counts,reliable_counts=stage.reliable_counts,**arrays)
        rows.append(dict(task=task,coverage=coverage(stage),provenance=stage.provenance,metrics=calculated,
                         controls=controls,weights=statistics,predictions_sha256=sha256_file(path)))
        locks[str(task)]=stage_locks
        print(f'{split.upper()} {name} stage{task} {stage.targets.shape[1]} classes: '+json.dumps({m:calculated[m]['all']['mAP'] for m in methods}),flush=True)
    summary=summarize(rows,methods,smoke)
    comparisons=None
    if summary:
        comparisons={m:{label:dict(delta={k:summary[m]['all'][k]-ref[k] for k in KEYS},
                           both_at_least_reference=all(summary[m]['all'][k]>=ref[k] for k in KEYS))
                        for label,ref in dict(original_D=BENCHMARK[split],same_branches_fixed=summary['fixed']['all'],
                                              initial=summary['initial']['all']).items()} for m in methods}
    report=dict(split=split,smoke=smoke,stages=rows,summary=summary,comparisons=comparisons,
                fit_sha256=sha256_file(out/'fit.json'),native_result_sha256=sha256_file(source/'evaluation'/split/'result.json'),
                target_comparable=not smoke)
    write_verified(out/f'{split}_results.json',report)
    if split=='val':
        selection=dict(selected=select_on_val(summary,list(candidates(cfg))) if summary else None,
                       rule=cfg['selection'],val_sha256=sha256_file(out/'val_results.json'))
        write_verified(out/'selection.json',selection)
        write_verified(out/'validation_seal.json',dict(fit_sha256=sha256_file(out/'fit.json'),val_sha256=selection['val_sha256'],
                       selection_sha256=sha256_file(out/'selection.json'),mean_scores=locks))
    return report


def run(cfg,phase):
    record=record_protocol(cfg)
    if phase=='check':print(json.dumps(record,indent=2));return
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Run committed code only')
    smoke=phase=='preflight';base=Path(cfg['output'])/phase;base.mkdir(parents=True,exist_ok=True)
    if not smoke:
        prior=read(Path(cfg['output'])/'preflight/complete.json')
        if prior['digest']!=record['digest'] or prior['sources']!=record['sources']:raise ValueError('Missing matching preflight')
    write_verified(base/'protocol.json',record)
    tasks=[0,7] if smoke else cfg['tasks'];fitting=dict(cfg['fit'])
    if smoke:fitting.update(steps=5,permutations=2)
    for name in cfg['sources']:train(cfg,name,base/name,tasks,fitting,smoke)
    for split in ('val','test'):
        if split=='test':
            for name in cfg['sources']:
                seal=read(base/name/'validation_seal.json')
                for file,key in [('fit.json','fit_sha256'),('val_results.json','val_sha256'),('selection.json','selection_sha256')]:require_digest(base/name/file,seal[key])
        for name in cfg['sources']:evaluate(cfg,name,base/name,split,tasks,fitting,smoke)
        for task in tasks:
            with np.load(base/'equal90'/split/f'task{task}.npz',allow_pickle=False) as a,np.load(base/'full'/split/f'task{task}.npz',allow_pickle=False) as b:
                for key in ('image_ids','person_ids','inverse','targets','reliable','counts','reliable_counts'):np.testing.assert_array_equal(a[key],b[key])
    rows=[]
    for name in cfg['sources']:
        reports={s:read(base/name/f'{s}_results.json') for s in ('val','test')}
        for m in ['fixed','initial',*candidates(cfg)]:
            row=dict(coverage=name,method=m,spec=candidates(cfg).get(m),selected_by_val=read(base/name/'selection.json')['selected']==m)
            if not smoke:
                row.update({s:{k:reports[s]['summary'][m]['all'][k] for k in KEYS} for s in reports})
                row['comparisons']={s:reports[s]['comparisons'][m] for s in reports}
            rows.append(row)
    write_verified(base/'comparison.json',dict(tasks=tasks,smoke=smoke,target_comparable=not smoke,rows=rows,original_D=BENCHMARK,
                   bounds={str(a):full_bounds(a) for a in cfg['amplitudes']},metric='final=stage7 all26 mAP; average=mean eight all-seen stage mAPs'))
    write_verified(base/'complete.json',dict(digest=record['digest'],sources=record['sources'],comparison_sha256=sha256_file(base/'comparison.json')))


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--config',type=Path,default=CONFIG)
    cli.add_argument('--phase',required=True,choices=('check','preflight','formal'));args=cli.parse_args()
    if Path.cwd().resolve()!=ROOT:cli.error('Run from the Linux worktree root')
    cfg=read(args.config);check(cfg);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    state=dict(phase=args.phase,status='running',started=datetime.now(timezone.utc).isoformat());start=time.monotonic()
    write_json(out/'run_state.json',state)
    try:run(cfg,args.phase)
    except BaseException as error:
        state.update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-start);write_json(out/'run_state.json',state);raise
    state.update(status='complete',elapsed_seconds=time.monotonic()-start,finished=datetime.now(timezone.utc).isoformat())
    write_json(out/'run_state.json',state);print(json.dumps(state),flush=True)

if __name__=='__main__':main()
