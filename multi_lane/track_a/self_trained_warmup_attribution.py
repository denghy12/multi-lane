"""Read-only within-student comparisons before and after the extra five epochs."""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _sha256,_torch_load
from .head_anchored_eight import runtime
from .head_anchored_pilot import task_slices
from .historical_b32_backbone_attribution import build,method_state
from .paired_transforms import move_model_inputs
from .run_historical_b32_test_attribution import read,write
from .runner import dataset_view,seen_indices,set_seed,tensor_state_digest
from .self_trained_fast_evaluation import loader_for,validate_saved
from .self_trained_recovery import CORE_FILES
from .self_trained_view_transfer import ARMS
from .strong_full_evaluation import evaluate_fused_and_views

TASKS=(0,2,7)
CONDITIONS=('WARMUP_ON','WARMUP_OFF','FINAL_ON','FINAL_OFF')


def code_audit(training_commit,evaluation_commit):
    root=Path(__file__).resolve().parents[2]
    paths=(*CORE_FILES,'multi_lane/track_a/strong_full_evaluation.py',
           'multi_lane/track_a/historical_b32_backbone_attribution.py',
           'multi_lane/track_a/evaluate_joint26_checkpoint.py')
    for name in paths:
        old=subprocess.check_output(['git','show',training_commit+':'+name],cwd=root)
        assert old==(root/name).read_bytes(),('evaluation_dependency_changed',name)
    loader='multi_lane/track_a/self_trained_fast_evaluation.py'
    assert subprocess.check_output(['git','show',evaluation_commit+':'+loader],cwd=root)==(root/loader).read_bytes()
    return dict(existing_model_transform_metric_and_loader_code_exact=True,files=list(paths))


def compare_boundaries(warm,final,task,sizes,baseline=False):
    assert set(warm)==set(final)
    for key in ('head.weight','head.bias'):
        assert torch.equal(warm[key],final[key]),'The student head changed during extra adaptation'
    old_w=task_slices(warm,task,sizes,False);old_f=task_slices(final,task,sizes,False)
    assert all(torch.equal(v,old_f[k]) for k,v in old_w.items()),'Old task parameters changed'
    if baseline:assert all(torch.equal(v,final[k]) for k,v in warm.items())
    return dict(own_head_exact=True,old_task_parameters_exact=True,
        all_parameters_exact=all(torch.equal(v,final[k]) for k,v in warm.items()),
        warmup_model_sha256=tensor_state_digest(warm),final_model_sha256=tensor_state_digest(final))


def install_state(model,state,task):
    missing=model.load_state_dict(state,strict=False)
    assert not missing.unexpected_keys and all(k.startswith('visual_encoder.') for k in missing.missing_keys)
    model.restore_task(task);model.requires_grad_(False).eval()


def copy_complete(source,target,task,view):
    saved=validate_saved(source,task,view)
    if saved is None:return None
    for item in saved['files']:
        path=Path(item['path']);dst=target/path.relative_to(source)
        dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(path,dst)
        assert _sha256(dst)==item['sha256']
    return dict(source=str(source),source_files=saved['files'],validated=True)


def replay_final_prefix(model,loader,reference,enabled):
    batch=next(iter(loader));images,targets,ids=batch
    assert np.array_equal(np.asarray(ids),reference['sample_ids'][:len(ids)])
    assert np.array_equal(targets.numpy(),reference['targets'][:len(ids)])
    model.cross_view_runtime_enabled=enabled
    with torch.no_grad(),torch.cuda.amp.autocast(enabled=True):
        z,views=model.seen_logits_with_views(move_model_inputs(images,torch.device('cuda')))
    differences={}
    for name,value in dict(fused=z,**views).items():
        actual=value.float().cpu().numpy();expected=reference[name+'_logits'][:len(ids)]
        differences[name]=float(np.max(np.abs(actual-expected)))
    assert all(v==0. for v in differences.values()),('final_prefix_replay_mismatch',differences)
    return dict(people=len(ids),all_view_logits_exact=True,differences=differences)


def evaluate(args):
    git=runtime();set_seed(1,True);start=time.time()
    results=[];audits=[];hashes={}
    for arm in ARMS:
        training=args.source_root/arm/'train';cfg=read(training/'config.json')
        complete=read(training/'training_complete.json')
        assert complete['status']=='complete' and not cfg['smoke'] and cfg['tasks']==list(range(8))
        code=code_audit(cfg['git']['commit'],read(args.existing_eval_root/'completion.json')['git']['commit'])
        model=build(cfg['model_template'],Path(cfg['clip_checkpoint']))
        data=make_source(cfg['model_template'],Path(cfg['data_root']),Path(cfg['face_manifest_root']),args.split)
        for task in TASKS:
            wp=training/'warmup_checkpoints'/f'task{task}.pth'
            fp=training/'compact_checkpoints'/f'task{task}.pth'
            for p in (wp,fp):hashes[str(p)]=_sha256(p)
            assert hashes[str(fp)]==complete['checkpoints'][str(task)]['sha256']
            w=_torch_load(wp);f=_torch_load(fp)
            assert w['source_git']==f['source_git']==cfg['git'] and w['task_id']==f['task_id']==task
            pair=compare_boundaries(w['model'],f['model'],task,model._task_sizes,arm=='SELF_BASE')
            view=dataset_view(data,seen_indices(task),True)
            if args.smoke:view=torch.utils.data.Subset(view,range(64))
            loader=loader_for(view,task,2)
            # Replaying the same first B64 never changes the original scores or weights.
            install_state(model,f['model'],task)
            prefix=[]
            for mode in ('on',) if model.cross_view_bank is None else ('on','off'):
                path=args.existing_eval_root/arm/args.split/mode/'view_scores'/f'task{task}.npz'
                with np.load(path) as z:ref={k:z[k].copy() for k in z.files}
                prefix.append(replay_final_prefix(model,loader,ref,mode=='on' and model.cross_view_bank is not None))
            for condition in CONDITIONS:
                target=args.output/arm/condition
                mode='on' if condition.endswith('_ON') else 'off'
                warm=condition.startswith('WARMUP')
                install_state(model,w['model'] if warm else f['model'],task)
                model.cross_view_runtime_enabled=mode=='on' and model.cross_view_bank is not None
                before=tensor_state_digest(method_state(model));tick=time.time();reuse=None
                if not args.smoke:
                    existing=validate_saved(target,task,view)
                    if existing is not None:reuse=dict(source='this_run',validated=True)
                    elif not warm or pair['all_parameters_exact']:
                        source=args.existing_eval_root/arm/args.split/(mode if model.cross_view_bank is not None else 'on')
                        reuse=copy_complete(source,target,task,view)
                        assert reuse is not None
                if reuse is None:
                    with torch.no_grad():
                        value,views=evaluate_fused_and_views(model,loader,torch.device('cuda'),task,.5,True,
                            target/'scores'/f'task{task}.npz',target/'view_scores'/f'task{task}.npz')
                    write(target/f'task{task}.json',dict(metrics=asdict(value),views=views))
                assert before==tensor_state_digest(method_state(model))
                row=read(target/f'task{task}.json')
                results.append(dict(arm=arm,task=task,condition=condition,**row['metrics']))
                audits.append(dict(arm=arm,task=task,condition=condition,parameter_comparison=pair,
                    parameters_unchanged=True,reused=reuse,prefix_replay=prefix,elapsed_seconds=time.time()-tick))
                write(args.output/'progress.json',dict(status='running',split=args.split,metrics=results,audits=audits))
                print('WARMUP_ATTRIBUTION',arm,args.split,task,condition,row['metrics']['mAP'],flush=True)
            del loader,w,f
        del model
    assert hashes=={p:_sha256(Path(p)) for p in hashes}
    write(args.output/'result.json',dict(status='complete',git=git,split=args.split,tasks=list(TASKS),
        smoke=args.smoke,seed=1,metrics=results,audits=audits,code_audit=code,
        input_checkpoint_sha256=hashes,source_weights_unchanged=True,training_updates=0,
        attention_policy='ON_enables_all_trained_old_paths_and_current_initial_or_trained_path',
        evaluation_scope='samples_intersect_seen_classes',batch_size=64,workers=2,worker_threads=1,
        model_threads=72,amp=True,tf32=True,elapsed_seconds=time.time()-start))


def summarize_attribution(root):
    """Paired current-task effects, using the ORIGINAL matched four-arm baseline."""
    rows=[]
    for split in ('val','test'):
        assert read(root/split/'result.json')['status']=='complete'
        for arm in ARMS:
            for task in TASKS:
                parents={c:root/split/arm/c for c in CONDITIONS}
                values={c:read(p/f'task{task}.json')['metrics'] for c,p in parents.items()}
                base=read(root/split/'SELF_BASE/WARMUP_ON'/f'task{task}.json')['metrics']['mAP']
                w=values['WARMUP_ON']['mAP'];wo=values['WARMUP_OFF']['mAP']
                f=values['FINAL_ON']['mAP'];fo=values['FINAL_OFF']['mAP']
                start=sum((5,3,3,3,3,3,3,3)[:task])
                current={c:float(np.mean(v['per_class_ap'][start:])) for c,v in values.items()}
                arrays={}
                for c,p in parents.items():
                    with np.load(p/'view_scores'/f'task{task}.npz') as z:
                        arrays[c]={k:z[k].copy() for k in ('sample_ids','targets','class_indices','fused_logits')}
                ref=arrays['WARMUP_ON']
                assert all(np.array_equal(z[k],ref[k]) for z in arrays.values()
                           for k in ('sample_ids','targets','class_indices'))
                cold=arrays['WARMUP_OFF']['fused_logits'][:,:start]
                old=arrays['FINAL_OFF']['fused_logits'][:,:start]
                old_diff=float(np.max(np.abs(old-cold))) if start else 0.
                current_zero=float(np.max(np.abs(arrays['WARMUP_ON']['fused_logits'][:,start:]
                                                 -arrays['WARMUP_OFF']['fused_logits'][:,start:])))
                assert current_zero==0.,('untrained_current_attention_not_identity',arm,task,current_zero)
                rows.append(dict(arm=arm,split=split,task=task,warmup_on_mAP=w,final_on_mAP=f,
                    gap_already_present_at_warmup=w-base,current_parameter_effect=fo-wo,
                    added_current_attention_effect=(f-fo)-(w-wo),net_current_extra_stage_effect=f-w,
                    final_gap_vs_original_matched_base=f-base,
                    current_class_warmup_AP=current['WARMUP_ON'],current_class_final_AP=current['FINAL_ON'],
                    current_class_parameter_effect=current['FINAL_OFF']-current['WARMUP_OFF'],
                    current_class_attention_effect=current['FINAL_ON']-current['FINAL_OFF'],
                    old_class_max_abs_logit_difference_off=old_diff,initial_current_attention_logit_difference=current_zero))
                assert abs((fo-wo)+((f-fo)-(w-wo))-(f-w))<1e-10
    with (root/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(root/'completion.json',dict(status='complete',tasks=list(TASKS),seed=1,
        training_updates=0,original_matched_baseline_used=True,
        historical_rng_repaired_baseline_not_mixed_with_original_four_arm_attribution=True))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','existing-eval-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=('val','test'),required=True)
    p.add_argument('--smoke',action='store_true')
    evaluate(p.parse_args())
