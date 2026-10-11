"""Audit the three-task whole-transform freeze and replay complete validation."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .compare_joint26 import require
from .evaluate_joint26_checkpoint import build_model, make_source
from .export_compact_test_scores import _torch_load, _sha256, restore_compact_model_state
from .openai_clip_loader import load_openai_clip_visual
from .runner import (LabelView, TASK_SIZES, compact_model_state_dict, git_metadata,
                     set_seed, tensor_state_digest, validate_classes)
from .snapshot_validation import audit_source, infer, load_scores, replay, score_metrics


def audit_control(source, reference, smoke_updates=0):
    read=lambda root,name:json.loads((root/name).read_text())
    c,s,a,h,d=[read(source,n) for n in ('config.json','seed_summary.json','paired_protocol_audit.json','training_history.json','image_stream_transfer_audit.json')]
    rc,ra=read(reference,'config.json'),read(reference,'paired_protocol_audit.json')
    require(s['status']=='complete' and s['config']==c and not c['git']['dirty'],'Invalid frozen run')
    require(c['seed']==0 and c['max_tasks']==3 and c['task_sizes']==list(TASK_SIZES)
            and c['parax_mode']=='image' and c['parax_freeze_all_after_task0'] and c['image_stream_transfer_audit']
            and c['reporting_split']=='val' and not c['also_report_test'] and s['test_metrics'] is None,'Wrong freeze protocol')
    require(a['frozen_visual_unchanged'] and a['frozen_visual_sha256_before']==a['frozen_visual_sha256_after'],'CLIP changed')
    keys=('seed','task_sizes','class_order','train_batch_size','eval_batch_size','learning_rate','scheduler_mode',
          'input_normalization','train_crop_scale','selector_mode','prompt_mode','view_fusion','view_classifier_mode',
          'view_auxiliary_loss_weight','supervised_loss_scale','training_loss_mode','parameter_group_loss_routing',
          'adapter_mode','parax_mode','parax_layer_indices','parax_rank','parax_num_experts','parax_router_hidden',
          'parax_initialization','parax_residual_scale','parax_output_scale_mode','parax_freeze_center_after_task0',
          'parax_task_local_gate','parax_level_conditioned','amp_initial_scale','amp_growth_interval','tf32')
    require(all(c[k]==rc[k] for k in keys),'Frozen/source configurations differ beyond the control')
    require(a['initial_selector_prompt_classifier_sha256']==ra['initial_selector_prompt_classifier_sha256'],'Initialization mismatch')
    require(set(h)==set(d['tasks'])=={'0','1','2'},'Missing task')
    total=0
    for task in range(3):
        x,y=a['tasks'][str(task)],ra['tasks'][str(task)]
        for k in ('sampler_seed','train_instances','train_sample_ids_sha256','first_train_batch_ids_sha256',
                  'validation_sample_ids_sha256','reporting_sample_ids_sha256','frozen_visual_sha256_before'):
            require(x[k]==y[k],f'Paired data mismatch: {task}/{k}')
        require(not x['test_loaded'],'Frozen control loaded test')
        rows=h[str(task)];updates=sum(r['optimizer_steps'] for r in rows);total+=updates
        expected=smoke_updates or 30*math.ceil(x['train_instances']/c['train_batch_size'])
        require(updates==expected and all(r['skipped_optimizer_steps']==0 and r['amp_loss_scale_end']==1024
                and math.isfinite(r['current_loss']) and r['parax_grad_finite']==1 for r in rows),'Invalid updates/AMP')
        if not smoke_updates:require(len(rows)==30,'Incomplete epochs')
        row=d['tasks'][str(task)]
        require(row['old_lane_parameters_unchanged'],'Old task parameters changed')
        if task>0:
            require(row['whole_parax_freeze_verified'] and row['parax_trainable_parameters']==0
                    and not row['shared_parax_changed'] and row['shared_parax_state_hash_after']==d['tasks']['0']['shared_parax_state_hash_after'], 'Freeze failed')
            require(all(r['parax_grad_total_norm']==0 for r in rows),'Frozen ParaX has gradients')
    require(total==s['completed_optimizer_updates'],'Wrong total update count')
    task0_replay=None
    if not smoke_updates:
        p,q=load_scores(source/'val_scores/task0.npz'),load_scores(reference/'val_scores/task0.npz')
        require(all(np.array_equal(p[k],q[k]) for k in ('sample_ids','targets','class_indices')),'Task0 cohort mismatch')
        task0_replay=float(np.abs(p['logits']-q['logits']).max())
        require(task0_replay<=.01,f'Task0 training was not equivalent: {task0_replay}')
        require(c['parax_optimizer_initial_learning_rate']==h['0'][0]['adapter_learning_rate']==.0004,'ParaX lr mismatch')
    result=dict(status='passed',optimization_updates=total,smoke_updates=smoke_updates,
                task0_reference_max_logit_difference=task0_replay,paired_initialization_and_samples=True,
                whole_freeze_verified=True,test_loaded=False,fixed_anchor=d)
    (source/'freeze_control_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('WHOLE_PARAX_CONTROL_AUDIT_PASSED',flush=True)
    return c,result


def complete_validation(source,reference_root,output,clip_checkpoint,data_root,face_manifest_root):
    require(not output.exists(),'Output exists')
    config,audit=audit_control(source,reference_root/'THREE_VIEW_PARAX')
    require(not git_metadata(Path(__file__).resolve().parents[2])['dirty'],'Dirty evaluator')
    require(_sha256(clip_checkpoint)==config['clip_checkpoint_sha256'],'CLIP file changed')
    sources={'baseline':reference_root/'THREE_VIEW','shared_live':reference_root/'THREE_VIEW_PARAX','whole_frozen':source}
    output.mkdir(parents=True);results={};cohort_ids=None;cohort_targets=None;cohort_face=None
    for name,root in sources.items():
        c=json.loads((root/'config.json').read_text())
        require(Path(c['data_root']).resolve()==data_root.resolve(),'Dataset root changed')
        if name!='whole_frozen':audit_source(root)
        files=[root/'config.json']+[root/f'compact_checkpoints/task{t}.pth' for t in range(3)]
        hashes={str(p):_sha256(p) for p in files}
        set_seed(c['seed'],c['tf32']);device=torch.device('cuda')
        model=build_model(c,load_openai_clip_visual(clip_checkpoint),task_sizes=c['task_sizes']).float().to(device)
        clip_hash=tensor_state_digest(model.visual_encoder.state_dict())
        expected_clip=json.loads((root/'paired_protocol_audit.json').read_text())['frozen_visual_sha256_after']
        require(clip_hash==expected_clip,'Restored CLIP differs')
        for split in ('val',):
            require(_sha256(face_manifest_root/f'manifests/{split}.jsonl')==c['face_manifest']['artifact_sha256'][split+'_manifest'],'Manifest changed')
        validation=make_source(c,data_root,face_manifest_root,'val')
        validate_classes(validation)
        require(_sha256(face_manifest_root/'detector_config.json')==c['face_manifest']['artifact_sha256']['detector_config'],'Face detector config changed')
        require(len(validation)==2397,'Incomplete validation')
        loader=DataLoader(LabelView(validation,range(len(validation)),range(26),include_sample_id=True),
                          batch_size=64,num_workers=2,shuffle=False)
        rows={}
        for task in range(3):
            restore_compact_model_state(model,_torch_load(root/f'compact_checkpoints/task{task}.pth'),task,c['git'])
            model.requires_grad_(False).eval();difference=replay(model,validation,root,c,task,device)
            method_hash=tensor_state_digest(compact_model_state_dict(model))
            _,full=infer(model,loader,device,c['amp'],task,c['task_sizes'])
            require(tensor_state_digest(compact_model_state_dict(model))==method_hash
                    and all(p.grad is None for p in model.parameters()),'Inference changed model state or gradients')
            if cohort_targets is None:
                cohort_targets=full['full_targets'].copy();cohort_face=full['face_reliable'].copy()
            require(np.array_equal(cohort_targets,full['full_targets']) and np.array_equal(cohort_face,full['face_reliable']), 'Full labels or Face cohort differ')
            full['targets']=full.pop('full_targets')[:,:sum(TASK_SIZES[:task+1])]
            full['class_indices']=np.arange(full['targets'].shape[1])
            if cohort_ids is None:cohort_ids=full['sample_ids']
            require(np.array_equal(cohort_ids,full['sample_ids']),'Full cohort differs')
            p=output/name;p.mkdir(exist_ok=True)
            np.savez_compressed(p/f'task{task}.npz',**full)
            metrics=score_metrics(full,c['threshold']);start=sum(TASK_SIZES[:task])
            rows[str(task)]=dict(samples=2397,replay_max_logit_difference=difference,metrics=metrics,
                new_class_mAP=float(np.mean(metrics['fused']['per_class_ap'][start:])),
                old_class_mAP=float(np.mean(metrics['fused']['per_class_ap'][:start])) if start else None)
            print(f'WHOLE_FREEZE_FULL_VAL method={name} task={task} mAP={metrics["fused"]["mAP"]:.6f}',flush=True)
        require(tensor_state_digest(model.visual_encoder.state_dict())==clip_hash,'Inference changed CLIP')
        require(all(_sha256(Path(p))==v for p,v in hashes.items()),'Source changed')
        results[name]=dict(tasks=rows,source=str(root.resolve()),source_sha256=hashes)
        del model
        torch.cuda.empty_cache()
    summary=dict(status='complete',split='val',samples=2397,seed=0,training_during_evaluation=False,
                 test_loaded=False,source_configs_fixed=True,freeze_audit=audit,results=results)
    (output/'comparison.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    lines=['# task0后冻结整个ParaX：完整2397人validation对照','',
           '| 完成任务 | 无ParaX | 共享ParaX持续更新 | task0后完全冻结ParaX |','| --- | ---: | ---: | ---: |']
    for task in range(3):
        lines.append(f'| task{task} | '+' | '.join(f'{results[n]["tasks"][str(task)]["metrics"]["fused"]["mAP"]:.4f}' for n in sources)+' |')
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('source','reference-root'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--smoke-updates',type=int,default=0)
    for key in ('output','clip-checkpoint','data-root','face-manifest-root'):p.add_argument('--'+key,type=Path)
    a=p.parse_args()
    if a.smoke_updates:audit_control(a.source,a.reference_root/'THREE_VIEW_PARAX',a.smoke_updates)
    else:complete_validation(a.source,a.reference_root,a.output,a.clip_checkpoint,a.data_root,a.face_manifest_root)
