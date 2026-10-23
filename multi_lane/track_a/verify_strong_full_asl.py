"""Audit full-TRAIN budgets, parameter freezing and complete recovery before test."""
import argparse
import json
import math
from pathlib import Path
import torch
from .strong_full_asl import ARMS, arm_for, validate_config
from .verify_spatial_summary import read, load_boundary
from .export_compact_test_scores import _torch_load, _sha256
from .spatial_resume import checkpoint_source_git


def verify(root, tasks=8, smoke=False, arms=ARMS, output_name="training_verification.json"):
    reference = reference_protocol = splits_reference = None
    checks=[]
    keys=('seed','task_sizes','train_batch_size','eval_batch_size','epochs_per_task',
        'optimizer_updates_per_task','learning_rate','optimizer','scheduler','weight_decay',
        'calibration_fraction','view_auxiliary_loss_weight','train_crop_scale',
        'person_crop_margin','full_crop_mode','input_normalization','training_loss_mode',
        'workers','eval_workers','amp','tf32','amp_initial_scale','amp_growth_interval',
        'git','initial_selector_prompt_classifier_sha256','frozen_visual_sha256_before')
    for arm in arms:
        train=root/arm/'train';c=read(train/'config.json');s=read(train/'seed_summary.json')
        validate_config(c);assert arm_for(c)==arm
        assert s['status']=='complete' and s['config']==c and not c['git']['dirty']
        audit=read(train/'paired_protocol_audit.json');history=read(train/'training_history.json')
        assert audit['frozen_visual_unchanged'] and c['max_tasks']==tasks
        splits=[read(train/'calibration_split_provenance'/f'task{t}.json') for t in range(tasks)]
        assert all(not z['calibration_sample_ids'] and z['fit_sample_ids'] for z in splits)
        if reference is None:reference,reference_protocol,splits_reference=c,audit,splits
        else:
            assert all(c[k]==reference[k] for k in keys),(arm,'config')
            assert splits==splits_reference,(arm,'split')
        previous=None;updates=0
        for t in range(tasks):
            for key in ('train_sample_ids_sha256','first_train_batch_ids_sha256','validation_sample_ids_sha256','sampler_seed'):
                assert audit['tasks'][str(t)][key]==reference_protocol['tasks'][str(t)][key],(arm,t,key)
            path=train/'compact_checkpoints'/f'task{t}.pth';p=_torch_load(path);state=p['model']
            assert p['source_git']==checkpoint_source_git(c,t) and p['task_id']==t
            assert all(torch.isfinite(v).all() for v in state.values())
            if previous is not None:
                assert torch.equal(previous['selectors'][:t],state['selectors'][:t])
                for n in state:
                    if n.startswith('prompts.'):assert torch.equal(previous[n][:,:t],state[n][:,:t]),n
                    for prefix in ('adapter_bank.task_adapters.','cross_view_bank.tasks.','spatial_summary_bank.tasks.'):
                        if n.startswith(prefix) and int(n[len(prefix):].split('.')[0])<t:assert torch.equal(previous[n],state[n]),n
                k=sum(c['task_sizes'][:t])
                for n in ('head.weight','head.bias'):assert torch.equal(previous[n][:k],state[n][:k])
            rows=history[str(t)];assert sum(z['skipped_optimizer_steps'] for z in rows)==0
            count=sum(z['optimizer_steps'] for z in rows)
            assert count==(2 if smoke else 30*math.ceil(len(splits[t]['fit_sample_ids'])/64)),(arm,t,count)
            updates+=count
            assert c['adapter_parameters_per_task']==49952
            assert state[f'adapter_bank.task_adapters.{t}.1.up.weight'].abs().sum()>0
            for name,key in [('cross_view','old_fusion_unchanged'),('spatial_summary','old_spatial_unchanged')]:
                if (train/(name+'_audit.json')).exists():
                    row=read(train/(name+'_audit.json'))[str(t)]
                    assert row[key] and row['before']==row['after']
            recovery=load_boundary(train/'recovery_checkpoints'/f'task{t}_before_reporting.pth')
            assert recovery['model'].keys()==state.keys()
            assert all(torch.equal(v,recovery['model'][n]) for n,v in state.items())
            assert recovery['sampler_generator'] is not None and recovery['rng']['torch_cuda']
            assert all(k in recovery['training_state'] for k in ('optimizer','scheduler','scaler','optimizer_parameter_names'))
            previous=state
        assert updates==(tasks*2 if smoke else 13950)
        checks.append(dict(arm=arm,updates=updates,skipped=0,full_train=True,old_parameters_unchanged=True,
            recovery_equals_compact=True,checkpoint_sha256=_sha256(path),
            peak_allocated_mib=audit['peak_allocated_mib'],peak_reserved_mib=audit['peak_reserved_mib']))
    result=dict(status='verified',tasks=tasks,smoke=smoke,checks=checks)
    (root/output_name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--tasks',type=int,default=8);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();verify(a.root,a.tasks,a.smoke)
