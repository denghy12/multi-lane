"""Audit all eight factorial runs before evaluating held-out test."""
import argparse
import json
import math
from pathlib import Path
import torch
from .module_stacking import ARMS, modes, validate_config
from .spatial_resume import checkpoint_source_git
from .verify_spatial_summary import read, load_boundary
from .export_compact_test_scores import _torch_load, _sha256


def verify(root, tasks=8, smoke=False):
    reference = reference_protocol = reference_splits = None
    checks = []
    common_keys = ('seed', 'task_sizes', 'train_batch_size', 'eval_batch_size', 'epochs_per_task',
        'optimizer_updates_per_task', 'learning_rate', 'optimizer', 'scheduler', 'weight_decay',
        'calibration_fraction', 'view_auxiliary_loss_weight', 'view_fusion', 'view_classifier_mode',
        'train_crop_scale', 'person_crop_margin', 'full_crop_mode', 'input_normalization',
        'training_loss_mode', 'parameter_group_loss_routing', 'workers', 'eval_workers',
        'amp', 'tf32', 'amp_initial_scale', 'amp_growth_interval', 'module_stacking', 'git',
        'initial_selector_prompt_classifier_sha256', 'frozen_visual_sha256_before')
    for arm in ARMS:
        train = root / arm / 'train'
        config, summary = read(train/'config.json'), read(train/'seed_summary.json')
        validate_config(config)
        assert all(config[k] == v for k,v in modes(arm).items()), arm
        assert summary['status'] == 'complete' and summary['config'] == config
        assert not config['git']['dirty'] and config['max_tasks'] == tasks
        protocol, history = read(train/'paired_protocol_audit.json'), read(train/'training_history.json')
        assert protocol['frozen_visual_unchanged']
        splits = [read(train/'calibration_split_provenance'/f'task{task}.json') for task in range(tasks)]
        fit = {x.rsplit('#person=',1)[0] for s in splits for x in s['fit_sample_ids']}
        held = {x.rsplit('#person=',1)[0] for s in splits for x in s['calibration_sample_ids']}
        assert fit and held and not fit & held
        if reference is None:
            reference, reference_protocol, reference_splits = config, protocol, splits
        else:
            assert all(config[k] == reference[k] for k in common_keys), arm
            assert splits == reference_splits, arm
        previous = None; updates = 0
        for task in range(tasks):
            for key in ('train_sample_ids_sha256', 'first_train_batch_ids_sha256', 'validation_sample_ids_sha256', 'sampler_seed'):
                assert protocol['tasks'][str(task)][key] == reference_protocol['tasks'][str(task)][key], (arm,task,key)
            path = train/'compact_checkpoints'/f'task{task}.pth'
            payload = _torch_load(path); state = payload['model']
            assert payload['source_git'] == checkpoint_source_git(config, task) and payload['task_id'] == task
            assert all(torch.isfinite(v).all() for v in state.values())
            if previous is not None:
                assert torch.equal(previous['selectors'][:task], state['selectors'][:task])
                for name in state:
                    if name.startswith('prompts.'):
                        assert torch.equal(previous[name][:,:task], state[name][:,:task]), name
                    for prefix in ('adapter_bank.task_adapters.', 'cross_view_bank.tasks.', 'spatial_summary_bank.tasks.'):
                        if name.startswith(prefix) and int(name[len(prefix):].split('.')[0]) < task:
                            assert torch.equal(previous[name], state[name]), name
                classes = sum(config['task_sizes'][:task])
                for name in ('head.weight','head.bias'):
                    assert torch.equal(previous[name][:classes], state[name][:classes])
            rows = history[str(task)]
            assert sum(r['skipped_optimizer_steps'] for r in rows) == 0
            count = sum(r['optimizer_steps'] for r in rows)
            assert count == (2 if smoke else 30 * math.ceil(len(splits[task]['fit_sample_ids'])/64)), (arm,task,count)
            updates += count
            if arm[1] == '1':
                assert config['adapter_parameters_per_task'] == 49952
                assert state[f'adapter_bank.task_adapters.{task}.1.up.weight'].abs().sum() > 0
            if arm[2] == '1':
                audit = read(train/'cross_view_audit.json')[str(task)]
                assert audit['old_fusion_unchanged'] and audit['before'] == audit['after']
                assert config['cross_view_parameters_per_task'] == 196608
            if arm[3] == '1':
                audit = read(train/'spatial_summary_audit.json')[str(task)]
                assert audit['old_spatial_unchanged'] and audit['before'] == audit['after']
                assert config['spatial_summary_parameters_per_task'] == 60 and audit['parameter_norm'] > 0
                assert audit['final_gradient_norm'] is not None and math.isfinite(audit['final_gradient_norm'])
                assert set(audit['training_statistics']) == {f'task{task}/layer{layer}' for layer in range(12)}
            recovery = load_boundary(train/'recovery_checkpoints'/f'task{task}_before_reporting.pth')
            assert recovery['task_id'] == task and recovery['model'].keys() == state.keys()
            assert all(torch.equal(v, recovery['model'][n]) for n,v in state.items())
            assert recovery['sampler_generator'] is not None and recovery['rng']['torch_cuda']
            assert all(k in recovery['training_state'] for k in ('optimizer','scheduler','scaler','optimizer_parameter_names'))
            previous = state
        assert updates == (tasks*2 if smoke else 12660)
        checks.append(dict(arm=arm, updates=updates, skipped=0, image_holdout_disjoint=True,
            old_parameters_unchanged=True, recovery_equals_compact=True,
            checkpoint_sha256=_sha256(path), peak_allocated_mib=protocol['peak_allocated_mib'],
            peak_reserved_mib=protocol['peak_reserved_mib']))
    result = dict(status='verified', tasks=tasks, smoke=smoke, checks=checks)
    (root/'training_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('--tasks',type=int,default=8)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    verify(args.root,args.tasks,args.smoke)
