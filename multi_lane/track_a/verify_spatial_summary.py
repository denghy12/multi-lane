"""Verify paired training budgets, original-image holdout and old dependencies."""
import argparse
from .spatial_resume import checkpoint_source_git
import json
import math
from pathlib import Path
import torch
from .export_compact_test_scores import _torch_load, _sha256


def read(path):
    return json.loads(path.read_text())


def load_boundary(path):
    # These are our locally generated, trusted complete training snapshots.
    # RNG/optimizer provenance includes Python objects (e.g. TorchVersion),
    # unlike the compact tensor-only checkpoints loaded by _torch_load.
    return torch.load(path, map_location='cpu', weights_only=False)


def verify(root, tasks=8, smoke=False):
    checks = []
    reference = None
    reference_protocol = None
    reference_splits = None
    invariant = ('seed', 'task_sizes', 'train_batch_size', 'eval_batch_size', 'epochs_per_task',
                 'learning_rate', 'optimizer', 'scheduler', 'weight_decay', 'workers',
                 'calibration_fraction', 'view_auxiliary_loss_weight', 'view_fusion',
                 'view_classifier_mode', 'train_crop_scale', 'input_normalization', 'amp', 'tf32',
                 'full_crop_mode', 'person_crop_margin', 'training_loss_mode',
                 'amp_initial_scale', 'amp_growth_interval', 'max_tasks')
    for arm, mode in [('baseline', 'disabled'), ('absolute', 'absolute'), ('target', 'target')]:
        train = root / arm / 'train'
        config, summary = read(train / 'config.json'), read(train / 'seed_summary.json')
        protocol, history = read(train / 'paired_protocol_audit.json'), read(train / 'training_history.json')
        assert summary['status'] == 'complete' and summary['config'] == config
        assert config['git']['dirty'] is False and config['seed'] == 0 and config['max_tasks'] == tasks
        assert config['spatial_summary_mode'] == mode and config['view_auxiliary_loss_weight'] == .1
        assert config['calibration_fraction'] == .1 and config['spatial_summary_binding_loss_weight'] == 0
        assert config['adapter_mode'] == config['parax_mode'] == config['joint_summary_mode'] == config['cross_view_mode'] == 'disabled'
        assert protocol['frozen_visual_unchanged']
        splits = [read(train / 'calibration_split_provenance' / f'task{task}.json') for task in range(tasks)]
        all_fit = {x.rsplit('#person=', 1)[0] for split in splits for x in split['fit_sample_ids']}
        all_held = {x.rsplit('#person=', 1)[0] for split in splits for x in split['calibration_sample_ids']}
        assert all_fit and all_held and not all_fit & all_held
        if reference is None:
            reference, reference_protocol, reference_splits = config, protocol, splits
        else:
            assert all(config[key] == reference[key] for key in invariant)
            assert splits == reference_splits
            for key in ('initial_selector_prompt_classifier_sha256', 'frozen_visual_sha256_before'):
                assert protocol[key] == reference_protocol[key], key
        previous = None
        total = 0
        audits = read(train / 'spatial_summary_audit.json') if mode != 'disabled' else {}
        for task in range(tasks):
            for key in ('train_sample_ids_sha256', 'first_train_batch_ids_sha256', 'validation_sample_ids_sha256', 'sampler_seed'):
                assert protocol['tasks'][str(task)][key] == reference_protocol['tasks'][str(task)][key]
            path = train / 'compact_checkpoints' / f'task{task}.pth'
            payload = _torch_load(path); state = payload['model']
            assert payload['source_git'] == checkpoint_source_git(config, task) and payload['task_id'] == task
            assert all(torch.isfinite(v).all() for v in state.values())
            if previous is not None:
                assert torch.equal(previous['selectors'][:task], state['selectors'][:task])
                for name in state:
                    if name.startswith('prompts.'):
                        assert torch.equal(previous[name][:, :task], state[name][:, :task]), name
                    if name.startswith('spatial_summary_bank.tasks.') and int(name.split('.')[2]) < task:
                        assert torch.equal(previous[name], state[name]), name
                classes = sum(config['task_sizes'][:task])
                for name in ('head.weight', 'head.bias'):
                    assert torch.equal(previous[name][:classes], state[name][:classes]), name
            updates = sum(row['optimizer_steps'] for row in history[str(task)])
            skipped = sum(row['skipped_optimizer_steps'] for row in history[str(task)])
            assert skipped == 0 and updates == (2 if smoke else 30 * math.ceil(len(splits[task]['fit_sample_ids']) / 64))
            total += updates
            if mode != 'disabled':
                row = audits[str(task)]
                assert row['old_spatial_unchanged'] and row['before'] == row['after']
                assert row['parameter_norm'] > 0 and row['final_gradient_norm'] is not None and math.isfinite(row['final_gradient_norm'])
                assert set(row['training_statistics']) == {f'task{task}/layer{layer}' for layer in range(12)}
                assert config['spatial_summary_parameters_per_task'] == 60
                assert state[f'spatial_summary_bank.tasks.{task}'].shape == (10, 6)
            recovery = load_boundary(train / 'recovery_checkpoints' / f'task{task}_before_reporting.pth')
            assert recovery['task_id'] == task and recovery['kind'] == 'task_boundary_before_reporting'
            assert recovery['model'].keys() == state.keys()
            assert all(torch.equal(value, recovery['model'][name]) for name, value in state.items())
            assert all(key in recovery['training_state'] for key in ('optimizer', 'scheduler', 'scaler', 'optimizer_parameter_names'))
            assert recovery['sampler_generator'] is not None and recovery['rng']['torch_cuda']
            previous = state
        if not smoke:
            assert tasks == 8 and total == 12660
        checks.append(dict(arm=arm, updates=total, skipped=0, holdout_image_disjoint=True,
                           old_parameters_unchanged=True, recovery_compact_equal=True,
                           final_checkpoint_sha256=_sha256(train / 'compact_checkpoints' / f'task{tasks-1}.pth'),
                           peak_allocated_mib=protocol['peak_allocated_mib'], peak_reserved_mib=protocol['peak_reserved_mib']))
    result = dict(status='verified', smoke=smoke, tasks=tasks, checks=checks)
    (root / 'training_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--tasks', type=int, default=8)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    verify(args.root, args.tasks, args.smoke)
