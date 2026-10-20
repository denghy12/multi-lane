"""Audit four-arm joint-summary training without selecting on test scores."""
import argparse
import json
from pathlib import Path

import torch

from .export_compact_test_scores import _sha256, _torch_load


def verify(root, expected_tasks, baseline):
    read = lambda p: json.loads(p.read_text())
    train = root / 'train'
    config = read(train / 'config.json')
    summary = read(train / 'seed_summary.json')
    protocol = read(train / 'paired_protocol_audit.json')
    reference = read(baseline / 'paired_protocol_audit.json')
    reference_config = read(baseline / 'config.json')
    history = read(train / 'training_history.json')
    assert summary['status'] == 'complete' and summary['config'] == config
    assert config['git']['dirty'] is False and config['max_tasks'] == expected_tasks
    assert len(summary['task_metrics']) == expected_tasks and config['seed'] == 0
    assert config['view_auxiliary_loss_weight'] == 0 and config['adapter_mode'] == 'disabled'
    assert config['parax_mode'] == 'disabled' and protocol['frozen_visual_unchanged']
    for key in ('initial_selector_prompt_classifier_sha256', 'frozen_visual_sha256_before'):
        assert protocol[key] == reference[key], key
    for key in ('seed', 'train_batch_size', 'eval_batch_size', 'optimizer', 'learning_rate',
                'temperature', 'training_loss_mode', 'view_fusion', 'view_classifier_mode',
                'selector_mode', 'prompt_mode', 'input_normalization', 'train_crop_scale'):
        assert config[key] == reference_config[key], key
    if expected_tasks == 8:
        assert config['training_budget_mode'] == 'epochs'
        assert config['epochs_per_task'] == reference_config['epochs_per_task'] == 30
    audits = read(train / 'joint_summary_audit.json') if config['joint_summary_mode'] != 'disabled' else {}
    previous = None
    checks = []
    for task in range(expected_tasks):
        for key in ('train_sample_ids_sha256', 'first_train_batch_ids_sha256',
                    'validation_sample_ids_sha256', 'sampler_seed'):
            assert protocol['tasks'][str(task)][key] == reference['tasks'][str(task)][key], (task, key)
        path = train / 'compact_checkpoints' / f'task{task}.pth'
        payload = _torch_load(path)
        assert payload['source_git'] == config['git'] and payload['task_id'] == task
        state = payload['model']
        if previous is not None:
            assert torch.equal(previous['selectors'][:task], state['selectors'][:task])
            for name in state:
                if name.startswith('prompts.'):
                    assert torch.equal(previous[name][:, :task], state[name][:, :task]), name
                if name.startswith('joint_summary_bank.tasks.') and int(name.split('.')[2]) < task:
                    assert torch.equal(previous[name], state[name]), name
            old_classes = sum(config['task_sizes'][:task])
            for name in ('head.weight', 'head.bias'):
                assert torch.equal(previous[name][:old_classes], state[name][:old_classes]), name
        if audits:
            row = audits[str(task)]
            assert row['old_fusion_unchanged'] and row['before'] == row['after']
            assert row['training_statistics']
            for values in row['training_statistics'].values():
                assert values['samples'] > 0
                if config['joint_summary_mode'] == 'moe':
                    assert abs(sum(values['soft_mass']) - values['samples']) < 1e-4 * values['samples']
                    assert sum(values['hard_load']) == config['joint_summary_top_k'] * values['samples']
        recovery = torch.load(train / 'recovery_checkpoints' / f'task{task}_before_reporting.pth',
                              map_location='cpu', weights_only=False)
        assert recovery['kind'] == 'task_boundary_before_reporting' and recovery['task_id'] == task
        assert recovery['model'].keys() == state.keys()
        assert all(torch.equal(value, state[name]) for name, value in recovery['model'].items())
        assert all(key in recovery['training_state'] for key in ('optimizer', 'scheduler', 'scaler', 'optimizer_parameter_names'))
        assert recovery['sampler_generator'] is not None and recovery['rng']['torch_cuda']
        updates = sum(int(row['optimizer_steps']) for row in history[str(task)])
        skipped = sum(int(row['skipped_optimizer_steps']) for row in history[str(task)])
        assert updates > 0 and skipped == 0
        checks.append(dict(task=task, updates=updates, skipped=skipped,
                           checkpoint_sha256=_sha256(path), recovery_model_equal=True))
        previous = state
    result = dict(status='verified', checks=checks, source_git=config['git'],
                  original_B0=str(baseline), original_B0_auxiliary_weight=reference_config['view_auxiliary_loss_weight'],
                  current_auxiliary_weight=0, mode=config['joint_summary_mode'])
    (root / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--expected-tasks', type=int, default=8)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    verify(args.root, args.expected_tasks, args.baseline)
