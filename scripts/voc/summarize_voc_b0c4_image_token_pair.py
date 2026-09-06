#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_run(path: Path):
    config = json.loads((path / 'run_config.json').read_text(encoding='utf-8'))
    detail_files = sorted((path / 'detail').glob('*.json'))
    if len(detail_files) != 1:
        raise RuntimeError(f'Expected one detail JSON in {path}, found {len(detail_files)}')
    detail = json.loads(detail_files[0].read_text(encoding='utf-8'))
    rows = detail['overall_rows']
    if [row['task'] for row in rows] != list(range(5)):
        raise RuntimeError(f'{path} does not contain a complete five-task B0-C4 run')
    return config, detail


def validate_pair(control, adapter):
    ignored = {
        'name', 'notes', 'output_dir', 'adapter_mode', 'loss_routing',
        'rank', 'gpu', 'gpu_name', 'distributed', 'dist_backend',
        'trainable_parameters', 'adapter_parameters_per_task',
    }
    mismatches = {}
    for key in sorted(set(control) | set(adapter)):
        if key not in ignored and control.get(key) != adapter.get(key):
            mismatches[key] = [control.get(key), adapter.get(key)]
    if mismatches:
        raise RuntimeError(f'Paired configurations differ outside treatment: {mismatches}')
    if control.get('adapter_mode') != 'disabled' or control.get('loss_routing') != 'joint_bce':
        raise RuntimeError('Control is not the original Adapter-disabled BCE treatment')
    if adapter.get('adapter_mode') != 'image_token' or adapter.get('loss_routing') != 'adapter_asl':
        raise RuntimeError('Candidate is not Image-token Adapter ASL')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-root', type=Path, required=True)
    args = parser.parse_args()
    control_config, control_detail = load_run(args.batch_root / 'control')
    adapter_config, adapter_detail = load_run(args.batch_root / 'adapter')
    validate_pair(control_config, adapter_config)
    control_rows = control_detail['overall_rows']
    adapter_rows = adapter_detail['overall_rows']
    per_task = []
    for control_row, adapter_row in zip(control_rows, adapter_rows):
        per_task.append({
            'task': control_row['task'],
            'control_mAP': control_row['mAP'],
            'adapter_mAP': adapter_row['mAP'],
            'delta_mAP': adapter_row['mAP'] - control_row['mAP'],
        })
    control_average = sum(row['mAP'] for row in control_rows) / len(control_rows)
    adapter_average = sum(row['mAP'] for row in adapter_rows) / len(adapter_rows)
    summary = {
        'protocol': 'VOC2007 B0-C4, seed0, five tasks x four classes',
        'git_commit': control_config['git']['commit'],
        'paper_reference': {'average_mAP': 93.5, 'final_mAP': 88.8},
        'control': {
            'average_mAP': control_average,
            'final_mAP': control_rows[-1]['mAP'],
        },
        'image_token_adapter': {
            'average_mAP': adapter_average,
            'final_mAP': adapter_rows[-1]['mAP'],
        },
        'delta': {
            'average_mAP': adapter_average - control_average,
            'final_mAP': adapter_rows[-1]['mAP'] - control_rows[-1]['mAP'],
        },
        'per_task': per_task,
    }
    destination = args.batch_root / 'comparison_summary.json'
    destination.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Saved {destination}')


if __name__ == '__main__':
    main()
