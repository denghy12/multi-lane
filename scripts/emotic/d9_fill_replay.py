"""Exact branch and original-readout replay audit before a new task is published."""
import torch
from multi_lane.track_a.r1_crossvit_experiment import sha


def verify_task_replay(current, reference, task):
    path = reference / 'compact_checkpoints' / f'task{task}.pth'
    old = torch.load(path, map_location='cpu', weights_only=False)
    counts = {}
    if old['task'] != task or current['task'] != task:
        raise ValueError('Replay source task mismatch')
    for view in ('full', 'person', 'face'):
        a, b = current['branches'][view], old['branches'][view]
        if a.keys() != b.keys():
            raise ValueError(f'{view} branch state keys differ')
        for key in a:
            if not torch.equal(a[key].cpu(), b[key].cpu()):
                raise ValueError(f'Branch replay mismatch: task{task}/{view}/{key}')
        counts[view] = len(a)
    for key, value in old['gates'].items():
        if key not in current['gates'] or not torch.equal(current['gates'][key].cpu(), value.cpu()):
            raise ValueError(f'Original fusion readout changed: task{task}/{key}')
    if current['coverage'] != old['coverage']:
        raise ValueError('TRAIN branch coverage differs from replay source')
    for row, ref in zip(current['history'], old['history']):
        for key in ('task', 'epoch', 'examples', 'updates', 'skipped', 'fp32_retries', 'sampling', 'branch_lrs'):
            if row.get(key) != ref.get(key):
                raise ValueError(f'Branch training clock differs at task{task}/{row["epoch"]}/{key}')
    if len(current['history']) != len(old['history']):
        raise ValueError('Replay epoch history length differs')
    return dict(task=task, status='exact', source=str(path), source_sha256=sha(path),
                branch_tensors=counts, original_gate_tensors=len(old['gates']))


def verify_prediction_replay(root, reference, split, smoke):
    import json
    import numpy as np
    directory = 'smoke_evaluation' if smoke else 'evaluation'
    task_count = 2 if smoke else 8
    reports = []
    for task in range(task_count):
        now_path = root / directory / split / f'task{task}.npz'
        ref_path = reference / directory / split / f'task{task}.npz'
        with np.load(now_path, allow_pickle=False) as now, np.load(ref_path, allow_pickle=False) as old:
            now_ids, old_ids = now['sample_ids'].tolist(), old['sample_ids'].tolist()
            if len(set(now_ids)) != len(now_ids) or set(now_ids) != set(old_ids):
                raise ValueError('Prediction replay identities differ')
            order = {key: i for i, key in enumerate(old_ids)}
            aligned = [order[key] for key in now_ids]
            for key in ('targets', 'reliable', 'branch_logits', 'fixed_weights', 'fixed_logits'):
                if not np.array_equal(now[key], old[key][aligned]):
                    raise ValueError(f'Prediction replay mismatch: {split}/task{task}/{key}')
        reports.append(dict(task=task, people=len(now_ids), status='exact', source_sha256=sha(ref_path)))
    from multi_lane.track_a.r1_crossvit_experiment import write
    write(root / 'branch_replay' / f'{split}_predictions.json', dict(status='exact', stages=reports))


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('val', 'test'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    verify_prediction_replay(args.root, args.reference, args.split, args.smoke)
