"""Verify exact protected branch replay at every completed task."""
import argparse
from pathlib import Path

import torch

from multi_lane.track_a.r1_crossvit_experiment import read, write


def verify(root, reference, smoke):
    tasks = 2 if smoke else 8
    new, old = (read(p / 'config.json') for p in (root, reference))
    if new['mode'] not in ('multi', 'direction') or old['mode'] != 'fixed':
        raise ValueError('Expected new multi-readout model and historical fixed model')
    for config in (new, old):
        if config['smoke'] != smoke or config['tasks'] != tasks or not config['protect_branches']:
            raise ValueError('Smoke/full, task count or branch protection mismatch')
    keys = ('branches', 'seed', 'epochs', 'batch_size', 'micro_batch_size', 'workers',
            'precision', 'clip_checkpoint', 'deterministic_algorithms')
    if any(new[k] != old[k] for k in keys):
        raise ValueError('Branch recipe differs')
    result = []
    for task in range(tasks):
        artifact = Path('compact_checkpoints') / f'task{task}.pth'
        states = [torch.load(base / artifact, map_location='cpu', weights_only=False)
                  for base in (root, reference)]
        a, b = states
        if a['coverage'] != b['coverage'] or a['task'] != task or b['task'] != task:
            raise ValueError(f'Cohort mismatch at task {task}')
        tensors = 0
        for view in ('full', 'person', 'face'):
            left, right = (s['branches'][view] for s in states)
            if left.keys() != right.keys():
                raise ValueError(f'Branch state keys differ, task {task} {view}')
            for name, value in left.items():
                ref = right[name]
                if ref.dtype != value.dtype or not torch.equal(value, ref):
                    raise ValueError(f'Branch replay mismatch, task {task} {view}/{name}')
                tensors += 1
        updates = sum(r['updates'] for r in a['history'])
        if updates != sum(r['updates'] for r in b['history']) or any(r['skipped'] for r in a['history']):
            raise ValueError(f'Update budget mismatch, task {task}')
        row = dict(task=task, tensors=tensors, updates=updates, coverage=a['coverage'])
        if new['mode'] == 'direction':
            budget = 2 if smoke else 300
            expected_gates = {'center', 'a', 'b', 'c', 'center_b'}
            for epoch, record in enumerate(a['history'], start=1):
                gates = record['train_readouts']
                expected_updates = budget // new['epochs']
                if set(gates) != expected_gates or any(
                    gate['gate_updates'] != expected_updates
                    or gate['gate_total_updates'] != epoch * expected_updates
                    for gate in gates.values()
                ):
                    raise ValueError(f'Direction update budget differs at task {task}, epoch {epoch}')
                for key in ('direction_supervision_counts', 'direction_positive_counts'):
                    if gates['b'][key] != gates['c'][key]:
                        raise ValueError(f'Direction target counts differ at task {task}, epoch {epoch}')
            row['updates_per_gate'] = budget
        result.append(row)
        print(f'Exact branch state replay task {task}: {tensors} tensors, {updates} updates', flush=True)
    write(root / 'protected_replay.json', dict(reference=str(reference), smoke=smoke, tasks=result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    verify(args.root, args.reference_root, args.smoke)
