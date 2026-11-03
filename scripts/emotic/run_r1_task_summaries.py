"""Train shared R1 senders and compare two task-summary Face readers."""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from multi_lane.track_a.evaluation_scores import align_evaluation_scores, load_evaluation_scores
from multi_lane.track_a.fuse_face_endpoint_validation import load_face_reliability
from multi_lane.track_a.fuse_fixed_three_view_test import (
    LOCKED_FALLBACK_WEIGHTS, LOCKED_RELIABLE_WEIGHTS,
)
from multi_lane.track_a.r1_crossvit_experiment import flags, validation, write
from multi_lane.track_a.runner import compute_metrics, summarize_tasks


def main():
    smoke = '--smoke' in sys.argv
    batch = 'seed1_20261103_smoke' if smoke else 'seed1_20261103'
    root = Path('output/r1_task_summaries') / batch
    logs = Path('logs/r1_task_summaries') / batch
    logs.mkdir(parents=True, exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('GPU0 required')
    reference = Path('output/r1_crossvit/seed1_20261101_02')
    inputs = dict(
        reference_root=Path('../emotic_benchmark_runs'),
        data_root=Path('../multi-lane-main/datasets/EMOTIC'),
        clip_checkpoint=Path('../CODE_DDP/pretrained/clip/ViT-B-16.pt'),
        face_manifest_root=Path('../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909'),
    )

    def train(kind, view):
        args = SimpleNamespace(output=root, **inputs)
        command = flags(args, kind, view, smoke)
        if view == 'face':
            command += ['--r1-sender-root', str(root / 'native')]
        with (logs / f'{kind}_{view}.log').open('a') as log:
            subprocess.run([sys.executable, '-u', '-m', 'multi_lane.track_a.runner', *command],
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def parallel(jobs):
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda job: train(*job), jobs))

    senders = [('native', 'full'), ('native', 'person')]
    receivers = [('pool', 'face'), ('attention', 'face')]
    parallel(senders)
    parallel(receivers)
    for kind, view in senders + receivers:
        validation(SimpleNamespace(**inputs, training_root=root / kind / view / 'train',
                                   output=root / kind / view / 'val', smoke=smoke))
    if smoke:
        write(root / 'completion.json', {'status': 'complete', 'smoke': True})
        return

    results = {}
    for split in ('val', 'test'):
        reliability = load_face_reliability(inputs['face_manifest_root'], split)
        rows = {name: [] for name in ('native', 'patch_pool', 'patch_attention', 'summary_pool', 'summary_attention')}
        subsets = {name: [] for name in rows}
        for stage in range(8):
            def load(directory, kind, view):
                scores = 'val/scores' if split == 'val' else 'train/test_scores'
                return load_evaluation_scores(directory / kind / view / scores / f'task{stage}.npz', 64)

            full = load(root, 'native', 'full')
            person = load(root, 'native', 'person')
            for view, current in [('full', full), ('person', person)]:
                _, actual, expected, _, ap, ep = align_evaluation_scores(current, load(reference, 'native', view))
                if not (np.array_equal(actual, expected) and np.array_equal(ap, ep)):
                    raise ValueError(f'Native {view} replay failed at {split} stage {stage}')
            ids, f, p, targets, _, _ = align_evaluation_scores(full, person)
            mask = np.array([reliability[i] for i in ids])
            weights = np.where(mask[:, None], LOCKED_RELIABLE_WEIGHTS, LOCKED_FALLBACK_WEIGHTS)
            sources = [
                ('native', reference, 'native'), ('patch_pool', reference, 'pool'),
                ('patch_attention', reference, 'attention'), ('summary_pool', root, 'pool'),
                ('summary_attention', root, 'attention'),
            ]
            for name, directory, kind in sources:
                _, _, z, _, _, _ = align_evaluation_scores(full, load(directory, kind, 'face'))
                logits = weights[:, 0, None] * f + weights[:, 1, None] * p + weights[:, 2, None] * z
                if name == 'native':
                    fallback = logits[~mask].copy()
                if not np.array_equal(logits[~mask], fallback):
                    raise ValueError(f'Unreliable Face replay failed at {split} stage {stage}')
                scores = torch.from_numpy(logits).sigmoid()
                labels = torch.from_numpy(targets)
                rows[name].append(compute_metrics(stage, scores, labels, .5))
                subsets[name].append(compute_metrics(stage, scores[mask], labels[mask], .5))
        results[split] = {
            name: dict(summary=summarize_tasks(values), reliable=summarize_tasks(subsets[name]),
                       stages=[row.mAP for row in values],
                       reliable_stages=[row.mAP for row in subsets[name]])
            for name, values in rows.items()
        }
    write(root / 'completion.json', dict(status='complete', replay_exact=True, results=results))


if __name__ == '__main__':
    main()
