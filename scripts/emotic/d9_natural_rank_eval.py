"""Evaluate a natural-image ordering loss against the paired BCE control."""
import argparse
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, TaskMetrics, summarize_tasks
from scripts.emotic.d9_a_eight_eval import GROUPS, predictions, read, require_digest, verify_seal
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.evaluate_d9_relation import metrics_for_stage, weighted_logits
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS
from scripts.emotic.fit_d9_task10_probe import weights


METHODS = ('fixed', 'paired', 'rank', 'rank_reliable_only', 'rank_unreliable_only')


def val_decision(summary):
    delta = {k: summary['rank']['all'][k] - summary['paired']['all'][k] for k in MAP_KEYS}
    passes = min(delta.values()) >= -1e-10 and max(delta.values()) > 1e-10
    return dict(winner='rank' if passes else 'paired', control='paired', population='all',
                rank_minus_paired=delta, eligible=['rank'] if passes else [], tolerance=1e-10)


def group_attributions(paired, rank, reliable):
    mask = reliable[:, None]
    return dict(rank_reliable_only=torch.where(mask, rank, paired),
                rank_unreliable_only=torch.where(mask, paired, rank))


def evaluate(root, historical, out, split, smoke=False):
    if split not in ('val', 'test'):
        raise ValueError('Expected a natural evaluation split')
    lock_path = out / 'centers_lock.json'
    lock = read(lock_path)
    tasks = lock['tasks']
    stage_count = 1 if smoke else len(TASK_SIZES)
    if lock['smoke'] != smoke or set(tasks) != {str(t) for t in range(stage_count)}:
        raise ValueError('TRAIN centers do not match the requested stages')
    require_digest(root / 'config.json', lock['config_sha256'])
    native = root / 'evaluation'
    native_val_sha = None
    lock_hash = sha256_file(lock_path)
    seal_path = out / 'validation_seal.json'

    # Complete all validation-seal checks before even opening the Test metadata.
    if split == 'test':
        seal = verify_seal(out, lock_hash)
        validation = read(out / 'val_results.json')
        if seal['decision'] != val_decision(validation['summary']) or seal['decision'] != validation['decision']:
            raise ValueError('Sealed decision differs from the Validation results')
        native_val_sha = seal['native_val_sha256']
        require_digest(native / 'val/result.json', native_val_sha)
        require_digest(historical / 'val_results.json', seal['historical_val_sha256'])

    previous_path = historical / f'{split}_results.json'
    previous = read(previous_path)
    old_lock_path = historical / 'centers_lock.json'
    require_digest(old_lock_path, previous['centers_sha256'])
    old_tasks = read(old_lock_path)['tasks']
    if split == 'test':
        require_digest(historical / 'validation_seal.json', previous['validation_seal_sha256'])
    source_path = native / split / 'result.json'
    require_digest(source_path, previous['native_results_sha256'])
    source = read(source_path)
    if (source['mode'] != 'sample_holdout' or source['smoke'] or previous['smoke']
            or any(run['split'] != split or len(run['stages']) != len(TASK_SIZES) for run in (source, previous))):
        raise ValueError('Expected the historical eight-stage protected holdout evaluation')

    centers = {k: task['paired'] for k, task in tasks.items()}
    stages = []
    for t in range(stage_count):
        key = str(t)
        old, stage = previous['stages'][t], source['stages'][t]
        expected = dict(task=t, split=split, smoke=False, config_sha256=lock['config_sha256'],
                        checkpoint_sha256=tasks[key]['checkpoint_sha256'], validation_result_sha256=native_val_sha)
        if (old['stage'] != t or old['scores_sha256'] != stage['scores_sha256']
                or any(stage['protocol'][k] != v for k, v in expected.items())
                or centers[key]['center'] != old_tasks[key]['paired']['center']):
            raise ValueError(f'Historical source or paired center changed at stage {t}')
        archive = native / split / f'task{t}.npz'
        require_digest(archive, old['scores_sha256'])
        torch.set_num_threads(1)
        with np.load(archive, allow_pickle=False) as data:
            baseline, targets, reliable = predictions(data, centers, t)
            by_head = [weights(torch.as_tensor(tasks[str(h)]['rank']['center'], dtype=torch.float32), reliable)
                       for h in range(t + 1)]
            rank = weighted_logits(torch.from_numpy(data['branch_logits']), torch.stack(by_head, dim=1),
                                   TASK_SIZES[:t+1])
            if not torch.isfinite(rank).all():
                raise ValueError(f'Nonfinite ranking logits at stage {t}')
            paired = baseline['A']
            logits = dict(fixed=baseline['fixed'], paired=paired, rank=rank,
                          **group_attributions(paired, rank, reliable))
        torch.set_num_threads(2)
        metrics = metrics_for_stage(t, targets, reliable, logits)
        if any(metrics[k] != old['metrics'][k] for k in ('fixed', 'paired')):
            raise ValueError(f'Historical evaluation failed to replay at {split}/task{t}')
        stages.append(dict(stage=t, people=len(targets), reliable_face=int(reliable.sum()),
                           scores_sha256=old['scores_sha256'], metrics=metrics))
    summary = {m: {g: summarize_tasks([TaskMetrics(**s['metrics'][m][g]) for s in stages])
                   for g in GROUPS} for m in METHODS}
    decision = val_decision(summary) if split == 'val' else seal['decision']
    result = dict(split=split, smoke=smoke, summary=summary, stages=stages, decision=decision,
                  centers_sha256=lock_hash, historical_sha256=sha256_file(previous_path),
                  native_results_sha256=previous['native_results_sha256'])
    result_path = out / f'{split}_results.json'
    if split == 'test':
        result['validation_seal_sha256'] = sha256_file(seal_path)
    write_verified(result_path, result)
    if split == 'val':
        write_verified(seal_path, dict(decision=decision, centers_sha256=lock_hash,
                                       val_results_sha256=sha256_file(result_path),
                                       native_val_sha256=result['native_results_sha256'],
                                       historical_val_sha256=result['historical_sha256']))
    print(split, {m: {k: summary[m]['all'][k] for k in MAP_KEYS} for m in METHODS}, decision, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ('root', 'historical', 'out'):
        cli.add_argument('--' + arg, type=Path, required=True)
    cli.add_argument('--smoke', action='store_true')
    cli.add_argument('--split', choices=('val','test'), required=True)
    evaluate(**vars(cli.parse_args()))
