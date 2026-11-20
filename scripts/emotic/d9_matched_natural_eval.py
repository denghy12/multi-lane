"""Measure the effect of lowered TRAIN resolution using untouched natural evaluation images."""
import argparse
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, TaskMetrics, summarize_tasks
from scripts.emotic.d9_a_eight_eval import GROUPS, predictions, read, require_digest
from scripts.emotic.d9_paired_a_evaluate import METHODS, diff_by_stage, write_verified
from scripts.emotic.evaluate_d9_relation import metrics_for_stage, weighted_logits
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS
from scripts.emotic.fit_d9_task10_probe import weights


def evaluate(root, out, split):
    lock_path = out/'centers_lock.json'
    lock_hash = sha256_file(lock_path)
    tasks = read(lock_path)['tasks']
    if set(tasks) != {str(t) for t in range(len(TASK_SIZES))}:
        raise ValueError('Incomplete set of task centers')
    seal_path = out/'validation_seal.json'
    native_dir = root/'evaluation'
    reference_dir = root/'paired_A'
    val_sha = None
    if split == 'test':
        seal = read(seal_path)
        require_digest(lock_path, seal['centers_sha256'])
        require_digest(out/'val_results.json', seal['val_results_sha256'])
        val_sha = seal['native_val_sha256']
        require_digest(native_dir/'val/result.json', val_sha)
        require_digest(reference_dir/'val_results.json', seal['paired_val_sha256'])
    native_path = native_dir/split/'result.json'
    native_hash = sha256_file(native_path)
    native = read(native_path)
    reference_path = reference_dir/f'{split}_results.json'
    prior = read(reference_path)
    if (prior['native_result_sha256'] != native_hash or native['smoke'] or prior['smoke']
            or native['mode'] != 'sample_holdout' or prior['split'] != split or native['split'] != split
            or len(native['stages']) != len(TASK_SIZES) or len(prior['stages']) != len(TASK_SIZES)):
        raise ValueError(f'Different historical evaluation source for {split}')
    require_digest(reference_dir/'centers_lock.json', prior['centers_sha256'])
    config_sha = sha256_file(root/'config.json')
    paired_centers = {k: row['paired'] for k, row in tasks.items()}
    stages = []
    for t, stage in enumerate(native['stages']):
        protocol = stage['protocol']
        expected = dict(task=t, split=split, smoke=False, config_sha256=config_sha,
                        checkpoint_sha256=tasks[str(t)]['checkpoint_sha256'], validation_result_sha256=val_sha)
        if any(protocol[k] != value for k, value in expected.items()):
            raise ValueError(f'TRAIN and evaluation sources differ for {split}/task{t}')
        path = native_dir/split/f'task{t}.npz'
        require_digest(path, stage['scores_sha256'])
        previous = prior['stages'][t]
        if previous['stage'] != t or previous['scores_sha256'] != stage['scores_sha256']:
            raise ValueError(f'Historical paired A used another archive for task{t}')
        torch.set_num_threads(1)
        with np.load(path, allow_pickle=False) as z:
            scores, y, valid = predictions(z, paired_centers, t)
            branch = torch.from_numpy(z['branch_logits'])
            heads = [weights(torch.as_tensor(tasks[str(h)]['natural']['center'], dtype=torch.float32), valid)
                     for h in range(t + 1)]
            natural = weighted_logits(branch, torch.stack(heads, dim=1), TASK_SIZES[:t+1])
        torch.set_num_threads(2)
        metrics = metrics_for_stage(t, y, valid, dict(fixed=scores['fixed'], paired=scores['A'], natural=natural))
        if metrics['fixed'] != stage['metrics']['fixed'] or metrics['paired'] != previous['metrics']['A']:
            raise ValueError(f'Official per-class results did not replay for {split}/task{t}')
        stages.append(dict(stage=t, people=len(y), reliable_face=int(valid.sum()), scores_sha256=stage['scores_sha256'],
                           metrics=metrics, differences=diff_by_stage(t, metrics)))
    summary = {m: {g: summarize_tasks([TaskMetrics(**s['metrics'][m][g]) for s in stages])
                   for g in GROUPS} for m in METHODS}
    if summary['fixed'] != native['summary']['fixed'] or summary['paired'] != prior['summary']['A']:
        raise ValueError(f'Official eight-stage summary differs for {split}')
    differences = {name: {g: {k: summary['paired'][g][k] - summary[name][g][k] for k in MAP_KEYS}
                          for g in GROUPS} for name in ('natural','fixed')}
    result = dict(split=split, centers_sha256=lock_hash, native_results_sha256=native_hash,
                  paired_reference_sha256=sha256_file(reference_path), summary=summary,
                  paired_minus=differences, stages=stages)
    if split == 'test':
        result['validation_seal_sha256'] = sha256_file(seal_path)
    dest = out/f'{split}_results.json'
    write_verified(dest, result)
    if split == 'val':
        write_verified(seal_path, dict(centers_sha256=lock_hash, val_results_sha256=sha256_file(dest),
                                       paired_val_sha256=result['paired_reference_sha256'],
                                       native_val_sha256=native_hash))
    print(split, {m: {k: summary[m]['all'][k] for k in MAP_KEYS} for m in METHODS}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ('root','out'):
        cli.add_argument('--'+arg, type=Path, required=True)
    cli.add_argument('--split', choices=('val','test'), required=True)
    evaluate(**vars(cli.parse_args()))
