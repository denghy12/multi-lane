"""Evaluate expanded TRAIN populations with unchanged natural Validation and Test ranks."""
import argparse
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import CLASS_ORDER, TASK_SIZES, TaskMetrics, summarize_tasks
from scripts.emotic.d9_a_eight_eval import GROUPS, predictions, read, require_digest, verify_seal
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.evaluate_d9_relation import metrics_for_stage, weighted_logits
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS
from scripts.emotic.fit_d9_task10_probe import weights


METHODS = ('fixed', 'paired', 'rank', 'expanded_bce', 'expanded_rank')


def val_decision(summary):
    tolerance = 1e-10

    def differences(candidate, reference):
        return {k: summary[candidate]['all'][k] - summary[reference]['all'][k] for k in MAP_KEYS}

    def advances(change):
        return min(change.values()) >= -tolerance and max(change.values()) > tolerance

    c_a = differences('expanded_bce', 'paired')
    d_a = differences('expanded_rank', 'paired')
    d_c = differences('expanded_rank', 'expanded_bce')
    eligible = []
    if advances(c_a):
        eligible.append('expanded_bce')
    if advances(d_a) and advances(d_c):
        eligible.append('expanded_rank')
    return dict(winner=eligible[-1] if eligible else 'paired', eligible=eligible, control='paired',
                population='all', tolerance=tolerance, C_minus_A=c_a, D_minus_A=d_a, D_minus_C=d_c)


def top_k(stage, logits, targets, ap):
    """For each seen class, count errors among the top K people, K = its positive count."""
    scores, labels = logits.sigmoid().numpy(), targets.numpy()
    by_class = {}
    first = 0
    for size in TASK_SIZES[:stage+1]:
        no_head_emotion = ~labels[:, first:first+size].astype(bool).any(axis=1)
        new = []
        for column in range(first, first+size):
            k = int(labels[:, column].sum())
            # Use exactly the project's average_precision ranking and tie convention.
            best = scores[:, column].argsort()[::-1][:k]
            wrong = labels[best, column] == 0
            fp = int(wrong.sum())
            row = dict(k=k, tp=k-fp, fp=fp, fp_head_all_negative=int(no_head_emotion[best][wrong].sum()),
                       precision=(k-fp)/k if k else None, ap=ap[column])
            by_class[CLASS_ORDER[column]] = row
            new.append(row)
        first += size
    current = {key: sum(r[key] for r in new) for key in ('k','tp','fp','fp_head_all_negative')}
    precision = [r['precision'] for r in new if r['k']]
    current.update(macro_precision=float(np.mean(precision)) if precision else None,
                   mean_ap=float(np.mean([r['ap'] for r in new])))
    return dict(by_class=by_class, current=current)


def evaluate(root, historical, out, split, smoke=False):
    if split not in ('val', 'test'):
        raise ValueError('Expected a natural evaluation split')
    lock_path = out / 'centers_lock.json'
    lock = read(lock_path)
    tasks = lock['tasks']
    stage_count = 2 if smoke else len(TASK_SIZES)
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
                or any(tasks[key][m]['center'] != old_tasks[key][m]['center'] for m in ('paired', 'rank'))):
            raise ValueError(f'Historical source or paired center changed at stage {t}')
        archive = native / split / f'task{t}.npz'
        require_digest(archive, old['scores_sha256'])
        torch.set_num_threads(1)
        with np.load(archive, allow_pickle=False) as data:
            baseline, targets, reliable = predictions(data, centers, t)
            z = torch.from_numpy(data['branch_logits'])
            logits = dict(fixed=baseline['fixed'], paired=baseline['A'])
            for m in METHODS[2:]:
                by_head = [weights(torch.as_tensor(tasks[str(h)][m]['center'], dtype=torch.float32), reliable)
                           for h in range(t + 1)]
                logits[m] = weighted_logits(z, torch.stack(by_head, dim=1), TASK_SIZES[:t+1])
                if not torch.isfinite(logits[m]).all():
                    raise ValueError(f'Nonfinite {m} logits at stage {t}')
        if t == 0 and (not torch.equal(logits['paired'], logits['expanded_bce'])
                       or not torch.equal(logits['rank'], logits['expanded_rank'])):
            raise ValueError('Zero extra TRAIN people at task0 must reproduce both parent readouts')
        torch.set_num_threads(2)
        metrics = metrics_for_stage(t, targets, reliable, logits)
        if any(metrics[k] != old['metrics'][k] for k in ('fixed', 'paired', 'rank')):
            raise ValueError(f'Historical evaluation failed to replay at {split}/task{t}')
        tail = {m: top_k(t, logits[m], targets, metrics[m]['all']['per_class_ap']) for m in METHODS}
        stages.append(dict(stage=t, people=len(targets), reliable_face=int(reliable.sum()),
                           scores_sha256=old['scores_sha256'], metrics=metrics, top_k=tail))
    summary = {m: {g: summarize_tasks([TaskMetrics(**s['metrics'][m][g]) for s in stages])
                   for g in GROUPS} for m in METHODS}
    decision = val_decision(summary) if split == 'val' else seal['decision']
    result = dict(split=split, smoke=smoke, summary=summary, stages=stages, decision=decision,
                  top_k_definition='Full natural evaluation population, K is the number of positives per seen '
                                   'class, historical sigmoid / numpy descending argsort. fp_head_all_negative '
                                   'counts false positives with zero positive labels in that class introduction '
                                   'head. current totals count class-person assignments. Precision is a fraction; '
                                   'AP uses the historical 0-100 scale.',
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
