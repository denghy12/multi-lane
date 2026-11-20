"""Evaluate the reliable-Face perturbation factorial with a sealed Validation decision."""
import argparse
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, TaskMetrics, summarize_tasks
from scripts.emotic.d9_a_eight_eval import GROUPS, predictions, read, require_digest
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.evaluate_d9_relation import metrics_for_stage, weighted_logits
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS
from scripts.emotic.fit_d9_task10_probe import weights


CANDIDATES = ('natural', 'person', 'face')
METHODS = ('fixed', *CANDIDATES, 'paired')


def val_decision(summary):
    control = summary['paired']['all']
    passed = []
    for name in CANDIDATES:
        value = summary[name]['all']
        delta = [value[k] - control[k] for k in MAP_KEYS]
        if min(delta) >= -1e-10 and max(delta) > 1e-10:
            passed.append(name)
    winner = max(passed, key=lambda m: tuple(summary[m]['all'][k] for k in MAP_KEYS)) if passed else 'paired'
    return dict(winner=winner, eligible=passed, control='paired', population='all',
                criteria='Both Validation mAPs at least paired and one strictly higher at tolerance 1e-10; '
                         'break ties by final then average mAP, then natural, person, face.')


def evaluate(root, historical, out, split, smoke):
    lock_path = out / 'centers_lock.json'
    lock = read(lock_path)
    tasks = lock['tasks']
    task_count = 1 if smoke else len(TASK_SIZES)
    require_digest(root / 'config.json', lock['config_sha256'])
    if lock['smoke'] != smoke or set(tasks) != {str(t) for t in range(task_count)}:
        raise ValueError('TRAIN centers do not match the requested stages')

    native_dir = root / 'evaluation'
    native = read(native_dir / split / 'result.json')
    old_path = historical / f'{split}_results.json'
    old = read(old_path)
    require_digest(native_dir / split / 'result.json', old['native_results_sha256'])
    require_digest(historical / 'centers_lock.json', old['centers_sha256'])
    require_digest(root / 'paired_A' / f'{split}_results.json', old['paired_reference_sha256'])
    old_seal = read(historical / 'validation_seal.json')
    require_digest(historical / 'val_results.json', old_seal['val_results_sha256'])
    seal_path = out / 'validation_seal.json'
    native_val_sha = None
    if split == 'test':
        require_digest(historical / 'validation_seal.json', old['validation_seal_sha256'])
        seal = read(seal_path)
        require_digest(lock_path, seal['centers_sha256'])
        validation_path = out / 'val_results.json'
        require_digest(validation_path, seal['val_results_sha256'])
        if read(validation_path)['decision'] != seal['decision']:
            raise ValueError('Sealed Validation decision differs')
        native_val_sha = seal['native_val_sha256']
        require_digest(native_dir / 'val/result.json', native_val_sha)
        require_digest(historical / 'val_results.json', seal['historical_val_sha256'])
    if (native['split'] != split or old['split'] != split or native['smoke']
            or native['mode'] != 'sample_holdout' or len(native['stages']) != len(TASK_SIZES)
            or len(old['stages']) != len(TASK_SIZES)):
        raise ValueError('Unexpected original evaluation protocol')

    paired_centers = {k: r['paired'] for k,r in tasks.items()}
    stages = []
    for t in range(task_count):
        official = native['stages'][t]
        previous = old['stages'][t]
        expected = dict(task=t, split=split, smoke=False, config_sha256=lock['config_sha256'],
                        checkpoint_sha256=tasks[str(t)]['checkpoint_sha256'], validation_result_sha256=native_val_sha)
        if (any(official['protocol'][k] != v for k,v in expected.items())
                or previous['stage'] != t or previous['scores_sha256'] != official['scores_sha256']):
            raise ValueError(f'Incorrect stage archive or source checkpoint at task{t}')
        archive = native_dir / split / f'task{t}.npz'
        require_digest(archive, previous['scores_sha256'])
        torch.set_num_threads(1)
        with np.load(archive, allow_pickle=False) as data:
            baseline, labels, reliable = predictions(data, paired_centers, t)
            branch = torch.from_numpy(data['branch_logits'])
            logits = dict(fixed=baseline['fixed'], paired=baseline['A'])
            for name in CANDIDATES:
                head_weights = [weights(torch.as_tensor(tasks[str(h)][name]['center'], dtype=torch.float32), reliable)
                                for h in range(t+1)]
                logits[name] = weighted_logits(branch, torch.stack(head_weights, dim=1), TASK_SIZES[:t+1])
                if not torch.equal(logits[name][~reliable], logits['paired'][~reliable]):
                    raise ValueError(f'Face-unavailable scores changed for {name}/task{t}')
        torch.set_num_threads(2)
        metrics = metrics_for_stage(t, labels, reliable, logits)
        if (metrics['fixed'] != official['metrics']['fixed'] or metrics['paired'] != previous['metrics']['paired']
                or metrics['natural']['reliable'] != previous['metrics']['natural']['reliable']):
            raise ValueError(f'Historical control failed to replay for {split}/task{t}')
        if any(metrics[name]['unreliable'] != metrics['paired']['unreliable'] for name in CANDIDATES):
            raise ValueError('Changed Face-unavailable metrics')
        stages.append(dict(stage=t, people=len(labels), reliable_face=int(reliable.sum()),
                           scores_sha256=previous['scores_sha256'], metrics=metrics))
    summary = {m: {g: summarize_tasks([TaskMetrics(**s['metrics'][m][g]) for s in stages])
                   for g in GROUPS} for m in METHODS}
    decision = val_decision(summary) if split == 'val' else seal['decision']
    dest = out / f'{split}_results.json'
    result = dict(split=split, smoke=smoke, stages=stages, summary=summary, decision=decision,
                  centers_sha256=sha256_file(lock_path), native_results_sha256=old['native_results_sha256'],
                  historical_sha256=sha256_file(old_path))
    if split == 'test':
        result['validation_seal_sha256'] = sha256_file(seal_path)
    write_verified(dest, result)
    if split == 'val':
        write_verified(seal_path, dict(decision=decision, centers_sha256=result['centers_sha256'],
                                       val_results_sha256=sha256_file(dest), native_val_sha256=old['native_results_sha256'],
                                       historical_val_sha256=result['historical_sha256']))
    print(split, {m: {k: summary[m]['all'][k] for k in MAP_KEYS} for m in METHODS}, decision, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('root','historical','out'):
        parser.add_argument('--'+arg, type=Path, required=True)
    parser.add_argument('--split', choices=('val','test'), required=True)
    parser.add_argument('--smoke', action='store_true')
    evaluate(**vars(parser.parse_args()))
