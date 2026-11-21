"""Evaluate reliable-Face blur strength and natural-image loss mass on eight frozen stages."""
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


CANDIDATES = ('gentle_168', 'clean_112', 'clean_168')
METHODS = ('fixed', 'paired_112', *CANDIDATES)


def val_decision(summary):
    reference = summary['paired_112']['all']
    differences = {name: {key: summary[name]['all'][key] - reference[key] for key in MAP_KEYS}
                   for name in CANDIDATES}
    eligible = [name for name, change in differences.items()
                if min(change.values()) >= -1e-10 and max(change.values()) > 1e-10]
    winner = max(eligible, key=lambda name: tuple(summary[name]['all'][key] for key in MAP_KEYS)) if eligible else 'paired_112'
    return dict(winner=winner, eligible=eligible, control='paired_112', population='all',
                tolerance=1e-10, candidate_minus_control=differences,
                tiebreak='final mAP, average mAP, then gentle_168, clean_112, clean_168')


def evaluate(root, historical, out, split, smoke=False):
    if split not in ('val', 'test'):
        raise ValueError('Expected a natural-image evaluation split')
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

    # Validate this run's choice before opening any Test metadata or prediction file.
    if split == 'test':
        seal = verify_seal(out, lock_hash)
        validation = read(out / 'val_results.json')
        if seal['decision'] != val_decision(validation['summary']) or seal['decision'] != validation['decision']:
            raise ValueError('Sealed choice differs from the Validation results')
        native_val_sha = seal['native_val_sha256']
        require_digest(native / 'val/result.json', native_val_sha)
        require_digest(historical / 'val_results.json', seal['historical_val_sha256'])

    old_path = historical / f'{split}_results.json'
    old = read(old_path)
    old_lock_path = historical / 'centers_lock.json'
    require_digest(old_lock_path, old['centers_sha256'])
    old_tasks = read(old_lock_path)['tasks']
    if split == 'test':
        require_digest(historical / 'validation_seal.json', old['validation_seal_sha256'])
    source_path = native / split / 'result.json'
    require_digest(source_path, old['native_results_sha256'])
    source = read(source_path)
    if (source['mode'] != 'sample_holdout' or source['smoke'] or old['smoke']
            or any(run['split'] != split or len(run['stages']) != len(TASK_SIZES) for run in (source, old))):
        raise ValueError('Expected historical eight-stage protected holdout evaluation')

    centers = {key: row['paired_112'] for key, row in tasks.items()}
    stages = []
    for stage_index in range(stage_count):
        key = str(stage_index)
        prior = old['stages'][stage_index]
        stage = source['stages'][stage_index]
        expected = dict(task=stage_index, split=split, smoke=False, config_sha256=lock['config_sha256'],
                        checkpoint_sha256=tasks[key]['checkpoint_sha256'], validation_result_sha256=native_val_sha)
        if (prior['stage'] != stage_index or prior['scores_sha256'] != stage['scores_sha256']
                or any(stage['protocol'][field] != value for field, value in expected.items())
                or centers[key]['center'] != old_tasks[key]['paired']['center']):
            raise ValueError(f'Historical source or paired center changed at stage {stage_index}')
        archive = native / split / f'task{stage_index}.npz'
        require_digest(archive, prior['scores_sha256'])
        torch.set_num_threads(1)
        with np.load(archive, allow_pickle=False) as data:
            baseline, labels, reliable = predictions(data, centers, stage_index)
            branch = torch.from_numpy(data['branch_logits'])
            logits = dict(fixed=baseline['fixed'], paired_112=baseline['A'])
            for name in CANDIDATES:
                by_head = [weights(torch.as_tensor(tasks[str(h)][name]['center'], dtype=torch.float32), reliable)
                           for h in range(stage_index + 1)]
                logits[name] = weighted_logits(branch, torch.stack(by_head, dim=1), TASK_SIZES[:stage_index+1])
                if not torch.isfinite(logits[name]).all():
                    raise ValueError(f'Nonfinite prediction for {name}/task{stage_index}')
                if not torch.equal(logits[name][~reliable], logits['paired_112'][~reliable]):
                    raise ValueError(f'Face-unavailable scores changed for {name}/task{stage_index}')
        torch.set_num_threads(2)
        metrics = metrics_for_stage(stage_index, labels, reliable, logits)
        if (metrics['fixed'] != prior['metrics']['fixed'] or metrics['fixed'] != stage['metrics']['fixed']
                or metrics['paired_112'] != prior['metrics']['paired']):
            raise ValueError(f'Historical fixed or paired evaluation failed to replay at {split}/task{stage_index}')
        if any(metrics[name]['unreliable'] != metrics['paired_112']['unreliable'] for name in CANDIDATES):
            raise ValueError('Face-unavailable metrics changed')
        stages.append(dict(stage=stage_index, people=len(labels), reliable_face=int(reliable.sum()),
                           scores_sha256=prior['scores_sha256'], metrics=metrics))
    summary = {method: {group: summarize_tasks([TaskMetrics(**stage['metrics'][method][group]) for stage in stages])
                        for group in GROUPS} for method in METHODS}
    decision = val_decision(summary) if split == 'val' else seal['decision']
    result = dict(split=split, smoke=smoke, stages=stages, summary=summary, decision=decision,
                  centers_sha256=lock_hash, native_results_sha256=old['native_results_sha256'],
                  historical_sha256=sha256_file(old_path))
    result_path = out / f'{split}_results.json'
    if split == 'test':
        result['validation_seal_sha256'] = sha256_file(seal_path)
    write_verified(result_path, result)
    if split == 'val':
        write_verified(seal_path, dict(decision=decision, centers_sha256=lock_hash,
                                       val_results_sha256=sha256_file(result_path),
                                       native_val_sha256=result['native_results_sha256'],
                                       historical_val_sha256=result['historical_sha256']))
    print(split, {method: {key: summary[method]['all'][key] for key in MAP_KEYS} for method in METHODS},
          decision, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('root', 'historical', 'out'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--smoke', action='store_true')
    evaluate(**vars(parser.parse_args()))
