"""Evaluate a single TRAIN-locked mixture of a task center and a per-person gate."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_direction_permutation import FORMAL_SEEDS, difference_summary, permutation_sources
from scripts.emotic.evaluate_d9_relation import write_json
from scripts.emotic.fit_d9_task10_probe import (
    apply_rule, build_features, combine_message, group_weights, load_arrays, mix, scores,
)


def read_json(path):
    return json.loads(path.read_text())


def blend(center, individual, fraction):
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Invalid fusion fraction')
    if fraction == 0:
        return center
    if fraction == 1:
        return individual
    return (1.0 - fraction) * center + fraction * individual


def group_scores(logits, data):
    reliable = data['reliable']
    return {name: scores(logits[mask], data['targets'][mask])
            for name, mask in [('all', np.ones(len(reliable), dtype=bool)),
                               ('reliable', reliable), ('unreliable', ~reliable)]}


def assigned_means(weights, reliable, saved_means):
    result = weights.clone()
    for g in (0, 1):
        mask = reliable == g
        if mask.any():
            if saved_means[g] is None:
                raise ValueError('Missing locked Validation mean for an active availability group')
            result[mask] = weights.new_tensor(saved_means[g])
    return result


def evaluate_task(data, model, task, fraction, seeds, val_means=None):
    features, message, _ = build_features(data, model['feature_stats'])
    reliable = data['reliable']
    ids = data['sample_ids']
    target = data['targets']
    z = torch.as_tensor(data['branch_logits'], dtype=torch.float32)
    a = apply_rule('A', data, features, model)
    d = apply_rule('D', data, features, model)
    actual = blend(a, d, fraction)
    group_mean = group_weights(actual, reliable) if val_means is None else val_means
    mean_weights = a if fraction == 0 else assigned_means(actual, reliable, group_mean)
    logits_by_method = {}
    for name in ('fixed', 'equal'):
        w = apply_rule(name, data, features, model)
        logits = mix(w, z)
        if name + '_logits' in data:
            archived = torch.as_tensor(data[name + '_logits'], dtype=torch.float32)
            torch.testing.assert_close(logits, archived, atol=2e-5, rtol=2e-6)
            logits = archived
        logits_by_method[name] = logits
    logits_by_method.update(A=mix(a, z), D=mix(d, z), blend=mix(actual, z),
                            group_mean=mix(mean_weights, z))
    metrics = {k: group_scores(v, data) for k, v in logits_by_method.items()}
    baseline = metrics['blend']['all']['mAP']
    swapped_weights, swapped_messages = [], []
    for seed in seeds:
        if fraction == 0:
            swapped_weights.append(baseline)
            swapped_messages.append(baseline)
            continue
        donor = permutation_sources(ids, reliable, task + 1, seed)[:, task]
        wrong_input = combine_message(features['C'], message[donor])
        wrong_gate = apply_rule('D', data, features, model, wrong_input)
        swapped_weights.append(scores(mix(actual[donor], z), target)['mAP'])
        swapped_messages.append(scores(mix(blend(a, wrong_gate, fraction), z), target)['mAP'])
    return {
        'metrics': metrics, 'actual_group_weights': group_weights(actual, reliable),
        'group_means_used': group_mean,
        'minus_A': baseline - metrics['A']['all']['mAP'],
        'minus_group_mean': baseline - metrics['group_mean']['all']['mAP'],
        'weight_shuffle': difference_summary(baseline, swapped_weights),
        'message_shuffle': difference_summary(baseline, swapped_messages),
    }


def check_replay(result, historical):
    for method in ('A', 'D', 'fixed', 'equal'):
        for group in ('all', 'reliable', 'unreliable'):
            old = historical[method]['metrics'][group]
            new = result['metrics'][method][group]
            if old is None or new is None:
                if old != new:
                    raise ValueError(f'Endpoint cohort differs: {method}/{group}')
                continue
            if old['people'] != new['people'] or old['positive_counts'] != new['positive_counts']:
                raise ValueError(f'Endpoint population differs: {method}/{group}')
            np.testing.assert_allclose(old['per_class_ap'], new['per_class_ap'], rtol=0., atol=1e-9)
            np.testing.assert_allclose(old['mAP'], new['mAP'], rtol=0., atol=1e-9)
            np.testing.assert_allclose(old['bce'], new['bce'], rtol=0., atol=1e-6)


def aggregate(rows):
    methods = ('fixed', 'equal', 'A', 'D', 'blend', 'group_mean')
    macro = {method: float(np.mean([r['metrics'][method]['all']['mAP'] for r in rows]))
             for method in methods}
    changes = {'vs_A': [r['minus_A'] for r in rows],
               'vs_message_shuffle': [r['message_shuffle']['mean'] for r in rows],
               'vs_weight_shuffle': [r['weight_shuffle']['mean'] for r in rows],
               'vs_group_mean': [r['minus_group_mean'] for r in rows]}
    contrasts = {k: dict(mean=float(np.mean(v)), positive_tasks=sum(x > 0 for x in v),
                         per_task=v) for k, v in changes.items()}
    passed = all(contrasts[k]['mean'] > 0 and contrasts[k]['positive_tasks'] > len(rows) / 2
                 for k in ('vs_A', 'vs_message_shuffle'))
    return dict(macro_current_head_mAP=macro, contrasts=contrasts, passes_criteria=passed)


def verify_digest(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f'Artifact SHA-256 mismatch: {path}')


def run(source, out, split, permutations=100):
    from scripts.emotic.d9_shrink_fit import read_lock
    if split not in ('val', 'test') or not 1 <= permutations <= len(FORMAL_SEEDS):
        raise ValueError('Invalid evaluation split or number of permutations')
    torch.set_num_threads(1)
    lock = read_lock(source, out)
    tasks, fraction = lock['tasks'], lock['lambda']
    old = source/'probe'
    old_lock_path = old/'validation_lock.json'
    old_lock = read_json(old_lock_path)
    if old_lock['protocol']['tasks'] != tasks:
        raise ValueError('Historical probe used different tasks')
    protocol = dict(version=1, tasks=tasks, fraction=fraction, permutation_seeds=list(FORMAL_SEEDS[:permutations]),
                    lambda_lock_sha256=sha256_file(out/'lambda_lock.json'),
                    original_validation_lock_sha256=sha256_file(old_lock_path))
    val_report = old_test = None
    seal_path = out/'validation_seal.json'
    if split == 'test':
        seal = read_json(seal_path)
        if seal['protocol'] != protocol:
            raise ValueError('Validation seal differs from current evaluation protocol')
        verify_digest(out/'val_results.json', seal['val_results_sha256'])
        val_report = read_json(out/'val_results.json')
        old_test = read_json(old/'test_results.json')
        if old_test['validation_lock_sha256'] != protocol['original_validation_lock_sha256']:
            raise ValueError('Historical Test refers to a different Validation lock')

    by_task, feature_hashes, reference_hashes = {}, {}, {}
    for task in tasks:
        key = str(task)
        if old_lock['train_val_hash'][key]['train'] != lock['source_hashes'][key]:
            raise ValueError(f'Task {task} TRAIN features differ between OOF and original probes')
        model_file = old/f'task{task}_model.json'
        verify_digest(model_file, old_lock['artifact_hashes'][model_file.name])
        feature = source/'features'/f'task{task}'/f'{split}.npz'
        reference = old/f'task{task}_{split}.json'
        if split == 'val':
            digest = old_lock['train_val_hash'][key]['val']
            verify_digest(reference, old_lock['artifact_hashes'][reference.name])
        else:
            digest = old_test['test_features_sha256'][key]
        verify_digest(feature, digest)
        means = None if split == 'val' else val_report['by_task'][key]['group_means_used']
        row = evaluate_task(load_arrays(feature, task), read_json(model_file), task, fraction,
                            FORMAL_SEEDS[:permutations], means)
        check_replay(row, read_json(reference))
        by_task[key] = row
        feature_hashes[key] = digest
        reference_hashes[key] = sha256_file(reference)
    summary = aggregate(list(by_task.values()))
    report = dict(protocol=protocol, split=split, feature_sha256=feature_hashes,
                  endpoint_reference_sha256=reference_hashes, endpoint_replay=True,
                  by_task=by_task, **summary)
    if split == 'test':
        report.update(validation_seal_sha256=sha256_file(seal_path),
                      validation_passed=val_report['passes_criteria'])
    dest = out/f'{split}_results.json'
    if split == 'val' and seal_path.exists():
        existing = read_json(seal_path)
        if existing['protocol'] != protocol or read_json(dest) != report:
            raise ValueError('A different sealed Validation evaluation already exists')
        verify_digest(dest, existing['val_results_sha256'])
        return
    write_json(dest, report)
    if split == 'val':
        write_json(seal_path, dict(protocol=protocol, val_results_sha256=sha256_file(dest)))
    print(split, 'locked lambda', fraction, summary, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--split', choices=('val','test'), required=True)
    cli.add_argument('--permutations', type=int, default=100)
    args = cli.parse_args()
    run(args.source, args.out, args.split, args.permutations)
