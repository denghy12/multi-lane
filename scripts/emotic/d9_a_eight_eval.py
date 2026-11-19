"""Evaluate frozen TRAIN-learned task centers against native fixed fusion over all eight stages."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import CLASS_ORDER, TASK_SIZES, TaskMetrics, summarize_tasks
from scripts.emotic.evaluate_d9_protected_gates import verify_sample_controls
from scripts.emotic.evaluate_d9_relation import VIEWS, metrics_for_stage, weighted_logits, write_json
from scripts.emotic.evaluate_d9_sample_fusion import PROTECTED_BASELINE, MAP_KEYS
from scripts.emotic.fit_d9_task10_probe import weights as masked_weights, scores

METHODS = ('A', 'fixed', 'equal')
GROUPS = ('all', 'reliable', 'unreliable')
REPLAY_TASKS = (0, 3, 6)


def read(path):
    return json.loads(path.read_text())


def require_digest(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f'Archive or seal changed: {path}')


def predictions(archive, centers, stage):
    sizes = TASK_SIZES[:stage+1]
    if list(archive['task_sizes']) != list(sizes) or tuple(archive['view_order']) != VIEWS:
        raise ValueError('Archived task or view order differs')
    target, valid, z = (torch.from_numpy(archive[k]) for k in ('targets', 'reliable', 'branch_logits'))
    ids = archive['sample_ids']
    if (len(ids) != len(np.unique(ids)) or not len(ids) or valid.dtype != torch.bool
            or valid.shape != (len(ids),) or z.dtype != torch.float32
            or target.shape != (len(ids), sum(sizes)) or z.shape != (*target.shape, 3)
            or not ((target == 0) | (target == 1)).all() or not torch.isfinite(z).all()):
        raise ValueError('Invalid archived subjects, labels, reliability, or scores')
    results = {name: (torch.from_numpy(archive[name + '_weights']), torch.from_numpy(archive[name + '_logits']))
               for name in ('fixed', 'equal')}
    verify_sample_controls(results, valid)
    for weight, logits in results.values():
        torch.testing.assert_close(weighted_logits(z, weight, sizes), logits, atol=2e-5, rtol=2e-6)
    by_head = [masked_weights(torch.as_tensor(centers[str(h)]['center'], dtype=torch.float32), valid)
               for h in range(stage+1)]
    result = {name: pair[1] for name, pair in results.items()}
    result['A'] = weighted_logits(z, torch.stack(by_head, dim=1), sizes)
    return result, target, valid


def paired(metrics, stage):
    size = TASK_SIZES[stage]
    result = {}
    for group, a in metrics['A'].items():
        fixed = metrics['fixed'][group]
        delta = np.asarray(a['per_class_ap']) - np.asarray(fixed['per_class_ap'])
        result[group] = dict(A_minus_fixed_map=a['mAP'] - fixed['mAP'],
                             old_classes_delta=float(delta[:-size].mean()) if len(delta) > size else None,
                             current_classes_delta=float(delta[-size:].mean()),
                             per_class_delta=dict(zip(CLASS_ORDER[:len(delta)], delta.tolist())))
    return result


def prior_replay(prior, split, stage, archive_hash, ids, target, valid, logits):
    meta = read(prior / 'features' / f'task{stage}' / f'{split}.json')
    if (meta['protocol']['source_cohort_or_archive_sha256'] != archive_hash or not meta['exact_eval_replay']
            or meta['sample_id_sha256'] != hashlib.sha256('\n'.join(ids.tolist()).encode()).hexdigest()):
        raise ValueError(f'Prior probe source differs at stage {stage}')
    ref = read(prior / 'probe' / f'task{stage}_{split}.json')['A']['metrics']
    size = TASK_SIZES[stage]
    current = {}
    for group, mask in zip(GROUPS, (torch.ones_like(valid), valid, ~valid)):
        m = scores(logits[mask, -size:].contiguous(), target[mask, -size:].numpy())
        old = ref[group]
        if m['people'] != old['people'] or m['positive_counts'] != old['positive_counts']:
            raise ValueError(f'Prior probe population differs at stage {stage}/{group}')
        np.testing.assert_allclose(m['per_class_ap'], old['per_class_ap'], rtol=0, atol=1e-9)
        current[group] = m['mAP']
    return current


def verify_controls(actual, historical):
    for name in ('fixed', 'equal'):
        if actual[name].keys() != historical[name].keys():
            raise ValueError(f'Group set differs for {name}')
        for group, current in actual[name].items():
            prev = historical[name][group]
            for k in current:
                np.testing.assert_allclose(current[k], prev[k], rtol=0, atol=1e-9,
                                           err_msg=f'Native {name}/{group}/{k} failed to replay')


def verify_seal(out, lock_hash):
    seal = read(out / 'validation_seal.json')
    if seal['centers_sha256'] != lock_hash:
        raise ValueError('Center lock differs from completed validation evaluation')
    require_digest(out / 'val_results.json', seal['val_results_sha256'])
    return seal


def run(source, prior, out, split):
    from scripts.emotic.d9_a_eight_fit import load_centers
    if split not in ('val','test'):
        raise ValueError('Invalid split')
    torch.set_num_threads(1)
    lock = load_centers(source, prior, out)
    lock_hash = sha256_file(out / 'centers_lock.json')
    seal = verify_seal(out, lock_hash) if split == 'test' else None
    directory = source / 'evaluation' / split
    historical = read(directory / 'result.json')
    config_hash = sha256_file(source / 'config.json')
    if (historical['split'] != split or historical['smoke'] or historical['mode'] != 'sample_holdout'
            or len(historical['stages']) != len(TASK_SIZES)):
        raise ValueError(f'Incomplete or unexpected source archive: {directory}')
    stages = []
    for stage, saved in enumerate(historical['stages']):
        path = directory / f'task{stage}.npz'
        require_digest(path, saved['scores_sha256'])
        protocol = saved['protocol']
        expected = dict(task=stage, split=split, smoke=False, config_sha256=config_hash,
                        checkpoint_sha256=lock['tasks'][str(stage)]['checkpoint_sha256'],
                        validation_result_sha256=seal['native_val_result_sha256'] if seal else None)
        if any(protocol[k] != v for k, v in expected.items()):
            raise ValueError(f'Checkpoint or source protocol mismatch at stage {stage}')
        with np.load(path, allow_pickle=False) as data:
            z, y, valid = predictions(data, lock['tasks'], stage)
            ids = data['sample_ids']
        # The original eight-stage evaluation used two CPU threads for sigmoid/AP.
        # The separate current-head probe used one; near-tied scores need its own setting.
        torch.set_num_threads(2)
        try:
            metrics = metrics_for_stage(stage, y, valid, z)
        finally:
            torch.set_num_threads(1)
        verify_controls(metrics, saved['metrics'])
        row = dict(stage=stage, people=len(y), reliable_face=int(valid.sum()),
                   scores_sha256=saved['scores_sha256'], checkpoint_sha256=expected['checkpoint_sha256'],
                   metrics=metrics, paired=paired(metrics, stage))
        if stage in REPLAY_TASKS:
            row['prior_current_A_replay'] = prior_replay(
                prior, split, stage, saved['scores_sha256'], ids, y, valid, z['A'])
        stages.append(row)
    summary = {name: {g: summarize_tasks([TaskMetrics(**r['metrics'][name][g]) for r in stages])
                      for g in GROUPS} for name in METHODS}
    for name in ('fixed','equal'):
        for key in MAP_KEYS:
            np.testing.assert_allclose(summary[name]['all'][key], historical['summary'][name]['all'][key],
                                       rtol=0, atol=1e-9)
    baselines = dict(same_branches_fixed=summary['fixed']['all'], same_branches_equal=summary['equal']['all'],
                     protected_full_train=PROTECTED_BASELINE[split])
    comparisons = {}
    for name, baseline in baselines.items():
        changes = {k: summary['A']['all'][k] - baseline[k] for k in MAP_KEYS}
        comparisons[name] = dict(delta=changes, both_at_least_baseline=min(changes.values()) >= 0)
    report = dict(split=split, centers_sha256=lock_hash, stages=stages, summary=summary,
                  protected_full_train=PROTECTED_BASELINE[split], comparisons=comparisons,
                  native_result_sha256=sha256_file(directory / 'result.json'))
    if seal:
        report['validation_seal_sha256'] = sha256_file(out / 'validation_seal.json')
    dest = out / f'{split}_results.json'
    if dest.exists() and read(dest) != report:
        raise ValueError(f'Will not overwrite incompatible evaluation at {dest}')
    write_json(dest, report)
    if split == 'val':
        write_json(out / 'validation_seal.json', dict(centers_sha256=lock_hash, val_results_sha256=sha256_file(dest),
                                                     native_val_result_sha256=report['native_result_sha256']))
    print(split, 'A', {k: summary['A']['all'][k] for k in MAP_KEYS}, comparisons, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('source','prior','out'):
        cli.add_argument('--' + name, type=Path, required=True)
    cli.add_argument('--split', choices=('val','test'), required=True)
    args = cli.parse_args()
    run(args.source, args.prior, args.out, args.split)
