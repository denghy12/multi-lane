"""Compare frozen fixed, natural-TRAIN and paired-TRAIN fusion at all eight stages."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import CLASS_ORDER, TASK_SIZES, TaskMetrics, summarize_tasks, task_indices
from scripts.emotic.d9_a_eight_eval import GROUPS, predictions, read, require_digest
from scripts.emotic.d9_a_eight_fit import PROTOCOL, REPLAY, check_cohort, group_proportions
from scripts.emotic.d9_matched_run import fit_average
from scripts.emotic.evaluate_d9_relation import metrics_for_stage, weighted_logits, write_json
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS, PROTECTED_BASELINE
from scripts.emotic.fit_d9_task10_probe import fit_one, weights

METHODS = ('fixed', 'natural', 'paired')
TASKS = tuple(range(len(TASK_SIZES)))


def write_verified(path, result):
    if path.exists():
        if read(path) != result:
            raise ValueError(f'Result differs from existing artifact: {path}')
    else:
        write_json(path, result)


def load_training_pair(pairs, roles, task, base, source_hashes):
    path = pairs / f'task{task}_held.npz'
    meta = read(path.with_suffix('.json'))
    expected = dict(split='train', group='held', smoke=False, task=task, **source_hashes,
                    cohort_sha256=base['cohort_sha256'], checkpoint_sha256=base['checkpoint_sha256'])
    if any(meta[key] != value for key, value in expected.items()):
        raise ValueError(f'Mismatched paired TRAIN protocol: task{task}')
    require_digest(path, meta['archive_sha256'])
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in ('sample_ids', 'targets', 'reliable', 'branch_logits')}
        variants = {view: {'branch_logits': z[f'{view}_branch_logits']} for view in ('person', 'face')}
    ids, valid, y, logits = (data[k] for k in ('sample_ids', 'reliable', 'targets', 'branch_logits'))
    if (valid.dtype != np.bool_ or valid.shape != (len(ids),) or len(np.unique(ids)) != len(ids)
            or y.shape != (len(ids), TASK_SIZES[task]) or not np.isin(y, [0, 1]).all()
            or not y.any(1).all() or logits.shape != (*y.shape, 3)
            or not np.isfinite(logits).all()
            or hashlib.sha256('\n'.join(ids.tolist()).encode()).hexdigest() != meta['sample_id_sha256']):
        raise ValueError(f'Invalid natural TRAIN data: task{task}')
    check_cohort(data, meta, base['fusion_cohort'], roles, task)
    for view, changed in variants.items():
        z = changed['branch_logits']
        if z.shape != logits.shape or not np.isfinite(z).all():
            raise ValueError(f'Invalid {view} variant: task{task}')
        np.testing.assert_array_equal(logits[:, :, 0], z[:, :, 0])
        if view == 'face':
            np.testing.assert_array_equal(logits[~valid], z[~valid])
    return data, variants, meta


def fit_centers(source, prior, matched, pairs, out):
    torch.set_num_threads(1)
    old = read(prior / 'centers_lock.json')
    expected_keys = {str(t) for t in TASKS}
    hashes = dict(config_sha256=sha256_file(source / 'config.json'),
                  roles_sha256=sha256_file(source / 'train_image_roles.json'))
    if (old['protocol'] != PROTOCOL or set(old['tasks']) != expected_keys
            or old['source_config_sha256'] != hashes['config_sha256']
            or old['source_roles_sha256'] != hashes['roles_sha256']):
        raise ValueError('Unexpected natural A training source')
    roles = read(source / 'train_image_roles.json')
    prior_three = matched / 'readout'
    three_seal = read(prior_three / 'validation_seal.json')
    centers = {}
    for t in TASKS:
        k = str(t)
        base = old['tasks'][k]
        old_center_path = prior / f'task{t}_center.json'
        require_digest(old_center_path, old['center_file_sha256'][k])
        require_digest(source / 'compact_checkpoints' / f'task{t}.pth', base['checkpoint_sha256'])
        require_digest(source / 'cohorts' / f'task{t}.json', base['cohort_sha256'])
        if (read(old_center_path) != base or base['task'] != t or base['fit_classes'] != list(task_indices(t))
                or read(source / 'cohorts' / f'task{t}.json')['fusion'] != base['fusion_cohort']):
            raise ValueError(f'Historical natural A provenance mismatch: task{t}')
        d, variants, meta = load_training_pair(pairs, roles, t, base, hashes)
        natural = fit_one(d, None, PROTOCOL['steps'])
        if natural != base['center'] or group_proportions(natural) != base['group_weights']:
            raise ValueError(f'Natural A training did not replay: task{t}')
        paired = fit_average(d, variants)
        if t in REPLAY:
            path = prior_three / f'task{t}_centers.json'
            require_digest(path, three_seal['artifacts'][path.name])
            if paired != read(path)['held']:
                raise ValueError(f'Paired A training did not replay: task{t}')
        centers[k] = dict(natural=dict(center=natural, weights=group_proportions(natural)),
                          paired=dict(center=paired['params']['bias'], weights=paired['weights'], fit=paired['fit']),
                          fit_classes=base['fit_classes'], people=len(d['targets']),
                          reliable_face=int(d['reliable'].sum()), checkpoint_sha256=base['checkpoint_sha256'],
                          cohort_sha256=base['cohort_sha256'], pairs_sha256=meta['archive_sha256'])
        print(f'task{t}: {centers[k]["people"]} TRAIN people', flush=True)
    lock = dict(tasks=centers, prior_centers_sha256=sha256_file(prior / 'centers_lock.json'), **hashes)
    write_verified(out / 'centers_lock.json', lock)


def diff_by_stage(stage, metrics):
    size = TASK_SIZES[stage]
    result = {}
    for name in ('natural', 'fixed'):
        groups = {}
        for g in GROUPS:
            reference = metrics[name][g]
            candidate = metrics['paired'][g]
            diff = np.subtract(candidate['per_class_ap'], reference['per_class_ap'])
            groups[g] = dict(mAP=candidate['mAP'] - reference['mAP'],
                             new_classes=float(diff[-size:].mean()),
                             old_classes=float(diff[:-size].mean()) if stage else None,
                             per_class=dict(zip(CLASS_ORDER[:len(diff)], diff.tolist())))
        result['paired_minus_' + name] = groups
    return result


def run_split(source, prior, out, split):
    torch.set_num_threads(1)
    path = out / 'centers_lock.json'
    lock = read(path)
    lock_hash = sha256_file(path)
    old = read(prior / f'{split}_results.json')
    digest = sha256_file(source / 'evaluation' / split / 'result.json')
    if (old['split'] != split or old['native_result_sha256'] != digest
            or old['centers_sha256'] != lock['prior_centers_sha256']
            or sha256_file(prior / 'centers_lock.json') != old['centers_sha256']
            or len(old['stages']) != len(TASKS)):
        raise ValueError(f'Historical evaluation provenance mismatch: {split}')
    prior_seal = read(prior / 'validation_seal.json')
    if split == 'val':
        require_digest(prior / 'val_results.json', prior_seal['val_results_sha256'])
    else:
        require_digest(prior / 'validation_seal.json', old['validation_seal_sha256'])
    natural = {k: row['natural'] for k, row in lock['tasks'].items()}
    stages = []
    for t in TASKS:
        baseline = old['stages'][t]
        cp_hash = lock['tasks'][str(t)]['checkpoint_sha256']
        archive_path = source / 'evaluation' / split / f'task{t}.npz'
        require_digest(archive_path, baseline['scores_sha256'])
        if baseline['stage'] != t or baseline['checkpoint_sha256'] != cp_hash:
            raise ValueError(f'Wrong historical stage or checkpoint: {split}/task{t}')
        with np.load(archive_path, allow_pickle=False) as archive:
            preds, y, valid = predictions(archive, natural, t)
            heads = [weights(torch.tensor(lock['tasks'][str(h)]['paired']['center']), valid) for h in range(t+1)]
            paired = weighted_logits(torch.from_numpy(archive['branch_logits']), torch.stack(heads, dim=1), TASK_SIZES[:t+1])
        if len(y) != baseline['people'] or int(valid.sum()) != baseline['reliable_face']:
            raise ValueError(f'Historical sample count mismatch: {split}/task{t}')
        readouts = dict(fixed=preds['fixed'], natural=preds['A'], paired=paired)
        # Repeat the two-thread CPU sigmoid used by the historical eight-stage metric.
        torch.set_num_threads(2)
        try:
            metrics = metrics_for_stage(t, y, valid, readouts)
        finally:
            torch.set_num_threads(1)
        for m, ref in [('natural', 'A'), ('fixed', 'fixed')]:
            if metrics[m] != baseline['metrics'][ref]:
                raise ValueError(f'Historical metrics did not replay: {split}/task{t}/{m}')
        stages.append(dict(stage=t, people=len(y), reliable_face=int(valid.sum()), metrics=metrics,
                           scores_sha256=baseline['scores_sha256'], differences=diff_by_stage(t, metrics)))
    summary = {m: {g: summarize_tasks([TaskMetrics(**r['metrics'][m][g]) for r in stages])
                   for g in GROUPS} for m in METHODS}
    for m, ref in [('natural','A'), ('fixed','fixed')]:
        if summary[m] != old['summary'][ref]:
            raise ValueError(f'Historical eight-stage summary did not replay: {split}/{m}')
    reference = {'natural': summary['natural']['all'], 'fixed': summary['fixed']['all'],
                 'protected_full_train_fixed': PROTECTED_BASELINE[split]}
    deltas = {name: {k: summary['paired']['all'][k] - x[k] for k in MAP_KEYS} for name, x in reference.items()}
    report = dict(split=split, centers_sha256=lock_hash, native_results_sha256=digest, stages=stages, summary=summary,
                  paired_deltas=deltas, both_at_least={name: min(x.values()) >= 0 for name,x in deltas.items()})
    if split == 'test':
        report['validation_seal_sha256'] = sha256_file(out / 'validation_seal.json')
    write_verified(out / f'{split}_results.json', report)
    print(split, {m:{k:summary[m]['all'][k] for k in MAP_KEYS} for m in METHODS}, deltas, flush=True)


def run(source, prior, matched, pairs, out, phase):
    out.mkdir(parents=True, exist_ok=True)
    if phase == 'fit-val':
        fit_centers(source, prior, matched, pairs, out)
        run_split(source, prior, out, 'val')
        write_verified(out / 'validation_seal.json', dict(centers_sha256=sha256_file(out/'centers_lock.json'),
                       val_results_sha256=sha256_file(out/'val_results.json')))
    else:
        seal = read(out / 'validation_seal.json')
        require_digest(out / 'centers_lock.json', seal['centers_sha256'])
        require_digest(out / 'val_results.json', seal['val_results_sha256'])
        run_split(source, prior, out, 'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source','prior','matched','pairs','out'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--phase', choices=('fit-val','test'), required=True)
    run(**vars(parser.parse_args()))
