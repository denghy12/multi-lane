"""Pair each split strategy with the historical equal-update 90% model on native evaluation images."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, TaskMetrics, fit_calibration_indices, summarize_tasks, task_indices
from scripts.emotic.d9_a_eight_eval import GROUPS, read, require_digest
from scripts.emotic.d9_equal_updates_compare import BRANCHES, read_population
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_sample_training import describe_cohort, task_cohorts
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS
from scripts.emotic.run_r1_frozen_cross_view import input_paths


def verify_roles(candidate, reference):
    counts = Counter(reference.values())
    held = (len(reference) + 5) // 10
    if (reference.keys() != candidate.keys() or counts != {'fusion': held, 'branch': len(reference) - held}
            or Counter(candidate.values()) != counts or candidate == reference):
        raise ValueError('Expected different 90/10 assignments of exactly the same original TRAIN images')
    return dict(counts=dict(counts), changed_roles=sum(v != reference[k] for k, v in candidate.items()))


def verify_recipe(config, reference):
    strategy = config['split90']
    if strategy not in ('hash2', 'meta', 'online') or config['image_split'] != 'global_10pct_' + strategy:
        raise ValueError('Unrecognized TRAIN split recipe')
    budget = dict(config['equal_updates'])
    roles_sha = budget.pop('reference_roles_sha256')
    budget['roles_sha256'] = roles_sha
    comparable = dict(config, equal_updates=budget, image_split=reference['image_split'])
    comparable.pop('split90')
    if (comparable != reference or reference['mode'] != 'sample_holdout' or reference['smoke']
            or not reference['protect_branches'] or reference['image_split'] != 'global_10pct_hash_rank'
            or reference['equal_updates']['mode'] != 'cycle_90pct_to_full_epoch_size'
            or (reference['epochs'], reference['tasks'], reference['seed'], reference['batch_size']) != (30, 8, 1, 64)):
        raise ValueError('Recipes differ beyond the declared image-role allocation')


def audit_training(roots):
    config = {k:read(p/'config.json') for k,p in roots.items()}
    verify_recipe(config['candidate'], config['reference'])
    roles = {}
    for k,p in roots.items():
        if read(p/'training_complete.json')['status'] != 'complete':
            raise ValueError(f'Incomplete training: {p}')
        path = p/'train_image_roles.json'
        require_digest(path, config[k]['equal_updates']['roles_sha256'])
        roles[k] = read(path)
    assignment = verify_roles(roles['candidate'], roles['reference'])
    source = build_joint_source(input_paths(), 'train')
    tasks = []
    for t in range(len(TASK_SIZES)):
        classes = task_indices(t)
        indices, _ = fit_calibration_indices(source, classes, 0.)
        target = config['reference']['equal_updates']['tasks'][str(t)]
        if len(indices) != target['examples']:
            raise ValueError(f'Different full-TRAIN population at task{t}')
        epochs_by_arm, cohorts, retries = {}, {}, {}
        for k,p in roots.items():
            b, f = task_cohorts(source, indices, roles[k], False)
            cohort = {n:describe_cohort(source, ids, classes) for n,ids in [('branch',b),('fusion',f)]}
            if read(p/'cohorts'/f'task{t}.json') != cohort:
                raise ValueError(f'Archived cohorts do not follow image assignment: {k} task{t}')
            hist = read(p/'history'/f'task{t}.json')
            if hist['coverage']['sample_id_sha256'] != cohort['branch']['sample_id_sha256']:
                raise ValueError(f'Incorrect branch history: {k} task{t}')
            epochs = hist['epochs']
            if len(epochs) != config[k]['epochs']:
                raise ValueError(f'Incomplete epoch history: {k} task{t}')
            extra_count = np.zeros(len(b), dtype=np.int64)
            for ep,row in enumerate(epochs, 1):
                sampling = row['sampling']
                more = np.asarray(sampling['extra_indices'], dtype=np.int64)
                if (row['skipped'] or (row['task'],row['epoch']) != (t,ep) or row['updates'] != target['updates']
                        or row['examples'] != len(indices) or sampling['unique_people'] != len(b)
                        or sampling['duplicate_within_batches'] or sampling['extra_examples'] != len(more)
                        or len(more) + len(b) != len(indices) or np.any(more < 0) or np.any(more >= len(b))
                        or sampling['batch_sizes'] != [64]*(target['updates']-1) + [target['last_batch_size']]
                        or row['gate_schedule']['people'] != len(f)
                        or row['gate_schedule']['updates_this_epoch'] != config[k]['gate_updates']//len(epochs)):
                    raise ValueError(f'Exposure or budget mismatch: {k} task{t} epoch{ep}')
                extra_count += np.bincount(more, minlength=len(b))
                if extra_count.max() - extra_count.min() > 1:
                    raise ValueError(f'Unequal supplemental exposures: {k} task{t} epoch{ep}')
            epochs_by_arm[k] = epochs
            cohorts[k] = cohort
            retries[k] = sum(row['fp32_retries'] for row in epochs)
        for x,y in zip(epochs_by_arm['candidate'], epochs_by_arm['reference']):
            if x['branch_lrs'] != y['branch_lrs']:
                raise ValueError(f'Different branch learning rates: task{t} epoch{x["epoch"]}')
        tasks.append(dict(task=t, cohorts=cohorts, fp32_retries=retries,
                          updates=target['updates']*len(epochs_by_arm['reference'])))
    return dict(strategy=config['candidate']['split90'], assignment=assignment,
                role_hashes={k:v['equal_updates']['roles_sha256'] for k,v in config.items()},
                total_updates_per_run=sum(x['updates'] for x in tasks), tasks=tasks)


def compare(root, reference, out, split):
    roots = dict(candidate=root, reference=reference)
    out.mkdir(parents=True, exist_ok=True)
    seal = out/'validation_seal.json'
    val_hashes = dict.fromkeys(roots)
    if split == 'test':
        locked = read(seal)
        require_digest(out/'val_comparison.json', locked['val_comparison_sha256'])
        val_hashes = locked['val_native_sha256']
        for k,p in roots.items():
            require_digest(p/'evaluation'/'val'/'result.json', val_hashes[k])
    audit = audit_training(roots)
    paths = {k:p/'evaluation'/split/'result.json' for k,p in roots.items()}
    reports = {k:read(p) for k,p in paths.items()}
    cfg_hashes = {k:sha256_file(p/'config.json') for k,p in roots.items()}
    for name,report in reports.items():
        if (report['split'] != split or report['smoke'] or report['mode'] != 'sample_holdout'
                or len(report['stages']) != len(TASK_SIZES)):
            raise ValueError(f'Unexpected {name} evaluation protocol')
    stages = []
    for t in range(len(TASK_SIZES)):
        rows = {k:r['stages'][t] for k,r in reports.items()}
        population = {k:read_population(p, split, t, rows[k], cfg_hashes[k], val_hashes[k]) for k,p in roots.items()}
        for key,values in population['reference'].items():
            np.testing.assert_array_equal(values, population['candidate'][key], err_msg=f'{split}/task{t}/{key}')
        stages.append(dict(task=t, people=len(population['candidate']['sample_ids']),
                           reliable_face=int(population['candidate']['reliable'].sum()),
                           scores_sha256={k:r['scores_sha256'] for k,r in rows.items()},
                           mAP={m:{g:{k:r['metrics'][m][g]['mAP'] for k,r in rows.items()} for g in GROUPS}
                                for m in BRANCHES}))
    summary = {}
    for method in BRANCHES:
        summary[method] = {}
        for group in GROUPS:
            values = {}
            for k,r in reports.items():
                native = r['summary'][method][group]
                reproduced = summarize_tasks([TaskMetrics(**stage['metrics'][method][group]) for stage in r['stages']])
                if native != reproduced:
                    raise ValueError(f'Mismatched native summary: {k}/{method}/{group}')
                values[k] = {key:native[key] for key in MAP_KEYS}
            values['delta'] = {key:values['candidate'][key]-values['reference'][key] for key in MAP_KEYS}
            summary[method][group] = values
    hashes = {k:sha256_file(p) for k,p in paths.items()}
    result = dict(strategy=audit['strategy'], split=split, training=audit, config_sha256=cfg_hashes,
                  native_sha256=hashes, stages=stages, summary=summary)
    dest = out/f'{split}_comparison.json'
    if split == 'test':
        result['validation_seal_sha256'] = sha256_file(seal)
    write_verified(dest, result)
    if split == 'val':
        write_verified(seal, dict(val_comparison_sha256=sha256_file(dest), val_native_sha256=hashes))
    print(split, {k:summary[k]['reliable' if k=='face' else 'all'] for k in BRANCHES}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('root','reference','out'):
        parser.add_argument('--'+option, required=True, type=Path)
    parser.add_argument('--split', choices=('val','test'), required=True)
    compare(**vars(parser.parse_args()))
