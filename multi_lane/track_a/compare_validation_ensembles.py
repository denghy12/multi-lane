"""Fixed-weight, matched auxiliary-seed full/full versus full/person validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from .evaluation_scores import align_evaluation_scores
from .fuse_validation_scores import COMMON_CONFIG_FIELDS, validated_run_scores
from .runner import compute_metrics, summarize_tasks

METRICS = ('final_mAP', 'average_mAP', 'final_cF1', 'final_oF1', 'forgetting')


def audit_validation_run(run: Path, view: str):
    config = json.loads((run / 'config.json').read_text())
    summary = json.loads((run / 'seed_summary.json').read_text())
    history = json.loads((run / 'training_history.json').read_text())
    if (config.get('reporting_split') != 'val'
            or config.get('evaluation_score_purpose') not in (None, 'validation_search')
            or config.get('save_evaluation_scores') is not True):
        raise ValueError('Only validation score runs are allowed; test is forbidden')
    if (config.get('input_mode') != view or config.get('seed') not in (0, 1, 2)
            or config.get('threshold') != 0.5 or config.get('save_checkpoints') is not False):
        raise ValueError('Input view, seed, threshold or checkpoint configuration drift')
    if view == 'person_crop' and config.get('person_transform_mode') != 'letterbox':
        raise ValueError('Person input must use letterbox')
    if config.get('git', {}).get('dirty') is not False or not config['git'].get('commit'):
        raise ValueError('A clean source commit is required')
    if (summary.get('status') != 'complete' or summary.get('completed_epochs') != 240
            or summary.get('completed_optimizer_updates') != 13950):
        raise ValueError('Incomplete training budget')
    if set(history) != {str(t) for t in range(8)} or any(len(history[str(t)]) != 30 for t in range(8)):
        raise ValueError('Missing task/epoch history')
    if any(e['skipped_optimizer_steps'] != 0 for rows in history.values() for e in rows):
        raise ValueError('Skipped optimizer updates')
    dumps, rows = validated_run_scores(run, 'val')
    # The original seed0 validation predates score-purpose metadata. Accept it
    # only as audited schema1 validation; do not extend this exception to v2.
    if config.get('evaluation_score_purpose') is None and any(
        not dump.probability_source.startswith('legacy_cpu_sigmoid_batch') for dump in dumps
    ):
        raise ValueError('Missing score purpose is only supported for legacy validation dumps')
    if any(not str(sample_id).startswith('val:') for dump in dumps for sample_id in dump.sample_ids):
        raise ValueError('Non-validation sample ID')
    return config, dumps, rows


def _row_record(rows):
    return {'metrics': summarize_tasks(rows), 'task_metrics': [asdict(row) for row in rows]}


def _fuse(primary, auxiliary):
    rows = []
    for task, (first, second) in enumerate(zip(primary, auxiliary)):
        _, _, _, targets, first_probs, second_probs = align_evaluation_scores(first, second)
        probabilities = 0.8 * first_probs + 0.2 * second_probs
        rows.append(compute_metrics(task, torch.from_numpy(probabilities), torch.from_numpy(targets), 0.5))
    return _row_record(rows)


def _stats(values):
    return {'values': values, 'mean': statistics.mean(values), 'sample_std': statistics.stdev(values)}


def compare_ensembles(full_runs, person_runs):
    if len(full_runs) != 3 or len(person_runs) != 3:
        raise ValueError('Exactly three full and three person validation runs are required')
    indexed, configs, sources = {}, {}, []
    for view, paths in (('full', full_runs), ('person_crop', person_runs)):
        reference = None
        for run in paths:
            config, dumps, rows = audit_validation_run(run, view)
            key = (view, config['seed'])
            if key in indexed:
                raise ValueError('Duplicate seed/view input')
            comparable = {k: v for k, v in config.items() if k not in ('seed', 'git')}
            comparable['evaluation_score_purpose'] = 'validation_search'
            if reference is not None and comparable != reference:
                raise ValueError(f'Per-view configuration drift for {view}')
            reference = comparable
            indexed[key] = (dumps, rows)
            configs[key] = config
            sources.append({'run': str(run.resolve()), 'view': view, 'seed': config['seed'], 'git': config['git'],
                            'config_sha256': hashlib.sha256((run/'config.json').read_bytes()).hexdigest(),
                            'probability_sources': [d.probability_source for d in dumps]})
    for seed in range(3):
        for field in COMMON_CONFIG_FIELDS:
            if configs[('full', seed)].get(field) != configs[('person_crop', seed)].get(field):
                raise ValueError(f'Full/person training configuration drift: {field}')
    groups = {name: [] for name in ('full_single', 'person_single', 'same_seed_full_person',
                                   'cyclic_full_full', 'cyclic_full_person')}
    for seed in range(3):
        auxiliary_seed = (seed + 1) % 3
        full, full_rows = indexed[('full', seed)]
        person, person_rows = indexed[('person_crop', seed)]
        groups['full_single'].append(dict(seed=seed, **_row_record(full_rows)))
        groups['person_single'].append(dict(seed=seed, **_row_record(person_rows)))
        groups['same_seed_full_person'].append(dict(seed=seed, auxiliary_seed=seed, **_fuse(full, person)))
        for name, view in (('cyclic_full_full', 'full'), ('cyclic_full_person', 'person_crop')):
            groups[name].append(dict(seed=seed, auxiliary_seed=auxiliary_seed,
                                     **_fuse(full, indexed[(view, auxiliary_seed)][0])))
    paired = {}
    for label, treatment, control in (
        ('matched_auxiliary_person_minus_full', 'cyclic_full_person', 'cyclic_full_full'),
        ('same_seed_fusion_minus_single', 'same_seed_full_person', 'full_single'),
    ):
        paired[label] = {metric: _stats([a['metrics'][metric]-b['metrics'][metric]
                                        for a, b in zip(groups[treatment], groups[control])]) for metric in METRICS}
    return {
        'schema_version': 1, 'evaluation_split': 'val', 'search_performed': False,
        'rule': {'mode': 'probability', 'primary_weight': 0.8, 'auxiliary_weight': 0.2, 'threshold': 0.5},
        'seeds': [0, 1, 2], 'auxiliary_seed_mapping': [[0, 1], [1, 2], [2, 0]], 'std_ddof': 1,
        'sources': sources, 'groups': groups,
        'aggregate': {name: {metric: _stats([r['metrics'][metric] for r in rows]) for metric in METRICS}
                      for name, rows in groups.items()},
        'paired_deltas': paired,
        'note': 'Descriptive paired comparison; cyclic pairs reuse models and are not independent replicates. No test selection or automatic test launch.',
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--full-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--person-runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.audit_only:
        for view, runs in (('full', args.full_runs), ('person_crop', args.person_runs)):
            for run in runs:
                audit_validation_run(run, view)
                print(f'VALIDATION_SCORE_AUDIT_OK {run}')
        return
    if args.output is None:
        parser.error('--output is required for comparison')
    result = compare_ensembles(args.full_runs, args.person_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result['paired_deltas'], indent=2))
    print('FIXED_VALIDATION_ENSEMBLE_COMPARISON_COMPLETE')


if __name__ == '__main__':
    main()
