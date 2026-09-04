"""Strict same-global-seed Full repeats versus existing same-seed Full+Person."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .compare_fixed_test_ensembles import RULE, audit_test_run, sha256
from .compare_validation_ensembles import METRICS, _fuse, _row_record, _stats
from .evaluation_scores import align_evaluation_scores
from .fuse_validation_scores import COMMON_CONFIG_FIELDS


def _indexed(paths, view):
    if len(paths) != 3:
        raise ValueError('Exactly three runs per role are required')
    result = {}
    for path in paths:
        config, dumps, rows = audit_test_run(path, view)
        seed = config['seed']
        if seed in result:
            raise ValueError('Duplicate seed')
        result[seed] = (path, config, dumps, rows)
    return result


def compare_same_seed(full_runs, repeat_runs, person_runs):
    roles = {'full': _indexed(full_runs, 'full'), 'repeat': _indexed(repeat_runs, 'full'),
             'person': _indexed(person_runs, 'person_crop')}
    groups = {name: [] for name in ('full_single', 'repeat_single', 'same_seed_full_full', 'same_seed_full_person')}
    diagnostics, sources = [], []
    for seed in range(3):
        full, config, first, rows = roles['full'][seed]
        repeat, repeat_config, second, repeat_rows = roles['repeat'][seed]
        person, person_config, person_dumps, _ = roles['person'][seed]
        if full.resolve() == repeat.resolve():
            raise ValueError('Repeat must be a distinct training run, not the same source directory')
        if config != repeat_config:
            raise ValueError('Strict same-seed repeat requires identical config including source commit')
        for field in COMMON_CONFIG_FIELDS:
            if config.get(field) != person_config.get(field):
                raise ValueError(f'Full/person configuration drift: {field}')
        groups['full_single'].append(dict(seed=seed, **_row_record(rows)))
        groups['repeat_single'].append(dict(seed=seed, **_row_record(repeat_rows)))
        groups['same_seed_full_full'].append(dict(seed=seed, auxiliary_seed=seed, **_fuse(first, second)))
        groups['same_seed_full_person'].append(dict(seed=seed, auxiliary_seed=seed, **_fuse(first, person_dumps)))
        task_differences = []
        for task, (a, b) in enumerate(zip(first, second)):
            _, logits_a, logits_b, _, probs_a, probs_b = align_evaluation_scores(a, b)
            task_differences.append({'task': task,
                                     'max_abs_logit_difference': float(np.max(np.abs(logits_a-logits_b))),
                                     'max_abs_probability_difference': float(np.max(np.abs(probs_a-probs_b))),
                                     'differing_probability_count': int(np.count_nonzero(probs_a != probs_b))})
        histories = [json.loads((run/'training_history.json').read_text()) for run in (full, repeat)]
        normalized = [{task: [{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in epochs]
                       for task, epochs in history.items()} for history in histories]
        diagnostics.append({'seed': seed, 'tasks': task_differences,
                            'training_history_exact_except_elapsed': normalized[0] == normalized[1]})
        for role, run in (('full', full), ('repeat', repeat), ('person', person)):
            files = [run/name for name in ('config.json', 'training_history.json', 'seed_summary.json')]
            files += [run/'test_scores'/f'task{t}.npz' for t in range(8)]
            sources.append({'role': role, 'seed': seed, 'run': str(run.resolve()),
                            'artifact_sha256': {str(p.relative_to(run)): sha256(p) for p in files}})
    paired = {}
    for name, treatment, control in (
        ('person_minus_repeat', 'same_seed_full_person', 'same_seed_full_full'),
        ('full_full_minus_single', 'same_seed_full_full', 'full_single'),
    ):
        paired[name] = {metric: _stats([a['metrics'][metric]-b['metrics'][metric]
                                       for a,b in zip(groups[treatment], groups[control])]) for metric in METRICS}
    return {'schema_version': 1, 'evaluation_split': 'test', 'search_performed_on_test': False,
            'rule': RULE, 'auxiliary_seed_mapping': [[0,0],[1,1],[2,2]], 'sources': sources,
            'groups': groups, 'paired_deltas': paired, 'repeat_diagnostics': diagnostics,
            'aggregate': {name: {m: _stats([r['metrics'][m] for r in rows]) for m in METRICS}
                          for name, rows in groups.items()},
            'note': 'Freshly trained exact-global-seed repeats, not independently seeded ensemble members. '
                    'Identical predictions are possible; floating-point fusion may still affect ties. '
                    'No test search or winner selection.'}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--full-runs', type=Path, nargs=3, required=True)
    parser.add_argument('--repeat-runs', type=Path, nargs=3)
    parser.add_argument('--person-runs', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.audit_only:
        _indexed(args.full_runs, 'full')
        _indexed(args.person_runs, 'person_crop')
        print('SAME_SEED_SOURCE_AUDIT_OK')
        return
    if args.repeat_runs is None or args.output is None:
        parser.error('--repeat-runs and --output are required for comparison')
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite existing comparison')
    result = compare_same_seed(args.full_runs, args.repeat_runs, args.person_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({name: metrics['final_mAP'] for name, metrics in result['aggregate'].items()}, indent=2))
    print('SAME_SEED_FULL_FULL_TEST_COMPLETE')


if __name__ == '__main__':
    main()
