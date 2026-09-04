"""Evaluate the predeclared cyclic Full+Full/Full+Person test control; no search."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .compare_validation_ensembles import _compare_ensembles
from .fuse_validation_scores import validated_run_scores
from .runner import CLASS_ORDER, TASK_SIZES

RULE = {'mode': 'probability', 'primary_weight': 0.8,
        'auxiliary_weight': 0.2, 'threshold': 0.5}
MAPPING = [[0, 1], [1, 2], [2, 0]]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_test_run(run, view):
    config = json.loads((run / 'config.json').read_text())
    summary = json.loads((run / 'seed_summary.json').read_text())
    history = json.loads((run / 'training_history.json').read_text())
    if (config.get('reporting_split') != 'test'
            or config.get('evaluation_score_purpose') != 'fixed_test_fusion'
            or config.get('save_evaluation_scores') is not True):
        raise ValueError('Only fixed_test_fusion held-out test runs are allowed')
    if (config.get('input_mode') != view or config.get('seed') not in (0, 1, 2)
            or config.get('threshold') != 0.5 or config.get('save_checkpoints') is not False):
        raise ValueError('Input view, seed, threshold or checkpoint configuration drift')
    if view == 'person_crop' and config.get('person_transform_mode') != 'letterbox':
        raise ValueError('Person input must use letterbox')
    if (config.get('task_sizes') != list(TASK_SIZES)
            or config.get('class_order') != list(CLASS_ORDER)):
        raise ValueError('Unexpected task sizes or class order')
    if config.get('git', {}).get('dirty') is not False or not config['git'].get('commit'):
        raise ValueError('A clean source commit is required')
    if (summary.get('status') != 'complete' or summary.get('completed_epochs') != 240
            or summary.get('completed_optimizer_updates') != 13950):
        raise ValueError('Incomplete training budget')
    if set(history) != {str(t) for t in range(8)} or any(len(history[str(t)]) != 30 for t in range(8)):
        raise ValueError('Missing task/epoch history')
    if any(e['skipped_optimizer_steps'] != 0 for rows in history.values() for e in rows):
        raise ValueError('Skipped optimizer updates')
    dumps, rows = validated_run_scores(run, 'test')
    if any(not str(sid).startswith('test:') for dump in dumps for sid in dump.sample_ids):
        raise ValueError('Non-test sample ID')
    return config, dumps, rows


def compare_fixed_test_ensembles(full_runs, person_runs, validation_reference):
    reference = json.loads(validation_reference.read_text())
    if (reference.get('evaluation_split') != 'val' or reference.get('search_performed') is not False
            or reference.get('rule') != RULE or reference.get('auxiliary_seed_mapping') != MAPPING):
        raise ValueError('Validation reference must contain the predeclared fixed rule and mapping')
    result = _compare_ensembles(full_runs, person_runs, audit_test_run, 'test')
    result.update(
        comparison='locked_matched_auxiliary_test_control',
        search_performed_on_test=False,
        training_performed=False,
        evaluated_weight_rules=1,
        validation_reference={'path': str(validation_reference.resolve()),
                              'sha256': sha256(validation_reference)},
        note=('Descriptive paired comparison; cyclic pairs reuse models and are not independent '
              'replicates. No test selection, weight search, new training or new inference.'),
    )
    for source in result['sources']:
        run = Path(source['run'])
        files = [run / name for name in ('config.json', 'seed_summary.json', 'training_history.json')]
        files += [run / 'test_scores' / f'task{t}.npz' for t in range(8)]
        source['artifact_sha256'] = {str(p.relative_to(run)): sha256(p) for p in files}
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--full-runs', type=Path, nargs=3, required=True)
    parser.add_argument('--person-runs', type=Path, nargs=3, required=True)
    parser.add_argument('--validation-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing results; use a new output path')
    result = compare_fixed_test_ensembles(args.full_runs, args.person_runs, args.validation_reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({name: metrics['final_mAP'] for name, metrics in result['aggregate'].items()}, indent=2))
    print('LOCKED_TEST_ENSEMBLE_COMPARISON_COMPLETE')


if __name__ == '__main__':
    main()
