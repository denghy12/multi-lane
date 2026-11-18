"""Compare sample-level readouts, optionally pairing the independent full-TRAIN ASL run."""
import argparse
import json
from pathlib import Path

MAP_KEYS = ('final_mAP', 'average_mAP')
SPLITS = ('val', 'test')


def metrics(report, method):
    return {key: report['summary'][method]['all'][key] for key in MAP_KEYS}


def difference(a, b):
    return {key: a[key] - b[key] for key in MAP_KEYS}


def load_reports(root, mode, smoke):
    directory = root / ('smoke_evaluation' if smoke else 'evaluation')
    reports = {}
    for split in SPLITS:
        report = json.loads((directory / split / 'result.json').read_text())
        if (report['split'] != split or report['mode'] != mode or report['smoke'] != smoke
                or len(report['stages']) != (2 if smoke else 8)):
            raise ValueError(f'Incomplete or mismatched {mode} {split} report')
        reports[split] = report
    return reports


def check_full_pair(original, extra):
    """Each full-TRAIN run must reproduce the same archived branch and fixed logits exactly."""
    for split in SPLITS:
        for task, (a, b) in enumerate(zip(original[split]['stages'], extra[split]['stages'])):
            ref = a['protocol']['reference_scores_sha256']
            if (not a['exact_replay'] or not b['exact_replay'] or not ref
                    or ref != b['protocol']['reference_scores_sha256']
                    or a['protocol']['task'] != task or b['protocol']['task'] != task):
                raise ValueError(f'Full-TRAIN runs lack a common exact reference at {split} task{task}')
        for name in ('fixed', 'equal'):
            if metrics(original[split], name) != metrics(extra[split], name):
                raise ValueError(f'Full-TRAIN control mismatch for {split} {name}')


def summarize(root, smoke=False, full_asl_root=None):
    paired = {}
    sources = {}
    for mode in ('sample_full', 'sample_holdout'):
        reports = sources[mode] = load_reports(root / mode, mode, smoke)
        methods = ['fixed', 'equal', 'mid_bce', 'late_bce']
        if mode == 'sample_holdout':
            methods.append('mid_asl')
        paired[mode] = {m: {s: metrics(r, m) for s, r in reports.items()} for m in methods}

    if full_asl_root is not None:
        extra = load_reports(full_asl_root, 'sample_full_asl', smoke)
        check_full_pair(sources['sample_full'], extra)
        paired['sample_full']['mid_asl'] = {s: metrics(r, 'mid_asl') for s, r in extra.items()}

    contrasts = {}
    for mode, rows in paired.items():
        for method, row in rows.items():
            if method in ('fixed', 'equal'):
                continue
            for split in SPLITS:
                result = row[split]
                result['delta_vs_own_fixed'] = difference(result, rows['fixed'][split])
                result['delta_vs_full_fixed'] = difference(result, paired['sample_full']['fixed'][split])
        if 'mid_asl' in rows:
            contrasts[mode] = {s: difference(rows['mid_asl'][s], rows['mid_bce'][s]) for s in SPLITS}
    result = {'smoke': smoke, 'training_cohorts': paired, 'paired_asl_minus_bce': contrasts['sample_holdout']}
    dest = root / 'comparison.json'
    if full_asl_root is not None:
        result.update(paired_asl_minus_bce_by_cohort=contrasts, full_asl_source=str(full_asl_root),
                      original_cohorts_source=str(root), full_branch_reference_replayed=True)
        dest = full_asl_root / 'comparison_with_full_asl.json'
    tmp = dest.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    tmp.replace(dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--full-asl-root', type=Path, help='Completed independent sample_full_asl directory')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    summarize(args.root, args.smoke, args.full_asl_root)
