"""Compare the five prespecified sample-dependent readouts across both training cohorts."""
import argparse
import json
from pathlib import Path


def metrics(stage_report, method):
    value = stage_report['summary'][method]['all']
    return {name: value[name] for name in ('final_mAP', 'average_mAP')}


def summarize(root, smoke=False):
    title = 'smoke_evaluation' if smoke else 'evaluation'
    paired = {}
    for mode in ('sample_full', 'sample_holdout'):
        methods = ['fixed', 'equal', 'mid_bce', 'late_bce']
        if mode == 'sample_holdout':
            methods.append('mid_asl')
        reports = {}
        for split in ('val', 'test'):
            report = json.loads((root / mode / title / split / 'result.json').read_text())
            if report['split'] != split or len(report['stages']) != (2 if smoke else 8):
                raise ValueError(f'Incomplete {mode} {split} report')
            reports[split] = report
        paired[mode] = {m: {s: metrics(r, m) for s, r in reports.items()} for m in methods}

    for mode, rows in paired.items():
        for method, row in rows.items():
            if method in ('fixed', 'equal'):
                continue
            for split in ('val', 'test'):
                result = row[split]
                result['delta_vs_own_fixed'] = {
                    key: result[key] - rows['fixed'][split][key]
                    for key in ('final_mAP', 'average_mAP')
                }
                result['delta_vs_full_fixed'] = {
                    key: result[key] - paired['sample_full']['fixed'][split][key]
                    for key in ('final_mAP', 'average_mAP')
                }
    holdout = paired['sample_holdout']
    contrast = {s: {k: holdout['mid_asl'][s][k] - holdout['mid_bce'][s][k]
                    for k in ('final_mAP', 'average_mAP')} for s in ('val', 'test')}
    result = {'smoke': smoke, 'training_cohorts': paired, 'paired_asl_minus_bce': contrast}
    dest = root / 'comparison.json'
    tmp = dest.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    tmp.replace(dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    summarize(args.root, args.smoke)
