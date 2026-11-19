"""Audit whether saved person-class weights identify reliable and useful single-view predictions."""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import CLASS_ORDER, TASK_SIZES, task_indices
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.evaluate_d9_relation import VIEWS, write_json

TASKS = (0, 3, 6)
TRANSFER = .05
PROTOCOL = dict(tasks=list(TASKS), method='person_class', head='current at task introduction',
                baseline_by_face=[[.80, .20, 0.], [.64, .16, .20]], full_transfer=TRANSFER,
                label_strata=['all', 'positive', 'negative'], extreme_percentile=.25,
                binning='within class, availability and label stratum; tied weights share midrank',
                aggregation='equal-weight class/availability cells, with counts of informative cells')


def correlation(first_rank, values):
    if len(values) < 2 or np.ptp(first_rank) == 0 or np.ptp(values) == 0:
        return None
    return float(np.corrcoef(first_rank, rankdata(values))[0, 1])


def describe(w, truth_prob, gain, positive):
    """Use midpoint percentile so tied weights are never split by row order."""
    n = len(w)
    ranks = rankdata(w)
    pct = (ranks - .5) / max(n, 1)
    result = {}
    for label, selected in [('overall', np.ones(n, dtype=bool)), ('low', pct <= .25),
                            ('high', pct >= .75)]:
        size = int(selected.sum())
        result[label] = dict(n=size, positives=int(positive[selected].sum()))
        for metric, values in [('weight', w), ('tcp', truth_prob), ('bce_gain', gain),
                               ('benefit_rate', gain > 0), ('tie_rate', gain == 0)]:
            result[label][metric] = float(np.mean(values[selected])) if size else None
    result['high_minus_low'] = {k: result['high'][k] - result['low'][k]
                                if result['high']['n'] and result['low']['n'] else None
                                for k in ('tcp', 'bce_gain', 'benefit_rate')}
    result['spearman'] = {'tcp': correlation(ranks, truth_prob), 'bce_gain': correlation(ranks, gain)}
    return result


def audit_rows(task, y, z, valid, weights):
    cols = np.asarray(task_indices(task))
    y, z, weights = y[:, cols], z[:, cols].astype('float64'), weights[:, cols]
    person_weights = np.asarray(PROTOCOL['baseline_by_face'])[valid.astype(int)]
    fixed = (z * person_weights[:, None, :]).sum(-1)
    sign = 2 * y.astype('float64') - 1
    base_loss = np.logaddexp(0., -sign * fixed)
    result = []
    for v in (1, 2):
        truth_prob = np.exp(-np.logaddexp(0., -sign * z[..., v]))
        revised = fixed + TRANSFER * (z[..., v] - z[..., 0])
        gain = base_loss - np.logaddexp(0., -sign * revised)
        for col, index in enumerate(cols):
            for reliable in ([True] if v == 2 else [False, True]):
                group = valid == reliable
                include = [('all', group), ('positive', group & (y[:, col] == 1)),
                           ('negative', group & (y[:, col] == 0))]
                for label, mask in include:
                    result.append(dict(task=task, view=VIEWS[v], emotion=CLASS_ORDER[index],
                                       class_index=int(index), reliable=reliable, stratum=label,
                                       **describe(weights[mask, col, v], truth_prob[mask, col],
                                                  gain[mask, col], y[mask, col])))
    return result


def summaries(rows):
    result = {}
    for view in VIEWS[1:]:
        result[view] = {}
        for stratum in PROTOCOL['label_strata']:
            cells = [r for r in rows if r['view'] == view and r['stratum'] == stratum]
            report = dict(cells=len(cells))
            for statistic, metrics in [('high_minus_low', ('tcp', 'bce_gain', 'benefit_rate')),
                                       ('spearman', ('tcp', 'bce_gain'))]:
                report[statistic] = {}
                for metric in metrics:
                    values = [r[statistic][metric] for r in cells if r[statistic][metric] is not None]
                    report[statistic][metric] = dict(mean=float(np.mean(values)) if values else None,
                                                     positive_cells=sum(v > 0 for v in values),
                                                     informative_cells=len(values))
            result[view][stratum] = report
    return result


def load_stage(root, split, task, stage):
    score_path = root/'scores'/split/f'task{task}.npz'
    feature_path = root/'features'/f'task{task}'/f'{split}.npz'
    if stage['stage'] != task:
        raise ValueError('Source stage order mismatch')
    require_digest(score_path, stage['predictions_sha256'])
    require_digest(feature_path, stage['features_sha256'])
    with np.load(feature_path, allow_pickle=False) as f, np.load(score_path, allow_pickle=False) as s:
        ids, valid, y, z = (f[k] for k in ('sample_ids', 'reliable', 'targets', 'branch_logits'))
        w = s['person_class_weights']
        size = (stage['people'], sum(TASK_SIZES[:task+1]))
        if (tuple(f['view_order']) != VIEWS or tuple(f['head_ids']) != tuple(range(task+1)) or
                tuple(s['task_sizes']) != TASK_SIZES[:task+1] or ids.shape != (size[0],) or
                len(np.unique(ids)) != size[0] or not np.array_equal(ids, s['sample_ids']) or
                valid.shape != ids.shape or valid.dtype != np.bool_ or
                int(valid.sum()) != stage['reliable_face'] or not np.array_equal(valid, s['reliable']) or
                y.shape != size or not np.isin(y, [0, 1]).all() or
                z.shape != size+(3,) or w.shape != z.shape or not np.isfinite(z).all() or
                not np.isfinite(w).all() or (w < 0).any() or w[~valid, :, 2].any()):
            raise ValueError('Invalid source axes, weights, people or labels')
        np.testing.assert_allclose(w.sum(-1), 1., atol=2e-6, rtol=0)
        np.testing.assert_allclose((z*w).sum(-1), s['person_class_logits'], rtol=2e-6, atol=2e-5)
    return audit_rows(task, y, z, valid, w)


def run(caran, out, split):
    out.mkdir(parents=True, exist_ok=True)
    lock_path = out/'diagnostic_val_lock.json'
    val_path = out/'diagnostic_val.json'
    source_seal = read(caran/'validation_seal.json')
    require_digest(caran/'caran_fit_lock.json', source_seal['caran_fit_lock_sha256'])
    require_digest(caran/'val_results.json', source_seal['val_results_sha256'])
    lock_core = dict(protocol=PROTOCOL, source_validation_seal_sha256=sha256_file(caran/'validation_seal.json'))
    if split == 'test':
        lock = read(lock_path)
        require_digest(val_path, lock['diagnostic_val_sha256'])
        if any(lock[k] != v for k, v in lock_core.items()):
            raise ValueError('Validation source or protocol differs from the locked diagnostic')
    source_path = caran/f'{split}_results.json'
    ref = read(source_path)
    if ref['split'] != split or ref['caran_fit_lock_sha256'] != source_seal['caran_fit_lock_sha256']:
        raise ValueError('Unexpected split or CARAN training source')
    if split == 'test' and ref['validation_seal_sha256'] != lock_core['source_validation_seal_sha256']:
        raise ValueError('Test CARAN predictions refer to another Validation seal')
    rows, sources = [], {}
    for task in TASKS:
        stage = ref['stages'][task]
        rows.extend(load_stage(caran, split, task, stage))
        sources[str(task)] = {k: stage[k] for k in ('checkpoint_sha256', 'features_sha256', 'predictions_sha256')}
    report = dict(split=split, protocol=PROTOCOL, source_report_sha256=sha256_file(source_path),
                  sources=sources, cells=rows, macro=summaries(rows),
                  by_task={str(t): summaries([r for r in rows if r['task'] == t]) for t in TASKS})
    path = out/f'diagnostic_{split}.json'
    if path.exists() and read(path) != report:
        raise ValueError(f'Incompatible previous diagnostic: {path}')
    write_json(path, report)
    if split == 'val':
        write_json(lock_path, dict(**lock_core, diagnostic_val_sha256=sha256_file(path)))
    print(split, {v: report['macro'][v]['all']['high_minus_low'] for v in VIEWS[1:]}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caran', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    run(**vars(parser.parse_args()))
