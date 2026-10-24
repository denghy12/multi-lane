"""CPU-only audit of the four completed historical feature-fusion controls."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ARMS = ('FULL_ASL', 'THREE_BCE_AUX', 'THREE_ASL_NO_AUX', 'THREE_ASL_AUX')
PAIRS = (('THREE_BCE_AUX', 'FULL_ASL'), ('THREE_ASL_NO_AUX', 'FULL_ASL'),
         ('THREE_ASL_AUX', 'FULL_ASL'), ('THREE_ASL_AUX', 'THREE_ASL_NO_AUX'),
         ('THREE_ASL_AUX', 'THREE_BCE_AUX'))
STARTS = (0, 5, 8, 11, 14, 17, 20, 23, 26)
COUNTS = dict(val=(2285, 2375, 2381, 2397, 2397, 2397, 2397, 2397),
              test=(3142, 4600, 4691, 5232, 5342, 5354, 5367, 5368))


def read(path):
    return json.loads(path.read_text())


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def ap(p, y):
    order = p.argsort(axis=0)[::-1]
    positive = np.take_along_axis(y, order, axis=0) == 1
    cumulative = positive.cumsum(axis=0)
    total = cumulative[-1].copy()
    cumulative[~positive] = 0
    return 100 * (cumulative / np.arange(1, len(p) + 1)[:, None]).sum(0) / (total + 1e-8)


def metrics(p, y):
    pred = p > .5
    tp = np.logical_and(pred, y == 1).sum(0)
    fp = np.logical_and(pred, y == 0).sum(0)
    fn = np.logical_and(~pred, y == 1).sum(0)
    # Match the runner's zero-denominator convention exactly.
    ratio = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    cf1 = float(100 * ratio(2 * precision * recall, precision + recall).mean())
    op, ore = float(ratio(tp.sum(), (tp + fp).sum())), float(ratio(tp.sum(), (tp + fn).sum()))
    of1 = 100 * (2 * op * ore / (op + ore) if op + ore else 0.)
    return dict(mAP=float(ap(p, y).mean()), cF1=cf1, oF1=of1)


def table(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, list(rows[0])); writer.writeheader(); writer.writerows(rows)


def ranks(values):
    result = np.empty(values.shape, dtype=float)
    for c in range(values.shape[1]):
        _, inverse, counts = np.unique(values[:, c], return_inverse=True, return_counts=True)
        ends = counts.cumsum()
        result[:, c] = ((ends - counts + ends - 1) / 2)[inverse]
    return result


def rank_correlation(a, b):
    a, b = ranks(a), ranks(b)
    a -= a.mean(0); b -= b.mean(0)
    den = np.sqrt((a * a).sum(0) * (b * b).sum(0))
    valid = den > 0
    return float(((a * b).sum(0)[valid] / den[valid]).mean())


def main(args):
    root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=False)
    assert read(root / 'completion.json')['status'] == 'complete'
    assert not (root / 'failure.json').exists()
    results, cells, views = {}, {}, {}
    stages, endpoints, subsets, class_rows, task_rows, correlations, contrasts = [], [], [], [], [], [], []
    max_error = 0.; native_exact = 0; summary = []
    classes = read(root / 'FULL_ASL/train/config.json')['class_order']
    for arm in ARMS:
        record = dict(arm=arm)
        source = read(root / arm / 'train/seed_summary.json')
        assert source['completed_optimizer_updates'] == 13950 and source['completed_epochs'] == 240
        for split in ('val', 'test'):
            path = root / arm / ('historical_' + split)
            d = read(path / 'result.json'); results[arm, split] = d
            assert not d['full_cohort'] and len(d['task_metrics']) == 8
            for row in d['task_metrics']:
                t = row['task_id']; z = load(path / 'scores' / f'task{t}.npz')
                cells[arm, split, t] = z
                ref = cells['FULL_ASL', split, t]
                assert len(z['targets']) == row['samples'] == COUNTS[split][t]
                assert (z['targets'] == 1).any(1).all()
                assert all(np.array_equal(z[k], ref[k]) for k in ('sample_ids', 'targets', 'class_indices'))
                m = metrics(z['probabilities'], z['targets'])
                for key, value in m.items():
                    error = abs(value - row[key]); max_error = max(max_error, error)
                    assert error < 1e-8, (arm, split, t, key, value, row[key])
                stages.append(dict(arm=arm, split=split, task=t, samples=len(z['targets']), **m,
                                   delta_vs_full=row['mAP'] - results['FULL_ASL', split]['task_metrics'][t]['mAP']))
                if split == 'test':
                    native = load(root / arm / 'train/test_scores' / f'task{t}.npz')
                    assert all(np.array_equal(z[k], native[k]) for k in ('sample_ids', 'targets', 'class_indices', 'logits', 'probabilities'))
                    native_exact += 1
                if arm != 'FULL_ASL':
                    v = load(path / 'view_scores' / f'task{t}.npz'); views[arm, split, t] = v
                    assert all(np.array_equal(v[k], z[k]) for k in ('sample_ids', 'targets', 'class_indices'))
                    assert np.array_equal(v['fused_probabilities'], z['probabilities'])
                    expected = views['THREE_BCE_AUX', split, t]
                    assert np.array_equal(v['face_reliable'], expected['face_reliable'])
                    vm = {name: float(ap(v[name + '_probabilities'], v['targets']).mean()) for name in ('full', 'person', 'face', 'fused')}
                    endpoints.append(dict(arm=arm, split=split, task=t, samples=len(v['targets']), **vm,
                        fusion_minus_own_full=vm['fused']-vm['full'],
                        own_full_minus_independent_full=vm['full']-results['FULL_ASL', split]['task_metrics'][t]['mAP']))
            for key in ('final_mAP', 'average_mAP', 'final_cF1', 'final_oF1'):
                record[split + '_' + key] = d['summary'][key]
        summary.append(record)
    for split in ('val', 'test'):
        ref = cells['FULL_ASL', split, 7]; ids, y = ref['sample_ids'], ref['targets']
        group_names = np.asarray([str(v).rsplit('#person=', 1)[0] for v in ids])
        _, inverse, group_counts = np.unique(group_names, return_inverse=True, return_counts=True)
        masks = dict(all_people=np.ones(len(y), bool), multiple_people=group_counts[inverse] > 1,
                     single_person=group_counts[inverse] == 1)
        reliable = views['THREE_BCE_AUX', split, 7]['face_reliable']
        masks.update(reliable_face=reliable, unreliable_face=~reliable)
        for name, mask in masks.items():
            for arm in ARMS:
                p = cells[arm, split, 7]['probabilities']
                row = dict(arm=arm, split=split, subset=name, samples=int(mask.sum()),
                    images=len(np.unique(group_names[mask])), **metrics(p[mask], y[mask]))
                row['full_endpoint_mAP'] = row['mAP'] if arm == 'FULL_ASL' else float(ap(views[arm, split, 7]['full_probabilities'][mask], y[mask]).mean())
                row['person_endpoint_mAP'] = '' if arm == 'FULL_ASL' else float(ap(views[arm, split, 7]['person_probabilities'][mask], y[mask]).mean())
                row['face_endpoint_mAP'] = '' if arm == 'FULL_ASL' else float(ap(views[arm, split, 7]['face_probabilities'][mask], y[mask]).mean())
                subsets.append(row)
        for arm in ARMS[1:]:
            v = views[arm, split, 7]
            correlations.append(dict(arm=arm, split=split,
                full_person_mean_class_spearman=rank_correlation(v['full_logits'], v['person_logits']),
                full_face_reliable_mean_class_spearman=rank_correlation(v['full_logits'][reliable], v['face_logits'][reliable]),
                person_face_reliable_mean_class_spearman=rank_correlation(v['person_logits'][reliable], v['face_logits'][reliable])))
        for left, right in PAIRS:
            a, b = results[left, split], results[right, split]
            delta = np.asarray(a['task_metrics'][7]['per_class_ap']) - np.asarray(b['task_metrics'][7]['per_class_ap'])
            contrasts.append(dict(left=left, right=right, split=split,
                final_delta=a['summary']['final_mAP']-b['summary']['final_mAP'],
                average_delta=a['summary']['average_mAP']-b['summary']['average_mAP'],
                positive_stages=sum(x['mAP'] > z['mAP'] for x, z in zip(a['task_metrics'], b['task_metrics'])),
                positive_classes=int((delta > 0).sum())))
            for i, name in enumerate(classes):
                class_rows.append(dict(left=left, right=right, split=split, class_name=name,
                    left_AP=a['task_metrics'][7]['per_class_ap'][i], right_AP=b['task_metrics'][7]['per_class_ap'][i], delta=delta[i],
                    positive_people=int(y[:, i].sum())))
            for t, (start, end) in enumerate(zip(STARTS[:-1], STARTS[1:])):
                task_rows.append(dict(left=left, right=right, split=split, task=t,
                    classes=' / '.join(classes[start:end]), mean_AP_delta=float(delta[start:end].mean()),
                    contribution_to_final_mAP=float(delta[start:end].sum()/26)))
    for name, rows in [('summary', summary), ('stage_metrics', stages), ('endpoints', endpoints),
                       ('subsets', subsets), ('class_deltas', class_rows), ('task_class_deltas', task_rows),
                       ('view_rank_correlations', correlations), ('contrasts', contrasts)]:
        table(out / (name + '.csv'), rows)
    audit = dict(status='verified',models=4,full_evaluations=8,phase_scores=64,native_test_score_replays_equal=native_exact,
        view_score_files=48,max_metric_error=max_error,training_seeds=1,evaluation_scope='samples_intersect_seen_classes',
        training_or_model_forward=False,all_model_people_labels_classes_equal=True)
    (out / 'analysis_integrity.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('INDEPENDENT_AUDIT', json.dumps(audit), flush=True)
    # Conditional uncertainty over people grouped by original image; this does
    # not measure training-seed variance or select a test model.
    bootstrap = {}
    for split in ('val', 'test'):
        z = cells['FULL_ASL', split, 7]; y = z['targets']; groups = {}
        for i, identifier in enumerate(z['sample_ids']):
            groups.setdefault(str(identifier).rsplit('#person=',1)[0], []).append(i)
        groups = list(groups.values()); rng = np.random.default_rng(20261024 + (split == 'test'))
        samples = {a+'-'+b: [] for a,b in PAIRS}
        for repeat in range(args.bootstrap):
            ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            scores = {arm: float(ap(cells[arm,split,7]['probabilities'][ix], y[ix]).mean()) for arm in ARMS}
            for a,b in PAIRS:
                samples[a+'-'+b].append(scores[a]-scores[b])
        bootstrap[split] = dict(repeats=args.bootstrap,unit='original_image',images=len(groups),people=len(y),training_seeds=1,
            contrasts={a+'-'+b:dict(point_pp=results[a,split]['summary']['final_mAP']-results[b,split]['summary']['final_mAP'],
                ci95_pp=np.percentile(samples[a+'-'+b], [2.5,97.5]).tolist()) for a,b in PAIRS})
        print('BOOTSTRAP',split,json.dumps(bootstrap[split]),flush=True)
    (out / 'bootstrap_final.json').write_text(json.dumps(bootstrap,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=500)
    main(parser.parse_args())
