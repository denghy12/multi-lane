"""Read-only NumPy analysis of the completed 18-model historical b32 search."""
import argparse
import json
from pathlib import Path
import numpy as np

from analyze_historical_feature_fusion import ap, metrics, load, read, table, rank_correlation, COUNTS, STARTS


def fixed_path_audit(root, out):
    rows = []
    for seed in (0, 1, 3):
        for profile in ('FULL_BASE', 'THREE_BASE'):
            for split in ('val', 'test'):
                folder = root / f'seed{seed}' / profile / ('historical_' + split) / 'scores'
                last = load(folder / 'task7.npz'); ids = {str(s): i for i, s in enumerate(last['sample_ids'])}
                for t in (0, 2, 6):
                    z = load(folder / f'task{t}.npz'); ix = np.array([ids[str(s)] for s in z['sample_ids']]); n = z['targets'].shape[1]
                    assert np.array_equal(last['targets'][ix, :n], z['targets'])
                    diff = np.abs(last['logits'][ix, :n].astype(float) - z['logits'].astype(float))
                    old = float(ap(z['probabilities'], z['targets']).mean())
                    new = float(ap(last['probabilities'][ix, :n], z['targets']).mean())
                    rows.append(dict(seed=seed, profile=profile, split=split, old_task=t, people=len(ix), classes=n,
                        logits_exact=bool(np.array_equal(last['logits'][ix, :n], z['logits'])),
                        max_abs_logit_diff=float(diff.max()), mean_abs_logit_diff=float(diff.mean()),
                        changed_fraction=float((diff != 0).mean()), old_mAP=old, replayed_mAP=new, delta=new-old))
    table(out / 'fixed_old_path_audit.csv', rows)


def within_image_audit(root, out):
    rows = []
    for seed in (0, 1, 3):
        for split in ('val', 'test'):
            v = load(root / f'seed{seed}/THREE_BASE' / ('historical_' + split) / 'view_scores/task7.npz')
            groups = {}
            for i, identifier in enumerate(v['sample_ids']):
                groups.setdefault(str(identifier).rsplit('#person=', 1)[0], []).append(i)
            groups = [ix for ix in groups.values() if len(ix) > 1]
            discordant = sum(bool(np.any(v['targets'][ix].max(0) != v['targets'][ix].min(0))) for ix in groups)
            for view in ('full', 'person', 'face', 'fused'):
                values = []
                for ix in groups:
                    if view == 'face':
                        ix = [i for i in ix if v['face_reliable'][i]]
                        if len(ix) < 2:
                            continue
                    x = v[view + '_logits'][ix].astype(float)
                    values.append(float((x.max(0) - x.min(0)).mean()))
                rows.append(dict(seed=seed, split=split, view=view, multi_person_images=len(groups),
                    label_discordant_images=discordant, groups_compared=len(values),
                    median_mean_class_logit_span=float(np.median(values)),
                    p90_mean_class_logit_span=float(np.percentile(values, 90)),
                    exact_identical_fraction=float(np.mean(np.asarray(values) == 0))))
    table(out / 'within_image_person_separation.csv', rows)


def main(args):
    root, out = args.root, args.output
    if out.exists():
        assert not any(out.iterdir()), 'Refuse to overwrite existing analysis artifacts'
    else:
        out.mkdir(parents=True)
    complete = read(root / 'completion.json')
    assert complete['status'] == 'complete' and not (root / 'failure.json').exists()
    cells = sorted((c['seed'], c['profile']) for c in complete['jobs'])
    assert len(cells) == 18 and {s for s, _ in cells} == {0, 1, 3}
    data, final, views, configs, reference, results = {}, {}, {}, {}, {}, {}
    summaries, stages, endpoints, training = [], [], [], []
    max_error = 0.; native_equal = 0; view_files = 0
    for seed, profile in cells:
        cell = root / f'seed{seed}' / profile
        c = read(cell / 'train/config.json'); configs[seed, profile] = c
        s = read(cell / 'train/seed_summary.json'); h = read(cell / 'train/training_history.json')
        audit = read(cell / 'training_verification.json')
        assert c['git']['commit'] == complete['head'] and not c['git']['dirty'] and s['config'] == c
        assert c['seed'] == seed and c['historical_b32_profile'] == profile
        assert s['completed_optimizer_updates'] == 13950 and s['completed_epochs'] == 240
        assert all(audit[k] for k in ('old_parameters_unchanged', 'frozen_visual_unchanged', 'recovery_equals_compact'))
        assert sum(x['skipped_optimizer_steps'] for rows in h.values() for x in rows) == 0
        assert len({(c[k]) for k in ('person_color_jitter_strength', 'person_color_jitter_probability')}) == 1
        assert c['person_color_jitter_strength'] == 0 and c['train_batch_size'] == 64
        record = dict(seed=seed, profile=profile, scale=c['adapter_residual_scale'],
            adapter_lr=c['adapter_learning_rate'], auxiliary=c['view_auxiliary_loss_weight'])
        for t in range(8):
            last = h[str(t)][-5:]
            optional = {'adapter_residual_ratio_person', 'adapter_residual_ratio_face'} if profile == 'FULL_BASE' else set()
            training.append(dict(seed=seed, profile=profile, task=t,
                **{key: float(np.mean([r.get(key, np.nan) if key in optional else r[key] for r in last])) for key in
                   ('adapter_residual_ratio_full', 'adapter_residual_ratio_person', 'adapter_residual_ratio_face',
                    'adapter_grad_shared_norm', 'current_loss', 'adapter_objective_loss')}))
        for split in ('val', 'test'):
            folder = cell / ('historical_' + split); d = read(folder / 'result.json')
            results[seed, profile, split] = d
            assert not d['full_cohort'] and len(d['task_metrics']) == 8
            for row in d['task_metrics']:
                t = row['task_id']; z = load(folder / 'scores' / f'task{t}.npz')
                assert len(z['targets']) == row['samples'] == COUNTS[split][t]
                assert (z['targets'] == 1).any(1).all()
                if (split, t) not in reference:
                    reference[split, t] = {k: z[k] for k in ('sample_ids', 'targets', 'class_indices')}
                assert all(np.array_equal(z[k], reference[split, t][k]) for k in reference[split, t])
                m = metrics(z['probabilities'], z['targets'])
                for key, value in m.items():
                    error = abs(value - row[key]); max_error = max(max_error, error)
                    assert error < 1e-8, (seed, profile, split, t, key, error)
                assert np.allclose(ap(z['probabilities'], z['targets']), row['per_class_ap'], atol=1e-8, rtol=0)
                stages.append(dict(seed=seed, profile=profile, split=split, task=t, samples=len(z['targets']), **m))
                if split == 'test':
                    native = load(cell / 'train/test_scores' / f'task{t}.npz')
                    assert all(np.array_equal(z[k], native[k]) for k in ('sample_ids', 'targets', 'class_indices', 'logits', 'probabilities'))
                    native_equal += 1
                if t == 7:
                    final[seed, profile, split] = z
                if profile != 'FULL_BASE':
                    v = load(folder / 'view_scores' / f'task{t}.npz'); view_files += 1
                    assert all(np.array_equal(v[k], z[k]) for k in ('sample_ids', 'targets', 'class_indices'))
                    assert np.array_equal(v['fused_probabilities'], z['probabilities'])
                    vm = {name: float(ap(v[name + '_probabilities'], v['targets']).mean()) for name in ('full', 'person', 'face', 'fused')}
                    for name in vm:
                        error = abs(vm[name] - d['view_diagnostics'][str(t)]['metrics'][name]['mAP'])
                        max_error = max(max_error, error); assert error < 1e-8
                    endpoints.append(dict(seed=seed, profile=profile, split=split, task=t, **vm,
                        fusion_minus_own_full=vm['fused'] - vm['full']))
                    if t == 7:
                        views[seed, profile, split] = v
            for key in ('final_mAP', 'average_mAP', 'final_cF1', 'final_oF1'):
                record[split + '_' + key] = d['summary'][key]
        summaries.append(record)
    summary_index = {(r['seed'], r['profile']): r for r in summaries}
    contrasts, class_rows, task_rows, groups, correlations = [], [], [], [], []
    for split in ('val', 'test'):
        ref = final[0, 'FULL_BASE', split]; y = ref['targets']
        names = np.asarray([str(x).rsplit('#person=', 1)[0] for x in ref['sample_ids']])
        _, inverse, counts = np.unique(names, return_inverse=True, return_counts=True)
        reliable = views[0, 'THREE_BASE', split]['face_reliable']
        masks = dict(all_people=np.ones(len(y), bool), reliable_face=reliable, unreliable_face=~reliable,
            multiple_people=counts[inverse] > 1, single_person=counts[inverse] == 1)
        for seed, profile in cells:
            p = final[seed, profile, split]['probabilities']
            for name, mask in masks.items():
                groups.append(dict(seed=seed, profile=profile, split=split, subset=name,
                    people=int(mask.sum()), images=len(np.unique(names[mask])), **metrics(p[mask], y[mask])))
            if profile != 'FULL_BASE':
                v = views[seed, profile, split]
                assert np.array_equal(v['face_reliable'], reliable)
                correlations.append(dict(seed=seed, profile=profile, split=split,
                    full_person=rank_correlation(v['full_logits'], v['person_logits']),
                    full_reliable_face=rank_correlation(v['full_logits'][reliable], v['face_logits'][reliable]),
                    person_reliable_face=rank_correlation(v['person_logits'][reliable], v['face_logits'][reliable])))
                for base in ('FULL_BASE', 'THREE_BASE'):
                    if profile == base:
                        continue
                    a, b = results[seed, profile, split], results[seed, base, split]
                    delta = np.asarray(a['task_metrics'][7]['per_class_ap']) - np.asarray(b['task_metrics'][7]['per_class_ap'])
                    contrasts.append(dict(seed=seed, profile=profile, reference=base, split=split,
                        final_delta=a['summary']['final_mAP']-b['summary']['final_mAP'],
                        average_delta=a['summary']['average_mAP']-b['summary']['average_mAP'],
                        positive_stages=sum(x['mAP'] > z['mAP'] for x, z in zip(a['task_metrics'], b['task_metrics'])),
                        positive_classes=int((delta > 0).sum())))
                    for i, name in enumerate(configs[seed, profile]['class_order']):
                        class_rows.append(dict(seed=seed, profile=profile, reference=base, split=split, class_name=name,
                            left_AP=a['task_metrics'][7]['per_class_ap'][i], right_AP=b['task_metrics'][7]['per_class_ap'][i],
                            delta=delta[i], positives=int(y[:, i].sum())))
                    for t, (start, end) in enumerate(zip(STARTS[:-1], STARTS[1:])):
                        task_rows.append(dict(seed=seed, profile=profile, reference=base, split=split, task=t,
                            mean_AP_delta=float(delta[start:end].mean()), contribution_to_final_mAP=float(delta[start:end].sum()/26)))
    aggregate = []
    for profile in sorted({p for _, p in cells}):
        rows = [r for r in summaries if r['profile'] == profile]
        for split in ('val', 'test'):
            for key in ('final_mAP', 'average_mAP'):
                scores = [r[split + '_' + key] for r in rows]
                deltas = [r[split + '_' + key] - summary_index[r['seed'], 'FULL_BASE'][split + '_' + key] for r in rows]
                aggregate.append(dict(profile=profile, seeds='/'.join(str(r['seed']) for r in rows), split=split, metric=key,
                    mean=float(np.mean(scores)), sample_std=float(np.std(scores, ddof=1)),
                    mean_paired_delta_vs_full=float(np.mean(deltas)), positive_seeds=sum(v > 0 for v in deltas)))
    for name, rows in [('summary', summaries), ('stage_metrics', stages), ('endpoints', endpoints),
            ('training_statistics', training), ('contrasts', contrasts), ('class_deltas', class_rows),
            ('task_class_deltas', task_rows), ('subgroups', groups), ('view_rank_correlations', correlations), ('aggregate', aggregate)]:
        table(out / (name + '.csv'), rows)
    historical = []
    if args.historical_archive:
        for t in range(8):
            a = load(args.historical_archive / 'test_scores' / f'task{t}.npz')
            b = load(root / 'seed1/THREE_BASE/train/test_scores' / f'task{t}.npz')
            assert all(np.array_equal(a[k], b[k]) for k in ('sample_ids', 'targets', 'class_indices', 'logits', 'probabilities'))
            historical.append(t)
    integrity = dict(status='verified', models=18, phase_scores=288, view_scores=view_files,
        native_test_replays_equal=native_equal, maximum_metric_error=max_error,
        historical_seed1_exact_stages=historical, same_people_labels_and_classes=True,
        training_or_model_forward=False, test_used_for_parameter_search=True)
    (out / 'analysis_integrity.json').write_text(json.dumps(integrity, indent=2)+'\n')
    fixed_path_audit(root, out)
    within_image_audit(root, out)
    print('AUDIT', json.dumps(integrity), flush=True)
    # Conditional image-sampling uncertainty. This is not a correction for the
    # parameter search and does not estimate variance over future training seeds.
    chosen = [(s,p) for s in (0,1,3) for p in ('FULL_BASE','THREE_BASE')]
    chosen += [(0,'THREE_AUX_02'),(3,'THREE_AUX_02'),(3,'THREE_LR_0002'),(0,'THREE_SCALE_001')]
    bootstrap = {}
    for split in ('val','test'):
        z = final[0,'FULL_BASE',split]; y=z['targets']; image_groups={}
        for i,identifier in enumerate(z['sample_ids']):
            image_groups.setdefault(str(identifier).rsplit('#person=',1)[0],[]).append(i)
        image_groups=list(image_groups.values()); rng=np.random.default_rng(20261025+(split=='test'))
        samples={}
        for repeat in range(args.bootstrap):
            ix=np.concatenate([image_groups[i] for i in rng.integers(0,len(image_groups),len(image_groups))])
            values={(s,p):float(ap(final[s,p,split]['probabilities'][ix],y[ix]).mean()) for s,p in chosen}
            delta={f'seed{s}/{p}-FULL_BASE':values[s,p]-values[s,'FULL_BASE'] for s,p in chosen if p!='FULL_BASE'}
            delta['mean_seed013_THREE_BASE-FULL_BASE']=float(np.mean([values[s,'THREE_BASE']-values[s,'FULL_BASE'] for s in (0,1,3)]))
            delta['mean_seed03_AUX02-THREE_BASE']=float(np.mean([values[s,'THREE_AUX_02']-values[s,'THREE_BASE'] for s in (0,3)]))
            for key,value in delta.items():samples.setdefault(key,[]).append(value)
        bootstrap[split]=dict(repeats=args.bootstrap,unit='original_image',images=len(image_groups),people=len(y),
            selection_adjusted=False,future_training_seed_variance_estimated=False,
            contrasts={key:dict(ci95_pp=np.percentile(value,[2.5,97.5]).tolist()) for key,value in samples.items()})
        print('BOOTSTRAP',split,json.dumps(bootstrap[split]),flush=True)
    (out/'bootstrap_final.json').write_text(json.dumps(bootstrap,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--historical-archive',type=Path)
    parser.add_argument('--bootstrap',type=int,default=500)
    main(parser.parse_args())
