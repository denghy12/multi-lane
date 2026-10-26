"""CPU audit of private Selectors and Face-only adaptation of a protected Person model."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_historical_feature_fusion import (
    COUNTS, STARTS, ap, load, metrics, rank_correlation, read, table,
)
from analyze_head_anchored_eight_face import bucket_masks


def cases(root, original, previous, split):
    result = {'B0': (original / 'BASE' / ('historical_' + split), None),
              'DUAL_NATIVE': (previous / 'FACE_JOINT' / ('native_' + split), 'on')}
    for arm, prefix, head, conditions in (
        ('REFERENCE_PERSON', 'PERSON', 'b0', ('on', 'off')),
        ('REFERENCE_DUAL', 'DUAL', 'b0', ('on', 'face_off', 'off')),
        ('PRIVATE_SELECTORS', 'PRIVATE', 'b0', ('on', 'face_off', 'off')),
        ('PRIVATE_SELECTORS', 'PRIVATE_NATIVE', 'native', ('on',)),
        ('FROZEN_PERSON_FACE', 'FROZEN_FACE', 'b0', ('on', 'face_off')),
    ):
        for mode in conditions:
            name = prefix + ('' if mode == 'on' else '_' + mode.upper())
            result[name] = (root / arm / (head + '_' + split), mode)
    return result


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    completion = read(args.root / 'completion.json')
    assert completion['status'] == 'complete' and completion['seed'] == 1
    assert completion['source_checkpoints_unchanged']
    assert not (args.root / 'failure.json').exists()
    data, final_views = {}, {}
    summary, stages, views, subgroups, classes, class_tasks, correlations = [], [], [], [], [], [], []
    error = 0.; references = protected = private_masks = 0
    class_names = read(args.original_root / 'BASE/train/config.json')['class_order']
    for split in ('val', 'test'):
        for name, (folder, mode) in cases(args.root, args.original_root, args.previous_root, split).items():
            result = read(folder / 'result.json')
            reports = result['task_metrics'] if mode is None else result['task_metrics'][mode]
            parent = folder if mode is None else folder / mode
            own = []
            for task in range(8):
                z = load(parent / 'scores' / f'task{task}.npz')
                v = load(parent / 'view_scores' / f'task{task}.npz')
                data[split, name, task] = z
                base = data[split, 'B0', task]
                assert all(np.array_equal(z[k], base[k]) for k in ('sample_ids', 'targets', 'class_indices'))
                assert all(np.array_equal(z[k], v[k]) for k in ('sample_ids', 'targets', 'class_indices'))
                assert len(z['targets']) == COUNTS[split][task] and (z['targets'] == 1).any(1).all()
                assert np.array_equal(z['probabilities'], v['fused_probabilities'])
                m = metrics(z['probabilities'], z['targets'])
                for k, value in m.items():
                    error = max(error, abs(value - reports[task][k]))
                    assert abs(value - reports[task][k]) < 1e-8
                assert np.allclose(ap(z['probabilities'], z['targets']), reports[task]['per_class_ap'], rtol=0, atol=1e-8)
                own.append(m)
                stages.append(dict(split=split, model=name, task=task, people=len(z['targets']), **m))
                vm = {key: float(ap(v[key + '_probabilities'], v['targets']).mean())
                      for key in ('full', 'person', 'face', 'fused')}
                diag = (result['view_diagnostics'][str(task)] if mode is None
                        else read(parent / f'task{task}.json')['views'])
                for key, value in vm.items():
                    error = max(error, abs(value - diag['metrics'][key]['mAP']))
                    assert abs(value - diag['metrics'][key]['mAP']) < 1e-8
                reliable = v['face_reliable'].astype(bool)
                views.append(dict(split=split, model=name, task=task,
                    **{k + '_mAP': value for k, value in vm.items()},
                    face_reliable_mAP=float(ap(v['face_probabilities'][reliable], v['targets'][reliable]).mean()),
                    fusion_minus_full=vm['fused'] - vm['full']))
                if name in ('PERSON', 'PERSON_OFF', 'DUAL', 'DUAL_FACE_OFF', 'DUAL_OFF'):
                    old_arm = 'PERSON_REFERENCE' if name.startswith('PERSON') else 'FACE_JOINT'
                    previous = args.previous_root / old_arm / ('b0_' + split) / mode
                    for group, current in (('scores', z), ('view_scores', v)):
                        old = load(previous / group / f'task{task}.npz')
                        assert current.keys() == old.keys()
                        assert all(np.array_equal(value, old[k]) for k, value in current.items())
                    references += 1
                if name.startswith('FROZEN_FACE'):
                    ref = load(args.root / 'REFERENCE_PERSON' / ('b0_' + split) / 'on/view_scores' / f'task{task}.npz')
                    assert all(np.array_equal(v[k], ref[k]) for k in ('sample_ids', 'targets', 'class_indices', 'face_reliable'))
                    for key in ('full', 'person'):
                        assert all(np.array_equal(v[key + suffix], ref[key + suffix])
                                   for suffix in ('_logits', '_probabilities'))
                    for key in ('face', 'fused'):
                        for suffix in ('_logits', '_probabilities'):
                            k = key + suffix
                            assert np.array_equal(v[k][~reliable], ref[k][~reliable])
                            if mode == 'face_off':
                                assert np.array_equal(v[k], ref[k])
                    protected += 1
                if name == 'PRIVATE_FACE_OFF':
                    on = load(folder / 'on/view_scores' / f'task{task}.npz')
                    for key in ('full', 'person'):
                        assert np.array_equal(v[key + '_logits'], on[key + '_logits'])
                    for key in ('face', 'fused'):
                        assert np.array_equal(v[key + '_logits'][~reliable], on[key + '_logits'][~reliable])
                    private_masks += 1
                if task == 7:
                    final_views[split, name] = v
                    for subset, mask in bucket_masks(z, v).items():
                        subgroups.append(dict(split=split, model=name, subset=subset, people=int(mask.sum()),
                                              **metrics(z['probabilities'][mask], z['targets'][mask])))
                    scores = ap(z['probabilities'], z['targets'])
                    for i, cls in enumerate(class_names):
                        classes.append(dict(split=split, model=name, emotion=cls, AP=float(scores[i])))
                    for t, (a, b) in enumerate(zip(STARTS[:-1], STARTS[1:])):
                        class_tasks.append(dict(split=split, model=name, class_task=t, AP=float(scores[a:b].mean())))
                    correlations.append(dict(split=split, model=name,
                        full_face_reliable_spearman=rank_correlation(v['full_logits'][reliable], v['face_logits'][reliable])))
            summary.append(dict(split=split, model=name, final_mAP=own[-1]['mAP'],
                average_mAP=float(np.mean([m['mAP'] for m in own])), final_cF1=own[-1]['cF1'], final_oF1=own[-1]['oF1']))
    for name, rows in (('summary', summary), ('stages', stages), ('views', views), ('subgroups', subgroups),
                       ('class_AP', classes), ('class_tasks_at_final', class_tasks), ('rank_correlations', correlations)):
        table(args.output / (name + '.csv'), rows)
    decompositions = []
    for split in ('val', 'test'):
        by = {r['model']: r for r in summary if r['split'] == split}
        for field in ('final_mAP', 'average_mAP'):
            shared = by['PRIVATE_OFF'][field] - by['DUAL_OFF'][field]
            person = ((by['PRIVATE_FACE_OFF'][field] - by['PRIVATE_OFF'][field])
                      - (by['DUAL_FACE_OFF'][field] - by['DUAL_OFF'][field]))
            face = ((by['PRIVATE'][field] - by['PRIVATE_FACE_OFF'][field])
                    - (by['DUAL'][field] - by['DUAL_FACE_OFF'][field]))
            net = by['PRIVATE'][field] - by['DUAL'][field]
            assert abs(shared + person + face - net) < 1e-10
            decompositions.append(dict(split=split, metric=field, trained_representation_head_change=shared,
                person_direct_contribution_change=person, face_direct_contribution_change=face, net_change=net))
    table(args.output / 'private_vs_shared_decomposition.csv', decompositions)
    effects = []; ranking = []; score_shifts = []
    for split in ('val', 'test'):
        p, f = final_views[split, 'PERSON'], final_views[split, 'FROZEN_FACE']
        reliable = f['face_reliable'].astype(bool)
        for subset, mask in bucket_masks(f, f).items():
            for key in ('face', 'fused'):
                change = f[key + '_logits'][mask] - p[key + '_logits'][mask]
                effects.append(dict(split=split, subset=subset, view=key, logit_abs_mean=float(abs(change).mean()),
                    logit_abs_max=float(abs(change).max()), changed_logits=int(np.count_nonzero(change)),
                    AP_change=float(ap(f[key+'_probabilities'][mask],f['targets'][mask]).mean()
                                    - ap(p[key+'_probabilities'][mask],p['targets'][mask]).mean())))
        for i, cls in enumerate(class_names):
            pos = f['targets'][:, i] == 1
            shift = f['fused_logits'][:, i] - p['fused_logits'][:, i]
            score_shifts.append(dict(split=split, emotion=cls,
                reliable_positive_logit_shift=float(shift[reliable & pos].mean()),
                reliable_negative_logit_shift=float(shift[reliable & ~pos].mean()),
                all_AP_change=float(ap(f['fused_probabilities'], f['targets'])[i]
                                    - ap(p['fused_probabilities'], p['targets'])[i]),
                reliable_AP_change=float(ap(f['fused_probabilities'][reliable], f['targets'][reliable])[i]
                                         - ap(p['fused_probabilities'][reliable], p['targets'][reliable])[i])))
            for pos_rel, neg_rel in ((True, True), (True, False), (False, True), (False, False)):
                pi = np.flatnonzero(pos & (reliable == pos_rel))
                ni = np.flatnonzero(~pos & (reliable == neg_rel))
                auc_delta = 0.; corrected = damaged = 0
                for start in range(0, len(pi), 128):
                    idx = pi[start:start+128]
                    old = p['fused_probabilities'][idx, i, None] - p['fused_probabilities'][ni, i][None, :]
                    new = f['fused_probabilities'][idx, i, None] - f['fused_probabilities'][ni, i][None, :]
                    corrected += int(((old <= 0) & (new > 0)).sum())
                    damaged += int(((old > 0) & (new <= 0)).sum())
                    auc_delta += float((new > 0).sum() + .5*(new == 0).sum()
                                       - (old > 0).sum() - .5*(old == 0).sum())
                pairs = len(pi) * len(ni)
                ranking.append(dict(split=split, emotion=cls, positive_reliable=pos_rel,
                    negative_reliable=neg_rel, pairs=pairs, corrected=corrected, damaged=damaged,
                    AUC_change_percentage_points=100*auc_delta/pairs if pairs else 0.))
    table(args.output / 'frozen_face_effects.csv', effects)
    table(args.output / 'frozen_face_cross_group_ranking.csv', ranking)
    table(args.output / 'frozen_face_class_score_shifts.csv', score_shifts)
    training = []; budgets = []; attention = []
    for name, folder in (('DUAL', args.previous_root / 'FACE_JOINT/train'),
                         ('PRIVATE', args.root / 'PRIVATE_SELECTORS/train'),
                         ('FROZEN_FACE', args.root / 'FROZEN_PERSON_FACE/train')):
        history = read(folder / 'training_history.json')
        for task, rows in history.items():
            row = dict(model=name, task=int(task), epochs=len(rows), updates=sum(r['optimizer_steps'] for r in rows),
                       skipped=sum(r['skipped_optimizer_steps'] for r in rows),
                       first_BCE=rows[0]['current_loss'], last_BCE=rows[-1]['current_loss'],
                       first_adapter_objective=rows[0]['adapter_objective_loss'], last_adapter_objective=rows[-1]['adapter_objective_loss'])
            for key in ('full', 'person', 'face'):
                vals = [r.get('strong_grad_selector_' + key + '_norm', 0.) for r in rows]
                row[key+'_selector_grad_mean'] = float(np.mean(vals))
            row['attention_grad_mean'] = float(np.mean([r.get('strong_grad_attention_norm',0.) for r in rows]))
            training.append(row)
        if name == 'FROZEN_FACE':
            d = read(folder / 'training_complete.json')
            budgets.append(dict(model=name, epochs=d['epochs'], updates=d['completed_optimizer_updates'],
                skipped=sum(r['skipped_optimizer_steps'] for rows in history.values() for r in rows),
                peak_allocated_mib=d['peak_allocated_mib'], elapsed_seconds=d['elapsed_seconds']))
        else:
            d = read(folder / 'seed_summary.json'); audit = read(folder.parent / 'training_verification.json')
            budgets.append(dict(model=name, epochs=d['completed_epochs'], updates=d['completed_optimizer_updates'],
                skipped=audit['skipped'], peak_allocated_mib=audit['peak_allocated_mib'], elapsed_seconds=d['elapsed_seconds']))
            for task, d in read(folder / 'cross_view_audit.json').items():
                for key, row in d['training_statistics'].items():
                    n = row['samples']
                    attention.append(dict(model=name, task=int(task), view='face' if key.endswith('/face') else 'person',
                        mean_residual_to_summary_norm=row['ratio_sum']/n, raw_ratio=row['raw_ratio_sum']/n,
                        attention_entropy=row['entropy_sum']/n))
    for name, rows in (('training_losses_gradients', training), ('training_budgets', budgets),
                       ('attention_training_summary', attention)):
        table(args.output / (name + '.csv'), rows)
    integrity = dict(status='verified',fused_predictions=len(stages),view_metrics=len(views)*4,
        maximum_metric_error=error,previous_reference_stage_replays=references,
        protected_face_stage_checks=protected,private_face_mask_stage_checks=private_masks,
        standard='samples_intersect_seen_classes',gpu_or_training=False)
    (args.output / 'integrity.json').write_text(json.dumps(integrity, indent=2)+'\n')
    print('INTEGRITY',json.dumps(integrity),flush=True)
    print('DECOMPOSITION',json.dumps(decompositions),flush=True)
    if args.bootstrap <= 0:
        return
    chosen=('PERSON','DUAL','DUAL_NATIVE','PRIVATE','PRIVATE_NATIVE','FROZEN_FACE')
    pairs=(('PRIVATE','DUAL'),('PRIVATE_NATIVE','DUAL_NATIVE'),('FROZEN_FACE','PERSON'))
    bootstrap={}
    for split in ('val','test'):
        images=np.asarray([str(s).rsplit('#person=',1)[0] for s in data[split,'B0',7]['sample_ids']])
        groups=np.unique(images)
        stage_groups=[]
        for task in range(8):
            names=np.asarray([str(s).rsplit('#person=',1)[0] for s in data[split,'B0',task]['sample_ids']])
            stage_groups.append([np.flatnonzero(names==g) for g in groups])
        gen=np.random.default_rng(20261026+(split=='test'))
        draws={kind:{a+'-'+b:[] for a,b in pairs} for kind in ('final','average')}
        for iteration in range(args.bootstrap):
            draw=gen.integers(0,len(groups),len(groups));values={name:[] for name in chosen}
            for task in range(8):
                ix=np.concatenate([stage_groups[task][i] for i in draw])
                y=data[split,'B0',task]['targets'][ix]
                for name in chosen:
                    values[name].append(float(ap(data[split,name,task]['probabilities'][ix],y).mean()))
            for a,b in pairs:
                draws['final'][a+'-'+b].append(values[a][-1]-values[b][-1])
                draws['average'][a+'-'+b].append(float(np.mean(values[a])-np.mean(values[b])))
            if (iteration+1)%100==0:print('BOOTSTRAP_PROGRESS',split,iteration+1,flush=True)
        bootstrap[split]=dict(repeats=args.bootstrap,unit='original_image',images=len(groups),
            stages_coupled=True,selection_adjusted=False,training_seed_variance_estimated=False,
            ci95={kind:{key:np.percentile(v,[2.5,97.5]).tolist() for key,v in result.items()}
                  for kind,result in draws.items()})
        print('BOOTSTRAP',split,json.dumps(bootstrap[split]),flush=True)
        (args.output/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','original-root','previous-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=500)
    main(p.parse_args())
