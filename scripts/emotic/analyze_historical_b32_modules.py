"""CPU-only, score-level analysis of the five seed1 summary-module controls."""
import argparse
import json
from pathlib import Path
import numpy as np

from analyze_historical_feature_fusion import ap, metrics, load, read, table, rank_correlation, COUNTS, STARTS

ARMS = ('BASE', 'ATTENTION', 'ATTENTION_BOUND', 'POSITION', 'POSITION_CALIBRATED')
VIEWS = ('full', 'person', 'face', 'fused')
FIELDS = ('sample_ids', 'targets', 'class_indices')


def main(args):
    root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=False)
    complete = read(root / 'completion.json')
    assert complete['status'] == 'complete' and not (root / 'failure.json').exists()
    assert read(root / 'baseline_reproduction.json')['status'] == 'exact'
    data, views, results, configs = {}, {}, {}, {}
    summary, stages, endpoints, training, module_stats, runtime = [], [], [], [], [], []
    classes = read(root / 'BASE/train/config.json')['class_order']
    initial, rng = set(), []
    max_error = 0.; native_equal = 0
    for arm in ARMS:
        cell = root / arm
        c = read(cell / 'train/config.json'); configs[arm] = c
        s = read(cell / 'train/seed_summary.json')
        h = read(cell / 'train/training_history.json')
        audit = read(cell / 'training_verification.json')
        assert c['git']['commit'] == complete['head'] and not c['git']['dirty']
        assert s['config'] == c and s['completed_epochs'] == 240 and s['completed_optimizer_updates'] == 13950
        assert c['seed'] == 1 and c['train_batch_size'] == 64 and c['adapter_residual_scale'] == .03
        assert c['view_auxiliary_loss_weight'] == .1 and c['parameter_group_loss_routing'] == 'adapter_asl'
        assert c['historical_b32_module_profile'] == arm
        assert all(audit[k] for k in ('old_parameters_unchanged', 'frozen_visual_unchanged',
            'recovery_equals_compact', 'global_sampler_rng', 'old_summary_modules_unchanged'))
        assert sum(r['skipped_optimizer_steps'] for rows in h.values() for r in rows) == 0
        initial.add(audit['initial_base_sha256']); rng.append(audit['cpu_rng_at_task_boundaries'])
        record = dict(arm=arm)
        for t in range(8):
            rows = h[str(t)][-5:]
            training.append(dict(arm=arm, task=t, **{k: float(np.mean([r.get(k, np.nan) for r in rows])) for k in
                ('current_loss', 'adapter_objective_loss', 'adapter_residual_ratio_full',
                 'adapter_residual_ratio_person', 'adapter_residual_ratio_face', 'adapter_grad_shared_norm',
                 'strong_grad_selectors_norm', 'strong_grad_prompts_norm', 'strong_grad_classifier_norm',
                 'strong_grad_adapter_norm', 'strong_grad_attention_norm', 'strong_grad_position_norm')}))
        for split in ('val', 'test'):
            folder = cell / ('historical_' + split); d = read(folder / 'result.json')
            results[arm, split] = d
            assert not d['full_cohort'] and d['checkpoint_tasks'] == list(range(8))
            for row in d['task_metrics']:
                t = row['task_id']; z = load(folder / 'scores' / f'task{t}.npz')
                data[arm, split, t] = z
                ref = data['BASE', split, t]
                assert all(np.array_equal(z[k], ref[k]) for k in FIELDS)
                assert len(z['targets']) == row['samples'] == COUNTS[split][t]
                assert (z['targets'] == 1).any(1).all()
                m = metrics(z['probabilities'], z['targets'])
                for k, v in m.items():
                    max_error = max(max_error, abs(v - row[k])); assert abs(v - row[k]) < 1e-8
                assert np.allclose(ap(z['probabilities'], z['targets']), row['per_class_ap'], rtol=0, atol=1e-8)
                stages.append(dict(arm=arm, split=split, task=t, **m,
                    delta_vs_base=m['mAP']-results['BASE', split]['task_metrics'][t]['mAP']))
                if split == 'test':
                    native = load(cell / 'train/test_scores' / f'task{t}.npz')
                    assert all(np.array_equal(z[k], native[k]) for k in (*FIELDS, 'logits', 'probabilities'))
                    native_equal += 1
                v = load(folder / 'view_scores' / f'task{t}.npz'); views[arm, split, t] = v
                assert all(np.array_equal(v[k], z[k]) for k in FIELDS)
                assert np.array_equal(v['fused_probabilities'], z['probabilities'])
                assert np.array_equal(v['face_reliable'], views['BASE', split, t]['face_reliable'])
                vm = {name: float(ap(v[name+'_probabilities'], v['targets']).mean()) for name in VIEWS}
                for name in VIEWS:
                    error = abs(vm[name] - d['view_diagnostics'][str(t)]['metrics'][name]['mAP'])
                    max_error = max(max_error, error); assert error < 1e-8
                endpoints.append(dict(arm=arm, split=split, task=t, **vm, fusion_minus_own_full=vm['fused']-vm['full']))
            for k in ('final_mAP', 'average_mAP', 'final_cF1', 'final_oF1'):
                record[split+'_'+k] = d['summary'][k]
            field = 'cross_view_statistics' if arm.startswith('ATTENTION') else 'spatial_summary_statistics'
            if arm != 'BASE':
                for key, values in d[field]['7'].items():
                    t, layer = (int(x.removeprefix(prefix)) for x, prefix in zip(key.split('/'), ('task', 'layer')))
                    stat = {k: float(v) for k, v in values.items() if k.endswith('_mean') and not isinstance(v, list)}
                    for k in ('query_std_across_images', 'attention_std_across_images'):
                        if k in values:
                            stat[k+'_mean'] = float(np.mean(values[k])); stat[k+'_max'] = float(np.max(values[k]))
                    module_stats.append(dict(arm=arm, split=split, task=t, layer=layer, **stat))
        summary.append(record)
        if arm != 'BASE':
            off_folder = cell / 'runtime_off'; off = read(off_folder / 'result.json')
            assert off['checkpoint_tasks'] == [0, 2, 7] and off['split'] == 'val' and not off['full_cohort']
            for row in off['task_metrics']:
                t = row['task_id']; z = load(off_folder / 'scores' / f'task{t}.npz')
                v = load(off_folder / 'view_scores' / f'task{t}.npz')
                data[arm, 'off', t] = z
                ref = data[arm, 'val', t]; on_views = views[arm, 'val', t]
                assert all(np.array_equal(z[k], ref[k]) for k in FIELDS)
                m = metrics(z['probabilities'], z['targets'])
                for k, value in m.items():
                    max_error = max(max_error, abs(value-row[k])); assert abs(value-row[k]) < 1e-8
                on = results[arm, 'val']['task_metrics'][t]['mAP']; base = results['BASE', 'val']['task_metrics'][t]['mAP']
                view_m = {name: float(ap(v[name+'_probabilities'], v['targets']).mean()) for name in VIEWS}
                runtime.append(dict(arm=arm, task=t, base_mAP=base, on_mAP=on, off_mAP=m['mAP'],
                    on_minus_off=on-m['mAP'], off_minus_base=m['mAP']-base, on_minus_base=on-base,
                    mean_abs_logit_change=float(np.abs(z['logits'].astype(float)-ref['logits']).mean()),
                    **{name+'_mAP_change':float(ap(on_views[name+'_probabilities'],v['targets']).mean())-view_m[name] for name in VIEWS},
                    **{name+'_logits_exact':bool(np.array_equal(v[name+'_logits'],on_views[name+'_logits'])) for name in VIEWS}))
    assert len(initial) == 1 and all(x == rng[0] for x in rng)
    subsets, correlations, contrasts, class_rows, task_rows, old_paths, within = [], [], [], [], [], [], []
    for split in ('val', 'test'):
        ref = data['BASE', split, 7]; y = ref['targets']
        image_names = np.asarray([str(s).rsplit('#person=', 1)[0] for s in ref['sample_ids']])
        names, inverse, counts = np.unique(image_names, return_inverse=True, return_counts=True)
        reliable = views['BASE', split, 7]['face_reliable']
        masks = dict(all_people=np.ones(len(y), bool), reliable_face=reliable, unreliable_face=~reliable,
            multiple_people=counts[inverse] > 1, single_person=counts[inverse] == 1)
        groups = [np.where(inverse == i)[0] for i in range(len(names)) if counts[i] > 1]
        for arm in ARMS:
            v = views[arm, split, 7]
            for subset, mask in masks.items():
                vm = {name: float(ap(v[name+'_probabilities'][mask],y[mask]).mean()) for name in VIEWS}
                subsets.append(dict(arm=arm, split=split, subset=subset, people=int(mask.sum()), images=len(np.unique(image_names[mask])),
                    **vm, delta_vs_base=vm['fused']-float(ap(data['BASE',split,7]['probabilities'][mask],y[mask]).mean())))
            correlations.append(dict(arm=arm, split=split,
                full_person=rank_correlation(v['full_logits'],v['person_logits']),
                full_face=rank_correlation(v['full_logits'][reliable],v['face_logits'][reliable]),
                person_face=rank_correlation(v['person_logits'][reliable],v['face_logits'][reliable])))
            for name in VIEWS:
                spans=[]
                for ix in groups:
                    if name=='face':ix=ix[reliable[ix]]
                    if len(ix)>1:
                        x=v[name+'_logits'][ix].astype(float);spans.append(float((x.max(0)-x.min(0)).mean()))
                within.append(dict(arm=arm,split=split,view=name,multiple_images=len(groups),
                    label_discordant_images=sum(bool(np.any(y[ix].max(0)!=y[ix].min(0))) for ix in groups),
                    compared_images=len(spans),median_mean_class_logit_span=float(np.median(spans)),p90_mean_class_logit_span=float(np.percentile(spans,90))))
            ids={str(s):i for i,s in enumerate(ref['sample_ids'])}
            final = data[arm,split,7]
            for t in (0,2,6):
                z=data[arm,split,t];ix=np.array([ids[str(s)] for s in z['sample_ids']]);n=z['targets'].shape[1]
                assert np.array_equal(final['targets'][ix,:n],z['targets'])
                delta=np.abs(final['logits'][ix,:n].astype(float)-z['logits'])
                before=float(ap(z['probabilities'],z['targets']).mean())
                after=float(ap(final['probabilities'][ix,:n],z['targets']).mean())
                old_paths.append(dict(arm=arm,split=split,old_task=t,max_abs_logit_change=float(delta.max()),mean_abs_logit_change=float(delta.mean()),mAP_change=after-before))
            if arm=='BASE':continue
            a,b=results[arm,split],results['BASE',split]
            delta=np.asarray(a['task_metrics'][7]['per_class_ap'])-np.asarray(b['task_metrics'][7]['per_class_ap'])
            contrasts.append(dict(arm=arm,split=split,final_delta=a['summary']['final_mAP']-b['summary']['final_mAP'],
                average_delta=a['summary']['average_mAP']-b['summary']['average_mAP'],positive_stages=sum(x['mAP']>z['mAP'] for x,z in zip(a['task_metrics'],b['task_metrics'])),positive_classes=int((delta>0).sum())))
            for i,name in enumerate(classes):
                class_rows.append(dict(arm=arm,split=split,class_name=name,AP=a['task_metrics'][7]['per_class_ap'][i],base_AP=b['task_metrics'][7]['per_class_ap'][i],delta=float(delta[i]),positives=int(y[:,i].sum())))
            for t,(start,end) in enumerate(zip(STARTS[:-1],STARTS[1:])):
                task_rows.append(dict(arm=arm,split=split,task=t,classes='/'.join(classes[start:end]),mean_AP_delta=float(delta[start:end].mean()),contribution_to_final_mAP=float(delta[start:end].sum()/26)))
    for name,rows in (('summary',summary),('stage_metrics',stages),('endpoints',endpoints),('training',training),
        ('runtime_off',runtime),('subgroups',subsets),('correlations',correlations),('contrasts',contrasts),
        ('class_deltas',class_rows),('task_class_deltas',task_rows),('old_path_replay',old_paths),('within_image',within)):
        table(out/(name+'.csv'),rows)
    # Attention and geometry statistics have intentionally different schemas.
    for prefix in ('ATTENTION','POSITION'):
        rows=[r for r in module_stats if r['arm'].startswith(prefix)]
        table(out/(prefix.lower()+'_statistics.csv'),rows)
    integrity=dict(status='verified',models=5,stage_scores=80,view_scores=80,runtime_off_scores=12,native_test_exact=native_equal,
        same_people_labels_classes=True,initialization_and_sampler_rng_equal=True,maximum_metric_error=max_error,
        historical_seed1_exact=True,training_or_model_forward=False,test_used_for_parameter_search=True)
    (out/'analysis_integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    print('AUDIT',json.dumps(integrity),flush=True)
    bootstrap={}
    for split in ('val','test'):
        z=data['BASE',split,7]; y=z['targets'];groups={}
        for i,s in enumerate(z['sample_ids']):groups.setdefault(str(s).rsplit('#person=',1)[0],[]).append(i)
        groups=list(groups.values());gen=np.random.default_rng(20261025+(split=='test'))
        probabilities={a:data[a,split,7]['probabilities'] for a in ARMS}
        if split=='val':probabilities.update({a+'_OFF':data[a,'off',7]['probabilities'] for a in ARMS[1:]})
        samples={}
        for repeat in range(args.bootstrap):
            ix=np.concatenate([groups[i] for i in gen.integers(0,len(groups),len(groups))])
            values={a:float(ap(p[ix],y[ix]).mean()) for a,p in probabilities.items()}
            for a in ARMS[1:]:
                samples.setdefault(a+'-BASE',[]).append(values[a]-values['BASE'])
                if split=='val':samples.setdefault(a+'-OFF',[]).append(values[a]-values[a+'_OFF'])
        bootstrap[split]=dict(repeats=args.bootstrap,unit='original_image',images=len(groups),people=len(y),
            selection_adjusted=False,future_training_seed_variance_estimated=False,
            contrasts={k:dict(ci95_pp=np.percentile(v,[2.5,97.5]).tolist()) for k,v in samples.items()})
        print('BOOTSTRAP',split,json.dumps(bootstrap[split]),flush=True)
    (out/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=500)
    main(p.parse_args())
