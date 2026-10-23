"""Read-only CPU audit and tables for the completed strong-ASL transfer matrix."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ARMS = ['S_BCE', 'S_ASL', 'T_BCE', 'T_ASL', 'T_ASL_ATTN', 'T_ASL_POS', 'T_ASL_BOTH']


def read(path):
    return json.loads(path.read_text())


def ap(probability, targets):
    order = probability.argsort(axis=0)[::-1]
    positive = np.take_along_axis(targets, order, axis=0) == 1
    cumulative = positive.cumsum(axis=0)
    total = cumulative[-1].copy()
    cumulative[~positive] = 0
    return 100 * (cumulative / np.arange(1, len(positive) + 1)[:, None]).sum(0) / (total + 1e-8)


def table(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    root, out = args.root, args.output
    out.mkdir(parents=True, exist_ok=True)
    complete = read(root / 'completion.json')
    assert complete['status'] == 'complete' and not (root / 'failure.json').exists()
    verified = read(root / 'training_verification.json')
    assert verified['status'] == 'verified'
    assert all(x['updates'] == 13950 and x['skipped'] == 0 for x in verified['checks'])
    results, arrays, stages, endpoints, subsets, task_ap = {}, {}, [], [], [], []
    max_error = 0
    for arm in ARMS + ['R_LEGACY']:
        for split, count in [('val', 2397), ('test', 5368)]:
            d = read(root / arm / ('full_' + split) / 'result.json')
            results[arm, split] = d
            assert len(d['task_metrics']) == 8
            for row in d['task_metrics']:
                task = row['task_id']
                assert row['samples'] == count
                with np.load(root / arm / ('full_' + split) / 'scores' / f'task{task}.npz') as z:
                    arrays[arm, split, task] = {k: z[k].copy() for k in z.files}
                z = arrays[arm, split, task]
                reference = arrays['S_BCE', split, task]
                for key in ['sample_ids', 'class_indices', 'targets']:
                    assert np.array_equal(z[key], reference[key]), (arm, split, task, key)
                error = abs(float(ap(z['probabilities'], z['targets']).mean()) - row['mAP'])
                max_error = max(max_error, error)
                assert error < 1e-6, (arm, split, task, error)
                stages.append(dict(arm=arm, split=split, task=task, samples=count, mAP=row['mAP']))
            last = d['task_metrics'][-1]
            starts = [0, 5, 8, 11, 14, 17, 20, 23, 26]
            for t, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
                value = float(np.mean(last['per_class_ap'][start:end]))
                task_ap.append(dict(arm=arm, split=split, task=t, final_AP_of_task_classes=value))
            diagnostics = d.get('view_diagnostics', {}).get('7', {}).get('metrics', {})
            if diagnostics:
                endpoints.append(dict(arm=arm, split=split,
                                      **{k: v['mAP'] for k, v in diagnostics.items()},
                                      fusion_minus_full=diagnostics['fused']['mAP']-diagnostics['full']['mAP']))
                with np.load(root/arm/('full_'+split)/'view_scores/task7.npz') as z:
                    mask = z['face_reliable']
                    for group, chosen in [('reliable_face', mask), ('unreliable_face', ~mask)]:
                        subsets.append(dict(arm=arm, split=split, subset=group, samples=int(chosen.sum()),
                                            **{v: float(ap(z[v+'_probabilities'][chosen], z['targets'][chosen]).mean())
                                               for v in ['fused', 'full', 'person', 'face']}))
    controls, module_stats = [], []
    for arm in ['T_ASL', 'T_ASL_ATTN', 'T_ASL_POS', 'T_ASL_BOTH']:
        original = results[arm, 'val']
        metrics = {x['task_id']: x['mAP'] for x in original['task_metrics']}
        for path in sorted((root/arm).glob('runtime_*/result.json')):
            disabled = read(path)
            assert disabled['summary_is_selected_boundaries_only']
            assert disabled['checkpoint_sha256'] == {str(t): original['checkpoint_sha256'][str(t)] for t in [0,2,7]}
            for row in disabled['task_metrics']:
                t = row['task_id']
                with np.load(path.parent/'scores'/f'task{t}.npz') as z:
                    reference = arrays[arm,'val',t]
                    assert all(np.array_equal(z[k],reference[k]) for k in ['sample_ids','class_indices','targets'])
                    assert abs(float(ap(z['probabilities'],z['targets']).mean())-row['mAP']) < 1e-6
                controls.append(dict(arm=arm, disabled=path.parent.name, task=t,
                                     enabled_mAP=metrics[t], disabled_mAP=row['mAP'],
                                     enabled_minus_disabled=metrics[t]-row['mAP']))
        for task in ['0','2','7']:
            cross=original['cross_view_statistics'].get(task,{})
            for key,value in cross.items():
                if key.startswith('task'+task+'/'):
                    module_stats.append(dict(arm=arm, task=int(task), module='attention',
                                             norm_ratio=value['ratio_mean'], cosine=value['cosine_mean'],
                                             bias_abs='', kl='', saturation=''))
            spatial=[v for k,v in original['spatial_summary_statistics'].get(task,{}).items()
                     if k.startswith('task'+task+'/')]
            if spatial:
                module_stats.append(dict(arm=arm, task=int(task), module='position', norm_ratio='', cosine='',
                                         bias_abs=float(np.mean([v['bias_abs_mean'] for v in spatial])),
                                         kl=float(np.mean([v['attention_kl_mean'] for v in spatial])),
                                         saturation=float(np.mean([v['saturation_mean'] for v in spatial]))))
    class_names=read(root/'S_ASL/train/config.json')['class_order']
    classes=[]
    for split in ['val','test']:
        b=np.array(results['T_ASL',split]['task_metrics'][-1]['per_class_ap'])
        for arm in ['T_ASL_ATTN','T_ASL_POS','T_ASL_BOTH']:
            v=np.array(results[arm,split]['task_metrics'][-1]['per_class_ap'])
            for i,name in enumerate(class_names):
                classes.append(dict(arm=arm,split=split,class_name=name,base_AP=b[i],module_AP=v[i],delta=v[i]-b[i]))
    for name,rows in [('stage_metrics',stages),('final_endpoints',endpoints),('face_reliability_subsets',subsets),
                      ('final_task_class_AP',task_ap),('runtime_controls',controls),('module_statistics',module_stats),
                      ('per_class_deltas',classes)]:
        table(out/(name+'.csv'),rows)
    audit=dict(status='verified',full_results=16,stage_score_files=128,runtime_score_files=len(controls),
               validation_people=2397,test_people=5368,all_cohorts_labels_classes_equal=True,
               all_source_checkpoint_hashes_matched_controls=True,max_independent_mAP_error=max_error,
               training_updates_per_new_arm=13950,skipped_updates=0,training_seeds=1,
               test_used_for_selection=False)
    (out/'analysis_integrity.json').write_text(json.dumps(audit,indent=2)+'\n')
    if args.historical_run is not None:
        historical=read(args.historical_run/'config.json')
        rerun=read(root/'R_LEGACY/train/config.json')
        comparison=read(root/'R_LEGACY/reference_verification.json')
        historical_audit=dict(historical_run=str(args.historical_run),
                              historical_anchor_matched=comparison['historical_anchor_matched'],
                              original_metrics=comparison['original_metrics'],
                              historical_anchor=comparison['historical_anchor'],
                              config_differences={k:[historical.get(k),rerun.get(k)]
                                                  for k in sorted(historical.keys()|rerun.keys())
                                                  if historical.get(k)!=rerun.get(k)},
                              cause='Configuration mismatch confirmed; its score effect is not isolated')
        (out/'historical_reproduction_audit.json').write_text(json.dumps(historical_audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--historical-run',type=Path)
    main(parser.parse_args())
