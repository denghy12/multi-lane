"""Recompute legacy seen-positive mAP from immutable full-cohort predictions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def average_precision(probabilities, targets):
    order=probabilities.argsort(axis=0)[::-1]
    positives=np.take_along_axis(targets,order,axis=0)==1
    cumulative=positives.cumsum(axis=0)
    counts=cumulative[-1].copy()
    cumulative[~positives]=0
    return 100*(cumulative/np.arange(1,len(positives)+1)[:,None]).sum(0)/(counts+1e-8)


def main(args):
    root,out=args.root,args.output
    out.mkdir(parents=True,exist_ok=True)
    arms=['S_BCE','S_ASL','T_BCE','T_ASL','T_ASL_ATTN','T_ASL_POS','T_ASL_BOTH','R_LEGACY']
    stage_rows=[];summary=[];reference={};archive_checks=[]
    for arm in arms:
        metrics={}
        for split in ['val','test']:
            full=[];historical=[];counts=[]
            for t in range(8):
                path=root/arm/('full_'+split)/'scores'/f'task{t}.npz'
                with np.load(path) as z:
                    y=z['targets'];p=z['probabilities'];ids=z['sample_ids']
                    keep=(y==1).any(axis=1)
                    assert keep.any()
                    if (split,t) not in reference:reference[split,t]=(ids[keep].copy(),y[keep].copy())
                    expected_ids,expected_targets=reference[split,t]
                    assert np.array_equal(ids[keep],expected_ids) and np.array_equal(y[keep],expected_targets)
                    if split=='test' and args.historical_run is not None:
                        with np.load(args.historical_run/'test_scores'/f'task{t}.npz') as old:
                            assert np.array_equal(ids[keep],old['sample_ids'])
                            assert np.array_equal(y[keep],old['targets'])
                            assert np.array_equal(z['class_indices'],old['class_indices'])
                        archive_checks.append((arm,t))
                    m=float(average_precision(p,y).mean());old_m=float(average_precision(p[keep],y[keep]).mean())
                    full.append(m);historical.append(old_m);counts.append(int(keep.sum()))
                    stage_rows.append(dict(arm=arm,split=split,task=t,full_people=len(y),historical_people=int(keep.sum()),
                                           full_mAP=m,historical_mAP=old_m,difference=old_m-m))
            assert counts[-1]==len(y)
            metrics[split+'_final']=historical[-1]
            metrics[split+'_average_historical']=float(np.mean(historical))
            metrics[split+'_average_full']=float(np.mean(full))
        summary.append(dict(arm=arm,**metrics))
    controls=[]
    for arm in arms:
        for path in (root/arm).glob('runtime_*/result.json'):
            d=json.loads(path.read_text())
            for row in d['task_metrics']:
                t=row['task_id']
                with np.load(path.parent/'scores'/f'task{t}.npz') as z:
                    keep=(z['targets']==1).any(axis=1)
                    off=float(average_precision(z['probabilities'][keep],z['targets'][keep]).mean())
                normal=next(r['historical_mAP'] for r in stage_rows if r['arm']==arm and r['split']=='val' and r['task']==t)
                controls.append(dict(arm=arm,disabled=path.parent.name,task=t,enabled_mAP=normal,disabled_mAP=off,
                                     enabled_minus_disabled=normal-off))
    for name,rows in [('summary_historical',summary),('stage_historical',stage_rows),('runtime_historical',controls)]:
        with (out/(name+'.csv')).open('w') as stream:
            w=csv.DictWriter(stream,list(rows[0]));w.writeheader();w.writerows(rows)
    audit=dict(status='verified',protocol='samples_intersect_seen_classes',selection='any current seen-class target == 1',
               averaging='arithmetic mean of eight cumulative seen-class mAP values',
               models=len(arms),same_people_across_models=True,final_cohort_unchanged=True,
               archived_test_cohort_comparisons=len(archive_checks),predictions_reused=True,
               training_or_model_forward=False,
               people={s:[len(reference[s,t][0]) for t in range(8)] for s in ['val','test']})
    (out/'historical_cohort_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
    for row in summary:print(row)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--historical-run',type=Path)
    main(p.parse_args())
