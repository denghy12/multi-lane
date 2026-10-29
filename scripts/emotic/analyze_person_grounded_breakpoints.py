"""Independent NumPy checks of query, spatial binding and classifier ranking."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_historical_feature_fusion import ap, metrics, load, read, table, STARTS, COUNTS

ARMS=tuple('PERSON_'+c+'_'+op for c in ('CONST','CONTENT','GROUND') for op in ('POOL','ATTN'))
TASKS=(0,2,6,7)
MODES=('native','early_swap','zero_condition','no_subject','no_face','late_query_swap')


def auc(z,y):
    pos=z[y==1];neg=np.sort(z[y==0])
    if not len(pos) or not len(neg):return float('nan')
    return float(100*np.mean((np.searchsorted(neg,pos,'left')+np.searchsorted(neg,pos,'right'))*.5)/len(neg))


def adjacent_pairs(base,native,projection,y):
    """Two adjacent baseline negatives per positive; no model-score selection."""
    positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
    if not len(positive) or not len(negative):return dict(pairs=0,correction_margin_mean=None,
        favorable_fraction=None,repaired_fraction=None,damaged_fraction=None)
    negative=negative[np.argsort(base[negative])];where=np.searchsorted(base[negative],base[positive])
    ps=[];ns=[]
    for offset in (-1,0):
        take=where+offset;valid=(take>=0)&(take<len(negative))
        ps.extend(positive[valid]);ns.extend(negative[take[valid]])
    ps,ns=np.asarray(ps),np.asarray(ns);old=base[ps]-base[ns];new=native[ps]-native[ns]
    correction=projection[ps]-projection[ns]
    return dict(pairs=len(ps),correction_margin_mean=float(correction.mean()),
        favorable_fraction=float((correction>0).mean()),repaired_fraction=float(((old<=0)&(new>0)).mean()),
        damaged_fraction=float(((old>0)&(new<=0)).mean()))


def cluster_interval(values,ids,repeats):
    if not len(values):return None,None
    groups=np.asarray([str(s).rsplit('#person=',1)[0] for s in ids])
    names,inverse=np.unique(groups,return_inverse=True)
    sums=np.bincount(inverse,weights=values);counts=np.bincount(inverse)
    rng=np.random.default_rng(20261029);draw=rng.integers(0,len(names),(repeats,len(names)))
    means=sums[draw].sum(1)/counts[draw].sum(1)
    return float(np.quantile(means,.025)),float(np.quantile(means,.975))


def strata(z,lane):
    valid=z['eligible'].astype(bool)&z['own_valid'].astype(bool)&z['other_valid'].astype(bool)
    high=z['high_visibility'].astype(bool)&valid
    ratio=z['own_area']/np.maximum(z['other_area'],1e-12)
    matched=(ratio>=2/3)&(ratio<=1.5);low=z['bbox_iou']<=.1
    exclusive=(z['own_exclusive_area']>=1/196)&(z['other_exclusive_area']>=1/196)
    diff=z['different_seen_labels'][:,STARTS[lane]:STARTS[lane+1]].any(-1)
    return dict(all_visible=valid,both_high_visibility=high,different_lane_labels=valid&diff,
        high_visibility_different_labels=high&diff,low_overlap=valid&low,size_matched=valid&matched,
        high_low_overlap_size_matched=high&low&matched,exclusive_size_matched=valid&exclusive&matched,
        high_exclusive_size_matched=high&exclusive&matched,
        small_own_box=valid&(z['own_area']<.1),medium_own_box=valid&(z['own_area']>=.1)&(z['own_area']<.3),
        large_own_box=valid&(z['own_area']>=.3))


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    complete=read(args.root/'completion.json')
    assert complete['status']=='complete' and complete['training_updates']==0
    rows={k:[] for k in ('full_metrics','class_ranking','trace','spatial','paired_metrics','intervals','resources')}
    hashes={};cases=0;negative_controls=0;head_error=0.
    for arm in ARMS:
        root=args.root/arm;result=read(root/'result.json');protocol=result['protocol']
        assert result['status']=='complete' and result['all_parameters_and_checkpoints_unchanged']
        assert protocol['training_updates']==0 and not protocol['parameter_assembly'] and not protocol['teacher_used']
        assert protocol['tasks']==list(TASKS) and protocol['splits']==['val','test']
        names=protocol['class_order']
        rows['resources'].append(dict(arm=arm,elapsed_seconds=result['elapsed_seconds'],peak_mib=result['peak_allocated_mib']))
        for case in result['cases']:
            task,split=case['task'],case['split'];folder=root/split/f'task{task}'
            assert case['parameters_unchanged'] and case['training_updates']==0
            assert case['full']['historical_native_and_three_views_exact'] and case['full']['historical_off_exact']
            assert case['paired']['native_early_and_late_replayed_exact'] and case['paired']['same_image_and_full_crop']
            for p in folder.glob('*.npz'):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            full,pair=load(folder/'full.npz'),load(folder/'pairs.npz')
            assert len(full['targets'])==COUNTS[split][task] and sum(full['batch_lengths'])==len(full['targets'])
            assert len(set(full['sample_ids']))==len(full['targets'])
            assert sum(pair['batch_lengths'])==len(pair['targets'])
            lookup={s:i for i,s in enumerate(full['sample_ids'])};positions=np.asarray([lookup[s] for s in pair['sample_ids']])
            assert np.array_equal(pair['targets'],full['targets'][positions])
            assert all(str(a).rsplit('#person=',1)[0]==str(b).rsplit('#person=',1)[0]
                       for a,b in zip(pair['sample_ids'],pair['donor_ids']))
            assert pair['same_full_crop'].all()
            pair_lookup={s:i for i,s in enumerate(pair['sample_ids'])}
            donor_positions=np.asarray([pair_lookup[s] for s in pair['donor_ids']])
            assert np.array_equal(pair['different_seen_labels'],pair['targets']!=pair['targets'][donor_positions])
            if '_CONST_' in arm:
                assert np.array_equal(pair['native_logits'],pair['early_swap_logits'])
                for k in ('shift','query','alpha','summary','hidden','delta'):
                    assert not pair[k+'_diff_rms'].any()
                negative_controls+=1
            for mode in ('native','base'):
                m=metrics(full[mode+'_probabilities'],full['targets'])
                rows['full_metrics'].append(dict(arm=arm,split=split,stage=task,mode=mode,samples=len(full['targets']),**m))
            on_ap,off_ap=ap(full['native_probabilities'],full['targets']),ap(full['base_probabilities'],full['targets'])
            for c,name in enumerate(names[:full['targets'].shape[1]]):
                y=full['targets'][:,c];w=full['w_delta_fp64'][:,c];old=full['base_logits'][:,c];new=full['native_logits'][:,c]
                rows['class_ranking'].append(dict(arm=arm,split=split,stage=task,class_index=c,class_name=name,
                    owner_task=int(np.searchsorted(STARTS[1:],c,side='right')),positives=int(y.sum()),
                    AP=float(on_ap[c]),base_AP=float(off_ap[c]),AP_delta=float(on_ap[c]-off_ap[c]),
                    AUC=auc(new,y),base_AUC=auc(old,y),AUC_delta=auc(new,y)-auc(old,y),
                    positive_w_delta=float(w[y==1].mean()) if (y==1).any() else None,
                    negative_w_delta=float(w[y==0].mean()) if (y==0).any() else None,
                    signed_projection_separation=float(w[y==1].mean()-w[y==0].mean()) if (y==1).any() and (y==0).any() else None,
                    **adjacent_pairs(old,new,w,y)))
            diagnostic_keys=[k for k,v in pair.items() if v.ndim==2 and v.shape[1]==task+1 and (
                k.endswith(('_rms','_relative_diff','_cosine')) or k in ('alpha_total_variation',
                'condition_saturated_gt2','condition_saturated_gt3','condition_tanh_derivative',
                'binding_mass_gain','binding_density_gain') or k.startswith(('native_own_','native_other_','swapped_own_','swapped_other_')))]
            for lane in range(task+1):
                groups=strata(pair,lane);start,stop=STARTS[lane:lane+2]
                for group,mask in groups.items():
                    if not mask.any():continue
                    common=dict(arm=arm,split=split,stage=task,lane=lane,subgroup=group,samples=int(mask.sum()),
                        images=len({str(s).rsplit('#person=',1)[0] for s in pair['sample_ids'][mask]}))
                    t=dict(common);spatial=dict(common)
                    for key in diagnostic_keys:
                        v=pair[key][mask,lane]
                        destination=spatial if key.startswith(('native_own_','native_other_','swapped_own_','swapped_other_','binding_')) else t
                        destination[key]=float(v.mean());destination[key+'_p95']=float(np.quantile(v,.95))
                    spatial['own_area']=float(pair['own_area'][mask].mean());spatial['other_area']=float(pair['other_area'][mask].mean())
                    spatial['bbox_iou']=float(pair['bbox_iou'][mask].mean())
                    spatial['own_minus_other_density']=float((pair['native_own_density']-pair['native_other_density'])[mask,lane].mean())
                    rows['trace'].append(t);rows['spatial'].append(spatial)
                    for mode in MODES:
                        p=pair[mode+'_probabilities'][mask,start:stop];y=pair['targets'][mask,start:stop]
                        n=pair['native_probabilities'][mask,start:stop];m=metrics(p,y);nm=metrics(n,y)
                        different=pair['different_seen_labels'][mask,start:stop]
                        change=pair['native_logits'][mask,start:stop]-pair[mode+'_logits'][mask,start:stop]
                        signed=(2*y-1)*change
                        rows['paired_metrics'].append(dict(common,mode=mode,owned_class_mAP=m['mAP'],native_owned_mAP=nm['mAP'],
                            native_minus_intervention=nm['mAP']-m['mAP'],mean_abs_logit_change=float(np.abs(change).mean()),
                            different_class_cells=int(different.sum()),signed_margin_different_cells=float(signed[different].mean()) if different.any() else None))
                    if lane==task and group in ('all_visible','high_exclusive_size_matched'):
                        for label,value in [('binding_density_gain',pair['binding_density_gain'][:,lane]),
                            ('own_minus_other_density',(pair['native_own_density']-pair['native_other_density'])[:,lane])]:
                            low,high=cluster_interval(value[mask],pair['sample_ids'][mask],args.bootstrap)
                            rows['intervals'].append(dict(common,statistic=label,mean=float(value[mask].mean()),lower95=low,upper95=high,
                                repeats=args.bootstrap,unit='original_image_cluster',multiple_comparisons_adjusted=False))
            cases+=1
        print('BREAKPOINT_ANALYSIS_VERIFIED',arm,flush=True)
    assert cases==48 and negative_controls==16
    for name,values in rows.items():table(args.output/(name+'.csv'),values)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    proof=dict(status='verified',cases=cases,constant_negative_control_cases=negative_controls,
        original_files_unchanged=True,training_updates=0,source_files=hashes,
        original_predictions_and_probability_replay=True,analytical_fp64_not_official_score=True,
        all_parameters_and_source_checkpoints_unchanged=True,bootstrap=args.bootstrap,
        training_seed_variance_measured=False,model_execution=False)
    (args.output/'integrity.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('BREAKPOINT_ANALYSIS_COMPLETE',cases,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=500)
    main(p.parse_args())
