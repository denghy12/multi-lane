"""Independent NumPy metric and source-integrity audit of the real R5 forward."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_historical_feature_fusion import COUNTS, ap, load, metrics, read, table

CONDITIONS=('A','B','R5','LOGIT_MEAN')
VIEWS=('fused','full','person','face')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    assert read(args.root/'completion.json')['status']=='complete'
    args.output.mkdir(parents=True,exist_ok=False)
    stages=[];summary=[];views=[];groups=[];checks=[];hashes={};metric_error=0.
    for split in ('val','test'):
        result=read(args.root/'evaluation'/split/'result.json')
        assert result['tasks']==list(range(8)) and result['training_updates']==0
        for task in range(8):
            ref={s:load(args.reference/name/('b0_'+split)/'on/view_scores'/f'task{task}.npz')
                 for s,name in (('A','REFERENCE_PERSON'),('B','REFERENCE_DUAL'))}
            arrays={}
            for arm in CONDITIONS:
                parent=args.root/'evaluation'/split/arm
                path=parent/'view_scores'/f'task{task}.npz'; hashes[str(path)]=sha(path)
                z=load(path); arrays[arm]=z
                score=load(parent/'scores'/f'task{task}.npz')
                assert len(z['targets'])==COUNTS[split][task] and (z['targets']==1).any(1).all()
                for key in ('sample_ids','targets','class_indices','face_reliable','batch_lengths'):
                    assert np.array_equal(z[key],ref['A'][key])
                assert np.array_equal(score['probabilities'],z['fused_probabilities'])
                assert np.array_equal(score['logits'],z['fused_logits'])
                m=metrics(z['fused_probabilities'],z['targets'])
                expected=result['metrics'][arm][task]
                for key,value in m.items():
                    metric_error=max(metric_error,abs(value-expected[key]))
                    assert abs(value-expected[key])<1e-8
                assert np.allclose(ap(z['fused_probabilities'],z['targets']),expected['per_class_ap'],rtol=0,atol=1e-8)
                stages.append(dict(split=split,condition=arm,task=task,people=len(z['targets']),**m))
                if arm in ('A','B'):
                    for view in VIEWS:
                        for suffix in ('_logits','_probabilities'):
                            assert np.array_equal(z[view+suffix],ref[arm][view+suffix])
                vm={v:float(ap(z[v+'_probabilities'],z['targets']).mean()) for v in VIEWS}
                reliable=z['face_reliable'].astype(bool)
                views.append(dict(split=split,condition=arm,task=task,**{v+'_mAP':m for v,m in vm.items()},
                    reliable_face_mAP=float(ap(z['face_probabilities'][reliable],z['targets'][reliable]).mean())))
                if task==7:
                    original=np.asarray([str(s).rsplit('#person=',1)[0] for s in z['sample_ids']])
                    names,counts=np.unique(original,return_counts=True)
                    multiple=np.isin(original,names[counts>1])
                    for name,mask in (('reliable_face',reliable),('unreliable_face',~reliable),
                                      ('multiple_eligible_people',multiple),('single_eligible_person',~multiple)):
                        groups.append(dict(split=split,condition=arm,subset=name,people=int(mask.sum()),
                            **metrics(z['fused_probabilities'][mask],z['targets'][mask])))
            for v in VIEWS:
                expected=.5*ref['A'][v+'_logits']+.5*ref['B'][v+'_logits']
                assert np.array_equal(expected,arrays['LOGIT_MEAN'][v+'_logits'])
                delta=float(np.abs(arrays['R5'][v+'_logits']-expected).max())
                assert delta<=.03125
                checks.append(dict(split=split,task=task,view=v,maximum_absolute_logit_difference=delta))
        for arm in CONDITIONS:
            s=[r for r in stages if r['split']==split and r['condition']==arm]
            base=[r for r in stages if r['split']==split and r['condition']=='A']
            offline=[r for r in stages if r['split']==split and r['condition']=='LOGIT_MEAN']
            current=np.asarray([r['mAP'] for r in s]);a=np.asarray([r['mAP'] for r in base])
            logit=np.asarray([r['mAP'] for r in offline])
            summary.append(dict(split=split,condition=arm,final_mAP=current[-1],average_mAP=float(current.mean()),
                final_cF1=s[-1]['cF1'],final_oF1=s[-1]['oF1'],final_delta_vs_A=current[-1]-a[-1],
                average_delta_vs_A=float((current-a).mean()),positive_stages_vs_A=int((current>a).sum()),
                final_delta_vs_logit_mean=current[-1]-logit[-1],average_delta_vs_logit_mean=float((current-logit).mean())))
    resources=read(args.root/'resource_summary.json');base=resources['A'];cost=[]
    for arm,row in resources.items():
        cost.append(dict(condition=arm,nonvisual_parameters=row['stored_nonvisual_parameters'],
            extra_parameters=row['stored_nonvisual_parameters']-base['stored_nonvisual_parameters'],
            gpu_ms_per_batch=row['gpu_milliseconds_per_batch'],latency_ratio=row['gpu_milliseconds_per_batch']/base['gpu_milliseconds_per_batch'],
            peak_allocated_mib=row['peak_allocated_mib'],frozen_view_forwards=row['frozen_view_forwards'],
            covered_flops_per_person=row['profile_reported_flops_per_batch']/64))
    for name,rows in (('summary',summary),('stages',stages),('views',views),('subgroups',groups),('numerical_differences',checks),('resources',cost)):
        table(args.output/(name+'.csv'),rows)
    assert all(sha(Path(p))==value for p,value in hashes.items())
    integrity=dict(status='verified',fused_stage_metrics=len(stages),source_replay_view_checks=16*2*4,
        logit_mean_exact_checks=len(checks),maximum_metric_error=metric_error,
        maximum_absolute_amp_logit_difference=max(r['maximum_absolute_logit_difference'] for r in checks),
        source_forward_arrays_unchanged=True,source_forward_sha256=hashes,
        training=False,model_execution=False,standard='samples_intersect_seen_classes')
    (args.output/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,integrity={k:v for k,v in integrity.items() if k!='source_forward_sha256'}),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
