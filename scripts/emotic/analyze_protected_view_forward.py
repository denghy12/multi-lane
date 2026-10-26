"""Independent CPU audit of complete feature-fusion forwards and source preservation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_historical_feature_fusion import COUNTS,ap,load,metrics,read,table

ARMS={'R0':('A','A','A'),'R1':('A','A','B'),'R2':('A','B','A'),
      'R3':('A','B','B'),'R4':('B','A','A')}
OFFLINE={'R0':'PERSON','R1':'FACE_SWAP_1','R2':'PERSON_SWAP_1','R3':'PERSON_FACE_SWAP_1','R4':'FULL_SWAP_1'}


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    assert read(args.root/'completion.json')['status']=='complete'
    data={};summary=[];stages=[];views=[];groups=[];maximum=0.;exact=0;protected=0
    offline=list(csv.DictReader((args.offline/'summary.csv').open()))
    for split in ('val','test'):
        result=read(args.root/'evaluation'/split/'result.json')
        assert result['tasks']==list(range(8)) and result['training_updates']==0
        for task in range(8):
            ref={}
            for key,name in (('A','REFERENCE_PERSON'),('B','REFERENCE_DUAL')):
                ref[key]=load(args.reference/name/('b0_'+split)/'on/view_scores'/f'task{task}.npz')
            for arm,choices in ARMS.items():
                parent=args.root/'evaluation'/split/arm
                z=load(parent/'scores'/f'task{task}.npz');v=load(parent/'view_scores'/f'task{task}.npz')
                data[split,arm,task]=z
                assert len(z['targets'])==COUNTS[split][task]
                assert (z['targets']==1).any(1).all()
                assert all(np.array_equal(z[k],ref['A'][k]) for k in ('sample_ids','targets','class_indices','batch_lengths'))
                assert np.array_equal(v['face_reliable'],ref['A']['face_reliable'])
                assert np.array_equal(z['probabilities'],v['fused_probabilities'])
                m=metrics(z['probabilities'],z['targets']);expected=result['metrics'][arm][task]
                for k,value in m.items():
                    maximum=max(maximum,abs(value-expected[k]));assert abs(value-expected[k])<1e-8
                assert np.allclose(ap(z['probabilities'],z['targets']),expected['per_class_ap'],rtol=0,atol=1e-8)
                stages.append(dict(split=split,arm=arm,task=task,people=len(z['targets']),**m))
                for view,source in zip(('full','person','face'),choices):
                    assert all(np.array_equal(v[view+suffix],ref[source][view+suffix])
                               for suffix in ('_logits','_probabilities'))
                    exact+=1
                invalid=~v['face_reliable'].astype(bool)
                if arm=='R0':
                    assert np.array_equal(z['logits'],ref['A']['fused_logits'])
                    assert np.array_equal(z['probabilities'],ref['A']['fused_probabilities'])
                if arm=='R1':
                    assert np.array_equal(z['logits'][invalid],ref['A']['fused_logits'][invalid])
                    assert np.array_equal(z['probabilities'][invalid],ref['A']['fused_probabilities'][invalid]);protected+=1
                vm={view:float(ap(v[view+'_probabilities'],v['targets']).mean()) for view in ('full','person','face','fused')}
                views.append(dict(split=split,arm=arm,task=task,**{k+'_mAP':value for k,value in vm.items()},
                    reliable_face_mAP=float(ap(v['face_probabilities'][~invalid],v['targets'][~invalid]).mean())))
                if task==7:
                    for name,mask in (('reliable_face',~invalid),('unreliable_face',invalid)):
                        groups.append(dict(split=split,arm=arm,subset=name,people=int(mask.sum()),
                                           **metrics(z['probabilities'][mask],z['targets'][mask])))
        for arm in ARMS:
            rows=[r for r in stages if r['split']==split and r['arm']==arm]
            old=next(r for r in offline if r['split']==split and r['method']==OFFLINE[arm])
            summary.append(dict(split=split,arm=arm,final_mAP=rows[-1]['mAP'],
                average_mAP=float(np.mean([r['mAP'] for r in rows])),final_cF1=rows[-1]['cF1'],final_oF1=rows[-1]['oF1'],
                final_minus_offline=rows[-1]['mAP']-float(old['final_mAP']),
                average_minus_offline=float(np.mean([r['mAP'] for r in rows]))-float(old['average_mAP'])))
    resources=read(args.root/'resource_summary.json');base=resources['R0'];cost=[]
    for arm,row in resources.items():
        cost.append(dict(arm=arm,nonvisual_parameters=row['stored_nonvisual_parameters'],
            added_parameters=row['stored_nonvisual_parameters']-base['stored_nonvisual_parameters'],
            trainable_parameters=row['trainable_parameters'],gpu_ms_per_batch=row['gpu_milliseconds_per_batch'],
            latency_ratio_vs_R0=row['gpu_milliseconds_per_batch']/base['gpu_milliseconds_per_batch'],
            peak_allocated_mib=row['peak_allocated_mib'],frozen_view_forwards=row['frozen_view_forwards'],
            profiler_covered_flops_per_person=row['profile_reported_flops_per_batch']/64))
    for name,rows in (('summary',summary),('stages',stages),('views',views),('subgroups',groups),('resources',cost)):
        table(args.output/(name+'.csv'),rows)
    integrity=dict(status='verified',fused_predictions=len(stages),source_view_exact_checks=exact,
        R1_invalid_exact_checks=protected,R0_all_stages_exact=True,maximum_metric_error=maximum,
        standard='samples_intersect_seen_classes',training=False,model_execution=False)
    (args.output/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True);print('INTEGRITY',json.dumps(integrity),flush=True)
    bootstrap={}
    for split in ('val','test'):
        final=data[split,'R0',7];names=[str(s).rsplit('#person=',1)[0] for s in final['sample_ids']]
        images=np.unique(names);stage_groups=[]
        for task in range(8):
            names=np.array([str(s).rsplit('#person=',1)[0] for s in data[split,'R0',task]['sample_ids']])
            stage_groups.append([np.flatnonzero(names==s) for s in images])
        gen=np.random.default_rng(20261026+(split=='test'));values={arm:[] for arm in ('R1','R2','R3')}
        for repeat in range(args.bootstrap):
            draw=gen.integers(0,len(images),len(images));deltas={arm:[] for arm in values}
            for task in range(8):
                ix=np.concatenate([stage_groups[task][i] for i in draw]);ref=data[split,'R0',task]
                y=ref['targets'][ix];base=float(ap(ref['probabilities'][ix],y).mean())
                for arm in values:
                    deltas[arm].append(float(ap(data[split,arm,task]['probabilities'][ix],y).mean())-base)
            for arm in values:values[arm].append([deltas[arm][-1],float(np.mean(deltas[arm]))])
        bootstrap[split]=dict(repeats=args.bootstrap,unit='original_image',stages_coupled=True,
            training_seed_variance_estimated=False,selection_adjusted=False,
            ci95={arm:np.percentile(v,[2.5,97.5],axis=0).tolist() for arm,v in values.items()},
            ci_columns=['final_mAP_delta','average_mAP_delta'])
        (args.output/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2)+'\n')
        print('BOOTSTRAP',split,json.dumps(bootstrap[split]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','reference','offline','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=500)
    main(p.parse_args())
