"""CPU verification and synthesis of checkpoint-component counterfactuals."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze_historical_feature_fusion import ap, metrics, load, read, table, rank_correlation, STARTS

ARMS=('ATTENTION','ATTENTION_BOUND','POSITION','POSITION_CALIBRATED')
VIEWS=('full','person','face','fused')
FIELDS=('sample_ids','targets','class_indices','logits','probabilities')


def main(args):
    root=args.root;out=args.output;out.mkdir(parents=True,exist_ok=False)
    primary=root/'backbone';extra=root/'transfer'
    assert read(primary/'completion.json')['status']=='complete'
    assert read(extra/'completion.json')['status']=='complete'
    records=[];data={};error=0.;anchors=0
    classes=read(args.original_root/'BASE/train/config.json')['class_order']
    for arm in ARMS:
        for split in ('val','test'):
            base=load(args.original_root/'BASE'/('historical_'+split)/'scores/task7.npz')
            for phase in ('anchors','evaluate'):
                result=read(primary/arm/split/phase/'evaluation.json')
                for value in result['conditions']:
                    c=value['condition'];folder=primary/arm/split/phase/c
                    z=load(folder/'scores/task7.npz');data['primary',arm,split,c]=z
                    assert all(np.array_equal(z[k],base[k]) for k in FIELDS[:3])
                    for k,v in metrics(z['probabilities'],z['targets']).items():
                        error=max(error,abs(v-value['metrics'][k]));assert abs(v-value['metrics'][k])<1e-8
                    assert np.allclose(ap(z['probabilities'],z['targets']),value['metrics']['per_class_ap'],rtol=0,atol=1e-8)
                    vm=load(folder/'view_scores/task7.npz')
                    endpoints={name:float(ap(vm[name+'_probabilities'],vm['targets']).mean()) for name in VIEWS}
                    for name,v in endpoints.items():
                        delta=abs(v-value['view_diagnostics']['metrics'][name]['mAP']);error=max(error,delta);assert delta<1e-8
                    if c in ('none','ASPH'):
                        ref_folder=(args.original_root/'BASE'/('historical_'+split) if c=='ASPH' else
                            args.bypass_root/arm/'off' if split=='test' else args.original_root/arm/'runtime_off')
                        ref=load(ref_folder/'scores/task7.npz')
                        assert all(np.array_equal(z[k],ref[k]) for k in FIELDS);anchors+=1
                    records.append(dict(source='primary',arm=arm,split=split,restored=c,**metrics(z['probabilities'],z['targets']),
                        **{name+'_mAP':v for name,v in endpoints.items()}))
    for arm in ARMS[:2]:
        for split in ('val','test'):
            result=read(extra/arm/split/'completion.json')
            for value in result['conditions']:
                c=value['condition'];folder=extra/arm/split/c;z=load(folder/'scores/task7.npz')
                data['transfer',arm,split,c]=z
                base=data['primary',arm,split,'ASPH']
                assert all(np.array_equal(z[k],base[k]) for k in FIELDS[:3])
                m=metrics(z['probabilities'],z['targets'])
                for k,v in m.items():
                    error=max(error,abs(v-value['metrics'][k]));assert abs(v-value['metrics'][k])<1e-8
                if c=='none':
                    ref=load(args.original_root/arm/('historical_'+split)/'scores/task7.npz')
                    assert all(np.array_equal(z[k],ref[k]) for k in FIELDS);anchors+=1
                vm=load(folder/'view_scores/task7.npz')
                endpoints={name:float(ap(vm[name+'_probabilities'],vm['targets']).mean()) for name in VIEWS}
                for name,v in endpoints.items():
                    delta=abs(v-value['view_diagnostics']['metrics'][name]['mAP']);error=max(error,delta);assert delta<1e-8
                records.append(dict(source='transfer',arm=arm,split=split,restored=c,**m,**{name+'_mAP':v for name,v in endpoints.items()}))
    assert len(records)==68 and anchors==20
    for r in records:
        base=next(x for x in records if x['source']=='primary' and x['arm']==r['arm'] and x['split']==r['split'] and x['restored']=='ASPH')
        none=next(x for x in records if x['source']=='primary' and x['arm']==r['arm'] and x['split']==r['split'] and x['restored']=='none')
        r['delta_vs_base']=r['mAP']-base['mAP'];r['delta_vs_joint_off']=r['mAP']-none['mAP']
    table(out/'summary.csv',records)
    probe_groups={};fingerprints={};feature_head_checks=0
    for arm in ARMS:
        probe=read(primary/arm/'probe/probe.json')['records'];assert len(probe)==96
        for r in probe:
            key=(r['task'],r['batch'])
            fingerprint=(r['sample_ids'],r['pixel_sha256'])
            fingerprints.setdefault(key,fingerprint);assert fingerprints[key]==fingerprint
            if r['condition']=='ASPH':
                assert all(v.get('feature_relative_l2',0)<1e-7 and v.get('logit_mean_abs_change',0)<1e-7 for v in r['views'].values())
            for name,v in r['views'].items():
                if not v['samples']:continue
                probe_groups.setdefault((arm,r['task'],r['condition'],name),[]).append(v)
        index={(r['task'],r['batch'],r['condition']):r for r in probe}
        for r in probe:
            if 'H' not in r['condition'] or r['condition']=='none':continue
            without=r['condition'].replace('H','') or 'none'
            other=index[r['task'],r['batch'],without]
            for name,v in r['views'].items():
                if v['samples']:
                    assert abs(v['feature_mean_cosine']-other['views'][name]['feature_mean_cosine'])<1e-6
                    feature_head_checks+=1
    probes=[]
    for (arm,task,condition,view),values in probe_groups.items():
        n=sum(v['samples'] for v in values)
        probes.append(dict(arm=arm,task=task,restored=condition,view=view,samples=n,
            **{key:sum(v[key]*v['samples'] for v in values)/n for key in
               ('feature_relative_l2','feature_mean_cosine','logit_mean_abs_change','bce')}))
    table(out/'train_probe_summary.csv',probes)
    subgroups=[];class_rows=[];task_rows=[];rank_rows=[]
    for split in ('val','test'):
        ref=data['primary','ATTENTION_BOUND',split,'ASPH'];y=ref['targets']
        names=np.asarray([str(s).rsplit('#person=',1)[0] for s in ref['sample_ids']])
        _,inverse,counts=np.unique(names,return_inverse=True,return_counts=True)
        mask=load(primary/'ATTENTION_BOUND'/split/'anchors/ASPH/view_scores/task7.npz')['face_reliable']
        masks=dict(all_people=np.ones(len(y),bool),reliable_face=mask,unreliable_face=~mask,multiple_people=counts[inverse]>1,single_person=counts[inverse]==1)
        for r in [x for x in records if x['split']==split]:
            z=data[r['source'],r['arm'],split,r['restored']]
            delta=ap(z['probabilities'],y)-ap(ref['probabilities'],y)
            for subset,ix in masks.items():
                m=metrics(z['probabilities'][ix],y[ix]);base=float(ap(ref['probabilities'][ix],y[ix]).mean())
                subgroups.append(dict(source=r['source'],arm=r['arm'],split=split,restored=r['restored'],subset=subset,people=int(ix.sum()),**m,delta_vs_base=m['mAP']-base))
            for i,name in enumerate(classes):
                class_rows.append(dict(source=r['source'],arm=r['arm'],split=split,restored=r['restored'],class_name=name,delta_vs_base=float(delta[i])))
            for t,(start,end) in enumerate(zip(STARTS[:-1],STARTS[1:])):
                task_rows.append(dict(source=r['source'],arm=r['arm'],split=split,restored=r['restored'],task=t,mean_AP_delta=float(delta[start:end].mean())))
            rank_rows.append(dict(source=r['source'],arm=r['arm'],split=split,restored=r['restored'],mean_class_spearman_vs_base=rank_correlation(z['logits'],ref['logits'])))
    for name,rows in (('subgroups',subgroups),('class_deltas',class_rows),('task_class_deltas',task_rows),('ranks',rank_rows)):
        table(out/(name+'.csv'),rows)
    fallback=[]
    for split in ('val','test'):
        base=data['primary','ATTENTION_BOUND',split,'ASPH']
        candidate=data['transfer','ATTENTION_BOUND',split,'H']
        valid=load(primary/'ATTENTION_BOUND'/split/'anchors/ASPH/view_scores/task7.npz')['face_reliable']
        policies=dict(baseline=base['probabilities'],candidate=candidate['probabilities'],
            candidate_if_reliable=np.where(valid[:,None],candidate['probabilities'],base['probabilities']),
            candidate_if_unreliable=np.where(valid[:,None],base['probabilities'],candidate['probabilities']))
        assert np.array_equal(policies['candidate_if_reliable'][~valid],base['probabilities'][~valid])
        assert np.array_equal(policies['candidate_if_unreliable'][valid],base['probabilities'][valid])
        for name,p in policies.items():
            m=metrics(p,base['targets'])
            fallback.append(dict(split=split,policy=name,**m,
                delta_vs_base=m['mAP']-float(ap(base['probabilities'],base['targets']).mean()),
                reliable_subset_mAP=float(ap(p[valid],base['targets'][valid]).mean()),
                unreliable_subset_mAP=float(ap(p[~valid],base['targets'][~valid]).mean())))
    table(out/'reliability_fallback.csv',fallback)
    audit=dict(status='verified',full_predictions=68,view_predictions=68,anchor_replays_exact=20,
        fixed_train_records=384,head_feature_independence_checks=feature_head_checks,
        same_train_pixels_and_samples=True,maximum_metric_error=error,training_or_gpu_forward=False,
        no_full_eight_task_average=True,hybrids_are_exploratory=True)
    (out/'analysis_integrity.json').write_text(json.dumps(audit,indent=2));print('AUDIT',json.dumps(audit),flush=True)
    boot={}
    for split in ('val','test'):
        ref=data['primary','ATTENTION_BOUND',split,'ASPH'];y=ref['targets'];groups={}
        for i,s in enumerate(ref['sample_ids']):groups.setdefault(str(s).rsplit('#person=',1)[0],[]).append(i)
        groups=list(groups.values());gen=np.random.default_rng(20261025+(split=='test'))
        pairs=dict(base=ref['probabilities'],H_on=data['transfer','ATTENTION_BOUND',split,'H']['probabilities'],
            H_off=data['primary','ATTENTION_BOUND',split,'H']['probabilities'],
            original_on=data['transfer','ATTENTION_BOUND',split,'none']['probabilities'],
            base_on=data['transfer','ATTENTION_BOUND',split,'ASPH']['probabilities'])
        contrasts=(('H_on','base'),('H_off','base'),('H_on','H_off'),('H_on','original_on'),('base_on','base'))
        samples={a+'-'+b:[] for a,b in contrasts}
        for _ in range(args.bootstrap):
            ix=np.concatenate([groups[i] for i in gen.integers(0,len(groups),len(groups))])
            values={k:float(ap(p[ix],y[ix]).mean()) for k,p in pairs.items()}
            for a,b in contrasts:samples[a+'-'+b].append(values[a]-values[b])
        boot[split]=dict(repeats=args.bootstrap,unit='original_image',images=len(groups),people=len(y),
            selection_adjusted=False,future_training_seed_variance_estimated=False,
            ci95={k:np.percentile(v,[2.5,97.5]).tolist() for k,v in samples.items()})
        print('BOOTSTRAP',split,json.dumps(boot[split]),flush=True)
    (out/'bootstrap.json').write_text(json.dumps(boot,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','original-root','bypass-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=500)
    main(p.parse_args())
