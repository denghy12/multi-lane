"""Independently audit and compare completed eight-stage predictions, CPU only."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_historical_feature_fusion import ap, metrics, load, read, table, rank_correlation, STARTS, COUNTS

ARMS=('HEAD_FIXED_NO_ATTENTION','HEAD_FIXED_ATTENTION','BASE_FIXED_ATTENTION','HEAD_TRAINABLE_ATTENTION')
VIEWS=('full','person','face','fused')


def cases(root,original,split):
    values={'B0':(original/'BASE'/('historical_'+split),None),
            'PERSON_NATIVE':(original/'ATTENTION_BOUND'/('historical_'+split),None)}
    for arm in ARMS:
        values[arm]=(root/arm/('native_'+split),'on')
        if arm!='HEAD_FIXED_NO_ATTENTION': values[arm+'_OFF']=(root/arm/('native_'+split),'off')
    for mode in ('on','off'):
        values['PERSON_B0'+('_OFF' if mode=='off' else '')]=(root/'PERSON_REFERENCE'/('b0_'+split),mode)
    for mode,suffix in (('on',''),('face_off','_FACE_OFF'),('off','_OFF')):
        values['FACE_B0'+suffix]=(root/'FACE_JOINT'/('b0_'+split),mode)
    values['FACE_NATIVE']=(root/'FACE_JOINT'/('native_'+split),'on')
    return values


def bucket_masks(z,v):
    names=np.asarray([str(s).rsplit('#person=',1)[0] for s in z['sample_ids']])
    _,inv,counts=np.unique(names,return_inverse=True,return_counts=True)
    reliable=v['face_reliable'].astype(bool)
    return dict(all=np.ones(len(names),bool),reliable_face=reliable,unreliable_face=~reliable,
                multiple_eligible_people=counts[inv]>1,single_eligible_person=counts[inv]==1)


def extras(args):
    correlations=[];composition=[];advantage=[];training=[];budgets=[]
    classes=read(args.original_root/'BASE/train/config.json')['class_order']
    for split in ('val','test'):
        options=cases(args.root,args.original_root,split);cached={}
        for model in ('B0','PERSON_B0','FACE_B0','FACE_B0_FACE_OFF'):
            folder,mode=options[model];folder=folder/mode if mode is not None else folder
            v=load(folder/'view_scores/task7.npz');cached[model]=v;mask=v['face_reliable'].astype(bool)
            correlations.append(dict(split=split,model=model,
                mean_class_spearman_full_face_reliable=rank_correlation(v['full_logits'][mask],v['face_logits'][mask])))
        v=cached['B0'];masks=bucket_masks(v,v)
        composition.append(dict(split=split,people=len(v['targets']),positive_label_density=float(v['targets'].mean()),
            positive_labels_per_person=float(v['targets'].sum(1).mean()),
            reliable_face_fraction=float(masks['reliable_face'].mean()),
            multiple_eligible_people_fraction=float(masks['multiple_eligible_people'].mean())))
        face=cached['FACE_B0'];person=cached['PERSON_B0'];mask=face['face_reliable'].astype(bool)
        fa=ap(face['face_probabilities'][mask],face['targets'][mask])
        fused=ap(person['fused_probabilities'][mask],person['targets'][mask])
        for i,name in enumerate(classes):
            advantage.append(dict(split=split,emotion=name,face_AP=float(fa[i]),
                strong_fused_AP=float(fused[i]),face_minus_strong_fused=float(fa[i]-fused[i])))
    for label,folder in (('PERSON_NATIVE',args.original_root/'ATTENTION_BOUND/train'),('FACE_NATIVE',args.root/'FACE_JOINT/train')):
        history=read(folder/'training_history.json')
        for task,rows in history.items():
            training.append(dict(model=label,task=int(task),first_BCE=rows[0]['current_loss'],last_BCE=rows[-1]['current_loss'],
                first_adapter_ASL=rows[0]['adapter_objective_loss'],last_adapter_ASL=rows[-1]['adapter_objective_loss']))
        summary=read(folder/'seed_summary.json');audit=read(folder.parent/'training_verification.json')
        budgets.append(dict(model=label,epochs=summary['completed_epochs'],updates=summary['completed_optimizer_updates'],
            skipped=sum(r['skipped_optimizer_steps'] for rows in history.values() for r in rows),
            peak_allocated_mib=audit['peak_allocated_mib'],elapsed_seconds=summary['elapsed_seconds']))
    for name,rows in (('full_face_rank_correlation',correlations),('cohort_composition',composition),
                      ('face_vs_strong_fused_by_class',advantage),('joint_training_losses',training),('joint_training_budgets',budgets)):
        table(args.output/(name+'.csv'),rows)
    good=[name for name in classes if all(next(r['face_minus_strong_fused'] for r in advantage
        if r['split']==split and r['emotion']==name)>0 for split in ('val','test'))]
    (args.output/'face_advantage_on_both_splits.json').write_text(json.dumps(good,indent=2)+'\n')
    print('FACE_CLASS_ADVANTAGE',good,flush=True)


def main(args):
    root,out=args.root,args.output;out.mkdir(parents=True,exist_ok=False)
    assert read(root/'completion.json')['status']=='complete' and not (root/'failure.json').exists()
    classes=read(args.original_root/'BASE/train/config.json')['class_order']
    data={};final_views={};stages=[];views=[];summary=[];subgroups=[];class_rows=[];task_classes=[];calibration=[]
    max_error=0.;prefix_exact=0;off_base_exact=0;native_exact=0;face_mask_exact=0
    for split in ('val','test'):
        for name,(folder,mode) in cases(root,args.original_root,split).items():
            result=read(folder/'result.json')
            reported=result['task_metrics'] if mode is None else result['task_metrics'][mode]
            own=[]
            for task in range(8):
                parent=folder if mode is None else folder/mode
                z=load(parent/'scores'/f'task{task}.npz')
                v=load(parent/'view_scores'/f'task{task}.npz')
                data[split,name,task]=z
                base=data[split,'B0',task]
                assert all(np.array_equal(z[k],base[k]) for k in ('sample_ids','targets','class_indices'))
                assert len(z['targets'])==COUNTS[split][task]
                assert (z['targets']==1).any(1).all()
                assert all(np.array_equal(z[k],v[k]) for k in ('sample_ids','targets','class_indices'))
                assert np.array_equal(z['probabilities'],v['fused_probabilities'])
                expected=reported[task]
                m=metrics(z['probabilities'],z['targets'])
                for key,value in m.items():
                    max_error=max(max_error,abs(value-expected[key]));assert abs(value-expected[key])<1e-8
                assert np.allclose(ap(z['probabilities'],z['targets']),expected['per_class_ap'],rtol=0,atol=1e-8)
                own.append(m)
                stages.append(dict(split=split,model=name,task=task,samples=len(z['targets']),**m,
                    delta_vs_b0=m['mAP']-metrics(base['probabilities'],base['targets'])['mAP']))
                diag=result['view_diagnostics'][str(task)] if mode is None else read(parent/f'task{task}.json')['views']
                vm={key:float(ap(v[key+'_probabilities'],v['targets']).mean()) for key in VIEWS}
                for key,value in vm.items():
                    max_error=max(max_error,abs(value-diag['metrics'][key]['mAP']))
                    assert abs(value-diag['metrics'][key]['mAP'])<1e-8
                reliable=v['face_reliable'].astype(bool)
                views.append(dict(split=split,model=name,task=task,**{k+'_mAP':x for k,x in vm.items()},
                    face_reliable_mAP=float(ap(v['face_probabilities'][reliable],v['targets'][reliable]).mean()),
                    fusion_minus_full=vm['fused']-vm['full']))
                if name=='BASE_FIXED_ATTENTION_OFF':
                    assert all(np.array_equal(z[k],base[k]) for k in ('logits','probabilities'));off_base_exact+=1
                if name in ARMS and split=='val' and task<3:
                    previous=load(args.pilot_root/name/'on/scores'/f'task{task}.npz')
                    assert all(np.array_equal(z[k],previous[k]) for k in ('sample_ids','targets','class_indices','logits','probabilities'))
                    prefix_exact+=1
                if name=='FACE_NATIVE' and split=='test':
                    previous=load(root/'FACE_JOINT/train/test_scores'/f'task{task}.npz')
                    assert all(np.array_equal(z[k],previous[k]) for k in ('sample_ids','targets','class_indices','logits','probabilities'))
                    native_exact+=1
                if task==7:
                    final_views[split,name]=v
                    masks=bucket_masks(z,v);y=z['targets'];p=z['probabilities']
                    for subset,mask in masks.items():
                        subgroups.append(dict(split=split,model=name,subset=subset,people=int(mask.sum()),
                                              **metrics(p[mask],y[mask])))
                    delta=ap(p,y)-ap(base['probabilities'],y)
                    for i,cls in enumerate(classes):
                        class_rows.append(dict(split=split,model=name,emotion=cls,AP=float(ap(p,y)[i]),delta_vs_b0=float(delta[i])))
                    for t,(a,b) in enumerate(zip(STARTS[:-1],STARTS[1:])):
                        task_classes.append(dict(split=split,model=name,class_task=t,mean_AP_delta_vs_b0=float(delta[a:b].mean())))
                    pred=p>.5;positive=y==1;tp=np.logical_and(pred,positive).sum()
                    calibration.append(dict(split=split,model=name,predicted_positive_rate=float(pred.mean()),
                        label_positive_rate=float(positive.mean()),oPrecision=float(100*tp/max(1,pred.sum())),
                        oRecall=float(100*tp/max(1,positive.sum())),mean_positive_score=float(p[positive].mean()),
                        mean_negative_score=float(p[~positive].mean())))
            summary.append(dict(split=split,model=name,final_mAP=own[-1]['mAP'],
                average_mAP=float(np.mean([r['mAP'] for r in own])),final_cF1=own[-1]['cF1'],final_oF1=own[-1]['oF1']))
        for task in range(8):
            a=load(root/'FACE_JOINT'/('b0_'+split)/'on/view_scores'/f'task{task}.npz')
            b=load(root/'FACE_JOINT'/('b0_'+split)/'face_off/view_scores'/f'task{task}.npz')
            invalid=~a['face_reliable'].astype(bool)
            assert all(np.array_equal(a[k+'_logits'],b[k+'_logits']) for k in ('full','person'))
            assert np.array_equal(a['face_logits'][invalid],b['face_logits'][invalid])
            assert np.array_equal(a['fused_logits'][invalid],b['fused_logits'][invalid]);face_mask_exact+=1
    for name,rows in (('summary',summary),('stages',stages),('views',views),('subgroups',subgroups),
                      ('class_AP',class_rows),('class_tasks_at_final',task_classes),('threshold_calibration',calibration)):
        table(out/(name+'.csv'),rows)
    decomposition=[]
    for split in ('val','test'):
        by={r['model']:r for r in summary if r['split']==split}
        for metric in ('final_mAP','average_mAP'):
            shared=by['FACE_B0_OFF'][metric]-by['PERSON_B0_OFF'][metric]
            person_new=by['FACE_B0_FACE_OFF'][metric]-by['FACE_B0_OFF'][metric]
            person_old=by['PERSON_B0'][metric]-by['PERSON_B0_OFF'][metric]
            face=by['FACE_B0'][metric]-by['FACE_B0_FACE_OFF'][metric]
            net=by['FACE_B0'][metric]-by['PERSON_B0'][metric]
            assert abs(shared+person_new-person_old+face-net)<1e-10
            decomposition.append(dict(split=split,metric=metric,shared_ASP_change=shared,
                person_contribution_change=person_new-person_old,new_face_conditional=face,total_change=net))
    table(out/'face_vs_person_decomposition.csv',decomposition)
    evidence=[]
    for split in ('val','test'):
        for model in ('B0','PERSON_B0','FACE_B0'):
            v=final_views[split,model];mask=v['face_reliable'].astype(bool);y=v['targets'][mask]
            values={view:ap(v[view+'_probabilities'][mask],y) for view in VIEWS}
            for i,c in enumerate(classes):
                evidence.append(dict(split=split,model=model,emotion=c,
                                     **{key+'_AP':float(value[i]) for key,value in values.items()}))
    table(out/'reliable_face_class_evidence.csv',evidence)
    stats=[]
    for label,folder in (('PERSON_NATIVE',args.original_root/'ATTENTION_BOUND/train'),('FACE_NATIVE',root/'FACE_JOINT/train')):
        audit=read(folder/'cross_view_audit.json')
        for task,row in audit.items():
            for key,v in row['training_statistics'].items():
                n=v['samples'];stats.append(dict(model=label,task=int(task),view='face' if key.endswith('/face') else 'person',
                    training_rows=n,mean_residual_to_summary_norm=v['ratio_sum']/n,
                    raw_residual_ratio=v['raw_ratio_sum']/n,attention_entropy=v['entropy_sum']/n))
    table(out/'attention_training_summary.csv',stats)
    integrity=dict(status='verified',fused_predictions=len(stages),view_metrics=len(views)*4,
        maximum_metric_error=max_error,pilot_prediction_replays=prefix_exact,base_frozen_off_replays=off_base_exact,
        native_face_test_replays=native_exact,face_mask_invariances=face_mask_exact,
        gpu_or_training=False,standard='samples_intersect_seen_classes')
    (out/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    print('INTEGRITY',json.dumps(integrity),flush=True)
    print('DECOMPOSITION',json.dumps(decomposition),flush=True)
    # One paired original-image draw is shared by every model and every stage.
    chosen=('B0','PERSON_B0','FACE_B0','FACE_B0_FACE_OFF','HEAD_FIXED_NO_ATTENTION',
            'HEAD_FIXED_ATTENTION','BASE_FIXED_ATTENTION')
    pairs=(('PERSON_B0','B0'),('FACE_B0','PERSON_B0'),('FACE_B0','B0'),
           ('FACE_B0','FACE_B0_FACE_OFF'),('HEAD_FIXED_ATTENTION','HEAD_FIXED_NO_ATTENTION'),
           ('BASE_FIXED_ATTENTION','B0'),('HEAD_FIXED_NO_ATTENTION','B0'))
    bootstrap={}
    for split in ('val','test'):
        final=data[split,'B0',7]
        images=np.asarray([str(s).rsplit('#person=',1)[0] for s in final['sample_ids']])
        groups=np.unique(images);lookup={name:i for i,name in enumerate(groups)}
        stage_groups=[]
        for task in range(8):
            z=data[split,'B0',task]
            names=[str(s).rsplit('#person=',1)[0] for s in z['sample_ids']]
            stage_groups.append([np.asarray([i for i,s in enumerate(names) if s==group],dtype=np.int64) for group in groups])
        gen=np.random.default_rng(20261025+(split=='test'));draws={kind:{a+'-'+b:[] for a,b in pairs} for kind in ('final','average')}
        for iteration in range(args.bootstrap):
            draw=gen.integers(0,len(groups),len(groups));values={name:[] for name in chosen}
            for task in range(8):
                ix=np.concatenate([stage_groups[task][i] for i in draw]);y=data[split,'B0',task]['targets'][ix]
                for name in chosen:
                    values[name].append(float(ap(data[split,name,task]['probabilities'][ix],y).mean()))
            for a,b in pairs:
                draws['final'][a+'-'+b].append(values[a][-1]-values[b][-1])
                draws['average'][a+'-'+b].append(float(np.mean(values[a])-np.mean(values[b])))
            if (iteration+1)%100==0:print('BOOTSTRAP_PROGRESS',split,iteration+1,flush=True)
        bootstrap[split]=dict(repeats=args.bootstrap,unit='original_image',images=len(groups),
            stages_coupled=True,selection_adjusted=False,training_seed_variance_estimated=False,
            ci95={kind:{name:np.percentile(v,[2.5,97.5]).tolist() for name,v in results.items()} for kind,results in draws.items()})
        print('BOOTSTRAP',split,json.dumps(bootstrap[split]),flush=True)
        (out/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','original-root','pilot-root','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--extras-only',action='store_true')
    args=p.parse_args()
    if not args.extras_only:main(args)
    extras(args)
