"""Read-only independent audit of four self-trained students, both splits and bypasses."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_historical_feature_fusion import COUNTS, STARTS, ap, load, metrics, read, table
from analyze_protected_view_mechanisms import masks

ARMS=('SELF_BASE','SELF_NO_CONSTRAINT','SELF_MAIN_CONSTRAINT','SELF_MAIN_ADAPTER_CONSTRAINT')
VIEWS=('fused','full','person','face')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    assert read(args.root/'completion.json')['status']=='complete'
    assert read(args.training/'training_verification.json')['status']=='verified'
    classes=read(args.historical/'config.json')['class_order']
    stages=[];summary=[];view_rows=[];groups=[];class_rows=[];direct=[];calibration=[]
    distance=[];errors=[];hashes={};cells={};records={};old_checks=[];student_checks=[]
    for arm in ARMS:
        cfg=read(args.training/arm/'train/config.json')
        done=read(args.training/arm/'train/training_complete.json')
        history=read(args.training/arm/'train/training_history.json')
        audits=read(args.training/arm/'train/audits.json')
        assert done['status']=='complete' and done['config']==cfg and cfg['tasks']==list(range(8))
        assert cfg['student_initialization']=='CLIP_visual_only_and_fresh_task_parameters_and_head'
        assert cfg['external_student_parameter_loads']==0 and not cfg['teacher_required_for_inference']
        assert cfg['evaluation_scope']=='samples_intersect_seen_classes'
        for task in range(8):
            h=history[str(task)];audit=audits[str(task)]
            assert len(h['warmup'])==30 and len(h['adaptation'])==(0 if arm==ARMS[0] else 5)
            assert all(r['skipped_optimizer_steps']==0 for phase in h.values() for r in phase)
            assert audit['old_task_parameters_exact'] and audit['old_TRAIN_probe_within_fp32_tolerance']
            student_checks.append(dict(arm=arm,task=task,old_parameters_exact=True,
                warmup_updates=sum(x['optimizer_steps'] for x in h['warmup']),
                extra_updates=sum(x['optimizer_steps'] for x in h['adaptation'])))
            if arm in ARMS[2:]:
                c=audit['functional_calibration']
                assert c['beta_model']>0 and (c['beta_adapter']>0 if arm==ARMS[-1] else c['beta_adapter']==0)
                assert audit['teacher_parameters_and_gradients_unchanged']
                calibration.append(dict(arm=arm,task=task,beta_model=c['beta_model'],beta_adapter=c['beta_adapter'],
                    model_base_norm=c['model_base_norm'],model_functional_norm=c['model_functional_norm'],
                    adapter_base_norm=c['adapter_base_norm'],adapter_functional_norm=c['adapter_functional_norm'],
                    first_teacher_mse=h['adaptation'][0]['functional_teacher_mse'],
                    last_teacher_mse=h['adaptation'][-1]['functional_teacher_mse']))
        for split in ('val','test'):
            parent=args.root/arm/split; result=read(parent/'result.json');records[arm,split]=result
            assert result['status']=='complete' and result['tasks']==list(range(8))
            assert not result['teacher_loaded_for_inference'] and not result['external_parameter_assembly']
            for mode in ('on',) if arm==ARMS[0] else ('on','off'):
                series=[];sv=[]
                for task in range(8):
                    p=parent/mode/'view_scores'/f'task{task}.npz'; hashes[str(p)]=sha(p)
                    z=load(p);cells[arm,split,mode,task]=z
                    f=load(parent/mode/'scores'/f'task{task}.npz')
                    reference=cells['SELF_BASE',split,'on',task]
                    assert len(z['targets'])==COUNTS[split][task]
                    assert (z['targets']==1).any(1).all()
                    for k in ('sample_ids','targets','class_indices','face_reliable','batch_lengths'):
                        assert np.array_equal(z[k],reference[k]),k
                    assert np.array_equal(z['fused_logits'],f['logits'])
                    assert np.array_equal(z['fused_probabilities'],f['probabilities'])
                    expected=read(parent/mode/f'task{task}.json')
                    m=metrics(z['fused_probabilities'],z['targets']);series.append(m['mAP'])
                    for k,v in m.items():errors.append(abs(v-expected['metrics'][k]))
                    for name in VIEWS:
                        actual=metrics(z[name+'_probabilities'],z['targets'])
                        errors.extend(abs(v-expected['views']['metrics'][name][k]) for k,v in actual.items())
                        assert np.allclose(ap(z[name+'_probabilities'],z['targets']),
                            expected['views']['metrics'][name]['per_class_ap'],atol=1e-8,rtol=0)
                    reliable=z['face_reliable'].astype(bool)
                    vm={name:float(ap(z[name+'_probabilities'],z['targets']).mean()) for name in VIEWS}
                    vm['reliable_face']=float(ap(z['face_probabilities'][reliable],z['targets'][reliable]).mean())
                    errors.append(abs(vm['reliable_face']-expected['views']['metrics']['face_reliable']['mAP']))
                    sv.append(vm)
                    view_rows.append(dict(arm=arm,split=split,mode=mode,task=task,**vm))
                    stages.append(dict(arm=arm,split=split,mode=mode,task=task,people=len(z['targets']),**m))
                    if task==7:
                        for name,mask in masks(z).items():
                            groups.append(dict(arm=arm,split=split,mode=mode,subset=name,people=int(mask.sum()),
                                **metrics(z['fused_probabilities'][mask],z['targets'][mask])))
                        values=ap(z['fused_probabilities'],z['targets'])
                        base=ap(reference['fused_probabilities'],reference['targets'])
                        for index,emotion in enumerate(classes):
                            class_rows.append(dict(arm=arm,split=split,mode=mode,emotion=emotion,
                                AP=float(values[index]),delta_vs_base=float(values[index]-base[index])))
                    if mode=='off':
                        on=cells[arm,split,'on',task]
                        direct.append(dict(arm=arm,split=split,task=task,
                            delta_mAP_on_minus_off=metrics(on['fused_probabilities'],on['targets'])['mAP']-m['mAP'],
                            full_logit_max_abs=float(np.max(np.abs(z['full_logits']-on['full_logits']))),
                            face_logit_max_abs=float(np.max(np.abs(z['face_logits']-on['face_logits']))),
                            person_logit_mean_abs=float(np.mean(np.abs(z['person_logits']-on['person_logits']))),
                            person_logit_max_abs=float(np.max(np.abs(z['person_logits']-on['person_logits']))),
                            fused_logit_mean_abs=float(np.mean(np.abs(z['fused_logits']-on['fused_logits'])))))
                    if args.teacher is not None:
                        pa=args.teacher/'evaluation'/split/'A/view_scores'/f'task{task}.npz'
                        pb=args.teacher/'evaluation'/split/'B/view_scores'/f'task{task}.npz'
                        a=load(pa);b=load(pb)
                        for t in (a,b):
                            assert all(np.array_equal(t[k],z[k]) for k in ('sample_ids','targets','class_indices','face_reliable'))
                        s=slice(STARTS[task],STARTS[task+1])
                        teacher=.5*a['fused_logits'][:,s]+.5*b['fused_logits'][:,s]
                        d=z['fused_logits'][:,s]-teacher
                        distance.append(dict(arm=arm,split=split,mode=mode,task=task,
                            current_class_logit_MSE=float(np.mean(d.astype(float)**2)),
                            current_class_mean_AP=float(ap(z['fused_probabilities'][:,s],z['targets'][:,s]).mean()),
                            reference_logit_mean_current_class_mAP=float(ap(1/(1+np.exp(-teacher.astype(float))),z['targets'][:,s]).mean())))
                expected=result['summary' if mode=='on' else 'off_summary']
                errors.extend((abs(series[-1]-expected['final_mAP']),abs(float(np.mean(series))-expected['average_mAP'])))
                base=np.asarray([v['mAP'] for v in records[ARMS[0],split]['metrics']])
                delta=np.asarray(series)-base
                row=dict(arm=arm,split=split,mode=mode,final_mAP=series[-1],average_mAP=float(np.mean(series)),
                    final_cF1=expected['final_cF1'],final_oF1=expected['final_oF1'],
                    final_delta_vs_base=float(delta[-1]),average_delta_vs_base=float(delta.mean()),
                    positive_stages_vs_base=int((delta>1e-8).sum()))
                row.update({'average_'+v:float(np.mean([x[v] for x in sv])) for v in sv[0]})
                summary.append(row)
    assert max(errors)<1e-8,max(errors)
    for task in range(8):
        a=[read(args.training/arm/'train/audits.json')[str(task)] for arm in ARMS]
        assert all(row['warmup_batches']==a[0]['warmup_batches'] for row in a)
        assert all(row['adaptation_batches']==a[1]['adaptation_batches'] for row in a[1:])
    old=read(args.historical/'training_history.json')
    new=read(args.training/ARMS[0]/'train/training_history.json')
    oldmetrics=read(args.historical/'task_metrics.json')
    for task in range(8):
        h=old[str(task)];n=new[str(task)]['warmup']
        old_checks.append(dict(task=task,all_30_epoch_losses_exact=all(
            a['current_loss']==b['current_loss'] and a['adapter_objective_loss']==b['adapter_objective_loss'] for a,b in zip(h,n)),
            historical_first_epoch_loss=h[0]['current_loss'],new_first_epoch_loss=n[0]['current_loss'],
            historical_test_mAP=oldmetrics[task]['mAP'],new_test_mAP=records[ARMS[0],'test']['metrics'][task]['mAP']))
    for name,rows in (('summary',summary),('stages',stages),('views',view_rows),('subgroups',groups),
                     ('classes',class_rows),('attention_direct',direct),('calibration',calibration),
                     ('teacher_distance',distance),('historical_baseline_divergence',old_checks),('training_integrity',student_checks)):
        if rows:table(args.output/(name+'.csv'),rows)
    assert all(sha(Path(p))==v for p,v in hashes.items())
    integrity=dict(status='verified',fused_stage_metrics=len(stages),view_metrics=len(view_rows)*5,
        maximum_metric_error=max(errors),training_updates=0,model_forward_executed=False,
        original_prediction_files_unchanged=True,prediction_sha256=hashes,
        all_four_arms_consumed_identical_augmented_warmup_batches=True,
        all_three_extra_stage_arms_consumed_identical_augmented_batches=True,
        standard='samples_intersect_seen_classes',test_has_participated_in_research=True)
    (args.output/'integrity.json').write_text(json.dumps(integrity,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,maximum_metric_error=max(errors),stages=len(stages)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','training','historical','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--teacher',type=Path)
    main(p.parse_args())
