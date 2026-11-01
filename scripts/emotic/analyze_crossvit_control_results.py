"""Verify paired historical cohorts and analyze completed independent controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.runner import compute_metrics


def read(p):return json.loads(p.read_text())
def metric(p,y,task=7):
    return compute_metrics(task,torch.from_numpy(p),torch.from_numpy(y),.5)
def load(p):
    with np.load(p,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}
def weights(mask,mode):
    a=np.array([.64,.16,.20] if mode=='W' else [1/3]*3,dtype=np.float32)
    b=np.array([.8,.2,0.] if mode=='W' else [.5,.5,0.],dtype=np.float32)
    return np.where(mask[:,None],a,b)


def main(args):
    torch.set_num_threads(2);root=args.snapshot
    names=read(root/'baseline/config.json')['class_order']
    heads=read(args.head_accounting) if args.head_accounting else None
    result=dict(verified_score_records=0,paired_cohorts=[],summary={},stages={},final_views={},
        classes={},counterfactual={},reliability={},score_scale={},training={},head_calibration={})
    for arm in ('I_W','I_M'):
        p=root/'experiment'/arm/'train'
        cfg=read(p/'config.json');hist=read(p/'training_history.json');audit=read(p/'audits.json')
        assert cfg['seed']==1 and cfg['evaluation_scope']=='samples_intersect_seen_classes'
        assert list(sorted(map(int,hist)))==list(range(8))
        assert all(len(hist[str(t)])==30 for t in range(8))
        assert all(audit[str(t)]['old_parameters_exact'] and audit[str(t)]['frozen_visual_exact'] for t in range(8))
        result['training'][arm]=dict(epochs=[len(hist[str(t)]) for t in range(8)],
            optimizer_steps=sum(r['optimizer_steps'] for rows in hist.values() for r in rows),
            skipped_steps=sum(r['skipped_optimizer_steps'] for rows in hist.values() for r in rows),
            old_parameters_exact=True,frozen_visual_exact=True,
            fixed_TRAIN_max_abs_logit_drift=max((max(v['old_prediction_probe'].get('max_abs_difference',{}).values(),default=0.) for v in audit.values()),default=0.))
    assert read(root/'experiment/I_W/train/config.json')['initial_common_sha256']==read(root/'experiment/I_M/train/config.json')['initial_common_sha256']
    for split in ('val','test'):
        base=read(root/'baseline'/split/'result.json')
        result['summary']['B0_'+split]=base['summary']
        result['stages'][split]={'B0':[r['mAP'] for r in base['task_metrics']]}
        for arm in ('I_W','I_M'):
            d=read(root/'experiment'/arm/split/'result.json')
            assert d['status']=='complete' and d['tasks']==list(range(8))
            assert d['summary']['average_mAP']==float(np.mean([r['mAP'] for r in d['metrics']]))
            result['summary'][arm+'_'+split]=d['summary']
            result['stages'][split][arm]=[r['mAP'] for r in d['metrics']]
            counter={'W':[],'M':[]};native_differences=[]
            calibration={k:[] for k in ('norm_to_full','bias_to_full','norm_and_bias_to_full')}
            for task in range(8):
                b=load(root/'baseline'/split/'view_scores'/f'task{task}.npz')
                x=load(root/'experiment'/arm/split/f'task{task}_scores.npz')
                for k in ('sample_ids','targets','class_indices','face_reliable'):
                    assert np.array_equal(b[k],x[k]),(arm,split,task,k)
                for data,expected in ((b,base['task_metrics'][task]),(x,d['metrics'][task])):
                    m=metric(data['fused_probabilities'],data['targets'],task)
                    assert abs(m.mAP-expected['mAP'])<1e-10
                    assert np.allclose(m.per_class_ap,expected['per_class_ap'],rtol=0,atol=1e-10)
                    result['verified_score_records']+=1
                result['paired_cohorts'].append(dict(arm=arm,split=split,task=task,samples=len(x['targets']),
                    ids_targets_classes_reliability_exact=True))
                for mode in ('W','M'):
                    w=weights(x['face_reliable'],mode)
                    z=sum(x[v+'_logits']*w[:,i,None] for i,v in enumerate(('full','person','face')))
                    p=torch.from_numpy(z).sigmoid().numpy()
                    counter[mode].append(metric(p,x['targets'],task).mAP)
                    if arm.endswith(mode):
                        native_differences.append(float(np.max(np.abs(z-x['fused_logits']))))
                if heads is not None:
                    count=x['targets'].shape[1]
                    reference=heads[arm]['full']
                    rn=np.asarray(reference['row_norms'][:count],dtype=np.float32)
                    rb=np.asarray(reference['bias'][:count],dtype=np.float32)
                    w=weights(x['face_reliable'],arm[-1])
                    for name in calibration:
                        calibrated=[]
                        for v in ('full','person','face'):
                            h=heads[arm][v]
                            norm=np.asarray(h['row_norms'][:count],dtype=np.float32)
                            bias=np.asarray(h['bias'][:count],dtype=np.float32)
                            scale=rn/norm if name!='bias_to_full' else np.ones_like(rn)
                            offset=rb if name!='norm_to_full' else bias
                            calibrated.append((x[v+'_logits']-bias)*scale+offset)
                        z=sum(v*w[:,i,None] for i,v in enumerate(calibrated))
                        calibration[name].append(metric(torch.from_numpy(z).sigmoid().numpy(),x['targets'],task).mAP)
                if task!=7:continue
                for label,data in (('B0',b),(arm,x)):
                    views={v:metric(data[v+'_probabilities'],data['targets']).mAP for v in ('fused','full','person','face')}
                    mask=data['face_reliable'];views['face_reliable']=metric(data['face_probabilities'][mask],data['targets'][mask]).mAP
                    result['final_views'][label+'_'+split]=views
                    groups={}
                    for tag,select in (('reliable',mask),('unreliable',~mask)):
                        groups[tag]=dict(samples=int(select.sum()),fused=metric(data['fused_probabilities'][select],data['targets'][select]).mAP,
                            full=metric(data['full_probabilities'][select],data['targets'][select]).mAP)
                    result['reliability'][label+'_'+split]=groups
                    scale={}
                    for view in ('full','person','face'):
                        values=data[view+'_logits'][mask].astype(np.float64)
                        scale[view]=dict(logit_rms=float(np.sqrt(np.mean(values**2))),
                            mean_class_std=float(values.std(0).mean()))
                    result['score_scale'][label+'_'+split]=scale
                delta=np.asarray(d['metrics'][task]['per_class_ap'])-np.asarray(base['task_metrics'][task]['per_class_ap'])
                result['classes'][arm+'_'+split]=dict(improved=int((delta>0).sum()),declined=int((delta<0).sum()),
                    rows=sorted([dict(label=n,delta=float(v),base_ap=float(a),ap=float(c),positives=int(x['targets'][:,i].sum()))
                        for i,(n,v,a,c) in enumerate(zip(names,delta,base['task_metrics'][task]['per_class_ap'],d['metrics'][task]['per_class_ap']))],key=lambda v:v['delta']))
            result['counterfactual'][arm+'_'+split]={mode:dict(stages=vals,final=vals[-1],average=float(np.mean(vals))) for mode,vals in counter.items()}
            result['counterfactual'][arm+'_'+split]['native_max_abs_logit_reconstruction_difference']=max(native_differences)
            if heads is not None:
                result['head_calibration'][arm+'_'+split]={k:dict(stages=v,final=v[-1],average=float(np.mean(v))) for k,v in calibration.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(score_records=result['verified_score_records'],paired_records=len(result['paired_cohorts']),
        final_views=result['final_views'],counterfactual=result['counterfactual']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--head-accounting',type=Path)
    main(p.parse_args())
