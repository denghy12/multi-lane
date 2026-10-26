"""Additional CPU-only mechanism checks; no changes to the evaluated R0--R4 models."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_historical_feature_fusion import ap,load,metrics,read,table


def masks(z):
    images=np.array([str(x).rsplit('#person=',1)[0] for x in z['sample_ids']])
    _,inverse,counts=np.unique(images,return_inverse=True,return_counts=True)
    reliable=z['face_reliable'].astype(bool)
    return {'all':np.ones(len(images),bool),'reliable':reliable,'unreliable':~reliable,
            'multiple_eligible_people':counts[inverse]>1,'single_eligible_person':counts[inverse]==1,
            'reliable_multiple':reliable & (counts[inverse]>1)}


def sigmoid(z):
    return np.exp(-np.logaddexp(0.,-z.astype(np.float64))).astype(np.float32)


def whole_controls(args):
    """Use existing B bypass predictions to separate direct attention in the mean."""
    args.output.mkdir(parents=True,exist_ok=True);stages=[];summary=[];sha={}
    for split in ('val','test'):
        for mode in ('on','face_off','off'):
            values=[]
            for task in range(8):
                pa=args.root/'evaluation'/split/'R0/view_scores'/f'task{task}.npz'
                pb=args.reference/'REFERENCE_DUAL'/('b0_'+split)/mode/'view_scores'/f'task{task}.npz'
                for p in (pa,pb):sha[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                a,b=load(pa),load(pb)
                assert all(np.array_equal(a[k],b[k]) for k in ('sample_ids','targets','class_indices','face_reliable'))
                p=sigmoid((a['fused_logits'].astype(np.float64)+b['fused_logits'])*.5)
                values.append(metrics(p,a['targets'])['mAP'])
                stages.append(dict(split=split,B_condition=mode,task=task,mAP=values[-1],offline_only=True))
            summary.append(dict(split=split,B_condition=mode,final_mAP=values[-1],average_mAP=float(np.mean(values)),offline_only=True))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sha.items())
    table(args.output/'whole_attention_controls.csv',summary)
    table(args.output/'whole_attention_controls_stages.csv',stages)
    (args.output/'whole_attention_controls_integrity.json').write_text(json.dumps(dict(
        status='verified',source_files=sha,training=False,model_execution=False,source_files_unchanged=True),indent=2)+'\n')
    print('WHOLE_ATTENTION_CONTROLS',json.dumps(summary),flush=True)


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    classes=read(args.reference/'REFERENCE_PERSON/b0_val/result.json')['source_config']['class_order']
    stages=[];summary=[];groups=[];class_rows=[];source_hashes={};interaction=[];uniform=[];data={}
    for split in ('val','test'):
        values={arm:[] for arm in ('R0','R1','R2','R3','R4')};whole=[]
        for task in range(8):
            cells={}
            for arm in values:
                p=args.root/'evaluation'/split/arm/'view_scores'/f'task{task}.npz'
                source_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();cells[arm]=load(p)
            for cell in cells.values():
                assert all(np.array_equal(cell[k],cells['R0'][k]) for k in ('sample_ids','targets','class_indices','face_reliable'))
            y=cells['R0']['targets']
            for arm,cell in cells.items():
                m=metrics(cell['fused_probabilities'],y)['mAP'];values[arm].append(m)
                stages.append(dict(split=split,task=task,arm=arm,mAP=m))
            # Fixed, post-hoc WHOLE-MODEL logit average; explicitly an offline
            # numerical control, never presented as a new model forward.
            p=args.reference/'REFERENCE_DUAL'/('b0_'+split)/'on/view_scores'/f'task{task}.npz'
            source_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();dual=load(p)
            assert all(np.array_equal(dual[k],cells['R0'][k]) for k in ('sample_ids','targets','class_indices','face_reliable'))
            avg=sigmoid((cells['R0']['fused_logits'].astype(np.float64)+dual['fused_logits'])*.5)
            whole.append(metrics(avg,y)['mAP'])
            for arm in ('R0','R2','R3'):
                data[split,arm,task]=(cells[arm]['fused_probabilities'],y,cells['R0']['sample_ids'])
            data[split,'WHOLE_AVERAGE_OFFLINE',task]=(avg,y,cells['R0']['sample_ids'])
            uniform.append(dict(split=split,task=task,whole_model_half_logit_average_mAP=whole[-1],
                R2_mAP=values['R2'][-1],R3_mAP=values['R3'][-1],offline_only=True))
            interaction.append(dict(split=split,task=task,face_on_A=values['R1'][-1]-values['R0'][-1],
                face_on_updated_Person=values['R3'][-1]-values['R2'][-1],
                finite_difference_interaction=values['R3'][-1]-values['R2'][-1]-values['R1'][-1]+values['R0'][-1]))
            if task==7:
                cells['WHOLE_AVERAGE_OFFLINE']=dict(cells['R0'],fused_probabilities=avg)
                for arm,cell in cells.items():
                    for name,mask in masks(cell).items():
                        groups.append(dict(split=split,arm=arm,subset=name,people=int(mask.sum()),
                            **metrics(cell['fused_probabilities'][mask],y[mask])))
                for mask_name,mask in masks(cells['R0']).items():
                    for arm in ('R1','R2','R3','R4','WHOLE_AVERAGE_OFFLINE'):
                        actual=ap(cells[arm]['fused_probabilities'][mask],y[mask])
                        base=ap(cells['R0']['fused_probabilities'][mask],y[mask])
                        person=ap(cells['R2']['fused_probabilities'][mask],y[mask])
                        for i,name in enumerate(classes):
                            class_rows.append(dict(split=split,arm=arm,subset=mask_name,emotion=name,
                                AP=float(actual[i]),delta_vs_R0=float(actual[i]-base[i]),delta_vs_R2=float(actual[i]-person[i])))
        baseline=np.array(values['R0'])
        for arm,v in values.items():
            delta=np.array(v)-baseline
            summary.append(dict(split=split,arm=arm,final_mAP=v[-1],average_mAP=float(np.mean(v)),
                positive_stages=int((delta>1e-10).sum()),final_delta_vs_R0=float(delta[-1]),
                average_delta_vs_R0=float(delta.mean()),task0_delta=float(delta[0]),
                average_delta_excluding_task0=float(delta[1:].mean()),
                task0_fraction_of_sum_delta=float(delta[0]/delta.sum()) if abs(delta.sum())>1e-10 else None))
        summary.append(dict(split=split,arm='WHOLE_AVERAGE_OFFLINE',final_mAP=whole[-1],average_mAP=float(np.mean(whole)),
            positive_stages=int((np.array(whole)-baseline>1e-10).sum()),final_delta_vs_R0=float(whole[-1]-baseline[-1]),
            average_delta_vs_R0=float(np.mean(whole)-baseline.mean()),task0_delta=float(whole[0]-baseline[0]),
            average_delta_excluding_task0=float((np.array(whole)-baseline)[1:].mean()),
            task0_fraction_of_sum_delta=float((whole[0]-baseline[0])/(np.array(whole)-baseline).sum())))
    for name,rows in (('stage_changes',summary),('stages',stages),('subgroups',groups),('class_AP_changes',class_rows),
                       ('face_interaction',interaction),('whole_model_average_control',uniform)):
        table(args.output/(name+'.csv'),rows)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in source_hashes.items())
    (args.output/'integrity.json').write_text(json.dumps(dict(status='verified',source_files=source_hashes,
        source_files_unchanged=True,training=False,model_execution=False,
        whole_model_average_is_offline_control=True,selection_adjusted=False),indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)
    print('INTERACTION',json.dumps([r for r in interaction if r['task']==7]),flush=True)
    bootstrap={};pairs=(('R3','R2'),('WHOLE_AVERAGE_OFFLINE','R0'),('WHOLE_AVERAGE_OFFLINE','R2'))
    for split in ('val','test'):
        images=np.unique([str(x).rsplit('#person=',1)[0] for x in data[split,'R0',7][2]])
        stage_groups=[]
        for task in range(8):
            names=np.array([str(x).rsplit('#person=',1)[0] for x in data[split,'R0',task][2]])
            stage_groups.append([np.flatnonzero(names==x) for x in images])
        gen=np.random.default_rng(20261026+(split=='test'));draws={a+'-'+b:[] for a,b in pairs}
        for repeat in range(args.bootstrap):
            draw=gen.integers(0,len(images),len(images));values={arm:[] for arm in ('R0','R2','R3','WHOLE_AVERAGE_OFFLINE')}
            for task in range(8):
                ix=np.concatenate([stage_groups[task][i] for i in draw]);y=data[split,'R0',task][1][ix]
                for arm in values:values[arm].append(float(ap(data[split,arm,task][0][ix],y).mean()))
            for a,b in pairs:draws[a+'-'+b].append([values[a][-1]-values[b][-1],float(np.mean(values[a])-np.mean(values[b]))])
        bootstrap[split]=dict(repeats=args.bootstrap,unit='original_image',stages_coupled=True,
            training_seed_variance_estimated=False,selection_adjusted=False,
            ci95={name:np.percentile(values,[2.5,97.5],axis=0).tolist() for name,values in draws.items()},
            ci_columns=['final_mAP_delta','average_mAP_delta'])
        print('BOOTSTRAP',split,json.dumps(bootstrap[split]),flush=True)
        (args.output/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','reference','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=500)
    p.add_argument('--controls-only',action='store_true')
    args=p.parse_args();(whole_controls if args.controls_only else main)(args)
