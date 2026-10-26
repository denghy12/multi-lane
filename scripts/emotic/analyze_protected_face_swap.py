"""Read-only combinations of saved, stage-matched B0-head view logits; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_historical_feature_fusion import COUNTS, ap, load, metrics, table


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -x.astype(np.float64))).astype(np.float32)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    summaries, stages, subgroups, class_changes, controls = [], [], [], [], []
    source_hashes = {}; cached = {}; replay_error = 0.
    for split in ('val', 'test'):
        scores = {}
        for task in range(8):
            values = []
            for arm, mode in (('REFERENCE_PERSON', 'on'), ('REFERENCE_DUAL', 'on'), ('REFERENCE_DUAL', 'face_off')):
                p = args.root / arm / ('b0_' + split) / mode / 'view_scores' / f'task{task}.npz'
                source_hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
                values.append(load(p))
            person, dual, face_off = values
            for v in values[1:]:
                assert all(np.array_equal(v[k], person[k]) for k in ('sample_ids', 'targets', 'class_indices', 'face_reliable'))
            y = person['targets']; reliable = person['face_reliable'].astype(bool)
            assert len(y) == COUNTS[split][task] and (y == 1).any(1).all()
            weight = np.stack([np.where(reliable, .64, .80), np.where(reliable, .16, .20),
                               np.where(reliable, .20, 0.)], axis=1)
            difference = {v: dual[v+'_logits'].astype(np.float64) - person[v+'_logits']
                          for v in ('full', 'person', 'face')}
            base = person['fused_logits'].astype(np.float64)
            options = {'PERSON': person['fused_probabilities'], 'DUAL': dual['fused_probabilities']}
            for coefficient in (.25, .5, .75, 1.):
                options[f'FACE_SWAP_{coefficient:g}'] = sigmoid(base + coefficient*weight[:, 2, None]*difference['face'])
            for view, i in (('FULL', 0), ('PERSON', 1)):
                options[view+'_SWAP_1'] = sigmoid(base + weight[:, i, None]*difference[view.lower()])
            options['PERSON_FACE_SWAP_1'] = sigmoid(base + weight[:, 1, None]*difference['person']
                                                   + weight[:, 2, None]*difference['face'])
            all_swap = base + sum(weight[:, i, None]*difference[v] for i, v in enumerate(('full', 'person', 'face')))
            options['ALL_SWAP_1'] = sigmoid(all_swap)
            options['FACE_ATTENTION_DELTA_ONLY'] = sigmoid(base + weight[:, 2, None]*(
                dual['face_logits'].astype(np.float64) - face_off['face_logits']))
            # Preserve saved probabilities where the mathematical perturbation is
            # zero; NumPy and torch sigmoid may differ by one float32 ULP.
            for name, p in options.items():
                if name.startswith('FACE_SWAP') or name == 'FACE_ATTENTION_DELTA_ONLY':
                    p[~reliable] = person['fused_probabilities'][~reliable]
            replay = metrics(sigmoid(base), y)['mAP'] - metrics(person['fused_probabilities'], y)['mAP']
            replay_error = max(replay_error, abs(replay))
            assert abs(replay) < .005, ('sigmoid rounding floor', split, task, replay)
            controls.append(dict(split=split, task=task, sigmoid_replay_AP_delta=replay,
                all_swap_vs_dual_AP_delta=metrics(options['ALL_SWAP_1'], y)['mAP']-metrics(options['DUAL'],y)['mAP'],
                all_swap_vs_dual_logit_abs_max=float(abs(all_swap-dual['fused_logits']).max())))
            for name, p in options.items():
                scores.setdefault(name, []).append(metrics(p, y))
                stages.append(dict(split=split, method=name, task=task, **scores[name][-1]))
                if name.startswith('FACE_SWAP') or name == 'FACE_ATTENTION_DELTA_ONLY':
                    assert np.array_equal(p[~reliable], person['fused_probabilities'][~reliable]), (split,task,name)
                if name in ('PERSON','FACE_SWAP_1','PERSON_SWAP_1','PERSON_FACE_SWAP_1','DUAL'):
                    cached[split,name,task] = (p, y, person['sample_ids'])
                if task == 7:
                    for group, mask in (('reliable',reliable),('unreliable',~reliable)):
                        subgroups.append(dict(split=split,method=name,subset=group,people=int(mask.sum()),**metrics(p[mask],y[mask])))
                    delta=ap(p,y)-ap(person['fused_probabilities'],y)
                    for i, cls in enumerate(person['class_indices']):
                        class_changes.append(dict(split=split,method=name,class_index=int(cls),AP_delta=float(delta[i])))
        for name, rows in scores.items():
            summaries.append(dict(split=split,method=name,final_mAP=rows[-1]['mAP'],
                average_mAP=float(np.mean([r['mAP'] for r in rows])),final_cF1=rows[-1]['cF1'],final_oF1=rows[-1]['oF1']))
    for name, rows in (('summary',summaries),('stages',stages),('subgroups',subgroups),
                       ('class_changes',class_changes),('numerical_controls',controls)):
        table(args.output/(name+'.csv'),rows)
    provenance=dict(status='complete',training=False,model_execution=False,seed=1,
        baseline='Person-only co-trained representation with stage-matched B0 head',
        donor='Person+reliable Face co-trained representation with stage-matched B0 head',
        standard='samples_intersect_seen_classes',source_files=source_hashes,
        source_files_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in source_hashes.items()),
        primary_comparison='FACE_SWAP_1-PERSON',exploratory_grid=[0,.25,.5,.75,1],
        formal_model_validation_required=True,max_sigmoid_replay_AP_error=replay_error,
        note='Logit-combination diagnostic; AMP feature-fusion/head rounding can prevent exact full-swap replay.')
    assert provenance['source_files_unchanged']
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('SUMMARY',json.dumps(summaries),flush=True)
    print('NUMERICAL_CONTROLS',json.dumps(controls),flush=True)
    bootstrap={}
    for split in ('val','test'):
        _, _, ids=cached[split,'PERSON',7]
        groups=np.unique([str(x).rsplit('#person=',1)[0] for x in ids]);stage_groups=[]
        for task in range(8):
            names=np.array([str(x).rsplit('#person=',1)[0] for x in cached[split,'PERSON',task][2]])
            stage_groups.append([np.flatnonzero(names==g) for g in groups])
        compared=('FACE_SWAP_1','PERSON_SWAP_1','PERSON_FACE_SWAP_1')
        gen=np.random.default_rng(20261026+(split=='test'))
        final={name:[] for name in compared};average={name:[] for name in compared}
        for iteration in range(args.bootstrap):
            draw=gen.integers(0,len(groups),len(groups));deltas={name:[] for name in compared}
            for task in range(8):
                ix=np.concatenate([stage_groups[task][i] for i in draw])
                old,y,_=cached[split,'PERSON',task];baseline=float(ap(old[ix],y[ix]).mean())
                for name in compared:
                    new=cached[split,name,task][0]
                    deltas[name].append(float(ap(new[ix],y[ix]).mean())-baseline)
            for name in compared:
                final[name].append(deltas[name][-1]);average[name].append(float(np.mean(deltas[name])))
        bootstrap[split]=dict(repeats=args.bootstrap,unit='original_image',stages_coupled=True,
            training_seed_variance_estimated=False,selection_adjusted=False,
            final_ci95={name:np.percentile(values,[2.5,97.5]).tolist() for name,values in final.items()},
            average_ci95={name:np.percentile(values,[2.5,97.5]).tolist() for name,values in average.items()})
        print('BOOTSTRAP',split,json.dumps(bootstrap[split]),flush=True)
    (args.output/'bootstrap.json').write_text(json.dumps(bootstrap,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=500)
    main(p.parse_args())
