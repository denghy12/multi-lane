"""Paired factorial summaries; no checkpoint updates or model forward passes."""
import csv
import json
from pathlib import Path
import numpy as np
from .module_stacking import ARMS, factorial_contrasts


def read(path):return json.loads(path.read_text())


def map_value(probabilities, targets):
    values=[]
    for j in range(targets.shape[1]):
        positive=targets[probabilities[:,j].argsort()[::-1],j] == 1
        cumulative=positive.cumsum();total=int(cumulative[-1])
        cumulative[~positive]=0
        values.append(float((cumulative/np.arange(1,len(positive)+1)).sum()/(total+1e-8)))
    return 100*float(np.mean(values))


def bootstrap_final(root, repeats=1000):
    cells={};ids=targets=None
    for arm in ARMS:
        with np.load(root/arm/'full_val/scores/task7.npz') as archive:
            if ids is None:ids=archive['sample_ids'].copy();targets=archive['targets'].copy()
            else:
                assert np.array_equal(ids,archive['sample_ids']) and np.array_equal(targets,archive['targets'])
            cells[arm]=archive['probabilities'].copy()
    groups={}
    for i,sample in enumerate(ids):groups.setdefault(str(sample).rsplit('#person=',1)[0],[]).append(i)
    groups=[np.asarray(x) for x in groups.values()];rng=np.random.default_rng(20261022)
    draws={}
    for _ in range(repeats):
        index=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
        values={arm:map_value(p[index],targets[index]) for arm,p in cells.items()}
        contrasts=factorial_contrasts(values)
        contrasts.update({f'{arm}_minus_F000':values[arm]-values['F000'] for arm in ARMS[1:]})
        for name,value in contrasts.items():draws.setdefault(name,[]).append(value)
    point={arm:map_value(p,targets) for arm,p in cells.items()}
    contrasts=factorial_contrasts(point)
    contrasts.update({f'{arm}_minus_F000':point[arm]-point['F000'] for arm in ARMS[1:]})
    return dict(repeats=repeats,unit='original_image_cluster',images=len(groups),people=len(ids),
        seed=20261022,training_seed_replicates=1,split='val',checkpoint=7,
        limitation='Conditional on fixed seed0 models; not training-seed uncertainty or a mechanism proof.',
        contrasts={k:dict(point_pp=contrasts[k],ci95_pp=np.percentile(v,[2.5,97.5]).tolist()) for k,v in draws.items()})


def summarize(root,smoke=False):
    tasks=2 if smoke else 8;rows=[];reference={};values={}
    for arm in ARMS:
        row=dict(arm=arm)
        train=root/arm/'train';training=read(train/'seed_summary.json');config=read(train/'config.json')
        audit=read(train/'paired_protocol_audit.json')
        row.update(training_elapsed_seconds=training['elapsed_seconds'],updates=training['completed_optimizer_updates'],
            peak_allocated_mib=audit['peak_allocated_mib'],peak_reserved_mib=audit['peak_reserved_mib'],
            added_parameters_per_task=(49952*int(arm[1])+196608*int(arm[2])+60*int(arm[3])))
        for folder in ('full_val','replay_val' if smoke else 'full_test'):
            result=read(root/arm/folder/'result.json')
            assert len(result['task_metrics'])==tasks
            if folder!='replay_val':assert result['full_cohort'] and result['cohort_samples']==(5368 if folder=='full_test' else 2397)
            for task in range(tasks):
                path=root/arm/folder/'scores'/f'task{task}.npz'
                with np.load(path) as data:
                    key=(folder,task)
                    if key not in reference:reference[key]=(data['sample_ids'].copy(),data['targets'].copy())
                    else:assert np.array_equal(reference[key][0],data['sample_ids']) and np.array_equal(reference[key][1],data['targets'])
                    calculated=map_value(data['probabilities'],data['targets'])
                    assert abs(calculated-result['task_metrics'][task]['mAP'])<1e-8
                    if folder=='replay_val':
                        with np.load(train/'val_scores'/f'task{task}.npz') as old:
                            for name in ('sample_ids','targets','logits','probabilities'):assert np.array_equal(old[name],data[name]),(arm,task,name)
            for metric in ('final_mAP','average_mAP'):
                name=folder+'_'+metric;row[name]=result['summary'][metric]
                values.setdefault(name,{})[arm]=row[name]
        rows.append(row)
    with (root/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    runtime={}
    for name in ('off_A','off_B','off_C','off_ABC'):
        data=read(root/'F111'/('runtime_'+name)/'result.json')
        runtime[name]=data['task_metrics']
        assert data['split']=='val' and data['module_stacking']
    result=dict(status='complete',smoke=smoke,rows=rows,
        contrasts={name:factorial_contrasts(cells) for name,cells in values.items()},
        runtime_validation=runtime,test_used_for_selection=False,
        eight_cells_complete=True,normal_scores_independently_recomputed=True,
        smoke_archive_replay=smoke,training_seed_replicates=1)
    (root/'interactions.json').write_text(json.dumps(result,indent=2)+'\n')
    if not smoke:
        (root/'bootstrap_validation_final.json').write_text(json.dumps(bootstrap_final(root),indent=2)+'\n')
    return result
