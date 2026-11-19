"""Determine whether a matched TRAIN holdout transfers; lock conditional readouts before Test."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import CLASS_ORDER, task_indices
from scripts.emotic.d9_aux_budget_metrics import _class_ap
from scripts.emotic.d9_matched_dynamic import apply_dynamic, fit_dynamic
from scripts.emotic.d9_quality_eval import evaluate
from scripts.emotic.d9_quality_fit import _fit_one
from scripts.emotic.d9_quality_run import summarize as summarize_dynamic
from scripts.emotic.d9_sample_training import image_key
from scripts.emotic.d9_tcp_probe import identify_source
from scripts.emotic.evaluate_d9_relation import write_json
from scripts.emotic.fit_d9_task10_probe import apply_rule, load_arrays, mix, weights as masked_weights


TASKS = (0, 3, 6)
GROUPS = ('seen', 'held')


def read(path):
    return json.loads(path.read_text())


def load_train_archives(pairs, source, task):
    expected = dict(split='train', smoke=False, task=task,
                    checkpoint_sha256=sha256_file(source/'compact_checkpoints'/f'task{task}.pth'),
                    config_sha256=sha256_file(source/'config.json'),
                    roles_sha256=sha256_file(source/'train_image_roles.json'),
                    cohort_sha256=sha256_file(source/'cohorts'/f'task{task}.json'),
                    match_sha256=sha256_file(pairs/f'task{task}_match.json'))
    roles = read(source/'train_image_roles.json')
    by_group, ids = {}, {}
    for group in GROUPS:
        p = pairs/f'task{task}_{group}.npz'
        spec = read(p.with_suffix('.json'))
        expected.update(group=group, archive_sha256=sha256_file(p))
        if any(spec.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Incorrect TRAIN pair identity {p}')
        with np.load(p, allow_pickle=False) as a:
            natural = {k: a[k] for k in ('sample_ids','reliable','targets','branch_logits','content_proj','relation')}
            variants = {view: {k: a[f'{view}_{k}'] for k in ('branch_logits','content_proj','relation')}
                        for view in ('person','face')}
        n = len(natural['sample_ids'])
        if spec['people'] != n or len(np.unique(natural['sample_ids'])) != n:
            raise ValueError('TRAIN row identity/count mismatch')
        group_role = 'fusion' if group == 'held' else 'branch'
        if any(roles.get(image_key(i)) != group_role for i in natural['sample_ids']):
            raise ValueError('Wrong TRAIN image role')
        for view, variant in variants.items():
            np.testing.assert_array_equal(natural['branch_logits'][...,0], variant['branch_logits'][...,0])
            if view == 'face':
                invalid = ~natural['reliable']
                for key in ('branch_logits','relation','content_proj'):
                    np.testing.assert_allclose(variant[key][invalid], natural[key][invalid], atol=1e-10, rtol=0)
        by_group[group] = (natural,variants)
        ids[group] = spec['archive_sha256']
    seen,held = [by_group[g][0] for g in GROUPS]
    if len(seen['targets']) != len(held['targets']) or seen['reliable'].sum() != held['reliable'].sum():
        raise ValueError('Comparison does not match group counts')
    if {image_key(v) for v in seen['sample_ids']} & {image_key(v) for v in held['sample_ids']}:
        raise ValueError('Original image overlaps both cohorts')
    ids.update(match=expected['match_sha256'],checkpoint=expected['checkpoint_sha256'],config=expected['config_sha256'])
    return by_group, ids


def fit_average(natural, variants):
    valid = torch.from_numpy(natural['reliable'])
    z = torch.from_numpy(np.stack([d['branch_logits'] for d in (natural,variants['person'],variants['face'])], axis=1)).float()
    y = torch.from_numpy(natural['targets']).float().unsqueeze(1)
    usable = torch.stack([torch.ones_like(valid),torch.ones_like(valid),valid],dim=1)
    # The group-constant version of the exact paired-quality objective has no feature columns.
    dummy = torch.zeros((len(y),3,0))
    params,diag,_ = _fit_one(dummy,z,y,valid,usable,'A',[])
    w = masked_weights(torch.tensor(params['bias']), torch.tensor([False,True])).tolist()
    return dict(weights=w, params=params, fit=diag)


def natural_eval(prior, source, task, split, ids):
    p,who = identify_source(prior,task,split,source_mode='sample_holdout')
    if who['checkpoint_sha256'] != ids['checkpoint'] or who['config_sha256'] != ids['config']:
        raise ValueError('Fitting and evaluation checkpoint/config mismatch')
    native = source/'evaluation'/split/f'task{task}.npz'
    archive_sha = sha256_file(native)
    if (read(native.with_suffix('.json'))['scores_sha256'] != archive_sha or
            read(p.with_suffix('.json'))['protocol']['source_cohort_or_archive_sha256'] != archive_sha):
        raise ValueError('Native evaluation archive mismatch')
    with np.load(native, allow_pickle=False) as orig, np.load(p, allow_pickle=False) as cached:
        for key in ('sample_ids','reliable','targets','branch_logits','fixed_logits'):
            ref = orig[key]
            if key not in ('sample_ids','reliable'):
                ref = ref[:,task_indices(task)]
            np.testing.assert_array_equal(cached[key],ref)
    return load_arrays(p,task),dict(**who,archive_sha256=archive_sha)


def center_rows(a,r):
    return np.asarray(a['weights'],dtype=np.float32)[r.astype(int)]


def evaluate_centers(data, centers, task):
    r,y = data['reliable'],data['targets']
    branches = torch.from_numpy(data['branch_logits']).float()
    fixed = mix(apply_rule('fixed',data,None,None),branches)
    if 'fixed_logits' in data:
        source_fixed = torch.from_numpy(data['fixed_logits']).float()
        torch.testing.assert_close(fixed,source_fixed,atol=2e-5,rtol=2e-6)
        fixed = source_fixed
    logits = {'fixed':fixed}
    for g in GROUPS:
        logits[g] = mix(torch.from_numpy(center_rows(centers[g],r)), branches)
    names = [CLASS_ORDER[c] for c in task_indices(task)]
    result = {}
    for g,mask in (('all',np.ones_like(r)),('reliable',r),('unreliable',~r)):
        aps = {k:_class_ap(z[mask],y[mask]) for k,z in logits.items()}
        result[g] = dict(people=int(mask.sum()), positives=y[mask].sum(0).astype(int).tolist(),
                         macro={k:float(a.mean()) for k,a in aps.items()},
                         classes={c:{k:float(a[i]) for k,a in aps.items()} for i,c in enumerate(names)})
    return result


def summarize_centers(rows):
    scores = {m:np.array([r['all']['macro'][m] for r in rows]) for m in ('fixed','seen','held')}
    d = scores['held'] - scores['seen']
    return dict(macro={m:float(v.mean()) for m,v in scores.items()}, held_minus_seen=d.tolist(),
                held_minus_fixed=(scores['held']-scores['fixed']).tolist(),
                transfers=bool(d.mean()>0 and (d>0).sum()>=2))


def run(source,prior,pairs,out,phase):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    out.mkdir(parents=True,exist_ok=True)
    seal = out/'validation_seal.json'
    training = phase == 'fit-val'
    split = 'val' if training else 'test'
    if not training or seal.exists():
        lock = read(seal)
        for name,h in lock['artifacts'].items():
            if sha256_file(out/name) != h:
                raise ValueError(f'Sealed file changed: {name}')
        if training:
            return
    files, identities, trainsets, evalsets, center_models, a_reports = {}, {}, {}, {}, {}, []
    def save(name, payload):
        write_json(out/name,payload)
        files[name] = sha256_file(out/name)
    for t in TASKS:
        name = f'task{t}_centers.json'
        if training:
            trainsets[t],ids = load_train_archives(pairs,source,t)
            centers = {g:fit_average(*trainsets[t][g]) for g in GROUPS}
            save(name,centers)
            save(f'task{t}_train_replay.json',{g:evaluate_centers(trainsets[t][g][0],centers,t) for g in GROUPS})
        else:
            centers = read(out/name)
            ids = lock['identities'][str(t)]['train']
        data,eid = natural_eval(prior,source,t,split,ids)
        ar = evaluate_centers(data,centers,t)
        identities[str(t)] = dict(train=ids,evaluation=eid)
        save(f'task{t}_{split}_A.json',ar)
        evalsets[t],center_models[t] = data,centers['held']
        a_reports.append(ar)
    summary = summarize_centers(a_reports)
    advance = summary['transfers'] if training else lock['stage1']['transfers']
    dyn_summary = {}
    if advance:
        reports = {'cohort':[],'train':[]}
        for t in TASKS:
            model_name = f'task{t}_dynamic.json'
            if training:
                model = fit_dynamic(*trainsets[t]['held'], center_models[t]['weights'])
                save(model_name,model)
            else:
                model = read(out/model_name)
            data = evalsets[t]
            for mode, rows in reports.items():
                w = apply_dynamic(data,model,center_mode=mode)
                w.update(fixed=apply_rule('fixed',data,None,None).numpy(),A=center_rows(center_models[t],data['reliable']))
                if mode == 'cohort':
                    for flag in (False,True):
                        mask = data['reliable']==flag
                        for k in ('B','C'):
                            np.testing.assert_allclose(w[k][mask].mean(0,dtype=np.float64),w['A'][mask][0],
                                                       rtol=0,atol=3e-6)
                result = evaluate(data,w,t)
                save(f'task{t}_{split}_{mode}.json',result)
                pp = out/f'task{t}_{split}_{mode}_weights.npz'
                np.savez_compressed(pp,sample_ids=data['sample_ids'],reliable=data['reliable'],**w)
                files[pp.name] = sha256_file(pp)
                rows.append(result)
        dyn_summary = {m:summarize_dynamic(v) for m,v in reports.items()}
    record = dict(tasks=list(TASKS),split=split,seed=1,stage1=summary,dynamic_ran=bool(advance),
                  stage2=dyn_summary,identities=identities,artifacts=files)
    if not training:
        record['validation_seal_sha256'] = sha256_file(seal)
    write_json(seal if training else out/'test_results.json',record)
    print(split,summary,dyn_summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('source','prior','pairs','out'):
        parser.add_argument('--'+k,required=True,type=Path)
    parser.add_argument('--phase',choices=('fit-val','test'),required=True)
    run(**vars(parser.parse_args()))
