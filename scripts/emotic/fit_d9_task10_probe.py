"""Fit paired, current-head Task10 fusion probes with protected checkpoints and a Validation lock."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, average_precision
from scripts.emotic.d9_direction_permutation import FORMAL_SEEDS, difference_summary, permutation_sources
from scripts.emotic.evaluate_d9_relation import write_json

TRAINED = ('B_raw', 'B_std', 'C', 'D')
METHODS = ('fixed', 'equal', 'A', *TRAINED)
LR = .01
STD_FLOOR = 1e-6
PROJECTION_SEED = 20261118


def load_arrays(path, task):
    with np.load(path, allow_pickle=False) as archive:
        d = {k: archive[k] for k in ('sample_ids', 'targets', 'reliable', 'branch_logits',
                                     'relation', 'content', 'delta')}
        for name in ('fixed_logits', 'equal_logits'):
            if name in archive:
                d[name] = archive[name]
    ids = d['sample_ids']
    n, classes = len(ids), TASK_SIZES[task]
    if (not n or ids.shape != (n,) or len(np.unique(ids)) != n or
            d['reliable'].shape != (n,) or not np.isin(d['reliable'], [False, True]).all()):
        raise ValueError(f'Missing/duplicate people or invalid Face availability: {path}')
    d['reliable'] = d['reliable'].astype(bool)
    if (d['branch_logits'].shape != (n, classes, 3) or d['targets'].shape != (n, classes)
            or not np.isin(d['targets'], [0, 1]).all()
            or d['relation'].shape != (n, 8) or d['content'].ndim != 3
            or d['content'].shape[:2] != (n, 3) or d['delta'].shape != (n, 2, d['content'].shape[-1])):
        raise ValueError(f'Unexpected current-head feature shapes or target values: {path}')
    invalid = ~d['reliable']
    d['branch_logits'][invalid, :, 2] = 0.
    d['relation'][invalid, 4:] = 0.
    d['content'][invalid, 2] = 0.
    d['delta'][invalid, 1] = 0.
    if not all(np.isfinite(d[k]).all() for k in ('branch_logits', 'relation', 'content', 'delta')):
        raise ValueError(f'Nonfinite active logits or features: {path}')
    return d


def project(d):
    width = d['content'].shape[-1]
    rng = np.random.default_rng(PROJECTION_SEED)
    # Independent directions per branch; Person/Face messages use the same axes as their own CLS.
    matrix = rng.standard_normal((3, width, 4)) / np.sqrt(width)
    content = np.einsum('nvd,vdk->nvk', d['content'], matrix).reshape(len(d['content']), 12)
    message = np.einsum('nvd,vdk->nvk', d['delta'], matrix[1:, :, 2:]).reshape(len(content), 4)
    return {'relation': d['relation'], 'content': content, 'message': message}


def stats(x, reliable, face_start):
    x = np.asarray(x, dtype=np.float64)
    mean = np.stack([x[reliable == flag].mean(axis=0) for flag in (False, True)])
    residual = x - mean[reliable.astype(int)]
    sd = np.sqrt((residual ** 2).mean(axis=0))
    sd[face_start:] = np.sqrt((residual[reliable, face_start:] ** 2).mean(axis=0))
    return {'mean': mean.tolist(), 'sd': sd.tolist(), 'face_start': face_start,
            'floored_columns': np.flatnonzero(sd <= STD_FLOOR).tolist()}


def normalized(x, reliable, spec):
    sd = np.asarray(spec['sd'])
    centered = x - np.asarray(spec['mean'])[reliable.astype(int)]
    result = np.divide(centered, sd, where=sd > STD_FLOOR, out=np.zeros_like(centered))
    result[~reliable, spec['face_start']:] = 0.
    return result.astype('float32')


def combine_message(content, message):
    combined = content.copy()
    combined[:, [6, 7, 10, 11]] += message
    return combined


def build_features(d, saved_stats=None):
    projected = project(d)
    reliable = d['reliable']
    if saved_stats is None:
        if not reliable.any() or reliable.all():
            raise ValueError('Both Face availability groups must exist in the fitting set')
        saved_stats = {k: stats(projected[k], reliable, i)
                       for k, i in (('relation', 4), ('content', 8), ('message', 2))}
    norm = {k: normalized(x, reliable, saved_stats[k]) for k, x in projected.items()}
    features = {'B_raw': projected['relation'].astype('float32'), 'B_std': norm['relation'],
                'C': norm['content'], 'D': combine_message(norm['content'], norm['message'])}
    return features, norm['message'], saved_stats


def weights(base, reliable, x=None, matrix=None):
    scores = base[reliable.long()]
    if x is not None:
        scores = scores + x @ matrix.T
    return torch.stack((scores[:, 0], scores[:, 1], scores[:, 2].masked_fill(~reliable, -torch.inf)), 1).softmax(-1)


def mix(w, z):
    return w[:, 0:1] * z[:, :, 0] + w[:, 1:2] * z[:, :, 1] + w[:, 2:3] * z[:, :, 2]


def fit_one(d, inputs, steps, base=None):
    reliable = torch.as_tensor(d['reliable'])
    z = torch.as_tensor(d['branch_logits'], dtype=torch.float32)
    y = torch.as_tensor(d['targets'], dtype=torch.float32)
    is_center = base is None
    param = torch.zeros((2, 3) if is_center else (3, inputs.shape[1]), requires_grad=True)
    center = param if is_center else torch.as_tensor(base, dtype=torch.float32)
    x = None if is_center else torch.as_tensor(inputs)
    opt = torch.optim.Adam([param], lr=LR)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(mix(weights(center, reliable, x, param), z), y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite probe fitting BCE')
        opt.zero_grad()
        loss.backward()
        opt.step()
    return param.detach().numpy().tolist()


def apply_rule(name, d, features, model, override=None):
    r = torch.from_numpy(d['reliable'])
    if name == 'fixed':
        return torch.where(r[:, None], torch.tensor([[.64,.16,.20]]), torch.tensor([[.8,.2,0.]]))
    if name == 'equal':
        return weights(torch.zeros((2, 3)), r)
    b = torch.tensor(model['center'])
    if name == 'A':
        return weights(b, r)
    x = features[name] if override is None else override
    return weights(b, r, torch.from_numpy(x), torch.tensor(model['matrices'][name]))


def scores(logits, target):
    if len(target) == 0:
        return None
    prob = logits.sigmoid().numpy()
    aps = [100. * average_precision(p, y) for p, y in zip(prob.T, target.T)]
    return {'mAP': float(np.mean(aps)), 'per_class_ap': [float(x) for x in aps],
            'bce': float(F.binary_cross_entropy_with_logits(logits, torch.as_tensor(target, dtype=torch.float32))),
            'people': len(target), 'positive_counts': target.sum(0).astype(int).tolist()}


def group_weights(w, reliable):
    return [w[reliable == flag].mean(0).tolist() if (reliable == flag).any() else None
            for flag in (False, True)]


def evaluation(d, model, task, permutations, val_means=None):
    features, message, _ = build_features(d, model['feature_stats'])
    r, ids, y = d['reliable'], d['sample_ids'], d['targets']
    z = torch.as_tensor(d['branch_logits'], dtype=torch.float32)
    donors = [permutation_sources(ids, r, task + 1, seed)[:, task] for seed in FORMAL_SEEDS[:permutations]]
    result = {}
    for name in METHODS:
        w = apply_rule(name, d, features, model)
        logits = mix(w, z)
        if name in ('fixed', 'equal') and name + '_logits' in d:
            reference = torch.as_tensor(d[name + '_logits'], dtype=torch.float32)
            torch.testing.assert_close(logits, reference, atol=2e-5, rtol=2e-6)
            logits = reference
        by_group = {key: scores(logits[mask], y[mask])
                    for key, mask in (('all', np.ones(len(y), bool)), ('reliable', r), ('unreliable', ~r))}
        means = group_weights(w, r)
        means_for_readout = means if val_means is None else val_means[name]
        assigned = w.clone()
        for g in (0, 1):
            if (r == g).any():
                if means_for_readout[g] is None:
                    raise ValueError('Missing validation group for locked-mean comparison')
                assigned[r == g] = torch.tensor(means_for_readout[g])
        # For truly constant groups, avoid float32 reductions turning identical scores into new ties.
        if name in ('fixed', 'equal', 'A'):
            assigned = w
        mean_map = by_group['all']['mAP'] if name in ('fixed', 'equal', 'A') else scores(mix(assigned, z), y)['mAP']
        row = dict(metrics=by_group, group_weights=means, group_mean_map=mean_map,
                   actual_minus_group_mean=by_group['all']['mAP'] - mean_map)
        if name in TRAINED:
            exchanged = [scores(mix(w[donor], z), y)['mAP'] for donor in donors]
            row['weight_shuffle'] = difference_summary(by_group['all']['mAP'], exchanged)
        if name == 'D':
            mismatch = []
            for donor in donors:
                wrong = combine_message(features['C'], message[donor])
                wrong_w = apply_rule(name, d, features, model, wrong)
                mismatch.append(scores(mix(wrong_w, z), y)['mAP'])
            row['message_shuffle'] = difference_summary(by_group['all']['mAP'], mismatch)
        result[name] = row
    return result


def selection(rows):
    macro = {name: float(np.mean([r[name]['metrics']['all']['mAP'] for r in rows])) for name in METHODS}
    decisions = {}
    for name in TRAINED:
        tests = {'vs_A': [r[name]['metrics']['all']['mAP'] - r['A']['metrics']['all']['mAP'] for r in rows],
                 'vs_group_mean': [r[name]['actual_minus_group_mean'] for r in rows],
                 'vs_weight_shuffle': [r[name]['weight_shuffle']['mean'] for r in rows]}
        if name == 'D':
            tests.update(vs_C=[r[name]['metrics']['all']['mAP'] - r['C']['metrics']['all']['mAP'] for r in rows],
                         vs_message_shuffle=[r[name]['message_shuffle']['mean'] for r in rows])
        decisions[name] = {'deltas': tests, 'eligible': all(np.mean(x) > 0 and
                           sum(v > 0 for v in x) > len(rows) / 2 for x in tests.values())}
    eligible = [n for n, d in decisions.items() if d['eligible']]
    return dict(macro_current_head_mAP=macro, evidence=decisions,
                winner=max(eligible, key=macro.get) if eligible else 'A')


def run(root, phase, tasks, steps, permutations):
    if (not tasks or len(set(tasks)) != len(tasks) or any(t < 0 or t >= len(TASK_SIZES) for t in tasks)
            or steps < 1 or not 1 <= permutations <= len(FORMAL_SEEDS)):
        raise ValueError('Invalid diagnostic task list, step count or permutation count')
    torch.set_num_threads(1)
    out = root / 'probe'
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(version=1, tasks=tasks, steps=steps, LR=LR, permutations=permutations,
               projection_seed=PROJECTION_SEED, dtype='float32', projection_dims=[4,4,4],
               message_coordinates=[6,7,10,11], message_coefficient=1., sd_floor=STD_FLOOR)
    lock_path = out/'validation_lock.json'
    def source(t, split):
        return root/'features'/f'task{t}'/f'{split}.npz'
    if phase == 'fit-val':
        source_hash = {str(t): {s: sha256_file(source(t,s)) for s in ('train','val')} for t in tasks}
        if lock_path.exists():
            lock = json.loads(lock_path.read_text())
            if lock['protocol'] != cfg or lock['train_val_hash'] != source_hash:
                raise ValueError('An incompatible validation lock already exists')
            if any(sha256_file(out/k) != v for k,v in lock['artifact_hashes'].items()):
                raise ValueError('Locked artifacts differ')
            print('Reused verified Validation lock', flush=True)
            return
        rows, artifacts = [], {}
        for task in tasks:
            train = load_arrays(source(task,'train'),task)
            features, _, spec = build_features(train)
            center = fit_one(train,None,steps)
            model = dict(feature_stats=spec,center=center,matrices={name:fit_one(train,features[name],steps,center)
                         for name in TRAINED}, output_classes=TASK_SIZES[task], input_width=train['content'].shape[-1])
            val = load_arrays(source(task,'val'),task)
            rows.append(evaluation(val,model,task,permutations))
            model['val_means'] = {k:v['group_weights'] for k,v in rows[-1].items()}
            for key, value in ((f'task{task}_model.json',model),(f'task{task}_val.json',rows[-1])):
                write_json(out/key,value)
                artifacts[key] = sha256_file(out/key)
        selected = selection(rows)
        write_json(lock_path, dict(protocol=cfg, train_val_hash=source_hash, artifact_hashes=artifacts,
                                   selection=selected, selection_split='val', mean_source='per-stage unlabelled val'))
        print('Locked Validation selection:',selected['winner'], selected['macro_current_head_mAP'],flush=True)
    elif phase == 'test':
        lock = json.loads(lock_path.read_text())
        if lock['protocol'] != cfg or any(sha256_file(out/k) != h for k,h in lock['artifact_hashes'].items()):
            raise ValueError('Validation lock is incompatible with requested test')
        rows, test_hash = [], {}
        for t in tasks:
            model = json.loads((out/f'task{t}_model.json').read_text())
            test = load_arrays(source(t,'test'),t)
            row=evaluation(test,model,t,permutations,model['val_means'])
            write_json(out/f'task{t}_test.json',row)
            rows.append(row)
            test_hash[str(t)]=sha256_file(source(t,'test'))
        decision = lock['selection']['winner']
        summary = {n: float(np.mean([r[n]['metrics']['all']['mAP'] for r in rows])) for n in METHODS}
        write_json(out/'test_results.json',dict(protocol=cfg,selected_on_validation=decision,
                   validation_lock_sha256=sha256_file(lock_path), test_features_sha256=test_hash,
                   macro_current_head_mAP=summary, selected_score=summary[decision]))
        print('Test of locked readouts:',summary,flush=True)
    else:
        raise ValueError('Unknown phase')


if __name__ == '__main__':
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument('--root',type=Path,required=True)
    cmd.add_argument('--phase',choices=('fit-val','test'),required=True)
    cmd.add_argument('--tasks',type=int,nargs='+',default=[0,3,6])
    cmd.add_argument('--steps',type=int,default=300)
    cmd.add_argument('--permutations',type=int,default=100)
    args=cmd.parse_args()
    run(args.root,args.phase,args.tasks,args.steps,args.permutations)
