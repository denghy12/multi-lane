"""Fit matched sample-level readouts with and without true-label confidence supervision."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_direction_permutation import FORMAL_SEEDS, difference_summary, permutation_sources
from scripts.emotic.evaluate_d9_relation import write_json
from scripts.emotic.fit_d9_task10_probe import (
    LR, PROJECTION_SEED, apply_rule, build_features, group_weights, load_arrays, mix, scores,
)

TASKS = (0, 3, 6)
METHODS = ('fixed', 'A', 'B_bce', 'C_tcp')
DYNAMIC = ('B_bce', 'C_tcp')
VIEWS = ('full', 'person', 'face')


def read(path):
    return json.loads(path.read_text())


def true_label_confidence(logits, targets):
    """For multilabel data use each view's mean P(observed binary label) over current classes."""
    return ((2 * targets.detach() - 1).unsqueeze(-1) * logits.detach()).sigmoid().mean(1)


def active_views(reliable):
    return torch.stack((torch.ones_like(reliable), torch.ones_like(reliable), reliable), dim=1)


def gate_scores(params, content, reliable):
    raw = params['bias'][reliable.long()]
    if 'coefficient' in params:
        raw = raw + F.linear(content.detach(), params['coefficient'])
    weights = raw.masked_fill(~active_views(reliable), -torch.inf).softmax(-1)
    return raw, weights


def fit(method, content, reliable, logits, targets, steps):
    if method not in METHODS[1:]:
        raise ValueError(f'Unknown learned readout {method}')
    params = {'bias': torch.zeros(2, 3, requires_grad=True)}
    if method in DYNAMIC:
        params['coefficient'] = torch.zeros(3, content.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam(params.values(), lr=LR)
    z, y = logits.detach(), targets.detach()
    truth = true_label_confidence(z, y)
    active = active_views(reliable)
    for _ in range(steps):
        raw, w = gate_scores(params, content, reliable)
        loss = F.binary_cross_entropy_with_logits(mix(w, z), y)
        if method == 'C_tcp':
            loss = loss + F.mse_loss(raw[active].sigmoid(), truth[active])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite fusion loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if not all(torch.isfinite(v).all() for v in params.values()):
        raise FloatingPointError('Nonfinite fusion parameters')
    return {k: v.detach().tolist() for k, v in params.items()}


def quality_report(raw, reliable, z, y):
    truth = true_label_confidence(z, y)
    prediction = raw.sigmoid()
    mask = active_views(reliable)
    details = {}
    for v, view in enumerate(VIEWS):
        rows = mask[:, v]
        if not rows.any():
            continue
        pred, actual = prediction[rows, v], truth[rows, v]
        corr = None
        if len(pred) > 1 and pred.min() != pred.max() and actual.min() != actual.max():
            corr = float(np.corrcoef(pred.numpy(), actual.numpy())[0, 1])
        details[view] = dict(n=int(rows.sum()), prediction_mean=float(pred.mean()),
                             target_mean=float(actual.mean()), mse=float(F.mse_loss(pred, actual)),
                             pearson_across_people=corr)
    return dict(mse=float(F.mse_loss(prediction[mask], truth[mask])), by_view=details)


def input_tensors(data, spec=None):
    features, _, spec = build_features(data, spec)
    return (torch.from_numpy(features['C']), torch.from_numpy(data['reliable']),
            torch.from_numpy(data['branch_logits']).float(), torch.from_numpy(data['targets']).float(), spec)


def evaluate(task, data, models, tensors, seeds, val_means=None):
    content, reliable, z, y, _ = tensors
    masks = {'all': np.ones(len(y), dtype=bool), 'reliable': data['reliable'],
             'unreliable': ~data['reliable']}
    donors = [permutation_sources(data['sample_ids'], data['reliable'], task + 1, seed)[:, task]
              for seed in seeds]
    rows = {}
    for method in METHODS:
        if method == 'fixed':
            w = apply_rule('fixed', data, None, None)
            row = {}
        else:
            params = {k: torch.tensor(v, dtype=torch.float32) for k, v in models[method].items()}
            raw, w = gate_scores(params, content, reliable)
            row = {'quality': quality_report(raw, reliable, z, y)}
        fused = mix(w, z)
        if method == 'fixed' and 'fixed_logits' in data:
            archived = torch.as_tensor(data['fixed_logits'], dtype=torch.float32)
            torch.testing.assert_close(fused, archived, atol=2e-5, rtol=2e-6)
            fused = archived
        row['metrics'] = {group: scores(fused[rows_], data['targets'][rows_])
                          for group, rows_ in masks.items()}
        row['group_weights'] = group_weights(w, data['reliable'])
        if method in DYNAMIC and seeds:
            means = row['group_weights'] if val_means is None else val_means[method]
            mean_w = w.clone()
            for group in (0, 1):
                selected = reliable == group
                if selected.any():
                    if means[group] is None:
                        raise ValueError('Missing locked validation availability group')
                    mean_w[selected] = torch.tensor(means[group])
            actual = row['metrics']['all']['mAP']
            row['group_mean_mAP'] = scores(mix(mean_w, z), data['targets'])['mAP']
            row['actual_minus_group_mean'] = actual - row['group_mean_mAP']
            shuffled = [scores(mix(w[idx], z), data['targets'])['mAP'] for idx in donors]
            row['weight_shuffle'] = difference_summary(actual, shuffled)
        rows[method] = row
    return rows


def summarize(rows):
    macro = {m: float(np.mean([r[m]['metrics']['all']['mAP'] for r in rows])) for m in METHODS}
    tcp = [r['C_tcp']['metrics']['all']['mAP'] for r in rows]
    gaps = {m: [actual - r[m]['metrics']['all']['mAP'] for actual, r in zip(tcp, rows)]
            for m in ('A', 'B_bce')}
    gaps['own_group_mean'] = [r['C_tcp']['actual_minus_group_mean'] for r in rows]
    gaps['person_shuffle'] = [r['C_tcp']['weight_shuffle']['mean'] for r in rows]
    passes = all(np.mean(d) > 0 and sum(x > 0 for x in d) > len(rows) / 2 for d in gaps.values())
    return dict(macro_current_head_mAP=macro, tcp_paired_differences=gaps,
                tcp_passes=bool(passes))


def identify_source(prior, task, split, source_mode='sample_holdout'):
    path = prior/'features'/f'task{task}'/f'{split}.npz'
    digest = sha256_file(path)
    meta = read(path.with_suffix('.json'))
    spec = meta['protocol']
    if (spec.get('source_mode', 'sample_holdout') != source_mode or
            spec['task'] != task or spec['split'] != split or spec['smoke'] or
            digest != meta['features_sha256'] or not meta['visibility_mask_applied'] or
            (split != 'train' and not meta['exact_eval_replay'])):
        raise ValueError(f'Unverified source archive {path}')
    return path, dict(sha256=digest, checkpoint_sha256=spec['checkpoint_sha256'],
                      config_sha256=spec['source_config_sha256'])


def run(prior, out, phase, tasks, steps, permutations, source_mode='sample_holdout'):
    if source_mode not in ('sample_holdout', 'sample_full'):
        raise ValueError(f'Unsupported branch training data mode {source_mode}')
    if (not tasks or len(set(tasks)) != len(tasks) or any(t not in TASKS for t in tasks) or
            steps < 1 or not 1 <= permutations <= len(FORMAL_SEEDS)):
        raise ValueError('Invalid tasks, step count, or permutation count')
    torch.set_num_threads(1)
    out.mkdir(exist_ok=True, parents=True)
    cfg = dict(version=1, tasks=list(tasks), seed=1, steps=steps, optimizer='full-batch Adam', lr=LR,
               features='12 centered and TRAIN-standardized Task10 content coordinates, 4 per view',
               projection_seed=PROJECTION_SEED, dynamic_shared_across_classes=True, tcp_mse_coefficient=1.,
               tcp_target='mean binary true-label probability across current classes per person and view',
               fit_data='current task 10% TRAIN original-image holdout', initialization='zero, available views equal',
               permutations=permutations, methods=list(METHODS),
               criterion='C minus B, A, own group mean, and within-group shuffle each mean>0 and majority>0')
    if source_mode == 'sample_full':
        cfg.update(source_mode=source_mode, fit_data='all current-task eligible TRAIN people also used by the branch protocol')
    lock_path = out/'tcp_validation_lock.json'
    seeds = FORMAL_SEEDS[:permutations]
    if phase == 'fit-val':
        sources = {str(t): {s: identify_source(prior, t, s, source_mode)[1] for s in ('train', 'val')} for t in tasks}
        if lock_path.exists():
            lock = read(lock_path)
            if (cfg != lock['protocol'] or lock['train_val_sources'] != sources or
                    any(sha256_file(out/k) != h for k, h in lock['artifact_hashes'].items())):
                raise ValueError('Incompatible or damaged Validation lock')
            print('Reused verified Validation lock', flush=True)
            return
        artifacts, rows = {}, []
        for task in tasks:
            origin = sources[str(task)]
            for key in ('checkpoint_sha256', 'config_sha256'):
                if origin['train'][key] != origin['val'][key]:
                    raise ValueError(f'Different TRAIN/Validation source at task {task}')
            directory = prior/'features'/f'task{task}'
            train = load_arrays(directory/'train.npz', task)
            tensors = input_tensors(train)
            models = {m: fit(m, *tensors[:4], steps) for m in METHODS[1:]}
            model = dict(models=models, feature_stats=tensors[-1], task=task)
            train_scores = evaluate(task, train, models, tensors, ())
            val = load_arrays(directory/'val.npz', task)
            row = evaluate(task, val, models, input_tensors(val, tensors[-1]), seeds)
            model['val_means'] = {m: row[m]['group_weights'] for m in DYNAMIC}
            for filename, payload in ((f'tcp_model_task{task}.json', model),
                                      (f'tcp_train_task{task}.json', train_scores),
                                      (f'tcp_val_task{task}.json', row)):
                write_json(out/filename, payload)
                artifacts[filename] = sha256_file(out/filename)
            rows.append(row)
        summary = summarize(rows)
        write_json(lock_path, dict(protocol=cfg, selection_split='val', summary=summary,
                                   train_val_sources=sources, artifact_hashes=artifacts))
        print('Validation', summary, flush=True)
        return
    if phase != 'test':
        raise ValueError(f'Unknown phase {phase}')
    lock = read(lock_path)
    if (lock['protocol'] != cfg or
            any(sha256_file(out/k) != h for k, h in lock['artifact_hashes'].items())):
        raise ValueError('Invalid Validation lock for Test')
    rows, sources = [], {}
    for task in tasks:
        path, src = identify_source(prior, task, 'test', source_mode)
        origin = lock['train_val_sources'][str(task)]['val']
        if any(src[key] != origin[key] for key in ('checkpoint_sha256', 'config_sha256')):
            raise ValueError(f'Different Test source at task {task}')
        sources[str(task)] = src
        model = read(out/f'tcp_model_task{task}.json')
        data = load_arrays(path, task)
        tensors = input_tensors(data, model['feature_stats'])
        row = evaluate(task, data, model['models'], tensors, seeds, model['val_means'])
        rows.append(row)
        write_json(out/f'tcp_test_task{task}.json', row)
    summary = summarize(rows)
    write_json(out/'tcp_test_results.json',
               dict(protocol=cfg, summary=summary, passed_on_validation=lock['summary']['tcp_passes'],
                    test_sources=sources, validation_lock_sha256=sha256_file(lock_path)))
    print('Test with sealed readouts', summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--source-mode', choices=('sample_holdout', 'sample_full'), default='sample_holdout')
    parser.add_argument('--phase', choices=('fit-val', 'test'), required=True)
    parser.add_argument('--tasks', nargs='+', type=int, default=list(TASKS))
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--permutations', type=int, default=100)
    args = parser.parse_args()
    run(args.prior, args.out, args.phase, args.tasks, args.steps, args.permutations, args.source_mode)
