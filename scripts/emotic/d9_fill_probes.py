"""Complete the held-TRAIN task6 and task0/3/6 rows on the equal-update 90% trajectory.

Only feature extraction uses CUDA. The existing, unchanged probe objectives run on
CPU; these results are current-head diagnostics, never eight-stage summaries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from multi_lane.track_a.d9_joint_data import build_joint_source
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import set_seed, task_indices
from scripts.emotic import d9_aux_budget_fit as budget
from scripts.emotic import d9_aux_budget_metrics as budget_metrics
from scripts.emotic import d9_disagreement_fit as disagreement
from scripts.emotic import d9_disagreement_metrics as disagreement_metrics
from scripts.emotic import d9_quality_fit as quality
from scripts.emotic import d9_quality_eval as quality_metrics
from scripts.emotic import d9_tcp_probe as tcp
from scripts.emotic.d9_aux_budget_run import summarize as summarize_budget
from scripts.emotic.d9_disagreement_run import summarize as summarize_disagreement
from scripts.emotic.d9_quality_extract import collect as collect_pairs
from scripts.emotic.d9_quality_run import load_pairs, summarize as summarize_quality
from scripts.emotic.d9_sample_training import fusion_source, global_image_roles, image_key
from scripts.emotic.evaluate_d9_relation import write_json
from scripts.emotic.extract_d9_task10_probe import extract as extract_native, train_loader
from scripts.emotic.fit_d9_task10_probe import PROJECTION_SEED, load_arrays
from scripts.emotic.run_d9_relation_training import load_evaluation_model
from scripts.emotic.run_r1_frozen_cross_view import input_paths


TASKS = (0, 3, 6)


def read(path):
    return json.loads(Path(path).read_text())


def require_source(source):
    cfg = read(source / 'config.json')
    expected = dict(mode='sample_holdout', seed=1, smoke=False, tasks=8,
                    epochs=30, batch_size=64, protect_branches=True,
                    image_split='global_10pct_hash_rank')
    if any(cfg.get(k) != v for k, v in expected.items()) or not cfg.get('equal_updates'):
        raise ValueError('This supplement requires the protected equal-update original-random 90% model')
    return cfg


def check_train_population(data, cohort, roles, classes):
    ids, y, valid = data['sample_ids'], data['targets'], data['reliable']
    images = {image_key(person) for person in ids}
    digest = hashlib.sha256('\n'.join(ids.tolist()).encode()).hexdigest()
    measured = dict(people=len(y), images=len(images), classes=list(classes),
                    sample_id_sha256=digest, reliable_face=int(valid.sum()),
                    positives=y.sum(0).astype(int).tolist(),
                    reliable_face_positives=y[valid].sum(0).astype(int).tolist())
    if (any(cohort['fusion'][k] != v for k, v in measured.items()) or
            not y.any(axis=1).all() or any(roles.get(i) != 'fusion' for i in images)):
        raise ValueError('Probe TRAIN must be precisely the permanent held original-image cohort')
    return digest


def load_natural(cache, source, task, split):
    require_source(source)
    path, ident = tcp.identify_source(cache, task, split, 'sample_holdout')
    meta = read(path.with_suffix('.json'))
    spec = meta['protocol']
    checkpoint = source / 'compact_checkpoints' / f'task{task}.pth'
    if (ident['checkpoint_sha256'] != sha256_file(checkpoint) or
            ident['config_sha256'] != sha256_file(source / 'config.json') or
            spec['batch_size'] != 64 or spec['schema'] != 2):
        raise ValueError('Natural feature cache belongs to a different model or numerical protocol')
    data = load_arrays(path, task)
    id_hash = hashlib.sha256('\n'.join(data['sample_ids'].tolist()).encode()).hexdigest()
    if (meta['people'] != len(data['sample_ids']) or meta['sample_id_sha256'] != id_hash or
            meta['reliable_face'] != int(data['reliable'].sum())):
        raise ValueError('Natural feature metadata differs from its arrays')
    if split == 'train':
        ref = source / 'cohorts' / f'task{task}.json'
        check_train_population(data, read(ref), read(source / 'train_image_roles.json'), task_indices(task))
    else:
        ref = source / 'evaluation' / split / f'task{task}.npz'
        reference_meta = read(ref.with_suffix('.json'))
        if reference_meta['scores_sha256'] != sha256_file(ref):
            raise ValueError('Native evaluation archive was modified')
        expected = dict(task=task, split=split, smoke=False, batch_size=64,
                        checkpoint_sha256=ident['checkpoint_sha256'], config_sha256=ident['config_sha256'])
        if any(reference_meta['protocol'].get(k) != v for k, v in expected.items()):
            raise ValueError('Native evaluation provenance differs')
        with np.load(ref, allow_pickle=False) as native, np.load(path, allow_pickle=False) as raw:
            for key in ('sample_ids', 'reliable', 'targets', 'branch_logits', 'fixed_logits', 'equal_logits'):
                actual = native[key]
                if key not in ('sample_ids', 'reliable'):
                    actual = actual[:, task_indices(task)]
                np.testing.assert_array_equal(raw[key], actual, err_msg=f'{task}/{split}/{key}')
    if spec['source_cohort_or_archive_sha256'] != sha256_file(ref):
        raise ValueError('Feature cache source cohort/archive hash changed')
    ident.update(task=task, split=split, people=len(data['sample_ids']), sample_id_sha256=id_hash,
                 roles_sha256=sha256_file(source / 'train_image_roles.json'),
                 reference_sha256=sha256_file(ref), source_mode='sample_holdout',
                 equal_updates=True, branch_train_exposure='never in this original-image split')
    return data, ident


def extract_pairs(source, cache, out, tasks, workers, smoke=False):
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    set_seed(1, True)
    train = build_joint_source(input_paths(), 'train')
    roles = global_image_roles(train, 'sample_holdout', 1)
    if roles != read(source / 'train_image_roles.json'):
        raise ValueError('Original TRAIN partition changed')
    train = fusion_source(train)
    out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        natural, ident = load_natural(cache, source, task, 'train')
        spec = dict(task=task, split='train', smoke=smoke, checkpoint_sha256=ident['checkpoint_sha256'],
                    natural_sha256=ident['sha256'], config_sha256=ident['config_sha256'], schema=1,
                    input_size=224, reduced_size=112, projection_seed=PROJECTION_SEED, batch_size=64,
                    source_mode='sample_holdout', equal_updates=True)
        views = []
        for view in ('person', 'face'):
            path = out / f'task{task}_{view}.npz'
            manifest = path.with_suffix('.json')
            if path.exists() and manifest.exists():
                old = read(manifest)
                if any(old.get(k) != v for k, v in dict(spec, view=view).items()):
                    raise ValueError(f'Incompatible existing pair manifest: {path}')
                if old['archive_sha256'] != sha256_file(path):
                    raise ValueError(f'Changed paired archive: {path}')
            else:
                views.append(view)
        if not views:
            print(f'task{task}: verified existing quality pairs', flush=True)
            continue
        model = load_evaluation_model(source, task).requires_grad_(False).eval()
        loader = train_loader(source, train, roles, task, workers, 64)
        for view, arrays in collect_pairs(model, loader, natural, views, smoke):
            path = out / f'task{task}_{view}.npz'
            temp = path.with_suffix(f'.{os.getpid()}.tmp')
            with temp.open('wb') as f:
                np.savez_compressed(f, **arrays)
            temp.replace(path)
            write_json(path.with_suffix('.json'), dict(spec, view=view, people=len(arrays['sample_ids']),
                       original_replayed=True, archive_sha256=sha256_file(path)))
            print(f'task{task} {view}: {len(arrays["sample_ids"])} verified held TRAIN pairs', flush=True)
        del model


def run_readout(source, cache, pairs, out, family, phase):
    """Reuse the historical fit/eval functions without weakening their numerical or data checks."""
    families = {
        'quality': (quality, quality_metrics, summarize_quality),
        'budget': (budget, budget_metrics, summarize_budget),
        'disagreement': (disagreement, disagreement_metrics, summarize_disagreement),
    }
    fit_module, metrics_module, summarize = families[family]
    torch.set_num_threads(1)
    torch.manual_seed(1)
    out.mkdir(parents=True, exist_ok=True)
    seal = out / 'validation_seal.json'
    training = phase == 'fit-val'
    split = 'val' if training else 'test'
    protocol = dict(schema=1, source_config_sha256=sha256_file(source / 'config.json'),
                    tasks=list(TASKS), family=family, source_mode='sample_holdout', equal_updates=True,
                    fit_data='permanent 10% TRAIN original-image holdout',
                    metric='mean of task0/3/6 current-head mAP; not eight-stage final or average',
                    historical_objective_and_budget_unchanged=True)
    sources = {str(t): load_natural(cache, source, t, 'train')[1] for t in TASKS}
    lock = None
    if seal.exists():
        lock = read(seal)
        if lock['protocol'] != protocol or lock['train_sources'] != sources:
            raise ValueError('Sealed source or protocol changed')
        for name, digest in lock['artifacts'].items():
            if sha256_file(out / name) != digest:
                raise ValueError(f'Sealed fit/Validation artifact changed: {name}')
        if family == 'quality':
            for task in TASKS:
                natural, identity = load_natural(cache, source, task, 'train')
                _, pair_ids = load_pairs(pairs, task, natural, identity)
                if pair_ids != lock['identities'][str(task)]['pairs']:
                    raise ValueError('Sealed quality training pairs changed')
        if training:
            print(f'{family}: reused verified Validation seal', flush=True)
            return
    elif not training:
        raise ValueError('Validation must be sealed before Test')
    files, identities, reports = {}, {}, []
    for task in TASKS:
        model_file = out / f'task{task}_model.json'
        pair_ids = None
        if training:
            natural, tid = load_natural(cache, source, task, 'train')
            if family == 'quality':
                variants, pair_ids = load_pairs(pairs, task, natural, tid)
                model = fit_module.fit_task(natural, variants)
            else:
                model = fit_module.fit_task(natural)
            write_json(model_file, model)
            train_report = metrics_module.evaluate(natural, fit_module.apply_task(natural, model), task)
            train_file = out / f'task{task}_train.json'
            write_json(train_file, train_report)
            for path in (model_file, train_file):
                files[path.name] = sha256_file(path)
        else:
            model = read(model_file)
            tid = lock['identities'][str(task)]['train']
            pair_ids = lock['identities'][str(task)].get('pairs')
        data, identity = load_natural(cache, source, task, split)
        if any(tid[k] != identity[k] for k in ('config_sha256', 'checkpoint_sha256')):
            raise ValueError('Train/evaluation model mismatch')
        predictions = fit_module.apply_task(data, model)
        report = metrics_module.evaluate(data, predictions, task)
        files_path = out / f'task{task}_{split}.json'
        write_json(files_path, report)
        arr_path = out / f'task{task}_{split}_outputs.npz'
        np.savez_compressed(arr_path, sample_ids=data['sample_ids'], reliable=data['reliable'], **predictions)
        for path in (files_path, arr_path):
            files[path.name] = sha256_file(path)
        identities[str(task)] = dict(train=tid, evaluation=identity, pairs=pair_ids)
        reports.append(report)
        print(f'{family} {split} task{task}: {report["macro"]}', flush=True)
    summary = summarize(reports)
    payload = dict(protocol=protocol, train_sources=sources, identities=identities,
                   artifacts=files, split=split, summary=summary)
    if not training:
        payload.update(validation_seal_sha256=sha256_file(seal),
                       eligible_on_validation=lock['summary']['eligible'])
    write_json(seal if training else out / 'test_results.json', payload)
    print(f'{family} {split}: {summary}', flush=True)


def smoke_fit(source, out, workers):
    """Exercise all four unchanged reader families on real task6 images, outside formal outputs."""
    natural = load_arrays(out / 'features' / 'task6' / 'train.npz', 6)
    val = load_arrays(out / 'features' / 'task6' / 'val.npz', 6)
    train = build_joint_source(input_paths(), 'train')
    roles = global_image_roles(train, 'sample_holdout', 1)
    loader = train_loader(source, fusion_source(train), roles, 6, workers, 64)
    model = load_evaluation_model(source, 6).requires_grad_(False).eval()
    variants = dict(collect_pairs(model, loader, natural, ('person', 'face'), True))
    del model
    torch.set_num_threads(1)
    tensors = tcp.input_tensors(natural)
    tcp_models = {m: tcp.fit(m, *tensors[:4], 5) for m in tcp.METHODS[1:]}
    tcp_report = tcp.evaluate(6, val, tcp_models, tcp.input_tensors(val, tensors[-1]), ())
    quality_model = quality.fit_task(natural, variants)
    quality_report = quality_metrics.evaluate(val, quality.apply_task(val, quality_model), 6, permutations=1)
    budget_model = budget.fit_task(natural)
    weights = budget.apply_task(val, budget_model)
    np.testing.assert_array_equal(weights['person'][:, 0], weights['group'][:, 0])
    np.testing.assert_array_equal(weights['person'][~val['reliable']], weights['group'][~val['reliable']])
    budget_report = budget_metrics.evaluate(val, weights, 6, permutations=1)
    disagreement_model = disagreement.fit_task(natural)
    disagreement_report = disagreement_metrics.evaluate(
        val, disagreement.apply_task(val, disagreement_model), 6, permutations=1)
    write_json(out / 'smoke_results.json', dict(
        task=6, train_people=len(natural['sample_ids']), val_people=len(val['sample_ids']),
        tcp={m: r['metrics']['all']['mAP'] for m, r in tcp_report.items()},
        quality=quality_report['macro'], budget=budget_report['macro'],
        disagreement=disagreement_report['macro'], formal_effect_claim=False))
    print('Real task6 feature, quality intervention, four-reader smoke passed', flush=True)


def consolidate(out):
    paths = [out / 'tcp' / 'tcp_test_results.json'] + [
        out / f / 'test_results.json' for f in ('quality', 'budget', 'disagreement')]
    if not all(p.exists() for p in paths):
        return
    sources = {}
    rows = {}
    for family in ('tcp', 'quality', 'budget', 'disagreement'):
        folder = out / family
        val_path = folder / ('tcp_validation_lock.json' if family == 'tcp' else 'validation_seal.json')
        test_path = folder / ('tcp_test_results.json' if family == 'tcp' else 'test_results.json')
        val, test = read(val_path), read(test_path)
        if test['validation_lock_sha256' if family == 'tcp' else 'validation_seal_sha256'] != sha256_file(val_path):
            raise ValueError('Consolidation found an inconsistent Validation/Test seal')
        key = 'macro_current_head_mAP' if family == 'tcp' else 'task_macro_ap'
        rows[family] = dict(validation=val['summary'][key], test=test['summary'][key],
                            tasks=[6] if family == 'tcp' else list(TASKS))
        sources[family] = dict(validation_sha256=sha256_file(val_path), test_sha256=sha256_file(test_path))
    write_json(out / 'results.json', dict(
        source='protected original-random equal-update 90% branches',
        fusion_training_data='permanent held 10% TRAIN original images',
        evaluation_scope='tcp: task6 current classes; others: task0/3/6 current-head mean; NOT eight-stage metrics',
        rows=rows, sources=sources))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--phase', choices=('smoke', 'extract', 'fit-val', 'test'), required=True)
    p.add_argument('--splits', nargs='+', choices=('train', 'val', 'test'), default=['train', 'val'])
    p.add_argument('--tasks', nargs='+', type=int, choices=TASKS, default=list(TASKS))
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--family', choices=('all', 'tcp', 'quality', 'budget', 'disagreement'), default='all')
    args = p.parse_args()
    require_source(args.source)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    cache = args.out
    if args.phase in ('smoke', 'extract'):
        smoke = args.phase == 'smoke'
        target = cache
        extract_native(SimpleNamespace(source=args.source, source_mode='sample_holdout', output=target,
                       tasks=[6] if smoke else args.tasks, splits=['train', 'val'] if smoke else args.splits,
                       workers=args.workers, batch_size=64, smoke=smoke))
        if smoke:
            smoke_fit(args.source, args.out, args.workers)
        elif 'train' in args.splits:
            extract_pairs(args.source, cache, args.out / 'pairs', args.tasks, args.workers)
        return
    if args.family in ('all', 'tcp'):
        for split in (('train', 'val') if args.phase == 'fit-val' else ('train', 'val', 'test')):
            load_natural(cache, args.source, 6, split)
        tcp.run(cache, args.out / 'tcp', args.phase, [6], 300, 100, source_mode='sample_holdout')
    for family in ('quality', 'budget', 'disagreement'):
        if args.family in ('all', family):
            run_readout(args.source, cache, args.out / 'pairs', args.out / family, family, args.phase)
    if args.phase == 'test':
        consolidate(args.out)


if __name__ == '__main__':
    main()
