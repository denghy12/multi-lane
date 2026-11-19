"""Fit on matched TRAIN examples and score only untouched Validation and Test images."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_aux_budget_run import verify_fixed_report
from scripts.emotic.d9_disagreement_source import load_source
from scripts.emotic.d9_quality_eval import evaluate
from scripts.emotic.d9_quality_fit import apply_task, fit_task
from scripts.emotic.evaluate_d9_relation import write_json


TASKS = (0, 3, 6)
METHODS = ('fixed', 'A', 'B', 'C', 'shuffle', 'mean_C')


def read(path):
    return json.loads(path.read_text())


def load_pairs(folder, task, train, identity):
    variants, records = {}, {}
    for view in ('person', 'face'):
        path = folder / f'task{task}_{view}.npz'
        meta = read(path.with_suffix('.json'))
        checks = dict(task=task, view=view, split='train', smoke=False,
                      checkpoint_sha256=identity['checkpoint_sha256'],
                      natural_sha256=identity['sha256'], archive_sha256=sha256_file(path))
        if any(meta.get(k) != v for k, v in checks.items()):
            raise ValueError(f'Incorrect paired TRAIN provenance: {path}')
        with np.load(path, allow_pickle=False) as paired:
            for k in ('sample_ids', 'reliable', 'targets'):
                np.testing.assert_array_equal(paired[k], train[k], err_msg=str(path))
            variants[view] = {k: paired[k].copy() for k in ('branch_logits', 'relation', 'content_proj')}
        z = variants[view]['branch_logits']
        if z.shape != train['branch_logits'].shape:
            raise ValueError('Paired current head shape mismatch')
        np.testing.assert_array_equal(z[..., 0], train['branch_logits'][..., 0])
        if view == 'face':
            rows = ~train['reliable']
            np.testing.assert_array_equal(z[rows], train['branch_logits'][rows])
        records[view] = checks
    return variants, records


def summarize(reports):
    aps = {m: np.array([t['macro'][m] for t in reports]) for m in METHODS}
    deltas = {m: aps['C'] - a for m, a in aps.items() if m != 'C'}
    criteria = {m: bool(deltas[m].mean() > 0 and (deltas[m] > 0).sum() >= 2)
                for m in ('A', 'B', 'shuffle')}
    criteria['fixed'] = bool(deltas['fixed'].mean() >= 0)
    return dict(task_macro_ap={m: float(a.mean()) for m, a in aps.items()},
                C_minus_by_task={m: a.tolist() for m, a in deltas.items()},
                criteria=criteria, eligible=all(criteria.values()))


def run(cache, cache6, source, pairs, out, phase):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    out.mkdir(parents=True, exist_ok=True)
    seal = out / 'validation_seal.json'
    training = phase == 'fit-val'
    split = 'val' if training else 'test'
    if not training or seal.exists():
        lock = read(seal)
        for name, expected in lock['artifacts'].items():
            if sha256_file(out / name) != expected:
                raise ValueError(f'Sealed artifact changed: {name}')
        if training:
            print('Reused verified Validation seal')
            return
    files, identities, reports = {}, {}, []
    for task in TASKS:
        root = cache6 if task == 6 else cache
        model_file = out / f'task{task}_model.json'
        if training:
            cohort, tid = load_source(root, source, task, 'train')
            variants, variants_id = load_pairs(pairs, task, cohort, tid)
            model = fit_task(cohort, variants)
            write_json(model_file, model)
            train_report = evaluate(cohort, apply_task(cohort, model), task)
            verify_fixed_report(cache.parent / 'readout', task, 'train', train_report)
            train_file = out / f'task{task}_train.json'
            write_json(train_file, train_report)
            for path in (model_file, train_file):
                files[path.name] = sha256_file(path)
            del variants, cohort
        else:
            model = read(model_file)
            train_identity = lock['identities'][str(task)]
            tid, variants_id = train_identity['train'], train_identity['pairs']
        cohort, evaluation_id = load_source(root, source, task, split)
        if any(evaluation_id[k] != tid[k] for k in ('config_sha256', 'checkpoint_sha256')):
            raise ValueError('Mismatched TRAIN and evaluation model')
        weights = apply_task(cohort, model)
        report = evaluate(cohort, weights, task)
        verify_fixed_report(cache.parent / 'readout', task, split, report)
        identities[str(task)] = dict(train=tid, pairs=variants_id, evaluation=evaluation_id)
        report_file = out / f'task{task}_{split}.json'
        write_json(report_file, report)
        people_file = out / f'task{task}_{split}_weights.npz'
        np.savez_compressed(people_file, sample_ids=cohort['sample_ids'], reliable=cohort['reliable'],
                            **weights)
        for path in (report_file, people_file):
            files[path.name] = sha256_file(path)
        reports.append(report)
        print(split, task, report['macro'], flush=True)
    summary = summarize(reports)
    record = dict(seed=1, split=split, tasks=list(TASKS), identities=identities, artifacts=files,
                  summary=summary)
    if not training:
        record.update(qualified_on_validation=lock['summary']['eligible'],
                      validation_seal_sha256=sha256_file(seal))
    write_json(seal if training else out / 'test_results.json', record)
    print(split, summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'cache6', 'source', 'pairs', 'out'):
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=('fit-val', 'test'))
    run(**vars(parser.parse_args()))
