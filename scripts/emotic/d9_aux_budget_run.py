"""Compare task and individual auxiliary allocations on protected Full-model predictions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_aux_budget_fit import fit_task, apply_task
from scripts.emotic.d9_aux_budget_metrics import evaluate
from scripts.emotic.d9_disagreement_source import load_source
from scripts.emotic.evaluate_d9_relation import write_json


TASKS = (0, 3, 6)


def read(path):
    return json.loads(path.read_text())


def summarize(reports):
    methods = ('fixed', 'group', 'person', 'shuffle', 'mean_person')
    task_ap = {m: np.array([r['macro'][m] for r in reports]) for m in methods}
    deltas = {m: task_ap['person'] - task_ap[m] for m in methods if m != 'person'}
    gates = {m: bool(d.mean() > 0 and np.sum(d > 0) >= 2)
             for m, d in deltas.items() if m != 'fixed'}
    gates['fixed'] = bool(deltas['fixed'].mean() >= 0)
    return {'task_macro_ap': {m: float(a.mean()) for m, a in task_ap.items()},
            'person_minus_control_by_task': {m: d.tolist() for m, d in deltas.items()},
            'criteria': gates, 'eligible': all(gates.values())}


def verify_fixed_report(reference_dir, task, split, report):
    prior = read(reference_dir / f'task{task}_{split}.json')
    np.testing.assert_allclose(report['macro']['fixed'], prior['macro']['ap']['fixed'],
                               rtol=0, atol=1e-8)


def smoke(cache6, source):
    data, _ = load_source(cache6, source, 6, 'train')
    chosen = np.concatenate([np.flatnonzero(data['reliable'] == flag)[:32] for flag in (False, True)])
    small = {key: values[chosen] for key, values in data.items()}
    for row in fit_and_predict_smoke(small):
        assert np.isfinite(row).all()
    print('Passed real task6 smoke on', len(chosen), 'TRAIN people', flush=True)


def fit_and_predict_smoke(data):
    model = fit_task(data)
    outputs = apply_task(data, model)
    np.testing.assert_array_equal(outputs['group'][:, 0], outputs['person'][:, 0])
    valid = data['reliable']
    np.testing.assert_array_equal(outputs['group'][~valid], outputs['person'][~valid])
    for method in ('group', 'person'):
        assert (outputs[method][:, 0] >= 0.5).all()
    return outputs.values()


def run(cache, cache6, source, out, phase):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    if phase == 'smoke':
        smoke(cache6, source)
        return
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
    ref = cache.parent / 'readout'
    for task in TASKS:
        root = cache6 if task == 6 else cache
        model_file = out / f'task{task}_model.json'
        if training:
            train, tid = load_source(root, source, task, 'train')
            model = fit_task(train)
            write_json(model_file, model)
            training_report = evaluate(train, apply_task(train, model), task)
            verify_fixed_report(ref, task, 'train', training_report)
            train_file = out / f'task{task}_train.json'
            write_json(train_file, training_report)
            for path in (train_file, model_file):
                files[path.name] = sha256_file(path)
            del train
        else:
            model = read(model_file)
            tid = lock['identities'][str(task)]['train']
        cohort, identity = load_source(root, source, task, split)
        if any(tid[k] != identity[k] for k in ('config_sha256', 'checkpoint_sha256')):
            raise ValueError('Mismatched TRAIN and evaluation model')
        weights = apply_task(cohort, model)
        report = evaluate(cohort, weights, task)
        verify_fixed_report(ref, task, split, report)
        identities[str(task)] = {'train': tid, split: identity}
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
    record = {'seed': 1, 'split': split, 'tasks': list(TASKS), 'summary': summary,
              'artifacts': files, 'identities': identities}
    if not training:
        record.update(validation_seal_sha256=sha256_file(seal),
                      qualified_on_validation=lock['summary']['eligible'])
    write_json(seal if training else out / 'test_results.json', record)
    print(split, summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('cache', 'cache6', 'source', 'out'):
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--phase', choices=('smoke', 'fit-val', 'test'), required=True)
    run(**vars(parser.parse_args()))
