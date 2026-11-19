"""Fit per-class disagreement readers on TRAIN; seal Validation before evaluating Test."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_disagreement_fit import apply_task, fit_task
from scripts.emotic.d9_disagreement_metrics import evaluate
from scripts.emotic.d9_disagreement_source import load_source
from scripts.emotic.evaluate_d9_relation import write_json

TASKS = (0, 3, 6)
CONTROLS = ('fixed', 'all', 'score', 'shuffle_mean')


def read(path):
    return json.loads(path.read_text())


def summarize(reports):
    aps = [r['macro']['ap'] for r in reports]
    macros = {m: float(np.mean([a[m] for a in aps])) for m in (*CONTROLS, 'middle')}
    diffs = {m: [a['middle'] - a[m] for a in aps] for m in CONTROLS}
    cross_task = {m: np.mean(v) > 0 and sum(x > 0 for x in v) >= 2 for m,v in diffs.items()}
    row = reports[-1]['classes']['Sadness']
    ap = row['ap']
    counts = row['top50']
    auc = row['candidate']['all']['auc_middle']
    have_support = counts['rescued_by_all'] > 0 and counts['false_alarms_by_all'] > 0
    top = (2 * counts['rescued_retained'] >= counts['rescued_by_all'] and
           counts['false_alarms_prevented'] >= 1) if have_support else None
    gates = dict(candidate_auc_above_half=auc is not None and auc > .5,
                 sadness_beats_fixed_and_all=ap['middle'] > max(ap['fixed'],ap['all']),
                 top50_retention_and_prevention=top)
    return dict(task_macro_ap=macros, middle_minus_control_by_task=diffs,
                cross_task_gates={k:bool(v) for k,v in cross_task.items()}, sadness_gates=gates,
                eligible=all(cross_task.values()) and all(x is True for x in gates.values()))


def run(cache, cache6, source, out, phase):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    seal = out/'validation_seal.json'
    files = {}
    reports = []
    identities = {}
    training = phase == 'fit-val'
    split = 'val' if training else 'test'
    if not training or seal.exists():
        lock = read(seal)
        for filename, digest in lock['artifacts'].items():
            if sha256_file(out/filename) != digest:
                raise ValueError(f'Sealed artifact differs: {filename}')
        if training:
            print('Reused verified Validation seal', flush=True)
            return
    for task in TASKS:
        root = cache6 if task == 6 else cache
        model_path = out/f'task{task}_model.json'
        if training:
            data, train_id = load_source(root, source, task,'train')
            model = fit_task(data)
            train_report = evaluate(data,apply_task(data,model),task)
            train_path = out/f'task{task}_train.json'
            write_json(train_path,train_report)
            write_json(model_path,model)
            for p in (train_path,model_path):
                files[p.name] = sha256_file(p)
        else:
            model = read(model_path)
        data, ident = load_source(root,source,task,split)
        fit_id = train_id if training else lock['identities'][str(task)]['train']
        if any(fit_id[k] != ident[k] for k in ('config_sha256','checkpoint_sha256')):
            raise ValueError('Split and TRAIN checkpoint/config mismatch')
        identities[str(task)] = {split:ident, 'train':fit_id}
        pred = apply_task(data,model)
        report = evaluate(data,pred,task)
        report_path = out/f'task{task}_{split}.json'
        write_json(report_path,report)
        people = out/f'task{task}_{split}_decisions.npz'
        np.savez_compressed(people,ids=data['sample_ids'],targets=data['targets'],reliable=data['reliable'],**pred)
        for p in (report_path,people):
            files[p.name] = sha256_file(p)
        reports.append(report)
        print(split, task, report['macro']['ap'], flush=True)
    summary = summarize(reports)
    result = dict(seed=1, split=split, tasks=list(TASKS), summary=summary,
                  artifacts=files, identities=identities)
    if not training:
        result.update(validation_seal_sha256=sha256_file(seal), qualified_on_validation=lock['summary']['eligible'])
    write_json(seal if training else out/'test_results.json',result)
    print(split,summary,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('cache','cache6','source','out'):
        parser.add_argument('--'+arg,required=True,type=Path)
    parser.add_argument('--phase',required=True,choices=('fit-val','test'))
    run(**vars(parser.parse_args()))
