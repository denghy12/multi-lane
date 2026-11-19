"""Evaluate fixed and paired TRAIN fusion independently for each Face availability group."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, TaskMetrics, summarize_tasks
from scripts.emotic.d9_a_eight_eval import GROUPS, predictions, read, require_digest
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.evaluate_d9_relation import metrics_for_stage
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS

METHODS = ('fixed', 'only_unreliable', 'only_reliable', 'paired')


def group_readouts(fixed, paired, reliable):
    """Select the sealed source's entire logit row for each person."""
    if (reliable.dtype != torch.bool or reliable.ndim != 1 or fixed.shape != paired.shape
            or fixed.ndim != 2 or len(fixed) != len(reliable) or fixed.dtype != paired.dtype):
        raise ValueError('Incompatible logits or Face mask')
    return dict(fixed=fixed, only_unreliable=torch.where(reliable[:, None], fixed, paired),
                only_reliable=torch.where(reliable[:, None], paired, fixed), paired=paired)


def check_metrics(metrics, prior_metrics):
    for m in ('fixed', 'paired'):
        if metrics[m] != prior_metrics[m]:
            raise ValueError(f'Historical control replay failed: {m}')
    for hybrid, switched, untouched in (('only_reliable','reliable','unreliable'),
                                       ('only_unreliable','unreliable','reliable')):
        if (metrics[hybrid][switched] != metrics['paired'][switched]
                or metrics[hybrid][untouched] != metrics['fixed'][untouched]):
            raise ValueError(f'Within-group replay failed: {hybrid}')


def run(source, prior, out, split):
    out.mkdir(parents=True, exist_ok=True)
    old_seal_path = prior / 'validation_seal.json'
    old_seal = read(old_seal_path)
    prior_seal_hash = sha256_file(old_seal_path)
    center_hash = old_seal['centers_sha256']
    require_digest(prior / 'centers_lock.json', center_hash)
    require_digest(prior / 'val_results.json', old_seal['val_results_sha256'])
    lock = read(prior / 'centers_lock.json')
    require_digest(source / 'config.json', lock['config_sha256'])
    require_digest(source / 'train_image_roles.json', lock['roles_sha256'])
    seal_path = out / 'validation_seal.json'
    if split == 'test':
        seal = read(seal_path)
        if seal['centers_sha256'] != center_hash or seal['prior_validation_seal_sha256'] != prior_seal_hash:
            raise ValueError('Test evaluation source differs from locked Validation source')
        require_digest(out / 'val_results.json', seal['val_results_sha256'])
    old_path = prior / f'{split}_results.json'
    old = read(old_path)
    if (old['split'] != split or old['centers_sha256'] != center_hash
            or (split == 'test' and old['validation_seal_sha256'] != prior_seal_hash)
            or len(old['stages']) != len(TASK_SIZES)):
        raise ValueError('Prior split, readouts or stage count mismatch')
    native = source / 'evaluation' / split
    require_digest(native / 'result.json', old['native_results_sha256'])
    centers = {k: row['paired'] for k, row in lock['tasks'].items()}
    stages = []
    for t, prev in enumerate(old['stages']):
        if prev['stage'] != t:
            raise ValueError('Invalid stage ordering')
        path = native / f'task{t}.npz'
        require_digest(path, prev['scores_sha256'])
        torch.set_num_threads(1)
        with np.load(path, allow_pickle=False) as archive:
            original, y, valid = predictions(archive, centers, t)
            ids_hash = hashlib.sha256('\n'.join(archive['sample_ids'].tolist()).encode()).hexdigest()
        if len(valid) != prev['people'] or int(valid.sum()) != prev['reliable_face']:
            raise ValueError(f'Population mismatch at {split}/task{t}')
        readouts = group_readouts(original['fixed'], original['A'], valid)
        # Match CPU sigmoid settings from the archived eight-stage evaluation.
        torch.set_num_threads(2)
        metrics = metrics_for_stage(t, y, valid, readouts)
        check_metrics(metrics, prev['metrics'])
        stages.append(dict(stage=t, people=len(valid), reliable_face=int(valid.sum()),
                           sample_id_sha256=ids_hash, scores_sha256=prev['scores_sha256'], metrics=metrics))
    summary = {m: {g: summarize_tasks([TaskMetrics(**s['metrics'][m][g]) for s in stages])
                   for g in GROUPS} for m in METHODS}
    check_metrics(summary, old['summary'])
    report = dict(split=split, centers_sha256=center_hash, prior_results_sha256=sha256_file(old_path),
                  prior_validation_seal_sha256=prior_seal_hash, native_results_sha256=old['native_results_sha256'],
                  methods=list(METHODS), stages=stages, summary=summary)
    if split == 'test':
        report['validation_seal_sha256'] = sha256_file(seal_path)
    dest = out / f'{split}_results.json'
    write_verified(dest, report)
    if split == 'val':
        write_verified(seal_path, dict(centers_sha256=center_hash, prior_validation_seal_sha256=prior_seal_hash,
                                      val_results_sha256=sha256_file(dest)))
    print(split, {m: {k: summary[m]['all'][k] for k in MAP_KEYS} for m in METHODS}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'prior', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    run(**vars(parser.parse_args()))
