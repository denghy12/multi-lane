"""Apply sealed TRAIN offsets and test mixed-group ranking on all eight EMOTIC stages."""
import argparse
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import CLASS_ORDER, TASK_SIZES, TaskMetrics, summarize_tasks
from scripts.emotic.d9_a_eight_eval import GROUPS, predictions, read, require_digest
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.evaluate_d9_relation import metrics_for_stage
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS, PROTECTED_BASELINE


METHODS = ('fixed', 'paired', 'offset')


def apply_offsets(logits, reliable, sizes, offsets):
    if (logits.dtype != torch.float32 or logits.ndim != 2 or reliable.dtype != torch.bool
            or reliable.shape != (len(logits),) or len(sizes) != len(offsets)
            or logits.shape[1] != sum(sizes) or not np.isfinite(offsets).all()):
        raise ValueError('Invalid readout or offset vector')
    per_class = torch.from_numpy(np.repeat(np.asarray(offsets, dtype=np.float64), sizes))
    exact = logits.double()
    exact[~reliable] += per_class
    # Match the historical FP32 sigmoid and evaluation precision, keeping the mathematical
    # translation in FP64 for the independent rank check before rounding.
    return exact, exact.float()


def audit_group_ranks(baseline, exact, official, labels, metrics):
    """Check the full weak ordering in score space and measure finite-precision changes."""
    source = baseline.numpy()
    moved = exact.numpy()
    sort_order = np.argsort(source, axis=0, kind='stable')
    def increments(a):
        return np.diff(np.take_along_axis(a, sort_order, axis=0), axis=0)
    if not np.array_equal(np.sign(increments(source)), np.sign(increments(moved))):
        raise ValueError('Constant shift changed FP64 within-group raw ranks or ties')
    before, after = (increments(z.sigmoid().numpy()) for z in (baseline, official))
    differences = np.subtract(metrics['after']['per_class_ap'], metrics['before']['per_class_ap'])
    return dict(people=len(source), raw_order_and_ties_preserved=True,
                fp32_probability_ties_created_adjacent=((before != 0) & (after == 0)).sum(axis=0).tolist(),
                fp32_probability_ties_broken_adjacent=((before == 0) & (after != 0)).sum(axis=0).tolist(),
                fp32_probability_adjacent_direction_reversed=(before*after < 0).sum(axis=0).tolist(),
                per_class_ap_change=dict(zip(labels, differences.tolist())),
                mAP_change=metrics['after']['mAP'] - metrics['before']['mAP'])


def cross_group_accuracy(logits, targets, reliable):
    """Percentage of correctly ordered opposite-status positive/negative pairs, ties half."""
    results = []
    for name, score, target in zip(CLASS_ORDER, logits.T, targets.T):
        pairs = 0
        wins = 0.
        for positive_group in (reliable, ~reliable):
            pos = score[positive_group & (target == 1)]
            neg = np.sort(score[(~positive_group) & (target == 0)])
            pairs += len(pos) * len(neg)
            lower = np.searchsorted(neg, pos, side='left')
            upper = np.searchsorted(neg, pos, side='right')
            wins += .5 * (lower + upper).sum(dtype=np.int64)
        results.append(dict(name=name, pairs=pairs, percent=100.*wins/pairs if pairs else None))
    valid = [r['percent'] for r in results if r['pairs']]
    return dict(percent=float(np.mean(valid)) if valid else None, valid_classes=len(valid), classes=results)


def run(source, previous, out, split):
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=True)
    prior = previous/'readout'
    centers_path = prior/'centers_lock.json'
    offset_path = out/'offsets_lock.json'
    offset_hash = sha256_file(offset_path)
    lock = read(offset_path)
    center_hash = lock['centers_sha256']
    require_digest(centers_path, center_hash)
    old_seal = read(prior/'validation_seal.json')
    if old_seal['centers_sha256'] != center_hash:
        raise ValueError('Original evaluation used different TRAIN group means')
    require_digest(prior/'val_results.json', old_seal['val_results_sha256'])
    centers = read(centers_path)
    for key, path in [('config_sha256','config.json'), ('roles_sha256','train_image_roles.json')]:
        require_digest(source/path, centers[key])
        if centers[key] != lock[key]:
            raise ValueError('Offsets and centers have different source')
    by_task = {k: row['paired'] for k, row in centers['tasks'].items()}
    tasks = {str(k) for k in range(len(TASK_SIZES))}
    if set(lock['tasks']) != tasks or set(by_task) != tasks:
        raise ValueError('Incomplete task lock')
    seal_path = out/'validation_seal.json'
    if split == 'test':
        seal = read(seal_path)
        if seal['offsets_sha256'] != offset_hash:
            raise ValueError('Offset lock differs from Validation')
        require_digest(out/'val_results.json', seal['val_results_sha256'])
    old_path = prior/f'{split}_results.json'
    old = read(old_path)
    if (old['centers_sha256'] != center_hash or old['split'] != split or len(old['stages']) != len(tasks)
            or (split == 'test' and old['validation_seal_sha256'] != sha256_file(prior/'validation_seal.json'))):
        raise ValueError('Historical readout, split or stage count mismatch')
    native = source/'evaluation'/split
    require_digest(native/'result.json', old['native_results_sha256'])
    stages = []
    for t, prev in enumerate(old['stages']):
        if prev['stage'] != t:
            raise ValueError('Non-sequential historical stages')
        path = native/f'task{t}.npz'
        require_digest(path, prev['scores_sha256'])
        torch.set_num_threads(1)
        with np.load(path, allow_pickle=False) as a:
            src, y, reliable = predictions(a, by_task, t)
        if len(y) != prev['people'] or int(reliable.sum()) != prev['reliable_face']:
            raise ValueError('Unexpected evaluation population')
        paired = src['A']
        sizes = TASK_SIZES[:t+1]
        offsets = [lock['tasks'][str(k)]['offset'] for k in range(t+1)]
        exact, official = apply_offsets(paired, reliable, sizes, offsets)
        torch.set_num_threads(2)
        metrics = metrics_for_stage(t, y, reliable, dict(fixed=src['fixed'], paired=paired, offset=official))
        for method in ('fixed','paired'):
            if metrics[method] != prev['metrics'][method]:
                raise ValueError(f'Endpoint replay failed: {split}/task{t}/{method}')
        if metrics['offset']['reliable'] != metrics['paired']['reliable']:
            raise ValueError('Reliable population changed')
        names = CLASS_ORDER[:sum(sizes)]
        checks = {}
        for g, mask in [('reliable', reliable), ('unreliable', ~reliable)]:
            checks[g] = audit_group_ranks(paired[mask], exact[mask], official[mask], names,
                                          dict(before=metrics['paired'][g], after=metrics['offset'][g]))
        accuracy = {name:cross_group_accuracy(z.numpy(), y.numpy(), reliable.numpy())
                    for name,z in [('paired',paired),('offset',exact)]}
        difference = {base:{g:dict(mAP=metrics['offset'][g]['mAP']-metrics[base][g]['mAP'],
                                    per_class=dict(zip(names, np.subtract(metrics['offset'][g]['per_class_ap'],
                                                                          metrics[base][g]['per_class_ap']).tolist())))
                            for g in GROUPS} for base in ('fixed','paired')}
        stages.append(dict(stage=t, people=len(y), reliable_face=int(reliable.sum()), metrics=metrics,
                           rankings=checks, cross_group_accuracy=accuracy, offset_minus=difference,
                           scores_sha256=prev['scores_sha256']))
    summary = {m:{g:summarize_tasks([TaskMetrics(**s['metrics'][m][g]) for s in stages]) for g in GROUPS}
               for m in METHODS}
    for m in ('fixed','paired'):
        if summary[m] != old['summary'][m]:
            raise ValueError('Failed to replay summary')
    refs = {m:summary[m]['all'] for m in ('paired','fixed')}
    refs['protected_full_train_fixed'] = PROTECTED_BASELINE[split]
    changes = {m:{k:summary['offset']['all'][k]-base[k] for k in MAP_KEYS} for m,base in refs.items()}
    report = dict(split=split, offsets_sha256=offset_hash, centers_sha256=center_hash,
                  historical_results_sha256=sha256_file(old_path), stages=stages, summary=summary,
                  offset_minus=changes)
    if split == 'test':
        report['validation_seal_sha256'] = sha256_file(seal_path)
    dest = out/f'{split}_results.json'
    write_verified(dest, report)
    if split == 'val':
        write_verified(seal_path, dict(offsets_sha256=offset_hash, val_results_sha256=sha256_file(dest)))
    print(split, {m:{k:summary[m]['all'][k] for k in MAP_KEYS} for m in METHODS}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('source','previous','out'):
        parser.add_argument('--'+option, type=Path, required=True)
    parser.add_argument('--split', choices=('val','test'), required=True)
    run(**vars(parser.parse_args()))
