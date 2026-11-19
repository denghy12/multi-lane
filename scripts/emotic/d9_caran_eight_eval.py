"""Paired eight-stage evaluation of frozen, TRAIN-learned view readouts."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import CLASS_ORDER, TASK_SIZES, TaskMetrics, summarize_tasks
from scripts.emotic.d9_a_eight_eval import GROUPS, predictions, read, require_digest, verify_controls
from scripts.emotic.d9_direction_permutation import FORMAL_SEEDS, batch_map, difference_summary, permutation_sources
from scripts.emotic.evaluate_d9_relation import VIEWS, metrics_for_stage, write_json
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS, PROTECTED_BASELINE

PERSON = ('person_task', 'person_class')


def combine(weights, logits):
    return (weights[..., 0] * logits[..., 0] + weights[..., 1] * logits[..., 1]
            + weights[..., 2] * logits[..., 2])


def group_means(weights, valid):
    if not valid.any() or valid.all():
        raise ValueError('Both Face availability groups must be represented')
    # Use double only for the across-person mean, then return to native scoring precision.
    return torch.stack([weights[valid == g].double().mean(0).float() for g in (False, True)])


def shuffle_scores(ids, valid, sizes, weights, branches, target, actual, seeds=FORMAL_SEEDS):
    """One donor per person and task, preserving every class and view in that head."""
    n, classes = target.shape
    if list(weights) != list(actual) or sum(sizes) != classes or len(ids) != n:
        raise ValueError('Unaligned shuffling inputs')
    col = np.arange(classes)
    heads = np.repeat(np.arange(len(sizes)), sizes)
    shuffled = {m: [] for m in weights}
    for seed in seeds:
        donors = permutation_sources(ids, valid, len(sizes), seed)[:, heads]
        for m, w in weights.items():
            logits = combine(w[donors, col], branches)
            shuffled[m].append(batch_map(logits[None], target)[0])
    return {m: difference_summary(actual[m], shuffled[m]) for m in weights}


def compare(metrics, lhs, rhs, stage):
    size = TASK_SIZES[stage]
    by_group = {}
    for group in GROUPS:
        delta = np.asarray(metrics[lhs][group]['per_class_ap']) - metrics[rhs][group]['per_class_ap']
        by_group[group] = dict(mAP=float(delta.mean()), current_classes=float(delta[-size:].mean()),
                               old_classes=float(delta[:-size].mean()) if stage else None,
                               per_class=dict(zip(CLASS_ORDER, delta.tolist())))
    return by_group


def load_stage(out, source, prior_a, split, stage, models, lock, native, previous, means):
    from scripts.emotic.d9_caran_eight_fit import METHODS, readout_weights
    name = f'task{stage}'
    sizes = TASK_SIZES[:stage+1]
    cp = lock['tasks'][str(stage)]['checkpoint_sha256']
    archive = source/'evaluation'/split/(name+'.npz')
    native_hash = native['scores_sha256']
    require_digest(archive, native_hash)
    config_hash = sha256_file(source/'config.json')
    if (native['protocol']['checkpoint_sha256'] != cp or native['protocol']['config_sha256'] != config_hash
            or previous['scores_sha256'] != native_hash or previous['checkpoint_sha256'] != cp):
        raise ValueError('TRAIN fit, native stage and previous A refer to different branch outputs')
    feature = out/'features'/name/(split+'.npz')
    meta_path = feature.with_suffix('.json')
    meta = read(meta_path)
    protocol = meta['protocol']
    if (protocol['task'] != stage or protocol['split'] != split or protocol['smoke']
            or not meta['exact_replay'] or not meta['visibility_mask_applied']
            or protocol['checkpoint_sha256'] != cp or protocol['source_config_sha256'] != config_hash
            or protocol['reference_sha256'] != native_hash):
        raise ValueError(f'Unverified feature source: {meta_path}')
    require_digest(feature, meta['features_sha256'])
    centers = read(prior_a/'centers_lock.json')['tasks']
    with np.load(archive, allow_pickle=False) as ref, np.load(feature, allow_pickle=False) as data:
        for key in ('sample_ids', 'reliable', 'targets', 'branch_logits'):
            if not np.array_equal(ref[key], data[key]):
                raise ValueError(f'Stage source differs in {key}')
        ids = ref['sample_ids']
        if (meta['sample_id_sha256'] != hashlib.sha256('\n'.join(ids.tolist()).encode()).hexdigest()
                or meta['people'] != len(ids) or tuple(data['head_ids']) != tuple(range(stage+1))
                or tuple(data['view_order']) != VIEWS):
            raise ValueError('Unexpected IDs, population or feature axes')
        branches = torch.from_numpy(ref['branch_logits'])
        logits, y, valid = predictions(ref, centers, stage)
        cls = torch.from_numpy(data['head_cls'])
    if (cls.ndim != 4 or cls.shape[:3] != (len(ids), stage+1, 3)
            or cls[~valid, :, 2].count_nonzero() or not torch.isfinite(cls).all()):
        raise ValueError('Invalid or unmasked head features')
    w_by_m = {}
    for m in METHODS:
        sum_by_class = []
        for task, size in enumerate(sizes):
            w = readout_weights(models[task][m], cls[:, task], valid, size)
            if w.shape != (len(y), size, 3) or not w.isfinite().all() or w[~valid, :, 2].count_nonzero():
                raise ValueError('Invalid or unmasked view weights')
            torch.testing.assert_close(w.sum(-1), torch.ones_like(w[..., 0]), rtol=0, atol=2e-6)
            sum_by_class.append(w)
        w = torch.cat(sum_by_class, 1)
        z = combine(w, branches)
        if m == 'A':
            torch.testing.assert_close(z, logits[m], rtol=0, atol=0)
        logits[m] = z
        if m in PERSON:
            w_by_m[m] = w
    del cls
    used_means = {}
    for m, w in w_by_m.items():
        mean = group_means(w, valid) if means is None else torch.tensor(means[m], dtype=torch.float32)
        used_means[m] = mean.tolist()
        logits[m+'_group_mean'] = combine(mean[valid.long()], branches)
    metrics = metrics_for_stage(stage, y, valid, logits)
    verify_controls(metrics, native['metrics'])
    for group in GROUPS:
        for key, value in metrics['A'][group].items():
            np.testing.assert_allclose(value, previous['metrics']['A'][group][key], atol=1e-9, rtol=0)
    measured = {m: metrics[m]['all']['mAP'] for m in PERSON}
    for m in PERSON:
        np.testing.assert_allclose(batch_map(logits[m].numpy()[None], y.numpy())[0], measured[m], atol=1e-9, rtol=0)
    shuffles = shuffle_scores(ids, valid.numpy(), sizes, {m:w.numpy() for m,w in w_by_m.items()},
                              branches.numpy(), y.numpy(), measured)
    pairs = [('class_static','A'),('person_task','A'),('person_class','class_static'),
             ('person_class','person_task'),('person_class','fixed')]
    for m in PERSON:
        pairs.append((m, m+'_group_mean'))
    comparisons = {f'{a}_minus_{b}':compare(metrics,a,b,stage) for a,b in pairs}
    # Preserve individual gate predictions for subsequent paired analyses.
    artifact = out/'scores'/split/(name+'.npz')
    artifact.parent.mkdir(parents=True, exist_ok=True)
    tmp = artifact.with_suffix(f'.{os.getpid()}.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, sample_ids=ids, reliable=valid.numpy(), task_sizes=sizes,
                            **{m+'_logits':logits[m].numpy() for m in METHODS},
                            **{m+'_weights':w.numpy() for m,w in w_by_m.items()})
    tmp.replace(artifact)
    return dict(stage=stage, people=len(y), reliable_face=int(valid.sum()), checkpoint_sha256=cp,
                native_scores_sha256=native_hash, features_sha256=meta['features_sha256'],
                predictions_sha256=sha256_file(artifact), metrics=metrics, comparisons=comparisons,
                group_means=used_means, same_group_shuffle=shuffles)


def summarize(stages, split):
    methods = list(stages[0]['metrics'])
    summary = {m:{g:summarize_tasks([TaskMetrics(**row['metrics'][m][g]) for row in stages])
                  for g in GROUPS} for m in methods}
    contrasts = {}
    for a in ('A','class_static','person_task','person_class'):
        for b in ('fixed','equal','A','class_static','person_task','protected_full_train'):
            if a == b:
                continue
            rhs = PROTECTED_BASELINE[split] if b == 'protected_full_train' else summary[b]['all']
            contrasts[f'{a}_minus_{b}'] = {k:summary[a]['all'][k]-rhs[k] for k in MAP_KEYS}
    permutations = {}
    for m in PERSON:
        # Preserve the seed pairing when averaging stages.
        shuffled = np.array([r['same_group_shuffle'][m]['differences'] for r in stages])
        permutations[m] = dict(final_actual_minus_shuffled=shuffled[-1].mean().item(),
                               average_actual_minus_shuffled=shuffled.mean(0).mean().item())
    return dict(summary=summary, contrasts=contrasts, person_pairing=permutations)


def validation_seal(out, fit_hash):
    seal = read(out/'validation_seal.json')
    if seal['caran_fit_lock_sha256'] != fit_hash:
        raise ValueError('Test requires a matching Validation seal')
    require_digest(out/'val_results.json', seal['val_results_sha256'])
    return seal


@torch.inference_mode()
def run(source, prior_a, out, split):
    from scripts.emotic.d9_caran_eight_fit import load_models
    torch.set_num_threads(2)
    lock, models = load_models(out)
    fit_hash = sha256_file(out/'caran_fit_lock.json')
    seal = validation_seal(out, fit_hash) if split == 'test' else None
    val = read(out/'val_results.json') if seal else None
    native_file = source/'evaluation'/split/'result.json'
    native = read(native_file)
    earlier = read(prior_a/f'{split}_results.json')
    require_digest(prior_a/'centers_lock.json', earlier['centers_sha256'])
    require_digest(native_file, earlier['native_result_sha256'])
    if native['smoke'] or native['split'] != split or native['mode'] != 'sample_holdout':
        raise ValueError('Unexpected native evaluation archive')
    stages = []
    for stage in range(len(TASK_SIZES)):
        row = load_stage(out, source, prior_a, split, stage, models, lock,
                         native['stages'][stage], earlier['stages'][stage],
                         val['stages'][stage]['group_means'] if val else None)
        stages.append(row)
        print(split, stage, {m:round(row['metrics'][m]['all']['mAP'],4) for m in row['metrics']}, flush=True)
    report = dict(split=split, caran_fit_lock_sha256=fit_hash, native_result_sha256=sha256_file(native_file),
                  prior_A_result_sha256=sha256_file(prior_a/f'{split}_results.json'), stages=stages,
                  protected_full_train=PROTECTED_BASELINE[split], permutation_seeds=list(FORMAL_SEEDS),
                  **summarize(stages, split))
    if seal:
        report['validation_seal_sha256'] = sha256_file(out/'validation_seal.json')
    dst = out/f'{split}_results.json'
    if dst.exists() and read(dst) != report:
        raise ValueError(f'Incompatible preexisting report: {dst}')
    write_json(dst, report)
    if split == 'val':
        write_json(out/'validation_seal.json', dict(caran_fit_lock_sha256=fit_hash,
                                                     val_results_sha256=sha256_file(dst)))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for option in ('source','prior-a','out'):
        p.add_argument('--'+option,type=Path,required=True)
    p.add_argument('--split',choices=('val','test'),required=True)
    run(**vars(p.parse_args()))
