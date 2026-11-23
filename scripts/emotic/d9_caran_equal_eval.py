"""Evaluate the four sealed equal-update CARAN heads on eight paired Val/Test stages."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES
from scripts.emotic.d9_a_eight_eval import predictions, read, require_digest, verify_controls
from scripts.emotic.d9_caran_eight_eval import (
    PERSON, combine, compare, group_means, shuffle_scores, summarize, validation_seal,
)
from scripts.emotic.d9_caran_equal_fit import METHODS, load_models, readout_weights, verify_source
from scripts.emotic.d9_direction_permutation import FORMAL_SEEDS, batch_map
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.evaluate_d9_relation import VIEWS, metrics_for_stage
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS, PROTECTED_BASELINE

FORMAL_ROWS = ('fixed', 'A', 'class_static', 'person_task', 'person_class')


def load_stage(out, source, split, stage, models, lock, native, means, native_val_hash):
    name = f'task{stage}'
    sizes = TASK_SIZES[:stage+1]
    cp = lock['tasks'][str(stage)]['checkpoint_sha256']
    native_path = source/'evaluation'/split/(name+'.npz')
    native_hash = native['scores_sha256']
    require_digest(native_path, native_hash)
    expected_native = dict(task=stage, split=split, smoke=False, batch_size=64,
        config_sha256=lock['source_config_sha256'], checkpoint_sha256=cp,
        validation_result_sha256=native_val_hash if split == 'test' else None)
    if any(native['protocol'].get(k) != v for k, v in expected_native.items()):
        raise ValueError('Native evaluation is not the sealed branch trajectory')
    feature = out/'features'/name/(split+'.npz')
    meta = read(feature.with_suffix('.json'))
    expected_feature = dict(schema=1, task=stage, split=split, smoke=False, batch_size=64,
        checkpoint_sha256=cp, source_config_sha256=lock['source_config_sha256'],
        reference_sha256=native_hash)
    if (meta['protocol'] != expected_feature or not meta['exact_replay']
            or not meta['visibility_mask_applied'] or meta['source_update_audit'] != lock['source_update_audit']):
        raise ValueError(f'Unverified evaluation features: {feature}')
    require_digest(feature, meta['features_sha256'])
    centers = {str(t): dict(center=models[t]['A']['bias'][:, 0].tolist()) for t in range(stage+1)}
    with np.load(native_path, allow_pickle=False) as ref, np.load(feature, allow_pickle=False) as data:
        for key in ('sample_ids', 'reliable', 'targets', 'branch_logits'):
            np.testing.assert_array_equal(ref[key], data[key], err_msg=f'Native replay: {key}')
        ids = ref['sample_ids']
        if (meta['sample_id_sha256'] != hashlib.sha256('\n'.join(ids.tolist()).encode()).hexdigest()
                or meta['people'] != len(ids) or tuple(data['head_ids']) != tuple(range(stage+1))
                or tuple(data['view_order']) != VIEWS):
            raise ValueError('Unexpected evaluation population or feature axes')
        branches = torch.from_numpy(ref['branch_logits'])
        logits, target, valid = predictions(ref, centers, stage)
        cls = torch.from_numpy(data['head_cls'])
    if (cls.shape != (len(ids), stage+1, 3, 512) or not cls.isfinite().all()
            or cls[~valid, :, 2].count_nonzero()):
        raise ValueError('Invalid or unmasked classifier inputs')
    person_weights = {}
    for method in METHODS:
        by_class = []
        for task, size in enumerate(sizes):
            w = readout_weights(models[task][method], cls[:, task], valid, size)
            if w.shape != (len(target), size, 3) or not w.isfinite().all() or w[~valid, :, 2].count_nonzero():
                raise ValueError('Invalid learned view weights')
            torch.testing.assert_close(w.sum(-1), torch.ones_like(w[..., 0]), rtol=0, atol=2e-6)
            by_class.append(w)
        w = torch.cat(by_class, 1)
        result = combine(w, branches)
        if method == 'A':
            # Independently compare the original per-task weighted_logits path.
            torch.testing.assert_close(result, logits['A'], rtol=0, atol=0)
        logits[method] = result
        if method in PERSON:
            person_weights[method] = w
    del cls
    used_means = {}
    for method, w in person_weights.items():
        mean = group_means(w, valid) if means is None else torch.tensor(means[method], dtype=torch.float32)
        if mean.shape != (2, sum(sizes), 3) or not mean.isfinite().all():
            raise ValueError('Invalid sealed group-mean diagnostic')
        used_means[method] = mean.tolist()
        logits[method+'_group_mean'] = combine(mean[valid.long()], branches)
    metrics = metrics_for_stage(stage, target, valid, logits)
    verify_controls(metrics, native['metrics'])
    actual = {method: metrics[method]['all']['mAP'] for method in PERSON}
    for method in PERSON:
        np.testing.assert_allclose(batch_map(logits[method].numpy()[None], target.numpy())[0],
                                   actual[method], rtol=0, atol=1e-9)
    shuffles = shuffle_scores(ids, valid.numpy(), sizes,
        {method: w.numpy() for method, w in person_weights.items()},
        branches.numpy(), target.numpy(), actual)
    pairs = [('class_static', 'A'), ('person_task', 'A'), ('person_class', 'class_static'),
             ('person_class', 'person_task'), ('person_class', 'fixed')]
    pairs.extend((method, method+'_group_mean') for method in PERSON)
    comparisons = {f'{a}_minus_{b}': compare(metrics, a, b, stage) for a, b in pairs}
    artifact = out/'scores'/split/(name+'.npz')
    artifact.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(sample_ids=ids, reliable=valid.numpy(), task_sizes=np.asarray(sizes),
                  **{method+'_logits': logits[method].numpy() for method in (*METHODS, 'fixed', 'equal')},
                  **{method+'_weights': w.numpy() for method, w in person_weights.items()})
    if artifact.exists():
        with np.load(artifact, allow_pickle=False) as previous:
            if set(previous.files) != set(arrays):
                raise ValueError('Incompatible existing evaluation scores')
            for key, value in arrays.items():
                np.testing.assert_array_equal(previous[key], value)
    else:
        tmp = artifact.with_suffix(f'.{os.getpid()}.tmp')
        with tmp.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        tmp.replace(artifact)
    return dict(stage=stage, people=len(target), reliable_face=int(valid.sum()), checkpoint_sha256=cp,
        native_scores_sha256=native_hash, features_sha256=meta['features_sha256'],
        features_meta_sha256=sha256_file(feature.with_suffix('.json')),
        predictions_sha256=sha256_file(artifact), metrics=metrics, comparisons=comparisons,
        group_means=used_means, same_group_shuffle=shuffles)


@torch.inference_mode()
def run(source, out, split):
    torch.set_num_threads(2)
    lock, models = load_models(out)
    verify_source(source, lock)
    fit_hash = sha256_file(out/'caran_fit_lock.json')
    seal = validation_seal(out, fit_hash) if split == 'test' else None
    validation = read(out/'val_results.json') if seal else None
    native_val_path = source/'evaluation'/'val'/'result.json'
    native_val_hash = sha256_file(native_val_path)
    if seal and seal['native_val_result_sha256'] != native_val_hash:
        raise ValueError('Native Validation results changed after the protocol seal')
    if validation is not None:
        for stage, row in enumerate(validation['stages']):
            require_digest(out/'features'/f'task{stage}'/'val.npz', row['features_sha256'])
            require_digest(out/'features'/f'task{stage}'/'val.json', row['features_meta_sha256'])
            require_digest(out/'scores'/'val'/f'task{stage}.npz', row['predictions_sha256'])
    native_path = source/'evaluation'/split/'result.json'
    native = read(native_path)
    if (native['smoke'] or native['split'] != split or native['mode'] != 'sample_holdout'
            or len(native['stages']) != len(TASK_SIZES)):
        raise ValueError('Incomplete native eight-stage evaluation')
    stages = []
    for stage in range(len(TASK_SIZES)):
        row = load_stage(out, source, split, stage, models, lock, native['stages'][stage],
                         validation['stages'][stage]['group_means'] if validation else None,
                         native_val_hash)
        stages.append(row)
        print(split, stage, {method: round(row['metrics'][method]['all']['mAP'], 4)
                             for method in FORMAL_ROWS}, flush=True)
    aggregate = summarize(stages, split)
    for method in ('fixed', 'equal'):
        for key in MAP_KEYS:
            np.testing.assert_allclose(aggregate['summary'][method]['all'][key],
                                       native['summary'][method]['all'][key], rtol=0, atol=1e-9)
    formal_table = {method: {key: aggregate['summary'][method]['all'][key] for key in MAP_KEYS}
                    for method in FORMAL_ROWS}
    report = dict(split=split, caran_fit_lock_sha256=fit_hash,
        native_result_sha256=sha256_file(native_path), stages=stages,
        protected_full_train=PROTECTED_BASELINE[split], permutation_seeds=list(FORMAL_SEEDS),
        formal_table=formal_table,
        group_mean_protocol='unlabelled Validation means, sealed and reused on Test; diagnostic only',
        **aggregate)
    if seal:
        report['validation_seal_sha256'] = sha256_file(out/'validation_seal.json')
    destination = out/f'{split}_results.json'
    write_verified(destination, report)
    if split == 'val':
        write_verified(out/'validation_seal.json', dict(caran_fit_lock_sha256=fit_hash,
            val_results_sha256=sha256_file(destination), native_val_result_sha256=native_val_hash))
    print(split, 'formal_table', formal_table, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'out'):
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--split', choices=('val', 'test'), required=True)
    run(**vars(p.parse_args()))
