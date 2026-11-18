"""Repeat person-weight shuffles on saved direction-gate outputs, with no fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, average_precision
from scripts.emotic.evaluate_d9_relation import SHUFFLE_SEED, shuffle_sources, write_json


METHODS = ('a', 'b', 'c', 'center_b')
FORMAL_SEEDS = (SHUFFLE_SEED, *range(1000, 1099))


def permutation_sources(ids, reliable, task_count, seed):
    """Same sorted-ID cyclic derangement as the evaluator, with an explicit seed."""
    ids = np.asarray(ids)
    reliable = np.asarray(reliable, dtype=bool)
    if ids.ndim != 1 or reliable.shape != ids.shape or len(set(ids.tolist())) != len(ids):
        raise ValueError('Unique aligned person IDs and availability flags are required')
    ordered = np.argsort(ids, kind='stable')
    donors = np.repeat(np.arange(len(ids))[:, None], task_count, axis=1)
    for task in range(task_count):
        for flag in (False, True):
            members = ordered[reliable[ordered] == flag]
            rng = np.random.default_rng(np.random.SeedSequence([seed, task, int(flag)]))
            cycle = rng.permutation(members)
            donors[cycle, task] = np.roll(cycle, 1)
    return donors


def batch_map(logits, targets):
    """Vectorize official AP across methods/classes, preserving its tie ordering.

    A contiguous final sample axis makes each quicksort and precision reduction
    identical to the official one-dimensional class calculation. Sigmoid uses
    PyTorch float32 as in the original evaluator; sorting raw logits can change
    ties after sigmoid saturation.
    """
    logits = np.asarray(logits)
    targets = np.asarray(targets)
    if logits.ndim != 3 or logits.shape[1:] != targets.shape or not len(targets):
        raise ValueError('Expected method x person x class logits and aligned targets')
    if not np.isfinite(logits).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError('Finite logits and binary targets are required')
    probabilities = torch.from_numpy(logits).float().sigmoid().numpy()
    by_class = np.ascontiguousarray(probabilities.transpose(0, 2, 1))
    order = np.argsort(by_class, axis=-1)[:, :, ::-1]
    truth = np.broadcast_to(targets.T[None], by_class.shape)
    positives = np.take_along_axis(truth, order, axis=-1) == 1
    cumulative = positives.cumsum(axis=-1)
    cumulative[~positives] = 0
    counts = positives.sum(axis=-1)
    precision_sum = (cumulative / np.arange(1, len(targets) + 1)).sum(axis=-1)
    class_ap = 100. * precision_sum / (counts + 1e-8)
    return class_ap.mean(axis=-1)


def shuffled_logits(branches, weights, actual, donors, sizes):
    """Use identical donors for every method, preserving constant-weight outputs."""
    heads = np.arange(len(sizes))
    columns = np.repeat(heads, sizes)
    donor_weights = weights[:, donors, heads]
    per_class = donor_weights[:, :, columns]
    fused = (branches[None, :, :, 0] * per_class[..., 0]
             + branches[None, :, :, 1] * per_class[..., 1]
             + branches[None, :, :, 2] * per_class[..., 2])
    same = (weights == donor_weights).all(-1)[:, :, columns]
    return np.where(same, actual, fused)


def difference_summary(actual, shuffled):
    delta = float(actual) - np.asarray(shuffled, dtype=np.float64)
    return dict(actual=float(actual), shuffled_mean=float(np.mean(shuffled)),
                mean=float(delta.mean()), minimum=float(delta.min()), maximum=float(delta.max()),
                percentile_2_5=float(np.quantile(delta, .025)),
                percentile_97_5=float(np.quantile(delta, .975)),
                positive_count=int((delta > 0).sum()), equal_count=int((delta == 0).sum()),
                negative_count=int((delta < 0).sum()), positive_fraction=float((delta > 0).mean()),
                differences=delta.tolist())


def evaluate(root, smoke, *, methods=METHODS, expected_mode="direction"):
    torch.set_num_threads(2)
    config = json.loads((root / 'config.json').read_text())
    if config['mode'] != expected_mode or config['smoke'] != smoke:
        raise ValueError(f'Expected a {expected_mode} run with matching smoke/full mode')
    title = 'smoke_evaluation' if smoke else 'evaluation'
    stage_count = 2 if smoke else len(TASK_SIZES)
    seeds = FORMAL_SEEDS[:3] if smoke else FORMAL_SEEDS
    result = dict(schema=1, trained_seed=config['seed'], smoke=smoke, permutation_seeds=list(seeds),
                  methods=list(methods), config_sha256=sha256_file(root / 'config.json'),
                  assignment='Within task head and Face reliability; sorted person IDs, cyclic derangement; singleton keeps own',
                  pairing='Same donors for every method at each stage and permutation seed',
                  score='Official float32 sigmoid then per-class AP, descending NumPy quicksort; average seen classes',
                  interpretation='Fixed checkpoints and evaluation sets. Differences quantify pairing sensitivity, not training-seed uncertainty. Evaluation labels only score fixed predictions.',
                  splits={})
    for split in ('val', 'test'):
        directory = root / title / split
        report_path = directory / 'result.json'
        report = json.loads(report_path.read_text())
        if report['split'] != split or report['smoke'] != smoke or len(report['stages']) != stage_count:
            raise ValueError(f'Complete matching {split} evaluation is required')
        actual_stages, shuffled_stages, artifacts = [], [], []
        for stage, saved in enumerate(report['stages']):
            archive = directory / f'task{stage}.npz'
            artifact_hash = sha256_file(archive)
            if artifact_hash != saved['scores_sha256'] or (expected_mode == 'direction' and not saved['exact_replay']):
                raise ValueError(f'Unverified saved scores at {split} stage {stage}')
            if saved['protocol']['shuffle_seed'] != SHUFFLE_SEED:
                raise ValueError('Formal permutation seed differs from protocol')
            with np.load(archive, allow_pickle=False) as data:
                ids, reliability, targets = data['sample_ids'], data['reliable'], data['targets']
                branches, sizes = data['branch_logits'], data['task_sizes'].tolist()
                if sizes != list(TASK_SIZES[:stage + 1]):
                    raise ValueError('Task class partition differs')
                weights = np.stack([data[name + '_weights'] for name in methods])
                actual = np.stack([data[name + '_logits'] for name in methods])
                archived_shuffle = np.stack([data[name + '_shuffled_logits'] for name in methods])
                archived_donors = data['shuffled_source_rows']
            scores = batch_map(actual, targets)
            expected_actual = [saved['metrics'][name]['all']['mAP'] for name in methods]
            np.testing.assert_allclose(scores, expected_actual, rtol=0., atol=1e-12)
            official = [np.mean([100. * average_precision(column, truth)
                                for column, truth in zip(torch.from_numpy(x).sigmoid().numpy().T, targets.T)])
                        for x in actual]
            np.testing.assert_allclose(scores, official, rtol=0., atol=1e-12)
            stage_permutations = []
            for index, seed in enumerate(seeds):
                donors = permutation_sources(ids, reliability, len(sizes), seed)
                permuted = shuffled_logits(branches, weights, actual, donors, sizes)
                permuted_scores = batch_map(permuted, targets)
                if index == 0:
                    np.testing.assert_array_equal(donors, archived_donors)
                    np.testing.assert_array_equal(donors, shuffle_sources(ids.tolist(), reliability, len(sizes)).numpy())
                    np.testing.assert_array_equal(permuted, archived_shuffle)
                    expected = [saved['metrics'][name + '_shuffled']['all']['mAP'] for name in methods]
                    np.testing.assert_allclose(permuted_scores, expected, rtol=0., atol=1e-12)
                stage_permutations.append(permuted_scores)
            actual_stages.append(scores)
            shuffled_stages.append(stage_permutations)
            artifacts.append(dict(stage=stage, people=len(ids), scores_sha256=artifact_hash,
                                  formal_donors_exact=True, formal_logits_exact=True,
                                  formal_map_max_abs=float(np.max(np.abs(np.asarray(stage_permutations[0]) - expected)))))
            print(f'{split} task{stage}: {len(seeds)} permutations, archived shuffle replayed', flush=True)
        actual_stages = np.asarray(actual_stages)
        shuffled_stages = np.asarray(shuffled_stages)
        by_method = {}
        for index, name in enumerate(methods):
            a = actual_stages[:, index]
            s = shuffled_stages[:, :, index]
            declared = report['summary'][name]['all']
            np.testing.assert_allclose([a[-1], a.mean()],
                                       [declared['final_mAP'], declared['average_mAP']], rtol=0., atol=1e-12)
            by_method[name] = dict(final=difference_summary(a[-1], s[-1]),
                                 average=difference_summary(a.mean(), s.mean(0)),
                                 actual_stages=a.tolist(),
                                 shuffled_stages_by_seed=s.T.tolist())
        result['splits'][split] = dict(result_sha256=sha256_file(report_path), methods=by_method,
                                      reproduction=artifacts)
    destination = root / 'diagnostics' / 'permutation_sensitivity.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)
    print(f'Saved {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    evaluate(args.root, args.smoke)


if __name__ == '__main__':
    main()
