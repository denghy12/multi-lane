"""Score independently trained view paths with the archived D readout and populations."""
import argparse
from collections import defaultdict
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.evaluation_scores import align_evaluation_scores, load_evaluation_scores
from multi_lane.track_a.fuse_face_endpoint_validation import load_face_reliability
from multi_lane.track_a.r1_crossvit_experiment import read, write
from multi_lane.track_a.runner import CLASS_ORDER, TASK_SIZES, compute_metrics, seen_indices, summarize_tasks
from scripts.emotic.r1_asymmetric_forward import fuse_logits
from scripts.emotic.run_r1_frozen_cross_view import input_paths, score_suffix

ANCHOR_VIEWS = ('full', 'person_all', 'face_all')


def branch_paths(root, archived, anchor_only):
    paths = {key: archived / 'branches' / key for key in ANCHOR_VIEWS}
    if not anchor_only:
        for mode in ('sync', 'independent'):
            for view in ('person', 'face'):
                paths[f'{mode}_{view}'] = root / mode / 'attention' / view
        paths.update(cap035_person=root / 'cap035/attention/person',
                     box_full=root / 'box/native/full', box_swap_full=root / 'box_swap/native/full')
    return paths


def fusion_groups(anchor_only):
    groups = {'D': ANCHOR_VIEWS}
    if not anchor_only:
        for mode in ('sync', 'independent'):
            groups[f'{mode}_person'] = ('full', f'{mode}_person', 'face_all')
            groups[f'{mode}_face'] = ('full', 'person_all', f'{mode}_face')
            groups[f'{mode}_both'] = ('full', f'{mode}_person', f'{mode}_face')
        groups.update(cap035_person=('full', 'cap035_person', 'face_all'),
                      box_full=('box_full', 'person_all', 'face_all'),
                      box_swap_full=('box_swap_full', 'person_all', 'face_all'))
    return groups


def pair_ranking(ids, logits, labels):
    """Count within-image positive/negative person comparisons separately per class."""
    images = defaultdict(list)
    for row, sample in enumerate(ids):
        image, sep, person = str(sample).rpartition('#person=')
        if not sep or not image or not person.isdigit():
            raise ValueError(f'Unexpected EMOTIC person ID: {sample}')
        images[image].append(row)
    nclass = labels.shape[1]
    pair_counts = np.zeros(nclass, dtype=np.int64)
    correct = np.zeros_like(pair_counts)
    ties = np.zeros_like(pair_counts)
    pairs = different = identical = 0
    multi = [rows for rows in images.values() if len(rows) >= 2]
    for rows in multi:
        for a, b in combinations(rows, 2):
            pairs += 1
            polarity = labels[a] - labels[b]
            differs = polarity != 0
            margin = (logits[a] - logits[b]) * polarity
            pair_counts += differs
            correct += differs & (margin > 0)
            ties += differs & (margin == 0)
            different += int(differs.any())
            identical += int(np.array_equal(logits[a], logits[b]))

    def stats(n, wins, draws):
        return dict(comparisons=int(n), correct=int(wins), ties=int(draws),
                    incorrect=int(n - wins - draws),
                    accuracy_half_credit=float((wins + .5 * draws) / n) if n else None)
    return dict(images=len(images), multi_images=len(multi), multi_people=sum(map(len, multi)),
                person_pairs=pairs, different_label_pairs=different,
                identical_logit_vector_pairs=identical,
                pooled=stats(pair_counts.sum(), correct.sum(), ties.sum()),
                per_class={name: stats(n, wins, draws)
                           for name, n, wins, draws in zip(CLASS_ORDER, pair_counts, correct, ties)})


def add_metrics(records, name, task, logits, targets, populations):
    for population, mask in populations.items():
        if not mask.any():
            raise ValueError(f'Empty {population} population at task {task}')
        # Match the historical raw-logit readout: subset, CPU float32 sigmoid, then per-class AP.
        metric = compute_metrics(task, logits[mask].sigmoid(), targets[mask], .5)
        records.setdefault(name, {}).setdefault(population, []).append(metric)


def finish_metrics(records):
    result = {}
    for name, populations in records.items():
        result[name] = {}
        for population, rows in populations.items():
            if [row.task_id for row in rows] != list(range(len(TASK_SIZES))):
                raise ValueError(f'Incomplete phase history for {name}/{population}')
            result[name][population] = dict(summary=summarize_tasks(rows),
                                            metrics=[asdict(row) for row in rows])
    return result


def run(root, anchor, manifest, anchor_only=False):
    archived = anchor / 'evaluation/trained'
    paths = branch_paths(root, archived, anchor_only)
    groups = fusion_groups(anchor_only)
    anchor_results = read(archived / 'result.json')['results']
    splits = {}
    for split in ('val', 'test'):
        reliability = load_face_reliability(manifest, split)
        branch_records, group_records, pair_records = {}, {}, defaultdict(list)
        for task in range(len(TASK_SIZES)):
            suffix = score_suffix(split, task)
            base = load_evaluation_scores(paths['full'] / suffix, 64)
            if base.task_id != task or not np.array_equal(base.class_indices, seen_indices(task)):
                raise ValueError(f'Historical scores have unexpected task/classes at {suffix}')
            ids, targets = base.sample_ids, torch.from_numpy(base.targets)
            reliable = torch.as_tensor([reliability[sample] for sample in ids], dtype=torch.bool)
            populations = {'all': torch.ones_like(reliable), 'reliable': reliable,
                           'unreliable': ~reliable}
            branches = {}
            for name, path in paths.items():
                raw = base.logits if name == 'full' else align_evaluation_scores(
                    base, load_evaluation_scores(path / suffix, 64))[2]
                branches[name] = torch.from_numpy(raw)
                add_metrics(branch_records, name, task, branches[name], targets, populations)
                if name in ('full', 'person_all', 'box_full', 'box_swap_full'):
                    ranking = pair_ranking(ids, raw, base.targets)
                    pair_records[name].append(dict(task=task, **ranking))
            archived_d = align_evaluation_scores(base, load_evaluation_scores(
                archived / 'fusion_scores/D_both_edges' / suffix, 64))[2]
            for group, (full, person, face) in groups.items():
                fused = fuse_logits(branches[full], branches[person], branches[face], reliable)
                if group == 'D' and not np.array_equal(fused.numpy(), archived_d):
                    raise AssertionError(f'D fusion logits failed replay at {split}/task{task}')
                if full == 'full' and person == 'person_all':
                    if not np.array_equal(fused[~reliable].numpy(), archived_d[(~reliable).numpy()]):
                        raise AssertionError(f'Fallback population changed in {group}')
                add_metrics(group_records, group, task, fused, targets, populations)
            for population in populations:
                expected = anchor_results[split]['D_both_edges'][population]['metrics'][task]['mAP']
                if abs(group_records['D'][population][-1].mAP - expected) > 1e-8:
                    raise AssertionError(f'Historical {split}/{population} mAP failed replay at task{task}')
            print('D9_REFINEMENT', split, task, len(ids), flush=True)
        splits[split] = dict(branches=finish_metrics(branch_records), fusion=finish_metrics(group_records),
                             same_image_full_diagnostic=dict(pair_records))
    return dict(status='complete', seed=1, branches={k: str(v) for k, v in paths.items()},
                fusion_sources=dict(groups), score_protocol='locked historical D logits',
                box_swap_role='same-checkpoint attribution', exact_D_replay=True,
                class_order=list(CLASS_ORDER), splits=splits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--anchor', type=Path, required=True, help='Archived D seed1_20261108 directory')
    parser.add_argument('--manifest', type=Path, default=input_paths().face_manifest_root)
    parser.add_argument('--anchor-only', action='store_true', help='Verify historical replay before training')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    results = run(args.root, args.anchor, args.manifest, args.anchor_only)
    filename = 'anchor_replay.json' if args.anchor_only else 'branch_refinement_results.json'
    write(args.output or args.root / filename, results)


if __name__ == '__main__':
    main()
