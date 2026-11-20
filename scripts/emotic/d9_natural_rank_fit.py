"""Fit task-private fusion with the original paired BCE and an optional clean ranking term."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_a_eight_fit import group_proportions
from scripts.emotic.d9_matched_natural_fit import checked_data
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_quality_fit import LR, STEPS, _balanced_bce, _gate_scores, _mixed, _row_loss


def class_pairs(labels, reliable):
    """Prepare natural-person pairs, including pairs from different Face availability groups."""
    pairs, counts = [], []
    for column, target in enumerate(labels.T):
        positive = torch.where(target == 1)[0]
        negative = torch.where(target == 0)[0]
        n_pos, n_neg = len(positive), len(negative)
        r_pos, r_neg = int(reliable[positive].sum()), int(reliable[negative].sum())
        counts.append(dict(positive=n_pos, negative=n_neg, pairs=n_pos * n_neg,
                           cross_group_pairs=r_pos * (n_neg - r_neg) + (n_pos - r_pos) * r_neg))
        if n_pos and n_neg:
            pairs.append((column, positive, negative))
    return pairs, counts


def natural_rank(logits, pairs):
    """Mean over eligible classes of their mean softplus(negative - positive)."""
    if not pairs:
        return logits.sum() * 0
    per_class = []
    for column, positive, negative in pairs:
        gap = logits[negative, column].unsqueeze(0) - logits[positive, column].unsqueeze(1)
        per_class.append(F.softplus(gap).mean())
    return torch.stack(per_class).mean()


def fit_arm(data, use_rank, steps=STEPS):
    reliable = torch.from_numpy(data['reliable'])
    usable = torch.stack([torch.ones_like(reliable), torch.ones_like(reliable), reliable], dim=1)
    stacked = np.stack([data[k] for k in ('branch_logits', 'person_branch_logits', 'face_branch_logits')], axis=1)
    branch = torch.from_numpy(stacked).float()
    labels = torch.from_numpy(data['targets']).float().unsqueeze(1)
    no_features = torch.empty((len(labels), 3, 0))
    pairs, counts = class_pairs(labels[:, 0], reliable)
    bias = torch.zeros(2, 3, requires_grad=True)
    optimizer = torch.optim.Adam([bias], lr=LR)

    def losses(include_rank):
        scores = _gate_scores(no_features, reliable, bias, None)
        logits = _mixed(branch, scores.softmax(-1))
        row = _row_loss(logits, labels)
        bce = _balanced_bce(row, usable)
        rank = natural_rank(logits[:, 0], pairs) if include_rank else None
        return bce, rank, row[:, 0].mean()

    @torch.no_grad()
    def diagnostic():
        bce, rank, clean_bce = losses(True)
        return dict(bce=bce.item(), natural_bce=clean_bce.item(), natural_rank=rank.item(),
                    objective=(bce + rank if use_rank else bce).item())

    initial = diagnostic()
    for _ in range(steps):
        bce, rank, _ = losses(use_rank)
        loss = bce + rank if use_rank else bce
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(bias.grad).all():
            raise FloatingPointError('Nonfinite TRAIN loss or fusion gradient')
        optimizer.step()
    final = diagnostic()
    if not np.isfinite(list(final.values())).all():
        raise FloatingPointError('Nonfinite final TRAIN diagnostic')
    center = bias.detach().tolist()
    return dict(center=center, weights=group_proportions(center), fit=dict(initial=initial, final=final)), counts


def fit(root, historical, out, smoke=False):
    torch.set_num_threads(1)
    cfg_path, roles_path = root/'config.json', root/'train_image_roles.json'
    cfg = read(cfg_path)
    expected = dict(mode='sample_holdout', seed=1, tasks=len(TASK_SIZES), smoke=False,
                    epochs=30, batch_size=64, protect_branches=True, image_split='global_10pct_hash_rank')
    if (any(cfg.get(k) != v for k, v in expected.items()) or
            cfg['equal_updates']['mode'] != 'cycle_90pct_to_full_epoch_size'):
        raise ValueError('Source must be the original equal-update random 90% seed1 TRAIN trajectory')
    require_digest(roles_path, cfg['equal_updates']['roles_sha256'])
    roles = read(roles_path)
    source_lock = root/'paired_A'/'centers_lock.json'
    reference_lock = historical/'centers_lock.json'
    seals, history = read(source_lock), read(reference_lock)
    keys = {str(k) for k in range(len(TASK_SIZES))}
    if (set(seals) != keys or set(history['tasks']) != keys or history['smoke']
            or history['protocol']['split'] != 'train' or history['protocol']['init'] != 'equal_available_views'
            or history['protocol']['steps'] != STEPS or history['protocol']['lr'] != LR):
        raise ValueError('Unexpected reference TRAIN centers')
    require_digest(source_lock, history['source_centers_sha256'])
    require_digest(cfg_path, history['config_sha256'])
    require_digest(roles_path, history['roles_sha256'])
    tasks = {}
    for task in range(1 if smoke else len(TASK_SIZES)):
        key = str(task)
        data, original, hashes, cohort = checked_data(root, task, seals.get(str(task-1)), seals[key], roles)
        ref = history['tasks'][key]
        ident = dict(task=task, archive_sha256=original['archive_sha256'], source_center_sha256=seals[key],
                     train_ids_sha256=cohort['sample_id_sha256'], people=cohort['people'],
                     reliable_face=cohort['reliable_face'], fit_classes=cohort['classes'], **hashes)
        if any(ref[k] != v for k, v in ident.items()):
            raise ValueError(f'Mismatched task{task} source roster or prediction archive')
        control, counts = fit_arm(data, False)
        for label, source_key in [('center', 'center'), ('weights', 'group_weights')]:
            np.testing.assert_array_equal(control[label], original[source_key])
            np.testing.assert_array_equal(control[label], ref['paired'][label])
        candidate, _ = fit_arm(data, True)
        tasks[key] = dict(**ident, class_counts=counts, paired=control, rank=candidate)
        print(f"task{task} TRAIN {cohort['people']} people; {sum(c['pairs'] for c in counts)} class pairs", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_verified(out/'centers_lock.json', dict(tasks=tasks, smoke=smoke, config_sha256=history['config_sha256'],
                   roles_sha256=history['roles_sha256'], source_centers_sha256=history['source_centers_sha256'],
                   historical_centers_sha256=sha256_file(reference_lock),
                   protocol=dict(split='train', init='equal_available_views', steps=STEPS, lr=LR, optimizer='Adam',
                                 seed=1, rank_coefficient=1., temperature=1., scale='raw fused logits',
                                 bce='natural/body-blur/reliable-face-blur; equal mass per person',
                                 rank='natural only; mean eligible class mean softplus(negative-positive); '
                                      'all within and across availability pairs in original held TRAIN cohort')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'historical', 'out'):
        parser.add_argument('--'+key, required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    fit(**vars(parser.parse_args()))
