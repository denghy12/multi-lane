"""Separate reliable-face degradation strength from its share of the fusion BCE."""
import argparse
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_a_eight_fit import group_proportions
from scripts.emotic.d9_matched_natural_fit import checked_data
from scripts.emotic.d9_matched_run import fit_average
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_paired_a_extract import NATURAL_KEYS, load_paired_archive, verify_pairs
from scripts.emotic.d9_quality_fit import LR, STEPS, _gate_scores, _mixed, _row_loss


METHODS = ('paired_112', 'gentle_168', 'clean_112', 'clean_168')


def arm_data(original, gentle, method):
    if method not in METHODS:
        raise ValueError(f'Unknown arm {method}')
    result = dict(original)
    if method.endswith('168'):
        for view in ('person', 'face'):
            key = view + '_branch_logits'
            result[key] = original[key].copy()
            result[key][original['reliable']] = gentle[key][original['reliable']]
    return result


def weighted_bce(row, reliable, prefer_clean):
    # The missing-Face row always keeps the original natural/body half-and-half.
    masses = torch.stack([torch.ones_like(reliable), torch.ones_like(reliable), reliable], dim=1).to(row.dtype)
    if prefer_clean:
        masses[reliable, 0] = 2
    return ((row * masses).sum(1) / masses.sum(1)).mean()


def fit_weighted(data, prefer_clean):
    valid = torch.from_numpy(data['reliable'])
    logits = torch.from_numpy(np.stack([data[k] for k in ('branch_logits', 'person_branch_logits',
                                                         'face_branch_logits')], axis=1)).float()
    labels = torch.from_numpy(data['targets']).float().unsqueeze(1)
    dummy = torch.zeros((len(valid), 3, 0))
    bias = torch.zeros(2, 3, requires_grad=True)
    optimizer = torch.optim.Adam([bias], lr=LR)
    for _ in range(STEPS):
        scores = _gate_scores(dummy, valid, bias, None)
        row = _row_loss(_mixed(logits, scores.softmax(-1)), labels)
        loss = weighted_bce(row, valid, prefer_clean)
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(bias.grad).all():
            raise FloatingPointError('Nonfinite TRAIN objective or fusion gradient')
        optimizer.step()
    center = bias.detach().tolist()
    return dict(center=center, weights=group_proportions(center))


def diagnostics(data, center, prefer_clean):
    valid = torch.from_numpy(data['reliable'])
    z = torch.from_numpy(np.stack([data[k] for k in ('branch_logits', 'person_branch_logits',
                                                    'face_branch_logits')], axis=1)).float()
    y = torch.from_numpy(data['targets']).float().unsqueeze(1)
    with torch.no_grad():
        scores = _gate_scores(torch.zeros((len(valid), 3, 0)), valid, torch.tensor(center), None)
        rows = _row_loss(_mixed(z, scores.softmax(-1)), y)
        result = dict(objective=weighted_bce(rows, valid, prefer_clean).item(), natural_bce=rows[:, 0].mean().item())
        for name, mask in [('reliable', valid), ('unreliable', ~valid)]:
            result[name] = {view: rows[mask, index].mean().item()
                            for index, view in enumerate(('natural_bce', 'person_bce', 'face_bce'))
                            if index != 2 or name == 'reliable'}
        return result


def fit_arm(original, gentle, method, reference):
    data = arm_data(original, gentle, method)
    prefer_clean = method.startswith('clean')
    if method == 'paired_112':
        base = fit_average(data, {v:dict(branch_logits=data[v+'_branch_logits']) for v in ('person', 'face')})
        result = dict(center=base['params']['bias'], weights=base['weights'])
        np.testing.assert_array_equal(result['center'], reference['center'])
        np.testing.assert_array_equal(result['weights'], reference['weights'])
    else:
        result = fit_weighted(data, prefer_clean)
    # Rows are independent parameters and have identical unavailable-person loss terms.
    np.testing.assert_array_equal(result['center'][0], reference['center'][0])
    np.testing.assert_array_equal(result['weights'][0], reference['weights'][0])
    result['fit'] = dict(initial=diagnostics(data, [[0., 0., 0.], [0., 0., 0.]], prefer_clean),
                         final=diagnostics(data, result['center'], prefer_clean))
    return result


def fit(root, historical, extra, out, smoke=False):
    torch.set_num_threads(1)
    config_path = root/'config.json'
    cfg = read(config_path)
    expected = dict(mode='sample_holdout', seed=1, epochs=30, tasks=len(TASK_SIZES), batch_size=64,
                    smoke=False, protect_branches=True, image_split='global_10pct_hash_rank')
    if (any(cfg.get(k) != v for k,v in expected.items())
            or cfg['equal_updates']['mode'] != 'cycle_90pct_to_full_epoch_size'):
        raise ValueError('Unexpected protected, equal-update original random TRAIN trajectory')
    roles_path = root/'train_image_roles.json'
    require_digest(roles_path, cfg['equal_updates']['roles_sha256'])
    roles = read(roles_path)
    source_lock = root/'paired_A'/'centers_lock.json'
    seals = read(source_lock)
    prior_path = historical/'centers_lock.json'
    prior = read(prior_path)
    keys = {str(t) for t in range(len(TASK_SIZES))}
    if (set(seals) != keys or set(prior['tasks']) != keys or prior['smoke']
            or prior['protocol']['split'] != 'train' or prior['protocol']['init'] != 'equal_available_views'
            or prior['protocol']['steps'] != STEPS or prior['protocol']['lr'] != LR):
        raise ValueError('Incorrect historical TRAIN reference')
    require_digest(config_path, prior['config_sha256'])
    require_digest(roles_path, prior['roles_sha256'])
    require_digest(source_lock, prior['source_centers_sha256'])
    tasks = {}
    for t in range(2 if smoke else len(TASK_SIZES)):
        key = str(t)
        data, original, hashes, cohort = checked_data(root, t, seals.get(str(t-1)), seals[key], roles)
        ref = prior['tasks'][key]
        identity = dict(task=t, source_center_sha256=seals[key], archive_sha256=original['archive_sha256'],
                        train_ids_sha256=cohort['sample_id_sha256'], people=cohort['people'],
                        reliable_face=cohort['reliable_face'], fit_classes=cohort['classes'], **hashes)
        if any(ref[k] != v for k,v in identity.items()):
            raise ValueError(f'Different historical people or source model at task{t}')
        path = extra/f'task{t}_168.npz'
        meta = read(path.with_suffix('.json'))
        require_digest(path, meta['archive_sha256'])
        expected_meta = dict(task=t, split='train', people=len(data['targets']),
                             reliable_face=int(data['reliable'].sum()), reduced_size=168, input_size=224,
                             sample_id_sha256=cohort['sample_id_sha256'],
                             original_archive_sha256=original['archive_sha256'], **hashes)
        if any(meta.get(k) != v for k,v in expected_meta.items()):
            raise ValueError(f'Invalid gentle TRAIN archive metadata at task{t}')
        gentle = load_paired_archive(path)
        for field in NATURAL_KEYS:
            np.testing.assert_array_equal(gentle[field], data[field])
        verify_pairs(gentle, data)
        for view in ('person', 'face'):
            if gentle[view+'_branch_logits'].dtype != data[view+'_branch_logits'].dtype:
                raise ValueError('Changed precision of perturbed TRAIN predictions')
        np.testing.assert_array_equal(ref['paired']['center'], original['center'])
        np.testing.assert_array_equal(ref['paired']['weights'], original['group_weights'])
        fits = {m:fit_arm(data, gentle, m, ref['paired']) for m in METHODS}
        tasks[key] = dict(**identity, gentle_archive_sha256=meta['archive_sha256'],
                          gentle_metadata_sha256=sha256_file(path.with_suffix('.json')), **fits)
        print(f"task{t} four arms on {cohort['people']} held TRAIN people, {cohort['reliable_face']} reliable", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    write_verified(out/'centers_lock.json', dict(tasks=tasks, smoke=smoke,
                   config_sha256=prior['config_sha256'], roles_sha256=prior['roles_sha256'],
                   source_centers_sha256=prior['source_centers_sha256'], historical_centers_sha256=sha256_file(prior_path),
                   protocol=dict(split='train', seed=1, steps=STEPS, lr=LR, optimizer='Adam', init='equal_available_views',
                                 people='original held TRAIN current-positive cohort',
                                 unreliable='original 112 natural/person halves with exact original weights',
                                 reliable={'paired_112':[1/3]*3,'gentle_168':[1/3]*3,
                                           'clean_112':[.5,.25,.25],'clean_168':[.5,.25,.25]},
                                 loss='per-person weighted version BCE, then mean people')))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'historical', 'extra', 'out'):
        cli.add_argument('--'+name, type=Path, required=True)
    cli.add_argument('--smoke', action='store_true')
    fit(**vars(cli.parse_args()))
