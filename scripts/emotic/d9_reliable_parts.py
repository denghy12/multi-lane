"""Factorial attribution of learned Face mass and the conditional Full/Person mix."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, TaskMetrics, summarize_tasks
from scripts.emotic.d9_a_eight_eval import GROUPS, predictions, read, require_digest
from scripts.emotic.d9_a_eight_fit import group_proportions
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.evaluate_d9_relation import metrics_for_stage, weighted_logits
from scripts.emotic.evaluate_d9_sample_fusion import MAP_KEYS

METHODS = ('base', 'face_only', 'ratio_only', 'both')


def weight_grid(learned):
    """Return four reliable-Face readouts in Full/Person/Face order."""
    if (learned.ndim != 2 or learned.shape[1] != 3 or learned.dtype != torch.float32
            or not torch.isfinite(learned).all() or (learned < 0).any()
            or (learned[:, :2].sum(dim=1) <= 0).any()):
        raise ValueError('Invalid learned reliable-Face weights')
    w = learned.double()
    torch.testing.assert_close(w.sum(dim=1), torch.ones_like(w[:, 0]), atol=2e-7, rtol=0)
    face = w[:, 2]
    full_fraction = w[:, 0] / w[:, :2].sum(dim=1)

    def compose(f, p):
        # Factor in double precision, then use the same float32 readout as the source model.
        return torch.stack(((1-f)*p, (1-f)*(1-p), f), dim=1).float()

    torch.testing.assert_close(compose(face, full_fraction), learned, atol=2e-7, rtol=0)
    return dict(base=learned.new_tensor([.64, .16, .20]).expand_as(learned),
                face_only=compose(face, torch.full_like(full_fraction, .8)),
                ratio_only=compose(torch.full_like(face, .2), full_fraction),
                both=learned)


def reliable_readouts(branches, valid, sizes, learned, fixed_logits, paired_logits):
    """Keep paired logits for every unavailable-Face person and vary reliable rows only."""
    if (valid.dtype != torch.bool or valid.shape != (len(branches),)
            or fixed_logits.shape != paired_logits.shape or fixed_logits.shape != branches.shape[:2]):
        raise ValueError('Incompatible Face availability or logit shapes')
    grid = weight_grid(learned)
    results = {}
    for name, matrix in grid.items():
        reliable = weighted_logits(branches[valid], matrix.unsqueeze(0).expand(int(valid.sum()), -1, -1), sizes)
        if name in ('base','both'):
            endpoint = fixed_logits if name == 'base' else paired_logits
            torch.testing.assert_close(reliable, endpoint[valid], atol=2e-5, rtol=2e-6)
            # Preserve source precision on both factorial corners.
            reliable = endpoint[valid]
        result = paired_logits.clone()
        result[valid] = reliable
        results[name] = result
    return results


def check_replay(metrics, historical):
    for name, prior in [('base','only_unreliable'), ('both','paired')]:
        if metrics[name] != historical[prior]:
            raise ValueError(f'Historical factorial corner disagrees: {name}')
    for name in METHODS:
        if metrics[name]['unreliable'] != historical['paired']['unreliable']:
            raise ValueError(f'Unavailable-Face population changed: {name}')


def run(source, paired, swap, out, split):
    out.mkdir(parents=True, exist_ok=True)
    paired_seal = read(paired/'validation_seal.json')
    center_hash = paired_seal['centers_sha256']
    require_digest(paired/'centers_lock.json', center_hash)
    lock = read(paired/'centers_lock.json')
    require_digest(source/'config.json', lock['config_sha256'])
    require_digest(source/'train_image_roles.json', lock['roles_sha256'])
    swap_seal_hash = sha256_file(swap/'validation_seal.json')
    swap_seal = read(swap/'validation_seal.json')
    if (swap_seal['centers_sha256'] != center_hash
            or swap_seal['prior_validation_seal_sha256'] != sha256_file(paired/'validation_seal.json')):
        raise ValueError('Group-swap reference uses different paired-A weights')
    require_digest(swap/'val_results.json', swap_seal['val_results_sha256'])
    seal_path = out/'validation_seal.json'
    if split == 'test':
        seal = read(seal_path)
        if seal['centers_sha256'] != center_hash or seal['swap_seal_sha256'] != swap_seal_hash:
            raise ValueError('Test source differs from locked Validation source')
        require_digest(out/'val_results.json', seal['val_results_sha256'])

    prior_path = swap/f'{split}_results.json'
    prior = read(prior_path)
    if (prior['split'] != split or len(prior['stages']) != len(TASK_SIZES)
            or prior['centers_sha256'] != center_hash
            or (split == 'test' and prior['validation_seal_sha256'] != swap_seal_hash)):
        raise ValueError('Wrong group-swap archive')
    require_digest(paired/f'{split}_results.json', prior['prior_results_sha256'])
    native = source/'evaluation'/split
    require_digest(native/'result.json', prior['native_results_sha256'])
    centers = {k: row['paired'] for k, row in lock['tasks'].items()}
    torch.set_num_threads(1)
    weights = []
    for t in range(len(TASK_SIZES)):
        center = centers[str(t)]
        if group_proportions(center['center']) != center['weights']:
            raise ValueError(f'Archived weights do not reproduce paired A at task{t}')
        weights.append(center['weights'][1])
    learned = torch.tensor(weights, dtype=torch.float32)
    stages = []
    for t, old in enumerate(prior['stages']):
        path = native/f'task{t}.npz'
        require_digest(path, old['scores_sha256'])
        torch.set_num_threads(1)
        with np.load(path, allow_pickle=False) as archive:
            src, y, valid = predictions(archive, centers, t)
            sid = hashlib.sha256('\n'.join(archive['sample_ids'].tolist()).encode()).hexdigest()
            if (old['stage'] != t or sid != old['sample_id_sha256'] or old['people'] != len(y)
                    or old['reliable_face'] != int(valid.sum())):
                raise ValueError(f'Changed population: {split}/stage{t}')
            readouts = reliable_readouts(torch.from_numpy(archive['branch_logits']), valid,
                                        TASK_SIZES[:t+1], learned[:t+1], src['fixed'], src['A'])
        torch.set_num_threads(2)
        metrics = metrics_for_stage(t, y, valid, readouts)
        check_replay(metrics, old['metrics'])
        stages.append(dict(stage=t, sample_id_sha256=sid, scores_sha256=old['scores_sha256'],
                           people=len(y), reliable_face=int(valid.sum()), metrics=metrics))
    summary = {m: {g: summarize_tasks([TaskMetrics(**s['metrics'][m][g]) for s in stages])
                   for g in GROUPS} for m in METHODS}
    check_replay(summary, prior['summary'])
    report = dict(split=split, stages=stages, summary=summary, centers_sha256=center_hash,
                  prior_sha256=sha256_file(prior_path), native_results_sha256=prior['native_results_sha256'],
                  reliable_head_weights={m:w.tolist() for m,w in weight_grid(learned).items()})
    if split == 'test':
        report['validation_seal_sha256'] = sha256_file(seal_path)
    dest = out/f'{split}_results.json'
    write_verified(dest, report)
    if split == 'val':
        write_verified(seal_path, dict(val_results_sha256=sha256_file(dest),
                                      swap_seal_sha256=swap_seal_hash, centers_sha256=center_hash))
    print(split, {m:{k:summary[m]['all'][k] for k in MAP_KEYS} for m in METHODS}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source','paired','swap','out'):
        parser.add_argument('--'+key, required=True, type=Path)
    parser.add_argument('--split', choices=('val','test'), required=True)
    run(**vars(parser.parse_args()))
