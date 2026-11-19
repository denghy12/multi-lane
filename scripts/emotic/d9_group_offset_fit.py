"""Learn one cross-reliability ranking offset per task from natural held-out TRAIN scores."""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import expit, log_expit, logsumexp

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import task_indices
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_a_eight_fit import group_proportions
from scripts.emotic.d9_paired_a_evaluate import TASKS, load_training_pair, write_verified
from scripts.emotic.fit_d9_task10_probe import mix, weights

XTOL = 1e-10


def fit_offset(logits, labels, reliable):
    """Minimize mean class pairwise logistic loss on opposite-label, opposite-group pairs."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels)
    if (z.ndim != 2 or not np.isfinite(z).all() or z.shape != y.shape
            or reliable.shape != (len(z),) or reliable.dtype != np.bool_
            or not np.isin(y, [0, 1]).all()):
        raise ValueError('Invalid natural TRAIN logits, binary labels or Face mask')

    terms, counts = [], []
    for scores, positives in zip(z.T, y.astype(bool).T):
        # In the first array raising b helps the positive. In the second it helps the negative.
        u = np.subtract.outer(scores[reliable & ~positives], scores[~reliable & positives]).ravel()
        r = np.subtract.outer(scores[~reliable & ~positives], scores[reliable & positives]).ravel()
        counts.append(dict(unreliable_positive=len(u), reliable_positive=len(r)))
        if len(u) + len(r):
            terms.append((u, r, len(u) + len(r)))

    def diagnostics(b):
        if not terms:
            return dict(loss=None, gradient=None)
        losses = [(np.logaddexp(0., u-b).sum() + np.logaddexp(0., r+b).sum()) / n for u, r, n in terms]
        grads = [(expit(r+b).sum() - expit(u-b).sum()) / n for u, r, n in terms]
        return dict(loss=float(np.mean(losses)), gradient=float(np.mean(grads)))

    def log_gradient_ratio(b):
        # Find equality of the two positive gradient magnitudes. Taking logs avoids
        # losing a direction when nearly all paired scores are very well separated.
        increase = [logsumexp(log_expit(r+b)) - np.log(n) for _, r, n in terms if len(r)]
        decrease = [logsumexp(log_expit(u-b)) - np.log(n) for u, _, n in terms if len(u)]
        return float(logsumexp(increase) - logsumexp(decrease))

    totals = {key: sum(row[key] for row in counts) for key in ('unreliable_positive', 'reliable_positive')}
    before = diagnostics(0.)
    b = 0.
    status = 'fitted'
    if not all(totals.values()):
        status = 'no_cross_group_pairs' if not terms else 'only_one_direction'
    else:
        start = log_gradient_ratio(0.)
        if start != 0:
            end = -1. if start > 0 else 1.
            while np.sign(log_gradient_ratio(end)) == np.sign(start):
                end *= 2.
                if not np.isfinite(end):
                    raise FloatingPointError('Cannot bracket a finite TRAIN optimum')
            bracket = sorted([0., end])
            b = float(brentq(log_gradient_ratio, *bracket, xtol=XTOL))
    after = diagnostics(b)
    if not np.isfinite(b) or (terms and after['loss'] > before['loss'] + 1e-11):
        raise FloatingPointError('Invalid fitted group offset')
    return b, dict(status=status, counts_by_class=counts, directional_pairs=totals,
                   active_classes=len(terms), before=before, after=after)


def fit(source, previous, out):
    torch.set_num_threads(1)
    cfg = read(source / 'config.json')
    if any(cfg.get(k) != v for k, v in dict(seed=1, tasks=8, mode='sample_holdout', smoke=False,
                                            protect_branches=True, image_split='global_10pct_hash_rank').items()):
        raise ValueError('Unexpected source branch configuration')
    roots = dict(config_sha256=sha256_file(source / 'config.json'),
                 roles_sha256=sha256_file(source / 'train_image_roles.json'))
    roles = read(source / 'train_image_roles.json')
    lock_path = previous / 'readout' / 'centers_lock.json'
    center_hash = read(previous / 'readout' / 'validation_seal.json')['centers_sha256']
    require_digest(lock_path, center_hash)
    lock = read(lock_path)
    if (set(lock['tasks']) != {str(t) for t in TASKS}
            or any(lock[k] != h for k, h in roots.items())):
        raise ValueError('Sealed fusion source and natural TRAIN source disagree')

    records = {}
    for t in TASKS:
        row = lock['tasks'][str(t)]
        require_digest(source / 'compact_checkpoints' / f'task{t}.pth', row['checkpoint_sha256'])
        cp = source / 'cohorts' / f'task{t}.json'
        require_digest(cp, row['cohort_sha256'])
        base = dict(row, fusion_cohort=read(cp)['fusion'])
        data, _, meta = load_training_pair(previous/'pairs', roles, t, base, roots)
        if (meta['seed'] != 1 or meta['archive_sha256'] != row['pairs_sha256']
                or row['fit_classes'] != list(task_indices(t))
                or len(data['targets']) != row['people']
                or int(data['reliable'].sum()) != row['reliable_face']
                or group_proportions(row['paired']['center']) != row['paired']['weights']):
            raise ValueError(f'Unexpected training cohort or sealed readout for task {t}')
        w = weights(torch.tensor(row['paired']['center']), torch.from_numpy(data['reliable']))
        z = mix(w, torch.from_numpy(data['branch_logits']).float())
        b, diag = fit_offset(z.numpy(), data['targets'], data['reliable'])
        records[str(t)] = {k: row[k] for k in ('people','reliable_face','fit_classes','pairs_sha256',
                                               'checkpoint_sha256','cohort_sha256')}
        records[str(t)].update(offset=b, fit=diag)
        print(f'task{t}: {diag["status"]}, b={b:+.8f}, {diag["directional_pairs"]}', flush=True)

    protocol = dict(seed=1, inputs='natural held-out TRAIN, current classes', initial_offset=0.,
                    application='add to unreliable Face only; sealed at introduction',
                    loss='cross-group pairwise logistic, all valid pairs per class, equal class mean',
                    optimizer='bracket from zero and Brent gradient root', xtol=XTOL)
    out.mkdir(parents=True, exist_ok=True)
    write_verified(out/'offsets_lock.json', dict(protocol=protocol, centers_sha256=center_hash, **roots, tasks=records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source','previous','out'):
        parser.add_argument('--'+key, type=Path, required=True)
    fit(**vars(parser.parse_args()))
