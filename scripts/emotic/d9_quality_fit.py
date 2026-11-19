"""Fit masked logit fusion with paired TRAIN supervision on changes in auxiliary utility."""
import numpy as np
import torch
import torch.nn.functional as F

from scripts.emotic.fit_d9_task10_probe import LR, PROJECTION_SEED, apply_rule, normalized, project, stats


STEPS = 300
VIEWS = ('person', 'face')


def _features(data):
    content = data.get('content_proj')
    if content is None:
        content = project(data)['content']
    return np.asarray(content), np.asarray(data['relation'])


def _inputs(raw, reliable, norm):
    content, relation = raw
    if (content.shape != (len(reliable), 12) or relation.shape != (len(reliable), 8)
            or not np.isfinite(content).all() or not np.isfinite(relation).all()):
        raise ValueError('Invalid projected content or Frozen9 relation features')
    x = np.column_stack((normalized(content, reliable, norm['content']),
                         normalized(relation, reliable, norm['relation'])))
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite standardized inputs')
    return torch.from_numpy(x)


def _gate_scores(x, reliable, bias, matrix):
    scores = bias[reliable.long()].unsqueeze(1).expand(-1, x.shape[1], -1)
    if matrix is not None:
        scores = scores + x @ matrix.T
    return torch.stack((scores[..., 0], scores[..., 1],
                        scores[..., 2].masked_fill(~reliable[:, None], -torch.inf)), dim=-1)


def _mixed(z, w):
    return (z * w.unsqueeze(-2)).sum(dim=-1)


def _row_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels.expand_as(logits), reduction='none').mean(-1)


def _balanced_bce(row_losses, usable):
    # A reliable person's three versions and an unreliable person's two get equal total mass.
    return ((row_losses * usable).sum(1) / usable.sum(1)).mean()


def _utility_targets(z, labels, base_scores, reliable):
    full = _row_loss(_mixed(z, base_scores.softmax(-1)), labels)
    targets = []
    for v in (1, 2):
        mask = torch.ones_like(reliable) if v == 1 else reliable
        subset = base_scores[mask].clone()
        subset[..., v] = -torch.inf
        remaining = _mixed(z[mask], subset.softmax(-1))
        utility = _row_loss(remaining, labels[mask]) - full[mask]
        # Current branch score and other views' exchanged messages may change between versions.
        targets.append((mask, (utility[:, 0] - utility[:, v]).detach()))
    return targets


def _odds_change(scores, mask, view):
    s = scores[mask]
    other = [v for v in range(3) if v != view]
    odds = s[..., view] - torch.logsumexp(s[..., other], dim=-1)
    return odds[:, 0] - odds[:, view]


def _paired_loss(scores, targets):
    terms = []
    for view, (mask, target) in enumerate(targets, start=1):
        change = _odds_change(scores, mask, view)
        terms.append(target.abs() * F.softplus(-target.sign() * change))
    return torch.cat(terms).mean()


def _fit_one(x, z, labels, reliable, usable, method, targets):
    bias = torch.zeros(2, 3, requires_grad=True)
    matrix = None if method == 'A' else torch.zeros(3, x.shape[-1], requires_grad=True)
    params = [bias] if matrix is None else [bias, matrix]
    optimizer = torch.optim.Adam(params, lr=LR)
    for _ in range(STEPS):
        scores = _gate_scores(x, reliable, bias, matrix)
        bce = _balanced_bce(_row_loss(_mixed(z, scores.softmax(-1)), labels), usable)
        loss = bce + _paired_loss(scores, targets) if method == 'C' else bce
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite objective for {method}')
        optimizer.zero_grad()
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError(f'Nonfinite gradient for {method}')
        optimizer.step()
    with torch.no_grad():
        scores = _gate_scores(x, reliable, bias, matrix)
        row = _row_loss(_mixed(z, scores.softmax(-1)), labels)
        bce = _balanced_bce(row, usable)
        pair = _paired_loss(scores, targets) if targets else bce.new_zeros(())
        if not torch.isfinite(bce + pair):
            raise FloatingPointError(f'Nonfinite final loss for {method}')
        fit = dict(bce=float(bce), natural_bce=float(row[:, 0].mean()),
                   paired_loss=float(pair) if targets else None,
                   objective=float(bce + pair if method == 'C' else bce),
                   last_gradient_norm=float(torch.cat([p.grad.flatten() for p in params]).norm()))
        weights = dict(bias=bias.tolist(), matrix=matrix.tolist() if matrix is not None else None)
    return weights, fit, scores


def _alignment(scores, targets):
    result = {}
    for view, (mask, d) in enumerate(targets, start=1):
        change = _odds_change(scores, mask, view)
        same = d.sign() * change.sign()
        meaningful = d != 0
        result[VIEWS[view - 1]] = dict(
            fraction_aligned=float((same[meaningful] > 0).float().mean()) if meaningful.any() else None,
            magnitude_weighted_fraction_aligned=float(d.abs()[same > 0].sum() / d.abs().sum())
            if meaningful.any() else None,
            mean_logodds_change=float(change.mean()),
        )
    return result


def fit_task(natural, variants):
    """Learn A, B and C from identical TRAIN people and view variants, each starting at equal weights."""
    valid = np.asarray(natural['reliable'])
    if valid.dtype != np.bool_ or valid.ndim != 1 or not valid.any() or valid.all():
        raise ValueError('Both Face availability groups must occur in the TRAIN sample')
    raw = _features(natural)
    norm = dict(content=stats(raw[0], valid, 8), relation=stats(raw[1], valid, 4))
    entries = [natural]
    xs = [_inputs(raw, valid, norm)]
    for name in VIEWS:
        variant = variants[name]
        for key in ('sample_ids', 'reliable', 'targets'):
            if key in variant and not np.array_equal(variant[key], natural[key]):
                raise ValueError(f'{name} TRAIN rows differ for {key}')
        entries.append(variant)
        xs.append(_inputs(_features(variant), valid, norm))
    x = torch.stack(xs, dim=1)
    z = torch.as_tensor(np.stack([row['branch_logits'] for row in entries], axis=1), dtype=torch.float32)
    labels = torch.as_tensor(natural['targets'], dtype=torch.float32).unsqueeze(1)
    if (labels.ndim != 3 or z.shape != (len(valid), 3, labels.shape[2], 3)
            or not torch.all((labels == 0) | (labels == 1)) or not torch.isfinite(z).all()):
        raise ValueError('Invalid protected branch logits or TRAIN labels')
    reliable = torch.from_numpy(valid)
    ones = torch.ones_like(reliable)
    usable = torch.stack((ones, ones, reliable), dim=1)
    model = dict(steps=STEPS, learning_rate=LR, paired_coefficient=1., projection_seed=PROJECTION_SEED,
                 stats=norm, weights={}, fit={}, alignment={})
    targets = []
    for method in ('A', 'B', 'C'):
        weights, fit, scores = _fit_one(x, z, labels, reliable, usable, method, targets)
        model['weights'][method] = weights
        model['fit'][method] = fit
        if method == 'A':
            targets = _utility_targets(z, labels, scores, reliable)
            model['pair_targets'] = {
                name: dict(count=len(d), clean_better=int((d > 0).sum()), degraded_better=int((d < 0).sum()),
                           tied=int((d == 0).sum()), mean_abs=float(d.abs().mean()),
                           mean_signed=float(d.mean()), max_abs=float(d.abs().max()))
                for name, (_, d) in zip(VIEWS, targets)
            }
        model['alignment'][method] = _alignment(scores, targets)
    return model


@torch.no_grad()
def apply_task(natural, model):
    """Predict task-level or person-dependent allocations from middle features alone."""
    valid = np.asarray(natural['reliable'])
    reliable = torch.from_numpy(valid)
    x = _inputs(_features(natural), valid, model['stats']).unsqueeze(1)
    pred = dict(fixed=apply_rule('fixed', natural, None, None).numpy())
    for method, params in model['weights'].items():
        w = params['matrix']
        matrix = torch.tensor(w, dtype=x.dtype) if w is not None else None
        scores = _gate_scores(x, reliable, torch.tensor(params['bias'], dtype=x.dtype), matrix)
        pred[method] = scores[:, 0].softmax(-1).numpy()
        if not np.isfinite(pred[method]).all():
            raise FloatingPointError('Nonfinite predicted weights')
    return pred
