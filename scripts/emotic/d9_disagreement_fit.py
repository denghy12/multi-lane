"""Fit frozen-branch disagreement probes; all thresholds and normalization use TRAIN only."""
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit

from scripts.emotic.fit_d9_task10_probe import apply_rule, build_features, mix


def evidence(data, stats=None):
    reliable = data['reliable']
    z = np.asarray(data['branch_logits'], dtype=np.float32)
    baseline = mix(apply_rule('fixed', data, None, None), torch.from_numpy(z)).numpy()
    if 'fixed_logits' in data:
        stored = np.asarray(data['fixed_logits'], dtype=np.float32)
        np.testing.assert_allclose(baseline, stored, atol=2e-5, rtol=2e-6)
        baseline = stored

    # Face can win only on reliable examples. Exact ties go to Person.
    face_wins = reliable[:, None] & (z[..., 2] > z[..., 1])
    receiver = np.where(face_wins, 2, 1).astype('int8')
    gap = np.where(face_wins, z[..., 2], z[..., 1]) - z[..., 0]
    features, _, stats = build_features(data, stats)
    middle = np.column_stack((features['C'], features['B_std']))
    return baseline, receiver, gap, middle, stats


def disagreement(gap, threshold):
    if threshold is None:
        return np.zeros(len(gap), dtype=bool)
    return (gap > 0) & (gap >= threshold)


def designs(baseline, gap, middle, reliable, receiver, candidate, normalization=None):
    """Both probes see the same two categorical flags; only the evidence columns differ."""
    observed = np.column_stack((baseline[candidate], gap[candidate])).astype(np.float64)
    if normalization is None:
        mean, scale = observed.mean(0), observed.std(0)
        scale[scale < 1e-6] = 1.0
        normalization = {'mean': mean.tolist(), 'scale': scale.tolist()}
    standardized = (observed - np.asarray(normalization['mean'])) / np.asarray(normalization['scale'])
    flags = np.column_stack((receiver[candidate] == 2, reliable[candidate]))
    return {'middle': np.column_stack((middle[candidate], flags)),
            'score': np.column_stack((standardized, flags))}, normalization


def fit_classifier(x, y):
    """Class-balanced binary logistic loss with L2 coefficient penalty, C=1."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sign = 2 * y - 1
    weights = np.where(y == 1, .5 * len(y) / y.sum(), .5 * len(y) / (len(y) - y.sum()))

    def objective(params):
        coef = params[:-1]
        margin = sign * (x @ coef + params[-1])
        loss = np.dot(weights, np.logaddexp(0., -margin)) + .5 * np.dot(coef, coef)
        derivative = -weights * sign * expit(-margin)
        grad = np.append(x.T @ derivative + coef, derivative.sum())
        return loss, grad

    opt = minimize(objective, np.zeros(x.shape[1]+1), method='L-BFGS-B',
                   jac=True, options={'maxiter': 500, 'gtol': 1e-6, 'ftol': 1e-12})
    if not opt.success or not np.isfinite(opt.x).all():
        raise RuntimeError(f'Disagreement gate did not converge: {opt.message}')
    return {'coef': opt.x[:-1].tolist(), 'intercept': float(opt.x[-1]), 'iterations': int(opt.nit)}


def fit_task(data):
    baseline, receiver, gap, middle, stats = evidence(data)
    classes = []
    for index, labels in enumerate(data['targets'].T):
        advantage = gap[:, index]
        positive_gaps = advantage[advantage > 0]
        threshold = float(np.median(positive_gaps)) if positive_gaps.size else None
        eligible = disagreement(advantage, threshold)
        y = labels[eligible]
        row = {'threshold': threshold, 'candidate_count': len(y), 'positive_count': int(y.sum()),
               'positive_gap_count': len(positive_gaps), 'gates': {}}
        if len(np.unique(y)) == 2:
            inputs, norm = designs(baseline[:, index], advantage, middle, data['reliable'],
                                   receiver[:, index], eligible)
            row['score_normalization'] = norm
            row['gates'] = {name: fit_classifier(x, y) for name, x in inputs.items()}
        classes.append(row)
    return {'feature_stats': stats, 'classes': classes}


def apply_task(data, model):
    """Generate decisions without looking at the targets of the current people."""
    baseline, receiver, gap, middle, _ = evidence(data, model['feature_stats'])
    if gap.shape[1] != len(model['classes']):
        raise ValueError('Class count differs between source logits and fitted disagreement gates')
    candidate = np.zeros(gap.shape, dtype=bool)
    probabilities = {name: np.zeros(gap.shape, dtype=np.float64) for name in ('middle', 'score')}
    for index, row in enumerate(model['classes']):
        eligible = disagreement(gap[:, index], row['threshold'])
        candidate[:, index] = eligible
        if not row['gates'] or not eligible.any():
            continue
        inputs, _ = designs(baseline[:, index], gap[:, index], middle, data['reliable'],
                            receiver[:, index], eligible, row['score_normalization'])
        for name, params in row['gates'].items():
            probabilities[name][eligible, index] = expit(inputs[name] @ params['coef'] + params['intercept'])
    return dict(candidate=candidate, receiver=receiver, gap=gap, fixed_logits=baseline,
                middle_probability=probabilities['middle'], score_probability=probabilities['score'])
