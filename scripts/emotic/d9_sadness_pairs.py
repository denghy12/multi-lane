"""Describe which weight changes move positive and negative people past each other."""
import numpy as np


# Attribution is evaluated in float64; a near-equal pair of contributions is reported as a tie.
TOL = 1e-8
VIEWS = ('full', 'person', 'face')


def _mean(values):
    return float(values.mean()) if len(values) else None


def _views(values):
    return {v: _mean(values[:, j]) for j, v in enumerate(VIEWS)}


def _change(values):
    n = len(values)
    up = int((values > TOL).sum())
    down = int((values < -TOL).sum())
    return dict(mean=_mean(values), increased=up, decreased=down, unchanged=n-up-down,
                fraction_increased=up/n if n else None, fraction_decreased=down/n if n else None)


def _person_group(mask, z, w_a, w_c, score_a, score_c, dr, terms):
    return dict(count=int(mask.sum()), branch_logit_mean=_views(z[mask]),
                weight_a_mean=_views(w_a[mask]), weight_c_mean=_views(w_c[mask]),
                weight_change_mean=_views((w_c-w_a)[mask]), logit_a_mean=_mean(score_a[mask]),
                logit_c_mean=_mean(score_c[mask]), logit_change=_change((score_c-score_a)[mask]),
                rank_change=_change(dr[mask]),
                person_term=_change(terms[mask, 0]), face_term=_change(terms[mask, 1]))


def _empty_crossings():
    return dict(count=0, person_sum=0., face_sum=0., person_supports_flip=0,
                face_supports_flip=0, person_larger_in_flip_direction=0,
                face_larger_in_flip_direction=0, contributions_tied=0, no_directional_raw_change=0)


def _add_crossings(record, person_term, face_term, direction):
    # A loss is a drop in (positive - negative) margin; a gain is an increase.
    record['count'] += len(person_term)
    record['person_sum'] += float(person_term.sum())
    record['face_sum'] += float(face_term.sum())
    p = direction * person_term
    f = direction * face_term
    record['person_supports_flip'] += int((p > TOL).sum())
    record['face_supports_flip'] += int((f > TOL).sum())
    record['person_larger_in_flip_direction'] += int((p - f > TOL).sum())
    record['face_larger_in_flip_direction'] += int((f - p > TOL).sum())
    record['contributions_tied'] += int((np.abs(p-f) <= TOL).sum())
    record['no_directional_raw_change'] += int((p+f <= TOL).sum())


def _pair_group(pos, neg, rank_a, rank_c, terms):
    result = dict(total=len(pos)*len(neg), correct_a=0, correct_c=0,
                  lost=_empty_crossings(), gained=_empty_crossings())
    for start in range(0, len(pos), 32):
        people = pos[start:start + 32]
        before = rank_a[people, None] < rank_a[neg]
        after = rank_c[people, None] < rank_c[neg]
        result['correct_a'] += int(before.sum())
        result['correct_c'] += int(after.sum())
        for name, flipped, direction in [('lost', before & ~after, -1), ('gained', ~before & after, 1)]:
            i, j = np.nonzero(flipped)
            diff = terms[people[i]] - terms[neg[j]]
            _add_crossings(result[name], diff[:, 0], diff[:, 1], direction)
    return result


def _finish_pairs(record):
    for name in ('lost', 'gained'):
        crosses = record[name]
        n = crosses['count']
        for view in ('person', 'face'):
            total = crosses.pop(view + '_sum')
            crosses[view + '_mean_margin_change'] = total/n if n else None
    return record


def attribute(ids, y, z, w_a, w_c, valid, rank_a, rank_c):
    """Ranks are distinct 1-based positions produced by the official AP sorting rule."""
    ids, y, valid, ra, rc = map(np.asarray, (ids, y, valid, rank_a, rank_c))
    n = len(ids)
    if (not n or any(arr.shape != (n,) for arr in (ids, y, valid, ra, rc))
            or len(np.unique(ids)) != n or not np.isin(y, [0, 1]).all()
            or not np.isin(valid, [0, 1]).all()):
        raise ValueError('People, binary labels and availability must align')
    for ranks in (ra, rc):
        if not np.issubdtype(ranks.dtype, np.integer) or not np.array_equal(np.sort(ranks), np.arange(1, n+1)):
            raise ValueError('Expected ranks to be a permutation of 1 through N')
    valid, y = valid.astype(bool), y.astype(bool)
    z, wa, wc = (np.asarray(a, dtype=np.float64) for a in (z, w_a, w_c))
    if any(a.shape != (n, 3) or not np.isfinite(a).all() for a in (z, wa, wc)):
        raise ValueError('Expected finite Full/Person/Face scores and weights')
    for w in (wa, wc):
        if np.any(w < 0) or np.any(w[~valid, 2] != 0) or not np.allclose(w.sum(1), 1, rtol=0, atol=2e-6):
            raise ValueError('Invalid normalized view weights or unavailable Face weight')

    # With normalized weights the Full weight is determined by the two auxiliaries.
    terms = (wc[:, 1:] - wa[:, 1:]) * (z[:, 1:] - z[:, :1])
    score_a, score_c = (wa*z).sum(1), (wc*z).sum(1)
    remainder = (score_c - score_a) - terms.sum(1)
    np.testing.assert_allclose(remainder, z[:, 0]*(wc.sum(1)-wa.sum(1)), rtol=1e-8, atol=1e-10)
    people = {}
    for name, mask in [('positive', y), ('negative', ~y)]:
        for suffix, group in [('all', mask), ('reliable', mask & valid), ('unreliable', mask & ~valid)]:
            people[f'{name}_{suffix}'] = _person_group(group, z, wa, wc, score_a, score_c,
                                                      rc.astype(np.int64)-ra.astype(np.int64), terms)

    pairs = {}
    total = dict(total=0, correct_a=0, correct_c=0, lost=_empty_crossings(), gained=_empty_crossings())
    for p in (False, True):
        for m in (False, True):
            row = _pair_group(np.flatnonzero(y & (valid == p)), np.flatnonzero(~y & (valid == m)), ra, rc, terms)
            for key in ('total', 'correct_a', 'correct_c'):
                total[key] += row[key]
            for key in ('lost', 'gained'):
                for stat, value in row[key].items():
                    total[key][stat] += value
            key = 'positive_{}_negative_{}'.format('reliable' if p else 'unreliable',
                                                   'reliable' if m else 'unreliable')
            pairs[key] = _finish_pairs(row)
    assert total['correct_c'] - total['correct_a'] == total['gained']['count'] - total['lost']['count']
    pairs['all'] = _finish_pairs(total)
    return dict(tolerance=TOL, maximum_absolute_weight_roundoff_logit=float(np.max(np.abs(remainder))),
                rank_change_definition='rank_C minus rank_A; positive means lower in the list',
                pair_basis='official ranks; unweighted positive-negative pairs, independent of AP position weights',
                contribution_basis='change in positive minus negative logit margin; Full is the reference',
                people_groups=people, pairs=pairs)
