"""Pair untouched and branch-trained TRAIN people using only legal task labels and Face status."""
import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment

from multi_lane.track_a.runner import fit_calibration_indices, task_indices
from scripts.emotic.d9_sample_training import describe_cohort, image_key, task_cohorts


def label_rows(source, rows, classes):
    return np.array([[c in source.targets[i] for c in classes] for i in rows],
                    dtype=bool).reshape(len(rows), len(classes))


def group_audit(source, rows, classes):
    report = describe_cohort(source, rows, classes)
    report['negatives'] = [len(rows) - pos for pos in report['positives']]
    report['by_reliability'] = {}
    for name, flag in [('reliable', True), ('unreliable', False)]:
        indices = [i for i in rows if bool(source.face_reliable[i]) == flag]
        counts = label_rows(source, indices, classes).sum(0)
        report['by_reliability'][name] = dict(people=len(indices), positives=counts.tolist(),
                                               negatives=(len(indices) - counts).tolist(),
                                               face_training_valid=sum(bool(source.face_training_valid[i])
                                                                       for i in indices))
    return report


def pair_within_reliability(source, seen_pool, held, classes, task, seed):
    if len(seen_pool) < len(held):
        raise ValueError('Too few branch-role people to match this Face reliability stratum')
    if not held:
        return []
    # A stable person hash chooses among equally good branch candidates without model predictions.
    def draw_key(i):
        person = str(source.sample_ids[i])
        return hashlib.sha256(f'd9-source-match:{seed}:{task}:{person}'.encode()).digest(), person

    candidates = sorted(seen_pool, key=draw_key)
    want = label_rows(source, held, classes)
    pool = label_rows(source, candidates, classes)
    mismatch = np.zeros((len(held), len(candidates)), dtype=np.int32)
    for a, b in zip(want.T, pool.T):
        mismatch += (a[:, None] != b)
    # Lexicographic assignment: minimize Face training-valid mismatches first, maximize exact
    # label vectors next, then minimize the remaining total number of mismatched labels.
    n = len(held)
    exact_cost = n * len(classes) + 1
    face_cost = n * (exact_cost + len(classes)) + 1
    valid = np.asarray(source.face_training_valid, dtype=bool)
    costs = mismatch.astype(np.int64) + exact_cost * (mismatch != 0)
    costs += face_cost * (valid[held, None] != valid[candidates])
    rows, columns = linear_sum_assignment(costs)
    if rows.tolist() != list(range(n)):
        raise RuntimeError('Incomplete TRAIN matching')
    return [candidates[j] for j in columns]


def select_groups(source, task, roles, expected_cohort, seed=1):
    """Return source row indices in paired order and JSON-serializable balance diagnostics."""
    classes = task_indices(task)
    eligible, _ = fit_calibration_indices(source, classes, 0.)
    ids = [str(source.sample_ids[i]) for i in eligible]
    if len(set(ids)) != len(ids) or any(roles.get(image_key(person)) not in ('branch','fusion') for person in ids):
        raise ValueError('Duplicate TRAIN person or missing original-image 90/10 role')
    pool, held = task_cohorts(source, eligible, roles, False)
    replay = {name: describe_cohort(source, rows, classes) for name, rows in [('branch', pool), ('fusion', held)]}
    if replay != expected_cohort:
        raise ValueError('Mismatch against registered 90/10 task TRAIN cohorts')
    seen_by_held = {}
    for flag in (False, True):
        held_stratum = [i for i in held if bool(source.face_reliable[i]) == flag]
        seen_stratum = [i for i in pool if bool(source.face_reliable[i]) == flag]
        matched = pair_within_reliability(source, seen_stratum, held_stratum, classes, task, seed)
        seen_by_held.update(zip(held_stratum, matched))
    seen = [seen_by_held[i] for i in held]
    held_images = {image_key(source.sample_ids[i]) for i in held}
    branch_images = {image_key(source.sample_ids[i]) for i in pool}
    if held_images & branch_images or len(set(seen)) != len(held):
        raise ValueError('Group image overlap or re-use of branch-trained people')
    seen_y, held_y = (label_rows(source, rows, classes) for rows in (seen, held))
    match = seen_y == held_y
    seen_status, held_status = ([bool(source.face_training_valid[i]) for i in rows] for rows in (seen, held))
    audit = dict(task=task, seed=seed, class_indices=list(classes), people_per_group=len(held),
                 exact_label_vectors=int(match.all(axis=1).sum()), label_mismatches_by_class=(~match).sum(0).tolist(),
                 total_label_mismatches=int((~match).sum()),
                 face_training_valid_mismatches=int(np.count_nonzero(np.asarray(seen_status) != held_status)),
                 face_reliability_mismatches=0, original_image_overlap=0,
                 seen=group_audit(source, seen, classes), held=group_audit(source, held, classes),
                 seen_pool=replay['branch'])
    return dict(seen=seen, held=held), audit
