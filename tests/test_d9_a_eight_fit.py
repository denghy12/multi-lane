import unittest

import numpy as np

from scripts.emotic.d9_a_eight_fit import check_cohort, group_proportions
from multi_lane.track_a.runner import task_indices


class ClosedHeadFitTest(unittest.TestCase):
    def test_equal_initialization_and_no_fixed_anchor(self):
        zero = np.zeros((2, 3))
        np.testing.assert_allclose(group_proportions(zero), [[.5, .5, 0], [1/3]*3])
        moved = group_proportions([[0, 7, 200], [0, 0, 7]])
        self.assertEqual(moved[0][2], 0)
        self.assertGreater(moved[1][2], .99)
        self.assertGreater(moved[0][1], .99)

    def test_other_role_and_wrong_current_classes_are_rejected(self):
        y = np.ones((2, len(task_indices(6))), dtype=int)
        d = dict(sample_ids=['train:a#person=0', 'train:b#person=0'], targets=y,
                 reliable=np.array([False, True]))
        meta = {'sample_id_sha256': 'hash'}
        c = dict(people=2, images=2, sample_id_sha256='hash', reliable_face=1,
                 positives=[2]*y.shape[1], reliable_face_positives=[1]*y.shape[1], classes=list(task_indices(6)))
        roles = {'train:a': 'fusion', 'train:b': 'fusion'}
        check_cohort(d, meta, c, roles, 6)
        with self.assertRaises(ValueError):
            check_cohort(d, meta, c, dict(roles, **{'train:a': 'branch'}), 6)
        with self.assertRaises(ValueError):
            check_cohort(d, meta, dict(c, classes=list(task_indices(7))), roles, 6)
