import unittest

import numpy as np

from scripts.emotic.d9_group_offset_fit import fit_offset


class OffsetFittingTest(unittest.TestCase):
    def test_exact_direction_counts_and_shared_shift(self):
        z = np.zeros((5, 1))
        y = np.array([[1], [0], [1], [1], [0]])
        reliable = np.array([True, True, False, False, False])
        b, diag = fit_offset(z, y, reliable)
        self.assertEqual(diag['directional_pairs'], dict(unreliable_positive=2, reliable_positive=1))
        self.assertAlmostEqual(b, np.log(2), places=9)
        self.assertLess(diag['after']['loss'], diag['before']['loss'])
        self.assertAlmostEqual(diag['after']['gradient'], 0., places=9)
        z[~reliable] += 1000.
        shifted, _ = fit_offset(z, y, reliable)
        self.assertAlmostEqual(shifted, b-1000., places=8)

    def test_unequal_class_pair_counts_do_not_reweight_classes(self):
        z = np.zeros((6, 2))
        y = np.array([[1, 1], [1, 1], [0, 0], [1, 0], [1, 0], [0, 0]])
        reliable = np.array([True, True, True, False, False, False])
        # Two of four pairs favor U in class one; zero of six in class two.
        # Equal class weights give P=1/4, pooling pairs would give P=1/5.
        b, diag = fit_offset(z, y, reliable)
        self.assertEqual(diag['active_classes'], 2)
        self.assertAlmostEqual(b, -np.log(3), places=9)

    def test_unidentified_and_one_sided_objectives_revert_to_zero(self):
        y = np.array([[0], [1]])
        z = np.zeros_like(y)
        for reliable, status in [(np.array([True, True]), 'no_cross_group_pairs'),
                                 (np.array([True, False]), 'only_one_direction')]:
            b, diag = fit_offset(z, y, reliable)
            self.assertEqual((b, diag['status']), (0., status))
