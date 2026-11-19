import unittest

import numpy as np
import torch

from scripts.emotic.d9_group_offset_eval import apply_offsets, audit_group_ranks, cross_group_accuracy


class OffsetEvaluationTest(unittest.TestCase):
    def test_task_specific_translation_with_raw_ties(self):
        z = torch.tensor([[2., 0., 1.], [1., 1., 4.], [2., 2., 3.], [1., 0., 5.]])
        r = torch.tensor([False, False, False, True])
        exact, official = apply_offsets(z, r, [2, 1], [-0.5, 0.25])
        torch.testing.assert_close(official[~r], z[~r] + torch.tensor([-0.5,-0.5,0.25]))
        torch.testing.assert_close(official[r], z[r], atol=0, rtol=0)
        m = dict(per_class_ap=[100.]*3, mAP=100.)
        audit_group_ranks(z[~r], exact[~r], official[~r], ['a','b','c'], dict(before=m, after=m))
        changed = exact[~r].clone()
        changed[0,0] += .1
        with self.assertRaises(ValueError):
            audit_group_ranks(z[~r], changed, official[~r], ['a','b','c'], dict(before=m, after=m))

    def test_cross_group_auc_counts_both_directions_and_half_ties(self):
        scores = np.array([[1., 2.], [2., 2.], [1., 2.], [0., 2.]])
        labels = np.array([[1,0], [0,0], [0,0], [1,0]])
        reliable = np.array([True, True, False, False])
        before = cross_group_accuracy(scores, labels, reliable)
        self.assertEqual(before['valid_classes'], 1)
        self.assertEqual(before['classes'][0]['pairs'], 2)
        self.assertEqual(before['percent'], 25.)
        scores[~reliable] += 3.
        self.assertEqual(cross_group_accuracy(scores, labels, reliable)['percent'], 50.)
