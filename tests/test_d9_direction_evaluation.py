import unittest

import numpy as np
import torch

from scripts.emotic.d9_direction_metrics import (
    direction_metrics, selection_statistics, tied_auc,
)


class DirectionEvaluationTest(unittest.TestCase):
    def test_auc_counts_exact_ties(self):
        self.assertEqual(tied_auc([0, 0, 0, 0], [False, True, False, True]), .5)
        self.assertEqual(tied_auc([1, 2], [False, True]), 1.)
        self.assertEqual(tied_auc([2, 1], [False, True]), 0.)
        self.assertEqual(tied_auc([1, 1, 0], [True, False, False]), .75)
        self.assertIsNone(tied_auc([1, 2], [True, True]))
        self.assertIsNone(tied_auc([], []))

    def test_selection_zero_is_no_improvement_and_threshold_fixed(self):
        row = selection_statistics([0, 1, -1], [0, 2, -2])
        self.assertEqual(row['improvement_positive'], 1)
        self.assertEqual(row['exact_zero_reduction'], 1)
        self.assertEqual(row['selected_precision'], 1.)
        self.assertEqual(row['selected_mean_bce_reduction'], 2.)
        self.assertEqual(row['auc'], 1.)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            selection_statistics([np.nan], [1])

    def test_center_b_uses_learned_center_and_caps_transfer(self):
        from multi_lane.track_a.d9_direction_objective import direction_benefits
        labels = torch.ones(2, 1)
        reliable = torch.tensor([True, False])
        branches = torch.tensor([[[0., 3., -2.]], [[0., 3., -2.]]])
        pred = torch.ones(2, 1, 2)
        center = torch.tensor([[[.01, .49, .50]], [[.02, .98, .0]]])
        result, arrays = direction_metrics(labels, reliable, branches,
                                            dict(a=pred, b=pred, c=pred, center_b=pred),
                                            [1], center_weights=center)
        expected, valid = direction_benefits(branches, labels, reliable,
                                             base_weights=center[:, 0])
        np.testing.assert_array_equal(arrays['center_direction_bce_reduction'][:, 0], expected.numpy())
        self.assertFalse(np.array_equal(arrays['direction_bce_reduction'],
                                        arrays['center_direction_bce_reduction']))
        baseline = (branches * center).sum(-1)
        shifted = baseline + center[:, :, 0] * (branches[:, :, 1] - branches[:, :, 0])
        loss = torch.nn.functional.binary_cross_entropy_with_logits
        manual = (loss(baseline, labels, reduction='none')
                  - loss(shifted, labels, reduction='none')).mean(1)
        torch.testing.assert_close(expected[:, 0], manual)
        with self.assertRaisesRegex(ValueError, 'learned center weights'):
            direction_metrics(labels, reliable, branches,
                              dict(a=pred, b=pred, c=pred, center_b=pred), [1])

    def test_task_and_availability_groups_and_finite_transfer(self):
        labels = torch.ones(4, 2)
        reliable = torch.tensor([True, True, False, False])
        # Positive auxiliary logits improve the positive class relative to Full.
        branches = torch.zeros(4, 2, 3)
        branches[:, 0, 1:] = torch.tensor([3., -3., 3., -3.])[:, None]
        branches[:, 1, 1:] = -branches[:, 0, 1:]
        pred = torch.tensor([[[1., 1.], [-1., -1.]],
                             [[-1., -1.], [1., 1.]],
                             [[1., 1.], [-1., -1.]],
                             [[-1., -1.], [1., 1.]]])
        result, arrays = direction_metrics(labels, reliable, branches,
                                            dict(a=pred, b=pred, c=-pred), [1, 1])
        self.assertEqual(arrays['direction_bce_reduction'].shape, (4, 2, 2))
        self.assertFalse(arrays['direction_valid'][~reliable, :, 1].any())
        for task in range(2):
            report = result['methods']['b'][task]['groups']
            self.assertEqual(report['reliable']['person']['auc'], 1.)
            self.assertEqual(report['reliable']['face']['auc'], 1.)
            self.assertEqual(report['unreliable']['person']['people'], 2)
            self.assertEqual(report['unreliable']['face']['people'], 0)
            self.assertIsNone(report['unreliable']['face']['auc'])
            self.assertEqual(result['methods']['c'][task]['groups']['reliable']['face']['auc'], 0.)
        # Reordering subjects must not change the group metrics.
        order = torch.tensor([3, 1, 0, 2])
        shuffled, _ = direction_metrics(labels[order], reliable[order], branches[order],
                                         dict(a=pred[order], b=pred[order], c=-pred[order]), [1, 1])
        self.assertEqual(result, shuffled)


if __name__ == '__main__':
    unittest.main()
