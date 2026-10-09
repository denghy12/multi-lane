import unittest
import numpy as np
import torch
from multi_lane.track_a.frozen_fusion_calibration_support import support, combine_cached
from multi_lane.track_a.frozen_view_ranking_fusion import fit_gate


class CalibrationSupportTest(unittest.TestCase):
    def test_support_uses_current_classes_only(self):
        labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1]])
        row = support(labels, [1, 2], 1)
        self.assertEqual(row['all_current_negative'], 1)
        self.assertEqual(row['positive_counts'], [1, 2])

    def test_existing_rows_preserved_exactly_in_dataset_order(self):
        def data(ids, offset):
            out = {'sample_ids': np.array(ids)}
            for key in ('targets', 'face_reliable', 'fused_logits', 'fused_probabilities',
                        'full_logits', 'person_logits', 'face_logits',
                        'full_probabilities', 'person_probabilities', 'face_probabilities'):
                out[key] = np.arange(len(ids))[:, None] + offset
            return out
        old, added = data(['train:b', 'train:d'], 10), data(['train:a'], 100)
        merged = combine_cached(old, added, ['train:a', 'train:b', 'train:d'])
        np.testing.assert_array_equal(merged['full_logits'][1:], old['full_logits'])
        with self.assertRaises(ValueError):
            combine_cached(old, added, ['train:b'])

    def test_epoch_samples_match_update_and_example_budgets(self):
        torch.manual_seed(9)
        views = torch.randn(24, 2, 3)
        fixed = views.mean(-1)
        labels = (torch.arange(48).reshape(24, 2) % 3 == 0).float()
        reliable = torch.ones(24, dtype=torch.bool)
        _, audit = fit_gate(views, fixed, labels, reliable, 'dynamic_ranking',
                            torch.device('cpu'), epochs=3, batch_size=4, epoch_samples=7)
        self.assertEqual(audit['epoch_samples'], 7)
        self.assertEqual(sum(row['updates'] for row in audit['history']), 6)
        self.assertEqual(sum(row['samples_used'] for row in audit['history']), 21)
        with self.assertRaises(ValueError):
            fit_gate(views, fixed, labels, reliable, 'dynamic_bce', torch.device('cpu'), epoch_samples=25)


if __name__ == '__main__':
    unittest.main()
