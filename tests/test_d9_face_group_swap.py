import unittest

import torch

from scripts.emotic.d9_face_group_swap import group_readouts
from scripts.emotic.evaluate_d9_relation import metrics_for_stage


class FaceGroupSwapTest(unittest.TestCase):
    def test_mask_selects_all_class_logits_for_each_person(self):
        fixed = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        paired = torch.tensor([[7., 8.], [9., 10.], [11., 12.]])
        reliable = torch.tensor([False, True, False])

        actual = group_readouts(fixed, paired, reliable)
        expected = {
            'fixed': [[1, 2], [3, 4], [5, 6]],
            'only_unreliable': [[7, 8], [3, 4], [11, 12]],
            'only_reliable': [[1, 2], [9, 10], [5, 6]],
            'paired': [[7, 8], [9, 10], [11, 12]],
        }
        self.assertEqual(set(actual), set(expected))
        for name, rows in expected.items():
            with self.subTest(name=name):
                torch.testing.assert_close(actual[name], fixed.new_tensor(rows), rtol=0, atol=0)

    def test_full_population_reranks_people_across_groups(self):
        reliable = torch.tensor([True, True, False, False])
        labels = torch.tensor([[1.], [0.], [1.], [0.]]).repeat(1, 5)
        fixed = torch.tensor([[4.], [2.], [3.], [1.]]).repeat(1, 5)
        paired = torch.tensor([[-1.], [-2.], [3.], [1.]]).repeat(1, 5)
        scores = group_readouts(fixed, paired, reliable)
        metrics = metrics_for_stage(0, labels, reliable, scores)
        # Both groups still rank their positive first. Across groups, the positives
        # fall at ranks one and three when only the reliable group changes.
        for group in ('reliable', 'unreliable'):
            self.assertAlmostEqual(metrics['only_reliable'][group]['mAP'], 100., places=5)
        self.assertAlmostEqual(metrics['fixed']['all']['mAP'], 100., places=5)
        self.assertAlmostEqual(metrics['only_reliable']['all']['mAP'], 100 * (1 + 2/3) / 2, places=5)
