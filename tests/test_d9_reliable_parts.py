import unittest

import torch

from scripts.emotic.d9_reliable_parts import reliable_readouts, weight_grid


class ReliableFusionPartsTest(unittest.TestCase):
    def setUp(self):
        self.learned = torch.tensor([[.45, .15, .40], [.20, .30, .50]])
        self.expected = {
            'base': [[.64, .16, .20], [.64, .16, .20]],
            'face_only': [[.48, .12, .40], [.40, .10, .50]],
            'ratio_only': [[.60, .20, .20], [.32, .48, .20]],
            'both': self.learned.tolist(),
        }

    def test_replaces_one_share_for_each_task(self):
        grid = weight_grid(self.learned)
        self.assertEqual(set(grid), set(self.expected))
        for name, rows in self.expected.items():
            with self.subTest(name=name):
                weights = grid[name]
                torch.testing.assert_close(weights, torch.tensor(rows), atol=1e-7, rtol=0)
                self.assertTrue((weights >= 0).all())
                torch.testing.assert_close(weights.sum(dim=1), torch.ones(2), atol=1e-7, rtol=0)

    def test_mask_and_task_of_introduction_choose_the_right_readout(self):
        reliable = torch.tensor([True, False, True])
        sizes = [2, 1]
        branches = torch.tensor([
            [[2., 8., 4.], [12., -6., 2.], [1., -2., -7.]],
            [[7., 2., 9.], [2., 20., -9.], [3., 11., 10.]],
            [[7., 6., -1.], [-3., 2., 1.], [10., -8., 2.]],
        ])

        def score_by_head(weights):
            by_class = torch.tensor([weights[0], weights[0], weights[1]])
            return (branches * by_class).sum(dim=2)

        fixed = score_by_head(self.expected['base'])
        fixed[~reliable] = score_by_head([[.8, .2, 0.], [.8, .2, 0.]])[~reliable]
        paired = score_by_head(self.expected['both'])
        paired[~reliable] = score_by_head([[.55, .45, 0.], [.85, .15, 0.]])[~reliable]
        results = reliable_readouts(branches, reliable, sizes, self.learned, fixed, paired)
        for method, rows in self.expected.items():
            with self.subTest(method=method):
                torch.testing.assert_close(results[method][~reliable], paired[~reliable], rtol=0, atol=0)
                torch.testing.assert_close(results[method][reliable], score_by_head(rows)[reliable], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(results['base'][reliable], fixed[reliable], rtol=0, atol=0)
        torch.testing.assert_close(results['both'], paired, rtol=0, atol=0)
