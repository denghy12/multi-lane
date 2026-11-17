import unittest

import numpy as np
import torch

from multi_lane.track_a.runner import average_precision
from scripts.emotic.d9_direction_permutation import (
    batch_map, difference_summary, permutation_sources, shuffled_logits,
)
from scripts.emotic.evaluate_d9_relation import SHUFFLE_SEED, shuffle_sources, weighted_logits


class DirectionPermutationTests(unittest.TestCase):
    def test_group_preservation_and_original_seed(self):
        ids = ['z', 'a', 'c', 'd', 'e']
        reliability = np.array([True, False, True, True, True])
        expected = shuffle_sources(ids, reliability, 2).numpy()
        donors = permutation_sources(ids, reliability, 2, SHUFFLE_SEED)
        np.testing.assert_array_equal(donors, expected)
        np.testing.assert_array_equal(reliability[donors], np.broadcast_to(reliability[:, None], donors.shape))
        self.assertTrue((donors[1] == 1).all())
        self.assertTrue((donors[reliability] != np.flatnonzero(reliability)[:, None]).all())
        for task in range(2):
            np.testing.assert_array_equal(np.sort(donors[:, task]), np.arange(len(ids)))

    def test_vectorized_map_matches_official_with_ties_and_saturation(self):
        rng = np.random.default_rng(31)
        logits = rng.integers(-3, 4, size=(4, 31, 5)).astype(np.float32)
        logits[:, 0:5, :] = 100.
        logits[:, 5:9, :] = -100.
        labels = rng.integers(0, 2, size=(31, 5)).astype(np.float32)
        labels[:, 0] = 0
        labels[:, 1] = 1
        expected = []
        for score in logits:
            p = torch.from_numpy(score).sigmoid().numpy()
            expected.append(np.mean([100. * average_precision(p[:, c], labels[:, c]) for c in range(5)]))
        np.testing.assert_allclose(batch_map(logits, labels), expected, rtol=0., atol=1e-12)

    def test_shared_donors_match_official_weighting_and_constant_replay(self):
        rng = np.random.default_rng(4)
        branches = rng.standard_normal((5, 5, 3)).astype(np.float32)
        weights = rng.uniform(size=(4, 5, 2, 3)).astype(np.float32)
        weights /= weights.sum(-1, keepdims=True)
        weights[0] = weights[0, :1]
        sizes = [2, 3]
        actual = np.stack([weighted_logits(torch.from_numpy(branches), torch.from_numpy(w), sizes).numpy()
                           for w in weights])
        # Mimic archived inference rounding on the method whose weights are constant.
        actual[0] = np.nextafter(actual[0], np.float32(np.inf))
        donors = permutation_sources(['a', 'b', 'c', 'd', 'e'], np.array([True] * 5), 2, 31)
        fused = shuffled_logits(branches, weights, actual, donors, sizes)
        np.testing.assert_array_equal(fused[0], actual[0])
        for method in (1, 2, 3):
            shifted = weights[method, donors, np.arange(2)]
            expected = weighted_logits(torch.from_numpy(branches), torch.from_numpy(shifted), sizes)
            np.testing.assert_array_equal(fused[method], expected.numpy())
        summary = difference_summary(2., [1., 2., 3.])
        self.assertEqual(summary['positive_count'], 1)
        self.assertEqual(summary['equal_count'], 1)
        self.assertEqual(summary['mean'], 0.)


if __name__ == '__main__':
    unittest.main()
