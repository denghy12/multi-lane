import unittest

import numpy as np
import torch

from multi_lane.track_a.runner import average_precision
from scripts.emotic.run_r1_frozen9_fusion_ap import (
    candidate_logits, candidate_map, candidate_weights, coarse_coordinates, fine_coordinates, search_grid,
)
from scripts.emotic.run_r1_frozen9_learned_fusion import blend


class Frozen9FusionAPTests(unittest.TestCase):
    def test_grid_includes_vertices_and_face_mask_is_exact(self):
        coarse = coarse_coordinates()
        self.assertEqual(len(coarse), 4851)
        for point in ((0, 0, 0), (100, 0, 100), (0, 100, 0)):
            self.assertIn(point, coarse)
            self.assertIn(point, fine_coordinates(point))
        w = candidate_weights(coarse)
        np.testing.assert_allclose(w.sum(axis=2), 1, rtol=0, atol=1e-7)
        self.assertTrue(np.all(w >= 0))
        np.testing.assert_array_equal(w[:, 0, 2], 0)

    def test_batch_ap_matches_reference_on_joint_population(self):
        rng = np.random.default_rng(17)
        logits = np.round(rng.normal(size=(113, 3, 3)), 2).astype(np.float32)
        targets = (rng.random((113, 3)) < .25).astype(np.float32)
        targets[:, 2] = 0  # Match the repository's AP=0 convention.
        reliable = rng.random(113) < .55
        points = [(0, 0, 0), (15, 25, 45), (0, 100, 100)]
        batched = candidate_logits(logits, reliable, points)
        expected = []
        for point, fused in zip(points, batched):
            z = blend(logits, reliable, candidate_weights([point])[0])
            np.testing.assert_array_equal(fused, z)
            probabilities = torch.from_numpy(z).sigmoid().numpy()
            expected.append(100 * np.mean([
                average_precision(probabilities[:, col], targets[:, col])
                for col in range(targets.shape[1])
            ]))
        np.testing.assert_allclose(candidate_map(batched, targets), expected, rtol=0, atol=1e-12)

    def test_search_can_learn_distinct_boundary_weights_without_prior(self):
        targets = np.array([[1], [0], [1], [0]], dtype=np.float32)
        reliable = np.array([True, True, False, False])
        scores = 4 * (2 * targets - 1)
        logits = np.stack((-scores, np.where(reliable[:, None], -scores, scores), scores), axis=1)
        best = search_grid(logits, targets, reliable, coarse_coordinates())
        self.assertEqual(best.coordinates, (0, 100, 100))
        self.assertGreater(best.mean_ap, 99.99)
