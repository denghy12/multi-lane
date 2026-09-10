from __future__ import annotations

import unittest

import numpy as np
import torch

from multi_lane.track_a.three_view_oof_router import SharedTaskBiasRouter
from multi_lane.track_a.three_view_router import INVALID_PRIOR, VALID_PRIOR


class ThreeViewOofRouterTest(unittest.TestCase):
    def test_initialization_matches_fixed_anchor_and_masks_face(self) -> None:
        router = SharedTaskBiasRouter(feature_dim=7, tasks=8, initialization_seed=3)
        features = torch.randn(4, 7)
        task_ids = torch.tensor([0, 1, 6, 7])
        reliable = torch.tensor([True, False, True, False])
        weights = router(features, task_ids, reliable).detach().numpy()
        np.testing.assert_allclose(
            weights[reliable.numpy()], np.tile(VALID_PRIOR, (2, 1)), atol=1e-7
        )
        np.testing.assert_allclose(
            weights[~reliable.numpy()], np.tile(INVALID_PRIOR, (2, 1)), atol=1e-7
        )
        self.assertTrue(np.array_equal(weights[~reliable.numpy(), 2], np.zeros(2)))

    def test_task_bias_changes_only_selected_task_rows(self) -> None:
        router = SharedTaskBiasRouter(feature_dim=3, tasks=2, initialization_seed=1)
        features = torch.zeros(2, 3)
        reliable = torch.ones(2, dtype=torch.bool)
        before = router(features, torch.tensor([0, 1]), reliable).detach()
        with torch.no_grad():
            router.task_bias.weight[1, 2] += 1.0
        after = router(features, torch.tensor([0, 1]), reliable).detach()
        torch.testing.assert_close(before[0], after[0])
        self.assertFalse(torch.equal(before[1], after[1]))

    def test_invalid_input_shapes_are_rejected(self) -> None:
        router = SharedTaskBiasRouter(feature_dim=3, tasks=2, initialization_seed=1)
        with self.assertRaises(ValueError):
            router(torch.zeros(2, 3), torch.zeros(1, dtype=torch.long), torch.ones(2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
