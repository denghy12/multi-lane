from __future__ import annotations

import unittest

import numpy as np
import torch

from multi_lane.track_a.class_aware_oof_stacking import (
    ClassAwareStacker,
    _named_final_ap_gains,
)
from multi_lane.track_a.runner import CLASS_ORDER
from multi_lane.track_a.three_view_router import INVALID_PRIOR, VALID_PRIOR


class ClassAwareOofStackingTest(unittest.TestCase):
    def test_bias_only_initialization_is_exact_r1_and_masks_face(self) -> None:
        model = ClassAwareStacker(classes=5)
        class_ids = torch.tensor([1, 3], dtype=torch.long)
        reliable = torch.tensor([True, False, True])
        weights, interaction = model(None, class_ids, reliable)
        np.testing.assert_allclose(
            weights.detach().numpy()[reliable.numpy()],
            np.tile(VALID_PRIOR, (2 * int(reliable.sum()), 1)).reshape(2, 2, 3),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            weights.detach().numpy()[~reliable.numpy()],
            np.tile(INVALID_PRIOR, (2, 1)).reshape(1, 2, 3),
            atol=1e-7,
        )
        self.assertEqual(torch.count_nonzero(interaction).item(), 0)
        self.assertEqual(torch.count_nonzero(weights[~reliable, :, 2]).item(), 0)

    def test_class_bias_changes_only_the_selected_class(self) -> None:
        model = ClassAwareStacker(classes=4)
        reliable = torch.ones(2, dtype=torch.bool)
        class_ids = torch.tensor([0, 2], dtype=torch.long)
        before, _ = model(None, class_ids, reliable)
        with torch.no_grad():
            model.class_bias[2, 2] += 1.0
        after, _ = model(None, class_ids, reliable)
        torch.testing.assert_close(before[:, 0], after[:, 0])
        self.assertFalse(torch.equal(before[:, 1], after[:, 1]))

    def test_rank_two_starts_as_exact_bias_model(self) -> None:
        bias = ClassAwareStacker(classes=4)
        rank_two = ClassAwareStacker(classes=4, feature_dim=6, rank=2, initialization_seed=7)
        with torch.no_grad():
            bias.class_bias[1] = torch.tensor([0.2, -0.1, -0.1])
            rank_two.class_bias.copy_(bias.class_bias)
        class_ids = torch.tensor([1, 3], dtype=torch.long)
        reliable = torch.tensor([True, False, True])
        expected, _ = bias(None, class_ids, reliable)
        actual, interaction = rank_two(torch.randn(3, 6), class_ids, reliable)
        torch.testing.assert_close(expected, actual)
        self.assertEqual(torch.count_nonzero(interaction).item(), 0)

    def test_invalid_dimensions_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            ClassAwareStacker(classes=3, feature_dim=4, rank=0)
        model = ClassAwareStacker(classes=3, feature_dim=4, rank=2)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3), torch.tensor([0]), torch.ones(2, dtype=torch.bool))

    def test_final_class_gains_use_serialized_per_class_ap_order(self) -> None:
        reference = {"task_metrics": [{"per_class_ap": [1.0] * len(CLASS_ORDER)}]}
        candidate = {
            "task_metrics": [
                {"per_class_ap": [1.0 + index / 100.0 for index in range(len(CLASS_ORDER))]}
            ]
        }
        gains = _named_final_ap_gains(candidate, reference)
        self.assertEqual(list(gains), list(CLASS_ORDER))
        self.assertAlmostEqual(gains[CLASS_ORDER[7]], 0.07)

        candidate["task_metrics"][-1]["per_class_ap"] = [1.0]
        with self.assertRaisesRegex(ValueError, "class order"):
            _named_final_ap_gains(candidate, reference)


if __name__ == "__main__":
    unittest.main()
