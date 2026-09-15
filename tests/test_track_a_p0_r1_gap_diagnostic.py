from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.p0_r1_gap_diagnostic import (
    _bootstrap_final_map_differences,
    fuse_logits,
    fuse_probabilities,
    load_view_scores,
)
from multi_lane.track_a.runner import evaluate_view_diagnostics
from test_track_a_end_to_end_view_fusion import three_view_batch, tiny_model


class P0R1GapDiagnosticTest(unittest.TestCase):
    def test_locked_probability_fusion_masks_face(self) -> None:
        endpoints = {
            "full": np.asarray([[0.1], [0.1]], dtype=np.float32),
            "person": np.asarray([[0.5], [0.5]], dtype=np.float32),
            "face": np.asarray([[0.9], [0.9]], dtype=np.float32),
        }
        result = fuse_probabilities(endpoints, np.asarray([True, False]))
        np.testing.assert_allclose(result[:, 0], [0.324, 0.18], rtol=0, atol=1e-7)

    def test_probability_and_logit_fusion_are_distinct(self) -> None:
        logits = {
            "full": np.asarray([[4.0]], dtype=np.float32),
            "person": np.asarray([[-2.0]], dtype=np.float32),
            "face": np.asarray([[0.5]], dtype=np.float32),
        }
        probabilities = {
            name: torch.sigmoid(torch.from_numpy(value)).numpy()
            for name, value in logits.items()
        }
        reliable = np.asarray([True])
        self.assertGreater(
            abs(float(fuse_probabilities(probabilities, reliable)[0, 0])
                - float(fuse_logits(logits, reliable)[0, 0])),
            1e-3,
        )

    def test_view_diagnostic_export_round_trip(self) -> None:
        model = tiny_model("fixed_three_view")
        images = three_view_batch()
        targets = torch.zeros(3, 5)
        targets[0, 0] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task0.npz"
            metrics = evaluate_view_diagnostics(
                model,
                [(images, targets, ["val:a#person=0", "val:a#person=1", "val:b#person=0"])],
                torch.device("cpu"),
                0,
                0.5,
                False,
                path,
            )
            dump = load_view_scores(path)
        self.assertEqual(dump.task_id, 0)
        self.assertEqual(dump.fused_logits.shape, (3, 5))
        self.assertEqual(set(dump.logits), {"full", "person", "face"})
        np.testing.assert_array_equal(dump.face_reliable, [True, False, True])
        self.assertEqual(metrics["metrics"]["fused"]["samples"], 3)

    def test_group_bootstrap_is_deterministic(self) -> None:
        ids = np.asarray([
            "val:a#person=0", "val:a#person=1", "val:b#person=0", "val:c#person=0"
        ])
        targets = np.asarray([[1], [0], [1], [0]], dtype=np.float32)
        methods = {
            "better": np.asarray([[0.9], [0.8], [0.7], [0.1]], dtype=np.float32),
            "worse": np.asarray([[0.1], [0.8], [0.2], [0.9]], dtype=np.float32),
        }
        first = _bootstrap_final_map_differences(
            ids, targets, methods, (("better", "worse"),), 20
        )
        second = _bootstrap_final_map_differences(
            ids, targets, methods, (("better", "worse"),), 20
        )
        self.assertEqual(first, second)
        self.assertEqual(first["group_count"], 3)


if __name__ == "__main__":
    unittest.main()
