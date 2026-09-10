from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multi_lane.track_a.summarize_full_anchored_residual import summarize


class FullAnchoredResidualSummaryTest(unittest.TestCase):
    def _run(self, root: Path, method: str, mode: str, final_map: float) -> Path:
        run = root / method
        run.mkdir()
        config = {
            "seed": 0,
            "reporting_split": "val",
            "view_fusion": mode,
            "view_auxiliary_loss_weight": 0.0,
            "view_residual_scale": None if method == "A0" else 0.1,
            "adapter_mode": "image_token",
            "git": {"commit": "abc123"},
        }
        summary = {
            "status": "complete",
            "seed": 0,
            "metrics": {"final_mAP": final_map, "average_mAP": final_map + 5},
            "task_metrics": [{"task_id": 7, "mAP": final_map}],
        }
        (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (run / "seed_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return run

    def test_a2_must_beat_both_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = {
                "A0": self._run(root, "A0", "disabled", 42.0),
                "A1": self._run(
                    root, "A1", "residual_full_person", 42.4
                ),
                "A2": self._run(
                    root, "A2", "residual_three_view", 42.5
                ),
            }
            result = summarize(runs)
            self.assertTrue(result["advance_A2_to_seed1_seed2_validation"])
            self.assertFalse(result["selection_uses_test"])
            self.assertFalse(result["test_permitted"])
            self.assertEqual(result["ranking"], ["A2", "A1", "A0"])

            summary = json.loads(
                (runs["A2"] / "seed_summary.json").read_text(encoding="utf-8")
            )
            summary["metrics"]["final_mAP"] = 42.3
            (runs["A2"] / "seed_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            rejected = summarize(runs)
            self.assertFalse(rejected["advance_A2_to_seed1_seed2_validation"])


if __name__ == "__main__":
    unittest.main()
