from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multi_lane.track_a.summarize_shared_dgl import summarize


class SharedDGLSummaryTest(unittest.TestCase):
    def _run(
        self, root: Path, method: str, routing: str, auxiliary: float,
        final: float, average: float, full: float,
    ) -> Path:
        run = root / method
        run.mkdir()
        config = {
            "git": {"commit": "abc123"},
            "reporting_split": "val",
            "save_checkpoints": False,
            "view_fusion": "fixed_three_view",
            "view_gradient_routing": routing,
            "view_auxiliary_loss_weight": auxiliary,
            "view_dgl_unimodal_weight": 1.0 if method == "G2" else None,
            "view_gradient_audit": True,
            "view_evaluation_diagnostics": True,
        }
        summary = {
            "status": "complete",
            "seed": 0,
            "metrics": {"final_mAP": final, "average_mAP": average},
            "task_metrics": [{"mAP": final}] * 8,
        }
        history = {
            str(task_id): [{
                "gradient_norm_fused": 1.0,
                "gradient_norm_full": 1.0,
                "gradient_cosine_fused_full": -0.2,
            }]
            for task_id in range(8)
        }
        diagnostics = {
            str(task_id): {
                "metrics": {"full": {"mAP": full}},
                "pairwise_ranking_vs_full": {
                    "positive_negative_pairs": 10,
                    "errors_corrected": 2,
                    "correct_pairs_damaged": 1,
                    "net_corrected_pairs": 1,
                    "per_class": [],
                },
            }
            for task_id in range(8)
        }
        for name, value in (
            ("config.json", config),
            ("seed_summary.json", summary),
            ("training_history.json", history),
            ("view_diagnostics.json", diagnostics),
        ):
            (run / name).write_text(json.dumps(value), encoding="utf-8")
        return run

    def test_all_advance_checks_must_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {
                "G0": self._run(root, "G0", "joint", 0.1, 42.5, 49.7, 40.0),
                "G1": self._run(
                    root, "G1", "fusion_detach", 0.1, 42.55, 49.8, 40.1
                ),
                "G2": self._run(root, "G2", "dgl", 0.0, 42.7, 49.8, 40.2),
            }
            result = summarize(runs)
            self.assertTrue(result["advance_G2_to_dynamic_router_stage"])
            self.assertEqual(result["ranking"][0], "G2")
            self.assertAlmostEqual(
                result["paired_differences"]["G2_minus_G0_final_mAP"], 0.2
            )


if __name__ == "__main__":
    unittest.main()
