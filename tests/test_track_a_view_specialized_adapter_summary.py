from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multi_lane.track_a.summarize_view_specialized_adapter import summarize


class ViewSpecializedAdapterSummaryTest(unittest.TestCase):
    def _run(
        self,
        root: Path,
        method: str,
        shared_dim: int,
        view_dim: int,
        parameters: int,
        final: float,
        average: float,
        full: float,
    ) -> Path:
        run = root / method
        run.mkdir()
        config = {
            "git": {"commit": "abc123"},
            "reporting_split": "val",
            "save_checkpoints": False,
            "view_fusion": "fixed_three_view",
            "view_gradient_routing": "joint",
            "view_auxiliary_loss_weight": 0.1,
            "adapter_bottleneck_dim": shared_dim,
            "adapter_view_bottleneck_dim": view_dim,
            "adapter_parameters_per_task": parameters,
            "view_path_gradient_audit_epochs": [0, 14, 29],
            "view_path_gradient_audit_batches_per_epoch": 3,
        }
        summary = {
            "status": "complete",
            "seed": 0,
            "metrics": {"final_mAP": final, "average_mAP": average},
            "task_metrics": [{"mAP": final}] * 8,
        }
        history = {
            str(task_id): [
                {
                    "epoch": epoch,
                    **(
                        {
                            "path_gradient_audit_samples": 3,
                            "path_gradient_representation_full_fused_norm": 1.0,
                        }
                        if epoch in (0, 14, 29) else {}
                    ),
                }
                for epoch in range(30)
            ]
            for task_id in range(8)
        }
        diagnostics = {
            str(task_id): {
                "metrics": {"full": {"mAP": full}},
                "pairwise_ranking_vs_full": {},
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

    def test_specialized_candidate_must_pass_all_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = {
                "P0": self._run(root, "P0", 32, 0, 49_952, 43.0, 50.0, 41.0),
                "P1": self._run(root, "P1", 45, 0, 69_933, 43.05, 50.0, 41.0),
                "P2": self._run(root, "P2", 32, 4, 70_700, 43.2, 50.1, 41.1),
            }
            result = summarize(runs)
            self.assertTrue(result["advance_P2_to_dynamic_router_stage"])
            self.assertEqual(result["ranking"][0], "P2")
            self.assertAlmostEqual(
                result["paired_differences"]["P2_minus_P1_final_mAP"], 0.15
            )


if __name__ == "__main__":
    unittest.main()
