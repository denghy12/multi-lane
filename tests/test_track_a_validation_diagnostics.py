from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multi_lane.track_a.summarize_validation_diagnostics import (
    summarize_diagnostics,
)


def write_run(root: Path, loss_mode: str, seed: int = 0) -> None:
    config = {
        "protocol_id": "emotic_b5c3_v0.1",
        "seed": seed,
        "class_order": [f"class{index}" for index in range(26)],
        "task_sizes": [5, 3, 3, 3, 3, 3, 3, 3],
        "reporting_split": "val",
        "training_label_scope": "current_classes_only",
        "epochs_per_task": 30,
        "train_batch_size": 64,
        "eval_batch_size": 64,
        "threshold": 0.5,
        "clip_checkpoint_sha256": "clip",
        "data_root": "/datasets/EMOTIC",
        "max_tasks": 8,
        "training_loss_mode": loss_mode,
        "git": {"commit": "commit", "tree": "tree", "dirty": False},
    }
    rows = []
    seen = 0
    for task_id, size in enumerate(config["task_sizes"]):
        seen += size
        rows.append({
            "task_id": task_id,
            "mAP": 40.0 + task_id,
            "cF1": 30.0 + task_id,
            "oF1": 50.0 + task_id,
            "per_class_ap": [50.0] * seen,
        })
    summary = {
        "status": "complete",
        "seed": seed,
        "completed_epochs": 240,
        "metrics": {
            "final_mAP": 47.0,
            "average_mAP": 43.5,
            "final_cF1": 37.0,
            "final_oF1": 57.0,
            "forgetting": 1.0,
        },
    }
    root.mkdir()
    for name, payload in (
        ("config.json", config),
        ("task_metrics.json", rows),
        ("seed_summary.json", summary),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")


class TrackAValidationDiagnosticsTest(unittest.TestCase):
    def test_summarizes_allowed_loss_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root / "legacy", "legacy_full_zero")
            write_run(root / "current", "current_only")
            result = summarize_diagnostics(
                [("legacy", root / "legacy"), ("current", root / "current")],
                ["training_loss_mode"],
            )
        self.assertEqual(len(result["runs"]), 2)
        self.assertEqual(
            result["runs"][1]["varying_config"]["training_loss_mode"],
            "current_only",
        )
        self.assertEqual(
            result["runs"][0]["focus_task"]["at_introduction_mean_ap"],
            50.0,
        )

    def test_rejects_non_seed0_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root / "first", "legacy_full_zero")
            write_run(root / "second", "current_only", seed=1)
            with self.assertRaisesRegex(ValueError, "complete seed0"):
                summarize_diagnostics(
                    [("first", root / "first"), ("second", root / "second")],
                    ["training_loss_mode"],
                )

    def test_rejects_unapproved_configuration_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root / "first", "legacy_full_zero")
            write_run(root / "second", "current_only")
            config_path = root / "second" / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["train_batch_size"] = 32
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train_batch_size"):
                summarize_diagnostics(
                    [("first", root / "first"), ("second", root / "second")],
                    ["training_loss_mode"],
                )


if __name__ == "__main__":
    unittest.main()
