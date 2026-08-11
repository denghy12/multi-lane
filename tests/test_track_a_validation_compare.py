from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multi_lane.track_a.compare_validation import compare_runs


CLASS_ORDER = [f"class{index}" for index in range(26)]
CLASS_ORDER[20:23] = ["Sadness", "Sensitivity", "Suffering"]
TASK_SIZES = [5, 3, 3, 3, 3, 3, 3, 3]


def write_run(
    root: Path,
    *,
    adapter_mode: str,
    task6_delta: float = 0.0,
    task7_delta: float = 0.0,
    focus_delta: float = 0.0,
    seed: int = 0,
) -> None:
    config = {
        "protocol_id": "emotic_b5c3_v0.1",
        "seed": seed,
        "class_order": CLASS_ORDER,
        "task_sizes": TASK_SIZES,
        "train_split": "train",
        "validation_split": "val",
        "reporting_split": "val",
        "training_label_scope": "current_classes_only",
        "evaluation_scope": "samples_intersect_seen_classes",
        "threshold": 0.5,
        "epochs_per_task": 30,
        "train_batch_size": 64,
        "eval_batch_size": 64,
        "workers": 2,
        "optimizer": "Adam_reset_per_task",
        "learning_rate": 0.0125,
        "scheduler": "CosineAnnealingLR_reset_per_task",
        "weight_decay": 0.0,
        "temperature": 1.0,
        "amp": True,
        "tf32": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "num_selectors": 10,
        "num_prompts": 10,
        "num_prompt_layers": 5,
        "normalize": "pre-head",
        "head_mode": "concat",
        "max_tasks": 8,
        "clip_checkpoint_sha256": "clip-hash",
        "data_root": "/datasets/EMOTIC",
        "git": {"commit": "commit", "tree": "tree", "dirty": False},
        "adapter_mode": adapter_mode,
        "adapter_task_initialization": (
            "copy_previous" if adapter_mode == "task_lane" else "independent"
        ),
        "adapter_initialization_rng": (
            "forked_global_state" if adapter_mode == "task_lane" else None
        ),
    }
    rows = []
    seen = 0
    for task_id, task_size in enumerate(TASK_SIZES):
        seen += task_size
        delta = task6_delta if task_id == 6 else task7_delta if task_id == 7 else 0.0
        per_class_ap = [50.0 + delta] * seen
        if adapter_mode == "task_lane" and task_id >= 6:
            per_class_ap[20:23] = [50.0 + focus_delta] * 3
        rows.append({
            "task_id": task_id,
            "seen_classes": seen,
            "samples": 100,
            "threshold": 0.5,
            "mAP": 50.0 + delta,
            "cF1": 40.0 + delta,
            "oF1": 60.0 + delta,
            "per_class_ap": per_class_ap,
        })
    metrics = {
        "final_mAP": rows[-1]["mAP"],
        "average_mAP": sum(row["mAP"] for row in rows) / len(rows),
        "final_cF1": rows[-1]["cF1"],
        "final_oF1": rows[-1]["oF1"],
        "forgetting": 1.0,
    }
    summary = {
        "status": "complete",
        "seed": seed,
        "completed_epochs": 240,
        "metrics": metrics,
    }
    root.mkdir()
    for name, payload in (
        ("config.json", config),
        ("seed_summary.json", summary),
        ("task_metrics.json", rows),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")


class TrackAValidationComparisonTest(unittest.TestCase):
    def test_continue_requires_all_late_task_improvements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root / "baseline", adapter_mode="disabled")
            write_run(
                root / "candidate",
                adapter_mode="task_lane",
                task6_delta=1.0,
                task7_delta=0.5,
                focus_delta=2.0,
            )
            result = compare_runs(root / "baseline", root / "candidate")
        self.assertTrue(result["decision"]["continue_task_lane_adapter"])
        self.assertEqual(result["focus_task"]["classes"], [
            "Sadness", "Sensitivity", "Suffering",
        ])
        self.assertEqual(
            result["focus_task"]["stages"]["at_introduction"]
            ["classes"]["Sadness"]["delta"],
            2.0,
        )

    def test_failed_final_improvement_stops_capacity_scaling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root / "baseline", adapter_mode="disabled")
            write_run(
                root / "candidate",
                adapter_mode="task_lane",
                task6_delta=1.0,
                task7_delta=-0.1,
                focus_delta=2.0,
            )
            result = compare_runs(root / "baseline", root / "candidate")
        self.assertFalse(result["decision"]["continue_task_lane_adapter"])
        self.assertEqual(
            result["decision"]["recommendation"],
            "stop_task_lane_adapter_capacity_scaling",
        )

    def test_rejects_non_seed0_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_run(root / "baseline", adapter_mode="disabled", seed=1)
            write_run(root / "candidate", adapter_mode="task_lane", seed=1)
            with self.assertRaisesRegex(ValueError, "only accepts seed0"):
                compare_runs(root / "baseline", root / "candidate")


if __name__ == "__main__":
    unittest.main()
