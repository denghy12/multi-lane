from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

from multi_lane.track_a.summarize_image_token_scheduler_search import (
    COMMON_CONFIG,
    EPOCHS,
    EXPECTED_ASL,
    NUM_TASKS,
    SPECS,
    TOTAL_UPDATES,
    expected_multiplier,
    summarize_scheduler_search,
)


TASK_UPDATES = (84, 69, 14, 156, 68, 40, 10, 24)


def write_run(root: Path, label: str, final_map: float) -> None:
    root.mkdir(parents=True)
    mode, min_ratio, warmup_ratio, scheduler_name = SPECS[label]
    config = dict(COMMON_CONFIG)
    config.update({
        "scheduler": scheduler_name,
        "scheduler_mode": mode,
        "scheduler_min_lr_ratio": min_ratio,
        "scheduler_warmup_ratio": warmup_ratio,
        "scheduler_warmup_epochs": (
            2 if warmup_ratio == 0.05 else 3 if warmup_ratio == 0.10 else 0
        ),
        "asl": dict(EXPECTED_ASL),
        "git": {"commit": "abc123", "tree": "tree123", "dirty": False},
    })
    rows = [
        {"task_id": task_id, "mAP": final_map - 0.7 + task_id * 0.1}
        for task_id in range(NUM_TASKS)
    ]
    history = {}
    for task_id in range(NUM_TASKS):
        task_history = []
        for step in range(EPOCHS):
            factor = expected_multiplier(label, step)
            next_factor = expected_multiplier(label, step + 1)
            task_history.append({
                "epoch": step,
                "optimizer_steps": TASK_UPDATES[task_id],
                "skipped_optimizer_steps": 0,
                "learning_rate": 0.0125 * factor,
                "next_learning_rate": 0.0125 * next_factor,
                "adapter_learning_rate": 0.0004 * factor,
                "next_adapter_learning_rate": 0.0004 * next_factor,
            })
        history[str(task_id)] = task_history
    summary = {
        "status": "complete",
        "seed": 0,
        "completed_epochs": EPOCHS * NUM_TASKS,
        "completed_optimizer_updates": TOTAL_UPDATES,
        "metrics": {
            "final_mAP": final_map,
            "average_mAP": final_map + 6.5,
            "final_cF1": 32.0,
            "final_oF1": 49.0,
            "forgetting": 4.5,
        },
    }
    for name, payload in (
        ("config.json", config),
        ("task_metrics.json", rows),
        ("training_history.json", history),
        ("seed_summary.json", summary),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")


def write_grid(root: Path, winner: str) -> List[Tuple[str, Path]]:
    runs = []
    for index, label in enumerate(SPECS):
        run_root = root / label
        write_run(run_root, label, 33.0 if label == winner else 32.0 - index / 100)
        runs.append((label, run_root))
    return runs


class TrackAImageTokenSchedulerSearchTest(unittest.TestCase):
    def test_selects_scheduler_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = summarize_scheduler_search(
                write_grid(Path(directory), "cosine_min001")
            )
        self.assertEqual(result["winner"]["label"], "cosine_min001")
        self.assertTrue(result["winner_exceeds_anchor"])
        self.assertEqual(result["next_action"], "lock_scheduler_winner_for_confirmation")

    def test_anchor_winner_closes_single_view_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = summarize_scheduler_search(
                write_grid(Path(directory), "cosine_anchor")
            )
        self.assertEqual(result["winner"]["label"], "cosine_anchor")
        self.assertFalse(result["winner_exceeds_anchor"])
        self.assertEqual(
            result["next_action"],
            "stop_single_view_scheduler_search_and_start_dual_view",
        )

    def test_rejects_lr_trajectory_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = write_grid(root, "cosine_anchor")
            path = root / "multistep" / "training_history.json"
            history = json.loads(path.read_text(encoding="utf-8"))
            history["6"][18]["learning_rate"] = 0.0125
            path.write_text(json.dumps(history), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "LR trajectory"):
                summarize_scheduler_search(runs)


if __name__ == "__main__":
    unittest.main()
