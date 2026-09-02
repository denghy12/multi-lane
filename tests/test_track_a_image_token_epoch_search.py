from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

from multi_lane.track_a.summarize_image_token_epoch_search import (
    COMMON_CONFIG,
    EPOCHS,
    EXPECTED_ASL,
    NUM_TASKS,
    UPDATES_PER_EPOCH,
    summarize_epoch_search,
)


TASK_UPDATES = (84, 69, 14, 156, 68, 40, 10, 24)


def write_run(root: Path, epochs: int, final_map: float, average_map: float) -> None:
    root.mkdir(parents=True)
    config = dict(COMMON_CONFIG)
    config.update({
        "epochs_per_task": epochs,
        "asl": dict(EXPECTED_ASL),
        "git": {"commit": "abc123", "tree": "tree123", "dirty": False},
    })
    rows = [
        {
            "task_id": task_id,
            "mAP": final_map - 0.7 + task_id * 0.1,
            "cF1": 30.0,
            "oF1": 45.0,
        }
        for task_id in range(NUM_TASKS)
    ]
    history = {
        str(task_id): [
            {
                "epoch": cycle,
                "optimizer_steps": TASK_UPDATES[task_id],
                "skipped_optimizer_steps": 0,
            }
            for cycle in range(epochs)
        ]
        for task_id in range(NUM_TASKS)
    }
    summary = {
        "status": "complete",
        "seed": 0,
        "completed_epochs": epochs * NUM_TASKS,
        "completed_optimizer_updates": epochs * UPDATES_PER_EPOCH,
        "metrics": {
            "final_mAP": final_map,
            "average_mAP": average_map,
            "final_cF1": 30.0,
            "final_oF1": 45.0,
            "forgetting": 4.0,
        },
    }
    for name, payload in (
        ("config.json", config),
        ("task_metrics.json", rows),
        ("training_history.json", history),
        ("seed_summary.json", summary),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")


def write_grid(root: Path, winner_epochs: int) -> List[Tuple[str, int, Path]]:
    runs = []
    for epochs in EPOCHS:
        run_root = root / f"epochs{epochs}"
        final_map = 33.0 if epochs == winner_epochs else 32.0 - abs(epochs - winner_epochs) / 100.0
        write_run(run_root, epochs, final_map, 39.0 - abs(epochs - winner_epochs) / 100.0)
        runs.append((f"epochs{epochs}", epochs, run_root))
    return runs


class TrackAImageTokenEpochSearchTest(unittest.TestCase):
    def test_selects_internal_winner_and_refinement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = summarize_epoch_search(write_grid(Path(directory), 34))
        self.assertEqual(result["winner"]["epochs_per_task"], 34)
        self.assertTrue(result["winner_exceeds_anchor"])
        self.assertEqual(result["refinement"]["action"], "refine_internal_winner")
        self.assertEqual(result["refinement"]["epochs"], [31, 32, 33, 35, 36, 37])

    def test_anchor_winner_stops_epoch_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = summarize_epoch_search(write_grid(Path(directory), 30))
        self.assertEqual(result["winner"]["epochs_per_task"], 30)
        self.assertFalse(result["winner_exceeds_anchor"])
        self.assertEqual(result["refinement"], {"action": "stop_epoch_search", "epochs": []})

    def test_rejects_protocol_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = write_grid(root, 34)
            path = root / "epochs48" / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["train_batch_size"] = 32
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train_batch_size"):
                summarize_epoch_search(runs)


if __name__ == "__main__":
    unittest.main()
