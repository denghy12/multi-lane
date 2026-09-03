from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.fuse_fixed_test_scores import fuse_fixed_test_runs
from multi_lane.track_a.fuse_validation_scores import COMMON_CONFIG_FIELDS
from multi_lane.track_a.runner import (
    CLASS_ORDER,
    TASK_SIZES,
    compute_metrics,
    summarize_tasks,
    write_evaluation_scores,
)


class FixedDualViewTest(unittest.TestCase):
    @staticmethod
    def _write_test_run(root: Path, input_mode: str, offset: float) -> None:
        config = {field: None for field in COMMON_CONFIG_FIELDS}
        config.update(
            {
                "protocol_id": "emotic_b5c3_v0.1",
                "seed": 0,
                "task_sizes": list(TASK_SIZES),
                "class_order": list(CLASS_ORDER),
                "reporting_split": "test",
                "max_tasks": 8,
                "save_checkpoints": False,
                "threshold": 0.5,
                "input_mode": input_mode,
                "person_transform_mode": (
                    "letterbox" if input_mode == "person_crop" else "legacy_crop"
                ),
                "save_evaluation_scores": True,
                "evaluation_score_purpose": "fixed_test_fusion",
                "git": {"commit": "abc123", "dirty": False},
            }
        )
        rows = []
        for task_id, seen_classes in enumerate(np.cumsum(TASK_SIZES)):
            sample_ids = tuple(f"test:sample{index}#person=0" for index in range(6))
            targets = torch.zeros(6, int(seen_classes))
            for class_id in range(int(seen_classes)):
                targets[class_id % 6, class_id] = 1.0
                targets[(class_id + 2) % 6, class_id] = 1.0
            logits = (targets * 4.0 - 2.0) + offset
            write_evaluation_scores(
                root / "test_scores" / f"task{task_id}.npz",
                task_id,
                sample_ids,
                logits,
                targets,
            )
            rows.append(
                compute_metrics(
                    task_id, torch.sigmoid(logits), targets, threshold=0.5
                )
            )
        summary = {
            "status": "complete",
            "config": config,
            "metrics": summarize_tasks(rows),
            "task_metrics": [asdict(row) for row in rows],
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "seed_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

    @staticmethod
    def _write_selection(path: Path, alpha: float = 0.20) -> None:
        path.write_text(
            json.dumps(
                {
                    "selection_split": "val",
                    "winner": {
                        "mode": "probability",
                        "alpha": alpha,
                        "metrics": {"final_mAP": 43.30347536439419},
                    },
                    "decision": {"advance_to_formal_test": True},
                }
            ),
            encoding="utf-8",
        )

    def test_applies_exactly_one_validation_locked_test_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full"
            person = root / "person"
            selection = root / "validation.json"
            self._write_test_run(full, "full", 0.0)
            self._write_test_run(person, "person_crop", -0.2)
            self._write_selection(selection)
            result = fuse_fixed_test_runs(
                full,
                person,
                selection,
                hashlib.sha256(selection.read_bytes()).hexdigest(),
            )
            self.assertEqual(result["evaluation_split"], "test")
            self.assertFalse(result["search_performed_on_test"])
            self.assertEqual(result["locked_rule"]["mode"], "probability")
            self.assertEqual(result["locked_rule"]["person_weight"], 0.20)
            self.assertNotIn("candidates", result)
            self.assertEqual(len(result["fixed_fusion"]["task_metrics"]), 8)

    def test_rejects_validation_rule_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full"
            person = root / "person"
            selection = root / "validation.json"
            self._write_test_run(full, "full", 0.0)
            self._write_test_run(person, "person_crop", -0.2)
            self._write_selection(selection, alpha=0.25)
            with self.assertRaisesRegex(ValueError, "locked fusion rule"):
                fuse_fixed_test_runs(
                    full,
                    person,
                    selection,
                    hashlib.sha256(selection.read_bytes()).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
