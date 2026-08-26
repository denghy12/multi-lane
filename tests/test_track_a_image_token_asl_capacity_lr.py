from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Tuple

from multi_lane.track_a.summarize_image_token_asl_capacity_lr import (
    ADAPTER_LEARNING_RATES,
    BASE_TRAINABLE_PARAMETERS,
    BOTTLENECKS,
    CANDIDATES,
    NUM_TASKS,
    _label,
    _parameters_per_task,
    summarize_capacity_lr_search,
)
from tests.test_track_a_image_token_layer_search import write_run


def _add_strict_numeric_metadata(root: Path) -> dict:
    path = root / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["tf32"] = False
    return config


def write_disabled(root: Path) -> None:
    write_run(root, "disabled", "joint_bce")
    config = _add_strict_numeric_metadata(root)
    config["adapter_parameters_per_task"] = 0
    config["adapter_parameters"] = 0
    config["trainable_parameters"] = BASE_TRAINABLE_PARAMETERS
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


def write_candidate(
    root: Path,
    bottleneck: int,
    learning_rate: float,
    delta: float,
    forgetting_delta: float,
    valid_classes: bool = True,
) -> None:
    class_deltas = {
        "Sadness": delta,
        "Sensitivity": delta,
        "Suffering": delta if valid_classes else -0.1,
    }
    write_run(
        root,
        "image_token",
        "adapter_asl",
        layer=8,
        final_delta=delta,
        task6_delta=delta,
        class_deltas=class_deltas,
        forgetting_delta=forgetting_delta,
    )
    config = _add_strict_numeric_metadata(root)
    per_task = _parameters_per_task(bottleneck)
    config["adapter_bottleneck_dim"] = bottleneck
    config["adapter_learning_rate"] = learning_rate
    config["adapter_parameters_per_task"] = per_task
    config["adapter_parameters"] = per_task * NUM_TASKS
    config["trainable_parameters"] = BASE_TRAINABLE_PARAMETERS + per_task
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


def write_grid(root: Path) -> List[Tuple[str, Path]]:
    runs: List[Tuple[str, Path]] = [("disabled_bce", root / "disabled_bce")]
    write_disabled(root / "disabled_bce")
    for bottleneck in BOTTLENECKS:
        for learning_rate in ADAPTER_LEARNING_RATES:
            label = _label(bottleneck, learning_rate)
            run_root = root / label
            if (bottleneck, learning_rate) == (32, 0.0004):
                write_candidate(run_root, bottleneck, learning_rate, 0.4, -0.1)
            elif (bottleneck, learning_rate) == (16, 0.0002):
                write_candidate(run_root, bottleneck, learning_rate, 1.2, -0.2)
            else:
                write_candidate(
                    run_root,
                    bottleneck,
                    learning_rate,
                    0.1,
                    0.1,
                    valid_classes=False,
                )
            runs.append((label, run_root))
    return runs


class TrackAImageTokenAslCapacityLrTest(unittest.TestCase):
    def test_selects_material_candidate_that_preserves_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = summarize_capacity_lr_search(write_grid(Path(directory)))
        self.assertEqual(result["winner_label"], "b16_lr2e4_asl")
        self.assertEqual(result["eligible_labels"], ["b16_lr2e4_asl"])
        self.assertTrue(result["continue_to_scale_activation"])

    def test_rejects_tf32_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = write_grid(root)
            path = root / "b64_lr4e4_asl" / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["tf32"] = True
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tf32"):
                summarize_capacity_lr_search(runs)

    def test_rejects_parameter_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = write_grid(root)
            label = next(iter(CANDIDATES))
            path = root / label / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["adapter_parameters_per_task"] += 1
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Adapter-ASL metadata"):
                summarize_capacity_lr_search(runs)


if __name__ == "__main__":
    unittest.main()
