from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Sequence, Tuple

from multi_lane.track_a.summarize_image_token_pair89_confirmation import (
    BASE_TRAINABLE_PARAMETERS,
    NUM_TASKS,
    PER_LAYER_PER_TASK_PARAMETERS,
    summarize_pair89_confirmation,
)
from tests.test_track_a_image_token_layer_search import write_run


def write_candidate(
    root: Path,
    layers: Sequence[int],
    delta: float,
    forgetting_delta: float,
) -> None:
    write_run(
        root,
        "image_token",
        "joint_bce",
        layer=int(layers[0]),
        final_delta=delta,
        task6_delta=delta,
        class_deltas={
            "Sadness": delta,
            "Sensitivity": delta,
            "Suffering": delta,
        },
        forgetting_delta=forgetting_delta,
    )
    path = root / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    per_task = len(layers) * PER_LAYER_PER_TASK_PARAMETERS
    config["adapter_layer_indices"] = list(layers)
    config["adapter_parameters_per_task"] = per_task
    config["adapter_parameters"] = per_task * NUM_TASKS
    config["trainable_parameters"] = BASE_TRAINABLE_PARAMETERS + per_task
    path.write_text(json.dumps(config), encoding="utf-8")


def write_grid(
    root: Path, pair_wins: bool = True
) -> List[Tuple[str, Path]]:
    runs = [("disabled_bce", root / "disabled_bce")]
    write_run(root / "disabled_bce", "disabled", "joint_bce")
    values = (
        ("single8_bce", (8,), 0.6, -0.10),
        ("single9_bce", (9,), 0.8, -0.15),
        ("pair8_9_bce", (8, 9), 1.2 if pair_wins else 0.7, -0.20),
    )
    for label, layers, delta, forgetting_delta in values:
        run_root = root / label
        write_candidate(run_root, layers, delta, forgetting_delta)
        runs.append((label, run_root))
    return runs


class TrackAImageTokenPair89ConfirmationTest(unittest.TestCase):
    def test_confirms_pair_when_it_strictly_preserves_single9_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = summarize_pair89_confirmation(write_grid(Path(directory)))
        self.assertTrue(result["pair8_9"]["confirmed_for_formal_test"])
        self.assertEqual(result["recommended_validation_structure"], "pair8_9_bce")

    def test_recommends_single9_when_pair_does_not_beat_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = summarize_pair89_confirmation(
                write_grid(Path(directory), pair_wins=False)
            )
        self.assertFalse(result["pair8_9"]["confirmed_for_formal_test"])
        self.assertEqual(result["recommended_validation_structure"], "single9_bce")

    def test_rejects_skipped_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = write_grid(root)
            path = root / "pair8_9_bce" / "training_history.json"
            history = json.loads(path.read_text(encoding="utf-8"))
            history["7"][-1]["skipped_optimizer_steps"] = 1
            path.write_text(json.dumps(history), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "optimizer-step"):
                summarize_pair89_confirmation(runs)


if __name__ == "__main__":
    unittest.main()
