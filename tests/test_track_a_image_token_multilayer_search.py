from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import List, Sequence, Tuple

from multi_lane.track_a.summarize_image_token_multilayer_search import (
    BASE_TRAINABLE_PARAMETERS,
    NUM_TASKS,
    PER_LAYER_PER_TASK_PARAMETERS,
    STRUCTURES,
    summarize_multilayer_search,
)
from tests.test_track_a_image_token_layer_search import write_run


def write_candidate(
    root: Path,
    routing: str,
    layers: Sequence[int],
    final_delta: float,
    task6_delta: float,
    class_delta: float,
    forgetting_delta: float,
) -> None:
    write_run(
        root,
        "image_token",
        routing,
        layer=int(layers[0]),
        final_delta=final_delta,
        task6_delta=task6_delta,
        class_deltas={
            "Sadness": class_delta,
            "Sensitivity": class_delta,
            "Suffering": class_delta,
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


def write_grid(root: Path) -> List[Tuple[str, Path]]:
    runs: List[Tuple[str, Path]] = [("disabled_bce", root / "disabled_bce")]
    write_run(root / "disabled_bce", "disabled", "joint_bce")
    for structure, layers, is_multilayer in STRUCTURES:
        bce_label = f"{structure}_bce"
        asl_label = f"{structure}_asl"
        bce_root = root / bce_label
        asl_root = root / asl_label
        if structure == "pair3_7":
            write_candidate(
                bce_root, "joint_bce", layers,
                final_delta=0.8, task6_delta=0.8,
                class_delta=0.8, forgetting_delta=-0.2,
            )
            write_candidate(
                asl_root, "adapter_asl", layers,
                final_delta=1.5, task6_delta=1.5,
                class_delta=1.5, forgetting_delta=-0.3,
            )
        elif is_multilayer:
            write_candidate(
                bce_root, "joint_bce", layers,
                final_delta=0.2, task6_delta=0.2,
                class_delta=-0.2, forgetting_delta=0.1,
            )
            write_candidate(
                asl_root, "adapter_asl", layers,
                final_delta=0.3, task6_delta=0.3,
                class_delta=-0.3, forgetting_delta=0.1,
            )
        else:
            single_delta = 0.5 if structure == "single3" else 0.4
            write_candidate(
                bce_root, "joint_bce", layers,
                final_delta=single_delta - 0.1,
                task6_delta=single_delta - 0.1,
                class_delta=single_delta - 0.1,
                forgetting_delta=-0.05,
            )
            write_candidate(
                asl_root, "adapter_asl", layers,
                final_delta=single_delta,
                task6_delta=single_delta,
                class_delta=single_delta,
                forgetting_delta=-0.1,
            )
        runs.extend(((bce_label, bce_root), (asl_label, asl_root)))
    return runs


class TrackAImageTokenMultilayerSearchTest(unittest.TestCase):
    def test_selects_multilayer_winner_above_fresh_single_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = summarize_multilayer_search(write_grid(Path(directory)))
        self.assertEqual(result["winner_bce_structure"], "pair3_7")
        self.assertEqual(result["winner_asl_structure"], "pair3_7")
        self.assertTrue(result["continue_with_bce"])
        self.assertTrue(result["continue_with_asl"])

    def test_rejects_layer_list_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = write_grid(root)
            path = root / "late8_9_asl" / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["adapter_layer_indices"] = [8]
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "wrong Adapter layers"):
                summarize_multilayer_search(runs)

    def test_rejects_skipped_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = write_grid(root)
            path = root / "pair3_11_asl" / "training_history.json"
            history = json.loads(path.read_text(encoding="utf-8"))
            history["7"][-1]["skipped_optimizer_steps"] = 1
            path.write_text(json.dumps(history), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "optimizer-step"):
                summarize_multilayer_search(runs)


if __name__ == "__main__":
    unittest.main()
