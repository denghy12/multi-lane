from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

from multi_lane.track_a.summarize_image_token_layer_search import (
    summarize_layer_search,
)


CLASS_ORDER = [f"class{index}" for index in range(20)] + [
    "Sadness", "Sensitivity", "Suffering", "Surprise", "Sympathy", "Yearning",
]
TASK_SIZES = [5, 3, 3, 3, 3, 3, 3, 3]


def write_run(
    root: Path,
    adapter_mode: str,
    routing: str,
    layer: int = 8,
    final_delta: float = 0.0,
    task6_delta: float = 0.0,
    class_deltas: Dict[str, float] = None,
    forgetting_delta: float = 0.0,
) -> None:
    class_deltas = class_deltas or {}
    enabled = adapter_mode == "image_token"
    config = {
        "seed": 0,
        "reporting_split": "val",
        "max_tasks": 8,
        "epochs_per_task": 30,
        "train_batch_size": 64,
        "eval_batch_size": 64,
        "threshold": 0.5,
        "training_loss_mode": "legacy_full_zero",
        "training_label_scope": "current_classes_only",
        "evaluation_scope": "samples_intersect_seen_classes",
        "input_mode": "full",
        "input_normalization": "clip",
        "train_crop_scale": [0.05, 1.0],
        "learning_rate": 0.0125,
        "optimizer": "Adam_reset_per_task",
        "scheduler": "CosineAnnealingLR_reset_per_task",
        "weight_decay": 0.0,
        "temperature": 1.0,
        "save_checkpoints": False,
        "amp": False,
        "adapter_mode": adapter_mode,
        "adapter_bottleneck_dim": 32,
        "adapter_layer_indices": [layer],
        "adapter_residual_scale": 0.1,
        "adapter_activation": "relu",
        "adapter_learning_rate": 0.0004 if enabled else None,
        "adapter_task_initialization": "independent",
        "adapter_target": "frozen_image_tokens_for_selector" if enabled else None,
        "adapter_image_token_scope": (
            "block_ln1_cls_plus_patch_tokens" if enabled else None
        ),
        "adapter_writes_back_to_frozen_visual_stream": False,
        "adapter_initialization_rng": "forked_global_state" if enabled else None,
        "task_sizes": TASK_SIZES,
        "class_order": CLASS_ORDER,
        "parameter_group_loss_routing": routing,
        "model_parameter_objective": "bce",
        "adapter_parameter_objective": "asl" if routing == "adapter_asl" else "bce",
        "asl": (
            {
                "gamma_neg": 9.8,
                "gamma_pos": 0.0,
                "clip": 0.05,
                "eps": 1e-8,
                "detach_focal_weight": True,
                "reduction": "mean_over_training_loss_view",
            }
            if routing == "adapter_asl" else None
        ),
        "git": {"commit": "commit", "tree": "tree", "dirty": False},
    }
    rows: List[Dict[str, object]] = []
    seen = 0
    for task_id, size in enumerate(TASK_SIZES):
        seen += size
        aps = [50.0] * seen
        if task_id >= 6:
            for name, delta in class_deltas.items():
                aps[CLASS_ORDER.index(name)] += delta
        rows.append({
            "task_id": task_id,
            "mAP": 40.0 + task_id + (task6_delta if task_id == 6 else 0.0)
            + (final_delta if task_id == 7 else 0.0),
            "cF1": 30.0 + task_id,
            "oF1": 50.0 + task_id,
            "per_class_ap": aps,
        })
    history = {
        str(task_id): [
            {
                "epoch": epoch,
                "optimizer_steps": 84 if task_id == 0 else (
                    57 if task_id == 1 else 54
                ),
                "skipped_optimizer_steps": 0,
            }
            for epoch in range(30)
        ]
        for task_id in range(8)
    }
    summary = {
        "status": "complete",
        "seed": 0,
        "completed_epochs": 240,
        "completed_optimizer_updates": 13950,
        "metrics": {
            "final_mAP": 47.0 + final_delta,
            "average_mAP": 43.5 + (final_delta + task6_delta) / 8.0,
            "final_cF1": 37.0,
            "final_oF1": 57.0,
            "forgetting": 1.0 + forgetting_delta,
        },
    }
    root.mkdir()
    for name, payload in (
        ("config.json", config),
        ("task_metrics.json", rows),
        ("training_history.json", history),
        ("seed_summary.json", summary),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")


def write_grid(root: Path) -> List[Tuple[str, Path]]:
    runs: List[Tuple[str, Path]] = [("disabled_bce", root / "disabled_bce")]
    write_run(root / "disabled_bce", "disabled", "joint_bce")
    for layer in range(12):
        bce_label = f"layer{layer}_bce"
        asl_label = f"layer{layer}_asl"
        bce_root = root / bce_label
        asl_root = root / asl_label
        if layer == 6:
            write_run(
                bce_root, "image_token", "joint_bce", layer,
                final_delta=0.5, task6_delta=0.5,
                class_deltas={"Sadness": 0.5, "Sensitivity": 0.2, "Suffering": 0.5},
                forgetting_delta=-0.1,
            )
            write_run(
                asl_root, "image_token", "adapter_asl", layer,
                final_delta=1.2, task6_delta=1.2,
                class_deltas={"Sadness": 1.2, "Sensitivity": 0.8, "Suffering": 1.2},
                forgetting_delta=-0.2,
            )
        else:
            write_run(
                bce_root, "image_token", "joint_bce", layer,
                final_delta=0.1, task6_delta=0.1,
                class_deltas={"Sadness": 0.1, "Sensitivity": 0.1, "Suffering": -0.1},
            )
            write_run(
                asl_root, "image_token", "adapter_asl", layer,
                final_delta=0.2, task6_delta=0.2,
                class_deltas={"Sadness": 0.2, "Sensitivity": 0.2, "Suffering": -0.2},
            )
        runs.extend(((bce_label, bce_root), (asl_label, asl_root)))
    return runs


class TrackAImageTokenLayerSearchTest(unittest.TestCase):
    def test_selects_paired_bce_and_asl_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = summarize_layer_search(write_grid(Path(directory)))
        self.assertEqual(result["winner_bce_layer"], 6)
        self.assertEqual(result["winner_asl_layer"], 6)
        self.assertTrue(result["continue_with_bce"])
        self.assertTrue(result["continue_with_asl"])

    def test_rejects_precision_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = write_grid(root)
            path = root / "layer11_asl" / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["amp"] = True
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "amp"):
                summarize_layer_search(runs)

    def test_rejects_skipped_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = write_grid(root)
            path = root / "layer11_asl" / "training_history.json"
            history = json.loads(path.read_text(encoding="utf-8"))
            history["7"][-1]["skipped_optimizer_steps"] = 1
            path.write_text(json.dumps(history), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "optimizer-step"):
                summarize_layer_search(runs)


if __name__ == "__main__":
    unittest.main()
