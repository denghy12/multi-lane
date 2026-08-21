from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multi_lane.track_a.summarize_image_token_asl_search import (
    CLIP_VALUES,
    GAMMA_NEG_VALUES,
    summarize_loss_search,
)


CLASS_ORDER = [f"class{index}" for index in range(20)] + [
    "Sadness", "Sensitivity", "Suffering", "Surprise", "Sympathy", "Yearning",
]


def write_run(
    root: Path,
    label: str,
    routing: str,
    gamma_neg: float = 9.8,
    clip: float = 0.05,
    final_delta: float = 0.0,
    suffering_delta: float = 0.0,
) -> None:
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
        "adapter_mode": "image_token",
        "adapter_bottleneck_dim": 32,
        "adapter_layer_indices": [8],
        "adapter_residual_scale": 0.1,
        "adapter_activation": "relu",
        "adapter_learning_rate": 0.0004,
        "adapter_task_initialization": "independent",
        "adapter_target": "frozen_image_tokens_for_selector",
        "adapter_image_token_scope": "block_ln1_cls_plus_patch_tokens",
        "adapter_writes_back_to_frozen_visual_stream": False,
        "adapter_initialization_rng": "forked_global_state",
        "learning_rate": 0.0125,
        "optimizer": "Adam_reset_per_task",
        "scheduler": "CosineAnnealingLR_reset_per_task",
        "weight_decay": 0.0,
        "temperature": 1.0,
        "save_checkpoints": False,
        "task_sizes": [5, 3, 3, 3, 3, 3, 3, 3],
        "class_order": CLASS_ORDER,
        "parameter_group_loss_routing": routing,
        "model_parameter_objective": "bce",
        "adapter_parameter_objective": "asl" if routing == "adapter_asl" else "bce",
        "asl": (
            {
                "gamma_neg": gamma_neg,
                "gamma_pos": 0.0,
                "clip": clip,
                "eps": 1e-8,
            }
            if routing == "adapter_asl" else None
        ),
        "git": {"commit": "commit", "tree": "tree", "dirty": False},
    }
    rows = []
    seen = 0
    for task_id, size in enumerate(config["task_sizes"]):
        seen += size
        aps = [50.0] * seen
        if task_id >= 6:
            aps[22] += suffering_delta
        rows.append({
            "task_id": task_id,
            "mAP": 40.0 + task_id + (final_delta if task_id >= 6 else 0.0),
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
            "average_mAP": 43.5 + final_delta,
            "final_cF1": 37.0,
            "final_oF1": 57.0,
            "forgetting": 1.0,
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


def write_grid(root: Path) -> list[tuple[str, Path]]:
    labeled_roots = [("joint_bce", root / "joint_bce")]
    write_run(root / "joint_bce", "joint_bce", "joint_bce")
    for gamma_neg in GAMMA_NEG_VALUES:
        for clip in CLIP_VALUES:
            label = f"g{gamma_neg}_c{clip}"
            run_root = root / label
            winning = gamma_neg == 4.0 and clip == 0.05
            write_run(
                run_root,
                label,
                "adapter_asl",
                gamma_neg,
                clip,
                final_delta=1.0 if winning else 0.1,
                suffering_delta=2.0 if winning else 0.1,
            )
            labeled_roots.append((label, run_root))
    return labeled_roots


def convert_to_failed_run(
    labeled_roots: list[tuple[str, Path]], label: str
) -> tuple[str, Path, Path]:
    matching = [item for item in labeled_roots if item[0] == label]
    if len(matching) != 1:
        raise AssertionError(f"Expected one matching run for {label}")
    _, run_root = matching[0]
    labeled_roots.remove(matching[0])
    (run_root / "seed_summary.json").unlink()
    rows_path = run_root / "task_metrics.json"
    rows = json.loads(rows_path.read_text(encoding="utf-8"))[:3]
    rows_path.write_text(json.dumps(rows), encoding="utf-8")
    history_path = run_root / "training_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history = {key: history[key] for key in ("0", "1", "2")}
    history_path.write_text(json.dumps(history), encoding="utf-8")
    log_path = run_root.parent / f"{label}.log"
    log_path.write_text(
        "Traceback\nFloatingPointError: ASL produced a non-finite loss\n",
        encoding="utf-8",
    )
    return label, run_root, log_path


class TrackAImageTokenASLSearchTest(unittest.TestCase):
    def test_validates_grid_and_selects_eligible_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = summarize_loss_search(write_grid(Path(directory)))
        self.assertEqual(result["winner"], "g4.0_c0.05")
        self.assertEqual(len(result["candidates"]), 20)
        self.assertEqual(result["joint_bce"]["summary_metrics"]["final_mAP"], 47.0)

    def test_rejects_configuration_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labeled_roots = write_grid(root)
            path = labeled_roots[-1][1] / "config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["adapter_bottleneck_dim"] = 64
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "adapter_bottleneck_dim"):
                summarize_loss_search(labeled_roots)

    def test_rejects_skipped_optimizer_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labeled_roots = write_grid(root)
            path = labeled_roots[-1][1] / "training_history.json"
            history = json.loads(path.read_text(encoding="utf-8"))
            history["7"][-1]["skipped_optimizer_steps"] = 1
            path.write_text(json.dumps(history), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "optimizer-step"):
                summarize_loss_search(labeled_roots)

    def test_records_verified_non_finite_candidate_as_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labeled_roots = write_grid(root)
            failed = convert_to_failed_run(labeled_roots, "g6.0_c0.025")
            result = summarize_loss_search(labeled_roots, [failed])
        self.assertEqual(result["failed_candidates"], ["g6.0_c0.025"])
        failed_result = next(
            item for item in result["candidates"]
            if item["label"] == "g6.0_c0.025"
        )
        self.assertFalse(failed_result["eligible"])
        self.assertEqual(failed_result["failure"]["completed_tasks"], 3)


if __name__ == "__main__":
    unittest.main()
