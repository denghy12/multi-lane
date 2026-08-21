"""Validate and rank the preregistered Image-token Adapter ASL loss grid."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


GAMMA_NEG_VALUES = (1.0, 2.0, 4.0, 6.0, 9.8)
CLIP_VALUES = (0.0, 0.025, 0.05, 0.1)
EXPECTED_CANDIDATES = {
    (gamma_neg, clip)
    for gamma_neg in GAMMA_NEG_VALUES
    for clip in CLIP_VALUES
}
EXPECTED_CONFIG = {
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
}


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing search artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _close(left: Any, right: Any) -> bool:
    if isinstance(right, float):
        if not isinstance(left, (int, float)):
            return False
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
    return left == right


def _validate_config(label: str, config: Dict[str, Any]) -> None:
    mismatches = [
        key for key, expected in EXPECTED_CONFIG.items()
        if not _close(config.get(key), expected)
    ]
    if mismatches:
        raise ValueError(f"Search run {label} has protocol mismatches: {mismatches}")
    if config.get("git", {}).get("dirty") is not False:
        raise ValueError(f"Search run {label} did not record clean Git")


def _load_run(label: str, root: Path) -> Dict[str, Any]:
    config = _load_json(root / "config.json")
    rows = _load_json(root / "task_metrics.json")
    history = _load_json(root / "training_history.json")
    summary = _load_json(root / "seed_summary.json")
    _validate_config(label, config)
    if summary.get("status") != "complete" or summary.get("seed") != 0:
        raise ValueError(f"Search run {label} is not a complete seed0 run")
    if len(rows) != 8 or [row.get("task_id") for row in rows] != list(range(8)):
        raise ValueError(f"Search run {label} does not contain ordered tasks 0--7")
    if summary.get("completed_epochs") != 240:
        raise ValueError(f"Search run {label} did not complete 240 epochs")
    if summary.get("completed_optimizer_updates") != 13950:
        raise ValueError(f"Search run {label} did not complete 13950 updates")
    if (root / "checkpoints").exists():
        raise ValueError(f"Search run {label} unexpectedly contains checkpoints")
    if list(history) != [str(index) for index in range(8)]:
        raise ValueError(f"Search run {label} has invalid training-history tasks")
    if any(len(history[str(index)]) != 30 for index in range(8)):
        raise ValueError(f"Search run {label} has incomplete per-task history")
    updates = sum(
        int(epoch["optimizer_steps"])
        for task_history in history.values()
        for epoch in task_history
    )
    skipped = sum(
        int(epoch["skipped_optimizer_steps"])
        for task_history in history.values()
        for epoch in task_history
    )
    if updates != summary.get("completed_optimizer_updates") or skipped != 0:
        raise ValueError(f"Search run {label} has invalid optimizer-step accounting")
    return {
        "label": label,
        "root": str(root.resolve()),
        "status": "complete",
        "config": config,
        "rows": rows,
        "summary": summary,
    }


def _load_failed_run(label: str, root: Path, log_path: Path) -> Dict[str, Any]:
    config = _load_json(root / "config.json")
    rows = _load_json(root / "task_metrics.json")
    history = _load_json(root / "training_history.json")
    _validate_config(label, config)
    if (root / "seed_summary.json").exists():
        raise ValueError(f"Failed search run {label} unexpectedly has a summary")
    if (root / "checkpoints").exists():
        raise ValueError(f"Failed search run {label} unexpectedly has checkpoints")
    if [row.get("task_id") for row in rows] != list(range(len(rows))):
        raise ValueError(f"Failed search run {label} has invalid completed tasks")
    if list(history) != [str(index) for index in range(len(rows))]:
        raise ValueError(f"Failed search run {label} has invalid partial history")
    if not rows or len(rows) >= 8:
        raise ValueError(f"Failed search run {label} must be a nonempty partial run")
    if any(len(history[str(index)]) != 30 for index in range(len(rows))):
        raise ValueError(f"Failed search run {label} has incomplete saved task history")
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing failure log for {label}: {log_path}")
    log_text = log_path.read_text(encoding="utf-8")
    failure_text = "FloatingPointError: ASL produced a non-finite loss"
    if failure_text not in log_text:
        raise ValueError(f"Failed search run {label} lacks the expected ASL failure")
    return {
        "label": label,
        "root": str(root.resolve()),
        "status": "failed_non_finite_asl_loss",
        "failure": {
            "error_type": "FloatingPointError",
            "reason": "ASL produced a non-finite loss",
            "completed_tasks": len(rows),
            "completed_epochs": sum(len(value) for value in history.values()),
            "log": str(log_path.resolve()),
        },
        "config": config,
        "rows": rows,
        "history": history,
    }


def _focus_metrics(run: Dict[str, Any], task_id: int = 6) -> Dict[str, Any]:
    config = run["config"]
    start = sum(int(size) for size in config["task_sizes"][:task_id])
    end = start + int(config["task_sizes"][task_id])
    classes = config["class_order"][start:end]
    aps = run["rows"][task_id]["per_class_ap"][start:end]
    return {
        "task_id": task_id,
        "classes": {name: float(ap) for name, ap in zip(classes, aps)},
        "mean_ap": float(statistics.mean(aps)),
    }


def summarize_loss_search(
    labeled_roots: Sequence[Tuple[str, Path]],
    failed_labeled_roots: Sequence[Tuple[str, Path, Path]] = (),
) -> Dict[str, Any]:
    if len(labeled_roots) + len(failed_labeled_roots) != 1 + len(EXPECTED_CANDIDATES):
        raise ValueError("Loss search requires one BCE control and 20 ASL outcomes")
    labels = [label for label, _ in labeled_roots] + [
        label for label, _, _ in failed_labeled_roots
    ]
    if len(labels) != len(set(labels)):
        raise ValueError("Search labels must be unique")
    runs = [_load_run(label, root) for label, root in labeled_roots] + [
        _load_failed_run(label, root, log_path)
        for label, root, log_path in failed_labeled_roots
    ]
    commits = {
        (run["config"]["git"].get("commit"), run["config"]["git"].get("tree"))
        for run in runs
    }
    if len(commits) != 1:
        raise ValueError("Search runs do not share one Git commit and tree")

    controls = [
        run for run in runs
        if run["config"].get("parameter_group_loss_routing") == "joint_bce"
    ]
    if len(controls) != 1:
        raise ValueError("Loss search requires exactly one joint-BCE control")
    control = controls[0]
    if control["status"] != "complete":
        raise ValueError("Joint-BCE control must complete successfully")
    if (
        control["config"].get("model_parameter_objective") != "bce"
        or control["config"].get("adapter_parameter_objective") != "bce"
        or control["config"].get("asl") is not None
    ):
        raise ValueError("Joint-BCE control has invalid objective metadata")
    candidates = [run for run in runs if run is not control]
    observed_grid = set()
    for run in candidates:
        config = run["config"]
        if (
            config.get("parameter_group_loss_routing") != "adapter_asl"
            or config.get("model_parameter_objective") != "bce"
            or config.get("adapter_parameter_objective") != "asl"
        ):
            raise ValueError(f"Candidate {run['label']} has invalid loss routing")
        asl = config.get("asl") or {}
        if not _close(asl.get("gamma_pos"), 0.0) or not _close(asl.get("eps"), 1e-8):
            raise ValueError(f"Candidate {run['label']} has invalid fixed ASL values")
        observed_grid.add((float(asl["gamma_neg"]), float(asl["clip"])))
    if observed_grid != EXPECTED_CANDIDATES:
        raise ValueError("Observed ASL candidates do not match the preregistered grid")

    baseline_summary = control["summary"]["metrics"]
    baseline_focus = _focus_metrics(control)
    results = []
    for run in candidates:
        if run["status"] != "complete":
            results.append({
                "label": run["label"],
                "root": run["root"],
                "status": run["status"],
                "asl": run["config"]["asl"],
                "failure": run["failure"],
                "eligibility_gates": {
                    "training_completed": False,
                },
                "eligible": False,
            })
            continue
        summary = run["summary"]["metrics"]
        focus = _focus_metrics(run)
        deltas = {
            key: float(summary[key]) - float(baseline_summary[key])
            for key in (
                "final_mAP", "average_mAP", "final_cF1",
                "final_oF1", "forgetting",
            )
        }
        task6_map_delta = (
            float(run["rows"][6]["mAP"]) - float(control["rows"][6]["mAP"])
        )
        class_deltas = {
            name: focus["classes"][name] - baseline_focus["classes"][name]
            for name in focus["classes"]
        }
        gates = {
            "task6_mAP_non_decreasing": task6_map_delta >= 0.0,
            "final_mAP_non_decreasing": deltas["final_mAP"] >= 0.0,
            "sadness_ap_non_decreasing": class_deltas["Sadness"] >= 0.0,
            "suffering_ap_non_decreasing": class_deltas["Suffering"] >= 0.0,
            "final_cF1_within_half_point": deltas["final_cF1"] >= -0.5,
            "final_oF1_within_half_point": deltas["final_oF1"] >= -0.5,
        }
        results.append({
            "label": run["label"],
            "root": run["root"],
            "status": "complete",
            "asl": run["config"]["asl"],
            "task_curve": [
                {
                    key: float(row[key]) if key != "task_id" else int(row[key])
                    for key in ("task_id", "mAP", "cF1", "oF1")
                }
                for row in run["rows"]
            ],
            "summary_metrics": {
                key: float(summary[key])
                for key in (
                    "final_mAP", "average_mAP", "final_cF1",
                    "final_oF1", "forgetting",
                )
            },
            "deltas_vs_joint_bce": deltas,
            "task6_mAP_delta": task6_map_delta,
            "task6_new_classes": focus,
            "task6_new_class_deltas": class_deltas,
            "eligibility_gates": gates,
            "eligible": all(gates.values()),
        })

    eligible = [result for result in results if result["eligible"]]
    eligible.sort(
        key=lambda result: (
            result["summary_metrics"]["final_mAP"],
            result["task6_new_classes"]["mean_ap"],
            result["summary_metrics"]["average_mAP"],
            result["summary_metrics"]["final_cF1"],
            result["summary_metrics"]["final_oF1"],
            -result["summary_metrics"]["forgetting"],
        ),
        reverse=True,
    )
    winner = eligible[0] if eligible else None
    top_labels = (
        [
            result["label"] for result in eligible
            if winner["summary_metrics"]["final_mAP"]
            - result["summary_metrics"]["final_mAP"] <= 0.30
        ]
        if winner is not None else []
    )
    return {
        "schema_version": 1,
        "status": "complete",
        "comparison": "image_token_adapter_asl_seed0_full_8task_validation_loss_grid",
        "git": {"commit": next(iter(commits))[0], "tree": next(iter(commits))[1]},
        "selection_policy": {
            "primary": "final_mAP among candidates passing every hard gate",
            "secondary": [
                "task6_new_class_mean_ap", "average_mAP", "final_cF1",
                "final_oF1", "lower_forgetting",
            ],
            "retain_within_final_mAP": 0.30,
        },
        "joint_bce": {
            "label": control["label"],
            "root": control["root"],
            "summary_metrics": {
                key: float(baseline_summary[key])
                for key in (
                    "final_mAP", "average_mAP", "final_cF1",
                    "final_oF1", "forgetting",
                )
            },
            "task6_new_classes": baseline_focus,
        },
        "candidates": results,
        "failed_candidates": [
            result["label"] for result in results
            if result["status"] != "complete"
        ],
        "eligible_ranking": [result["label"] for result in eligible],
        "winner": winner["label"] if winner is not None else None,
        "retained_within_0.30_final_mAP": top_labels,
    }


def _labeled_root(value: str) -> Tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("Run must have LABEL=PATH form")
    return label, Path(path).expanduser()


def _failed_labeled_root(value: str) -> Tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "Failed run must have LABEL=RUN_ROOT=LOG_PATH form"
        )
    return parts[0], Path(parts[1]).expanduser(), Path(parts[2]).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=_labeled_root, action="append", required=True)
    parser.add_argument(
        "--failed-run", type=_failed_labeled_root, action="append", default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize_loss_search(args.run, args.failed_run)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
