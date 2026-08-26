"""Validate and rank the strict Image-token Adapter-ASL capacity/LR grid."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


BOTTLENECKS = (8, 16, 32, 64)
ADAPTER_LEARNING_RATES = (0.0001, 0.0002, 0.0004)
ANCHOR_LABEL = "b32_lr4e4_asl"
BASE_TRAINABLE_PARAMETERS = 689178
NUM_TASKS = 8
MINIMUM_MATERIAL_FINAL_MAP_GAIN = 0.5
SUMMARY_KEYS = (
    "final_mAP",
    "average_mAP",
    "final_cF1",
    "final_oF1",
    "forgetting",
)
COMMON_CONFIG = {
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
    "tf32": False,
    "adapter_layer_indices": [8],
    "adapter_residual_scale": 0.1,
    "adapter_activation": "relu",
    "adapter_task_initialization": "independent",
    "adapter_writes_back_to_frozen_visual_stream": False,
}
EXPECTED_ASL = {
    "gamma_neg": 9.8,
    "gamma_pos": 0.0,
    "clip": 0.05,
    "eps": 1e-8,
    "detach_focal_weight": True,
    "reduction": "mean_over_training_loss_view",
}


def _label(bottleneck: int, learning_rate: float) -> str:
    lr_code = {0.0001: "1e4", 0.0002: "2e4", 0.0004: "4e4"}[learning_rate]
    return f"b{bottleneck}_lr{lr_code}_asl"


CANDIDATES = {
    _label(bottleneck, learning_rate): (bottleneck, learning_rate)
    for bottleneck in BOTTLENECKS
    for learning_rate in ADAPTER_LEARNING_RATES
}


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing capacity/LR artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _close(left: Any, right: Any) -> bool:
    if isinstance(right, float):
        return isinstance(left, (int, float)) and math.isclose(
            float(left), right, rel_tol=0.0, abs_tol=1e-12
        )
    return left == right


def _validate_common(label: str, config: Dict[str, Any]) -> None:
    mismatches = [
        key for key, expected in COMMON_CONFIG.items()
        if not _close(config.get(key), expected)
    ]
    if mismatches:
        raise ValueError(
            f"Capacity/LR run {label} has protocol mismatches: {mismatches}"
        )
    if config.get("git", {}).get("dirty") is not False:
        raise ValueError(f"Capacity/LR run {label} did not record clean Git")


def _load_complete_run(label: str, root: Path) -> Dict[str, Any]:
    config = _load_json(root / "config.json")
    rows = _load_json(root / "task_metrics.json")
    history = _load_json(root / "training_history.json")
    summary = _load_json(root / "seed_summary.json")
    _validate_common(label, config)
    if summary.get("status") != "complete" or summary.get("seed") != 0:
        raise ValueError(f"Capacity/LR run {label} is not a complete seed0 run")
    if len(rows) != 8 or [row.get("task_id") for row in rows] != list(range(8)):
        raise ValueError(f"Capacity/LR run {label} lacks ordered tasks 0--7")
    if summary.get("completed_epochs") != 240:
        raise ValueError(f"Capacity/LR run {label} did not complete 240 epochs")
    if summary.get("completed_optimizer_updates") != 13950:
        raise ValueError(f"Capacity/LR run {label} did not complete 13950 updates")
    if (root / "checkpoints").exists():
        raise ValueError(f"Capacity/LR run {label} unexpectedly has checkpoints")
    if list(history) != [str(index) for index in range(8)]:
        raise ValueError(f"Capacity/LR run {label} has invalid history tasks")
    if any(len(history[str(index)]) != 30 for index in range(8)):
        raise ValueError(f"Capacity/LR run {label} has incomplete task history")
    updates = sum(
        int(row["optimizer_steps"])
        for task_history in history.values()
        for row in task_history
    )
    skipped = sum(
        int(row["skipped_optimizer_steps"])
        for task_history in history.values()
        for row in task_history
    )
    if updates != 13950 or skipped != 0:
        raise ValueError(
            f"Capacity/LR run {label} has invalid optimizer-step accounting"
        )
    return {
        "label": label,
        "root": str(root.resolve()),
        "config": config,
        "rows": rows,
        "summary": summary,
    }


def _parameters_per_task(bottleneck: int) -> int:
    # Down: 768*b+b; up: b*768+768.
    return 1537 * bottleneck + 768


def _validate_disabled(run: Dict[str, Any]) -> None:
    config = run["config"]
    valid = (
        config.get("adapter_mode") == "disabled"
        and config.get("parameter_group_loss_routing") == "joint_bce"
        and config.get("model_parameter_objective") == "bce"
        and config.get("adapter_parameter_objective") == "bce"
        and config.get("asl") is None
        and config.get("adapter_target") is None
        and config.get("adapter_learning_rate") is None
        and config.get("adapter_parameters_per_task") == 0
        and config.get("adapter_parameters") == 0
        and config.get("trainable_parameters") == BASE_TRAINABLE_PARAMETERS
    )
    if not valid:
        raise ValueError("Disabled capacity/LR control has invalid metadata")


def _validate_candidate(
    run: Dict[str, Any], bottleneck: int, learning_rate: float
) -> None:
    config = run["config"]
    per_task = _parameters_per_task(bottleneck)
    asl = config.get("asl") or {}
    valid = (
        config.get("adapter_mode") == "image_token"
        and config.get("adapter_bottleneck_dim") == bottleneck
        and _close(config.get("adapter_learning_rate"), learning_rate)
        and config.get("adapter_target") == "frozen_image_tokens_for_selector"
        and config.get("adapter_image_token_scope")
        == "block_ln1_cls_plus_patch_tokens"
        and config.get("adapter_initialization_rng") == "forked_global_state"
        and config.get("parameter_group_loss_routing") == "adapter_asl"
        and config.get("model_parameter_objective") == "bce"
        and config.get("adapter_parameter_objective") == "asl"
        and all(_close(asl.get(key), value) for key, value in EXPECTED_ASL.items())
        and config.get("adapter_parameters_per_task") == per_task
        and config.get("adapter_parameters") == per_task * NUM_TASKS
        and config.get("trainable_parameters")
        == BASE_TRAINABLE_PARAMETERS + per_task
    )
    if not valid:
        raise ValueError(f"Run {run['label']} has invalid Adapter-ASL metadata")


def _summary_metrics(run: Dict[str, Any]) -> Dict[str, float]:
    metrics = run["summary"]["metrics"]
    return {key: float(metrics[key]) for key in SUMMARY_KEYS}


def _focus_metrics(run: Dict[str, Any], task_id: int = 6) -> Dict[str, Any]:
    config = run["config"]
    start = sum(int(size) for size in config["task_sizes"][:task_id])
    end = start + int(config["task_sizes"][task_id])
    names = config["class_order"][start:end]
    values = run["rows"][task_id]["per_class_ap"][start:end]
    return {
        "task_id": task_id,
        "classes": {name: float(value) for name, value in zip(names, values)},
        "mean_ap": float(statistics.mean(values)),
    }


def _comparison(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    candidate_summary = _summary_metrics(candidate)
    baseline_summary = _summary_metrics(baseline)
    candidate_focus = _focus_metrics(candidate)
    baseline_focus = _focus_metrics(baseline)
    return {
        "summary_deltas": {
            key: candidate_summary[key] - baseline_summary[key]
            for key in SUMMARY_KEYS
        },
        "task6_mAP_delta": (
            float(candidate["rows"][6]["mAP"])
            - float(baseline["rows"][6]["mAP"])
        ),
        "task6_new_class_deltas": {
            name: candidate_focus["classes"][name] - value
            for name, value in baseline_focus["classes"].items()
        },
        "task6_new_class_mean_delta": (
            candidate_focus["mean_ap"] - baseline_focus["mean_ap"]
        ),
    }


def _gates(comparison: Dict[str, Any]) -> Dict[str, bool]:
    summary = comparison["summary_deltas"]
    classes = comparison["task6_new_class_deltas"]
    return {
        "final_mAP_non_decreasing": summary["final_mAP"] >= 0.0,
        "average_mAP_non_decreasing": summary["average_mAP"] >= 0.0,
        "task6_mAP_non_decreasing": comparison["task6_mAP_delta"] >= 0.0,
        "sadness_ap_non_decreasing": classes["Sadness"] >= 0.0,
        "sensitivity_ap_non_decreasing": classes["Sensitivity"] >= 0.0,
        "suffering_ap_non_decreasing": classes["Suffering"] >= 0.0,
        "final_cF1_within_half_point": summary["final_cF1"] >= -0.5,
        "final_oF1_within_half_point": summary["final_oF1"] >= -0.5,
        "forgetting_non_increasing": summary["forgetting"] <= 0.0,
    }


def _run_result(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": run["label"],
        "root": run["root"],
        "summary_metrics": _summary_metrics(run),
        "task_curve": [
            {key: row[key] for key in ("task_id", "mAP", "cF1", "oF1")}
            for row in run["rows"]
        ],
        "task6_new_classes": _focus_metrics(run),
    }


def summarize_capacity_lr_search(
    labeled_roots: Sequence[Tuple[str, Path]],
) -> Dict[str, Any]:
    expected_labels = {"disabled_bce", *CANDIDATES}
    labels = [label for label, _ in labeled_roots]
    if set(labels) != expected_labels or len(labels) != len(expected_labels):
        raise ValueError(
            "Capacity/LR search requires disabled plus the complete 4x3 grid"
        )
    runs = {
        label: _load_complete_run(label, root) for label, root in labeled_roots
    }
    commits = {
        (run["config"]["git"].get("commit"), run["config"]["git"].get("tree"))
        for run in runs.values()
    }
    if len(commits) != 1:
        raise ValueError("Capacity/LR runs do not share one Git commit and tree")

    disabled = runs["disabled_bce"]
    anchor = runs[ANCHOR_LABEL]
    _validate_disabled(disabled)
    for label, (bottleneck, learning_rate) in CANDIDATES.items():
        _validate_candidate(runs[label], bottleneck, learning_rate)

    results = []
    for label, (bottleneck, learning_rate) in CANDIDATES.items():
        run = runs[label]
        vs_disabled = _comparison(run, disabled)
        vs_anchor = _comparison(run, anchor)
        disabled_gates = _gates(vs_disabled)
        anchor_gates = _gates(vs_anchor)
        material_gain = (
            vs_anchor["summary_deltas"]["final_mAP"]
            >= MINIMUM_MATERIAL_FINAL_MAP_GAIN
        )
        results.append({
            **_run_result(run),
            "bottleneck": bottleneck,
            "adapter_learning_rate": learning_rate,
            "adapter_parameters_per_task": _parameters_per_task(bottleneck),
            "vs_disabled": vs_disabled,
            "vs_anchor": vs_anchor,
            "disabled_gates": disabled_gates,
            "anchor_gates": anchor_gates,
            "material_final_mAP_gain_vs_anchor": material_gain,
            "eligible_for_stage2": (
                all(disabled_gates.values())
                and all(anchor_gates.values())
                and material_gain
            ),
        })

    eligible = [row for row in results if row["eligible_for_stage2"]]
    eligible.sort(
        key=lambda row: (
            row["summary_metrics"]["final_mAP"],
            min(row["vs_anchor"]["task6_new_class_deltas"].values()),
            row["summary_metrics"]["average_mAP"],
            -row["summary_metrics"]["forgetting"],
            -row["adapter_parameters_per_task"],
        ),
        reverse=True,
    )
    best_final = max(results, key=lambda row: row["summary_metrics"]["final_mAP"])
    commit, tree = next(iter(commits))
    return {
        "schema_version": 1,
        "status": "complete",
        "comparison": (
            "image_token_adapter_asl_seed0_strict_fp32_capacity_lr_search"
        ),
        "git": {"commit": commit, "tree": tree},
        "protocol": {
            "layer": 8,
            "bottlenecks": list(BOTTLENECKS),
            "adapter_learning_rates": list(ADAPTER_LEARNING_RATES),
            "loss": "model_bce_adapter_asl",
            "asl": dict(EXPECTED_ASL),
            "amp": False,
            "tf32": False,
            "minimum_material_final_mAP_gain": MINIMUM_MATERIAL_FINAL_MAP_GAIN,
            "selection": (
                "all gates vs disabled and b32/lr4e-4 anchor, then at least "
                "+0.5 final mAP vs anchor"
            ),
        },
        "disabled": _run_result(disabled),
        "anchor_label": ANCHOR_LABEL,
        "anchor": _run_result(anchor),
        "candidates": results,
        "eligible_labels": [row["label"] for row in eligible],
        "winner_label": eligible[0]["label"] if eligible else None,
        "best_final_mAP_label": best_final["label"],
        "continue_to_scale_activation": bool(eligible),
    }


def _labeled_root(value: str) -> Tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("Run must have LABEL=PATH form")
    return label, Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=_labeled_root, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize_capacity_lr_search(args.run)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
