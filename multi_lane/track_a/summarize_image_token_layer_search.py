"""Validate and rank the paired FP32 Image-token Adapter layer search."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


LAYERS = tuple(range(12))
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
    "adapter_bottleneck_dim": 32,
    "adapter_residual_scale": 0.1,
    "adapter_activation": "relu",
    "adapter_task_initialization": "independent",
    "adapter_writes_back_to_frozen_visual_stream": False,
}
SUMMARY_KEYS = (
    "final_mAP",
    "average_mAP",
    "final_cF1",
    "final_oF1",
    "forgetting",
)


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing layer-search artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _close(left: Any, right: Any) -> bool:
    if isinstance(right, float):
        return isinstance(left, (int, float)) and math.isclose(
            float(left), right, rel_tol=0.0, abs_tol=1e-12
        )
    return left == right


def _validate_common_config(label: str, config: Dict[str, Any]) -> None:
    mismatches = [
        key for key, expected in COMMON_CONFIG.items()
        if not _close(config.get(key), expected)
    ]
    if mismatches:
        raise ValueError(
            f"Layer-search run {label} has protocol mismatches: {mismatches}"
        )
    if config.get("git", {}).get("dirty") is not False:
        raise ValueError(f"Layer-search run {label} did not record clean Git")


def _load_complete_run(label: str, root: Path) -> Dict[str, Any]:
    config = _load_json(root / "config.json")
    rows = _load_json(root / "task_metrics.json")
    history = _load_json(root / "training_history.json")
    summary = _load_json(root / "seed_summary.json")
    _validate_common_config(label, config)
    if summary.get("status") != "complete" or summary.get("seed") != 0:
        raise ValueError(f"Layer-search run {label} is not a complete seed0 run")
    if len(rows) != 8 or [row.get("task_id") for row in rows] != list(range(8)):
        raise ValueError(f"Layer-search run {label} lacks ordered tasks 0--7")
    if summary.get("completed_epochs") != 240:
        raise ValueError(f"Layer-search run {label} did not complete 240 epochs")
    if summary.get("completed_optimizer_updates") != 13950:
        raise ValueError(f"Layer-search run {label} did not complete 13950 updates")
    if (root / "checkpoints").exists():
        raise ValueError(f"Layer-search run {label} unexpectedly has checkpoints")
    if list(history) != [str(index) for index in range(8)]:
        raise ValueError(f"Layer-search run {label} has invalid history tasks")
    if any(len(history[str(index)]) != 30 for index in range(8)):
        raise ValueError(f"Layer-search run {label} has incomplete task history")
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
            f"Layer-search run {label} has invalid optimizer-step accounting"
        )
    return {
        "label": label,
        "root": str(root.resolve()),
        "config": config,
        "rows": rows,
        "history": history,
        "summary": summary,
    }


def _validate_disabled(run: Dict[str, Any]) -> None:
    config = run["config"]
    if (
        config.get("adapter_mode") != "disabled"
        or config.get("parameter_group_loss_routing") != "joint_bce"
        or config.get("model_parameter_objective") != "bce"
        or config.get("adapter_parameter_objective") != "bce"
        or config.get("asl") is not None
        or config.get("adapter_target") is not None
        or config.get("adapter_learning_rate") is not None
    ):
        raise ValueError("Disabled layer-search control has invalid metadata")


def _validate_image_token(
    run: Dict[str, Any], layer: int, routing: str
) -> None:
    config = run["config"]
    if config.get("adapter_mode") != "image_token":
        raise ValueError(f"Run {run['label']} is not an Image-token Adapter")
    if config.get("adapter_layer_indices") != [layer]:
        raise ValueError(f"Run {run['label']} has the wrong layer")
    if (
        config.get("adapter_target") != "frozen_image_tokens_for_selector"
        or config.get("adapter_image_token_scope")
        != "block_ln1_cls_plus_patch_tokens"
        or not _close(config.get("adapter_learning_rate"), 0.0004)
        or config.get("adapter_initialization_rng") != "forked_global_state"
    ):
        raise ValueError(f"Run {run['label']} has invalid Adapter metadata")
    if config.get("parameter_group_loss_routing") != routing:
        raise ValueError(f"Run {run['label']} has the wrong loss routing")
    if config.get("model_parameter_objective") != "bce":
        raise ValueError(f"Run {run['label']} must retain model BCE")
    if routing == "joint_bce":
        if config.get("adapter_parameter_objective") != "bce" or config.get("asl") is not None:
            raise ValueError(f"Run {run['label']} has invalid BCE metadata")
    else:
        asl = config.get("asl") or {}
        expected_asl = {
            "gamma_neg": 9.8,
            "gamma_pos": 0.0,
            "clip": 0.05,
            "eps": 1e-8,
            "detach_focal_weight": True,
            "reduction": "mean_over_training_loss_view",
        }
        if (
            config.get("adapter_parameter_objective") != "asl"
            or any(not _close(asl.get(key), value) for key, value in expected_asl.items())
        ):
            raise ValueError(f"Run {run['label']} has invalid ASL metadata")


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


def _summary_metrics(run: Dict[str, Any]) -> Dict[str, float]:
    metrics = run["summary"]["metrics"]
    return {key: float(metrics[key]) for key in SUMMARY_KEYS}


def _deltas(
    candidate: Dict[str, float], baseline: Dict[str, float]
) -> Dict[str, float]:
    return {key: candidate[key] - baseline[key] for key in SUMMARY_KEYS}


def _comparison(
    candidate: Dict[str, Any], baseline: Dict[str, Any]
) -> Dict[str, Any]:
    candidate_summary = _summary_metrics(candidate)
    baseline_summary = _summary_metrics(baseline)
    candidate_focus = _focus_metrics(candidate)
    baseline_focus = _focus_metrics(baseline)
    class_deltas = {
        name: candidate_focus["classes"][name] - baseline_focus["classes"][name]
        for name in candidate_focus["classes"]
    }
    return {
        "summary_deltas": _deltas(candidate_summary, baseline_summary),
        "task6_mAP_delta": (
            float(candidate["rows"][6]["mAP"])
            - float(baseline["rows"][6]["mAP"])
        ),
        "task6_new_class_deltas": class_deltas,
        "task6_new_class_mean_delta": (
            candidate_focus["mean_ap"] - baseline_focus["mean_ap"]
        ),
    }


def _gates(comparison: Dict[str, Any]) -> Dict[str, bool]:
    summary = comparison["summary_deltas"]
    classes = comparison["task6_new_class_deltas"]
    gates = {
        "final_mAP_non_decreasing": summary["final_mAP"] >= 0.0,
        "task6_mAP_non_decreasing": comparison["task6_mAP_delta"] >= 0.0,
        "sadness_ap_non_decreasing": classes["Sadness"] >= 0.0,
        "sensitivity_ap_within_half_point": classes["Sensitivity"] >= -0.5,
        "suffering_ap_non_decreasing": classes["Suffering"] >= 0.0,
        "final_cF1_within_half_point": summary["final_cF1"] >= -0.5,
        "final_oF1_within_half_point": summary["final_oF1"] >= -0.5,
        "forgetting_non_increasing": summary["forgetting"] <= 0.0,
    }
    return gates


def summarize_layer_search(
    labeled_roots: Sequence[Tuple[str, Path]],
) -> Dict[str, Any]:
    expected_labels = {"disabled_bce"}
    for layer in LAYERS:
        expected_labels.add(f"layer{layer}_bce")
        expected_labels.add(f"layer{layer}_asl")
    labels = [label for label, _ in labeled_roots]
    if set(labels) != expected_labels or len(labels) != len(expected_labels):
        raise ValueError("Layer search requires disabled plus paired BCE/ASL layers 0--11")
    runs = {
        label: _load_complete_run(label, root) for label, root in labeled_roots
    }
    commits = {
        (run["config"]["git"].get("commit"), run["config"]["git"].get("tree"))
        for run in runs.values()
    }
    if len(commits) != 1:
        raise ValueError("Layer-search runs do not share one Git commit and tree")

    disabled = runs["disabled_bce"]
    _validate_disabled(disabled)
    disabled_summary = _summary_metrics(disabled)
    disabled_focus = _focus_metrics(disabled)
    results = []
    for layer in LAYERS:
        bce = runs[f"layer{layer}_bce"]
        asl = runs[f"layer{layer}_asl"]
        _validate_image_token(bce, layer, "joint_bce")
        _validate_image_token(asl, layer, "adapter_asl")
        bce_vs_disabled = _comparison(bce, disabled)
        asl_vs_bce = _comparison(asl, bce)
        asl_vs_disabled = _comparison(asl, disabled)
        bce_gates = _gates(bce_vs_disabled)
        asl_vs_bce_gates = _gates(asl_vs_bce)
        asl_vs_disabled_gates = _gates(asl_vs_disabled)
        asl_gates = {
            **{
                f"vs_paired_bce_{key}": value
                for key, value in asl_vs_bce_gates.items()
            },
            **{
                f"vs_disabled_{key}": value
                for key, value in asl_vs_disabled_gates.items()
            },
        }
        results.append({
            "layer": layer,
            "bce": {
                "label": bce["label"],
                "root": bce["root"],
                "summary_metrics": _summary_metrics(bce),
                "task_curve": [
                    {key: row[key] for key in ("task_id", "mAP", "cF1", "oF1")}
                    for row in bce["rows"]
                ],
                "task6_new_classes": _focus_metrics(bce),
                "vs_disabled": bce_vs_disabled,
                "eligibility_gates": bce_gates,
                "eligible": all(bce_gates.values()),
            },
            "asl": {
                "label": asl["label"],
                "root": asl["root"],
                "summary_metrics": _summary_metrics(asl),
                "task_curve": [
                    {key: row[key] for key in ("task_id", "mAP", "cF1", "oF1")}
                    for row in asl["rows"]
                ],
                "task6_new_classes": _focus_metrics(asl),
                "vs_paired_bce": asl_vs_bce,
                "vs_disabled": asl_vs_disabled,
                "eligibility_gates": asl_gates,
                "eligible": all(asl_gates.values()),
            },
        })

    eligible_asl = [row for row in results if row["asl"]["eligible"]]
    eligible_asl.sort(
        key=lambda row: (
            row["asl"]["summary_metrics"]["final_mAP"],
            row["asl"]["task6_new_classes"]["mean_ap"],
            row["asl"]["summary_metrics"]["average_mAP"],
            row["asl"]["summary_metrics"]["final_cF1"],
            row["asl"]["summary_metrics"]["final_oF1"],
            -row["asl"]["summary_metrics"]["forgetting"],
        ),
        reverse=True,
    )
    eligible_bce = [row for row in results if row["bce"]["eligible"]]
    eligible_bce.sort(
        key=lambda row: (
            row["bce"]["summary_metrics"]["final_mAP"],
            row["bce"]["task6_new_classes"]["mean_ap"],
            -row["bce"]["summary_metrics"]["forgetting"],
        ),
        reverse=True,
    )
    best_final_asl = max(
        results, key=lambda row: row["asl"]["summary_metrics"]["final_mAP"]
    )
    best_final_bce = max(
        results, key=lambda row: row["bce"]["summary_metrics"]["final_mAP"]
    )
    commit, tree = next(iter(commits))
    return {
        "schema_version": 1,
        "status": "complete",
        "comparison": "image_token_adapter_fp32_seed0_full_8task_validation_layer_search",
        "git": {"commit": commit, "tree": tree},
        "protocol": {
            "layers": list(LAYERS),
            "losses": ["joint_bce", "model_bce_adapter_asl"],
            "asl": {"gamma_neg": 9.8, "gamma_pos": 0.0, "clip": 0.05},
            "selection_primary": "final_mAP among candidates passing all hard gates",
        },
        "disabled": {
            "label": disabled["label"],
            "root": disabled["root"],
            "summary_metrics": disabled_summary,
            "task6_new_classes": disabled_focus,
        },
        "layers": results,
        "eligible_asl_layers": [row["layer"] for row in eligible_asl],
        "eligible_bce_layers": [row["layer"] for row in eligible_bce],
        "winner_asl_layer": eligible_asl[0]["layer"] if eligible_asl else None,
        "winner_bce_layer": eligible_bce[0]["layer"] if eligible_bce else None,
        "best_final_mAP_asl_layer": best_final_asl["layer"],
        "best_final_mAP_bce_layer": best_final_bce["layer"],
        "continue_with_asl": bool(eligible_asl),
        "continue_with_bce": bool(eligible_bce),
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
    result = summarize_layer_search(args.run)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
