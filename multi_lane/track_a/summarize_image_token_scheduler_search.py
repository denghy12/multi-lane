"""Validate and rank the strict Image-token Adapter scheduler search."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


EPOCHS = 30
NUM_TASKS = 8
TOTAL_UPDATES = 13950
ANCHOR_LABEL = "cosine_anchor"
MILESTONE_FRACTIONS = (0.6, 0.85)
MILESTONE_EPOCHS = (18, 26)
MULTISTEP_GAMMA = 0.1
SPECS = {
    "cosine_anchor": ("cosine", 0.0, 0.0, "CosineAnnealingLR_reset_per_task"),
    "cosine_min001": ("cosine", 0.01, 0.0, "RelativeMinCosineAnnealingLR_reset_per_task"),
    "cosine_min010": ("cosine", 0.10, 0.0, "RelativeMinCosineAnnealingLR_reset_per_task"),
    "cosine_warmup005": ("cosine", 0.0, 0.05, "LinearWarmupCosineAnnealingLR_reset_per_task"),
    "cosine_warmup010": ("cosine", 0.0, 0.10, "LinearWarmupCosineAnnealingLR_reset_per_task"),
    "linear": ("linear", 0.0, 0.0, "LinearLR_reset_per_task"),
    "constant": ("constant", 0.0, 0.0, "ConstantLR_reset_per_task"),
    "multistep": ("multistep", 0.0, 0.0, "MultiStepLR_reset_per_task"),
}
SUMMARY_KEYS = (
    "final_mAP", "average_mAP", "final_cF1", "final_oF1", "forgetting",
)
COMMON_CONFIG = {
    "seed": 0,
    "reporting_split": "test",
    "max_tasks": 8,
    "training_budget_mode": "epochs",
    "epochs_per_task": EPOCHS,
    "optimizer_updates_per_task": None,
    "train_batch_size": 64,
    "eval_batch_size": 64,
    "threshold": 0.5,
    "training_loss_mode": "legacy_full_zero",
    "training_label_scope": "current_classes_only",
    "evaluation_scope": "samples_intersect_seen_classes",
    "parameter_group_loss_routing": "adapter_asl",
    "model_parameter_objective": "bce",
    "adapter_parameter_objective": "asl",
    "input_mode": "full",
    "input_normalization": "clip",
    "train_crop_scale": [0.05, 1.0],
    "learning_rate": 0.0125,
    "optimizer": "Adam_reset_per_task",
    "weight_decay": 0.0,
    "temperature": 1.0,
    "save_checkpoints": False,
    "amp": True,
    "tf32": True,
    "adapter_mode": "image_token",
    "adapter_bottleneck_dim": 32,
    "adapter_layer_indices": [1],
    "adapter_residual_scale": 0.1,
    "adapter_residual_gate_mode": "fixed",
    "adapter_activation": "relu",
    "adapter_task_initialization": "independent",
    "adapter_writes_back_to_frozen_visual_stream": False,
    "adapter_learning_rate": 0.0004,
    "adapter_weight_decay": 0.0,
    "adapter_regularization": "none",
    "scheduler_multistep_milestone_fractions": [0.6, 0.85],
    "scheduler_multistep_milestone_epochs": [18, 26],
    "scheduler_multistep_gamma": 0.1,
    "scheduler_step_unit": "epoch",
}
EXPECTED_ASL = {
    "gamma_neg": 9.8,
    "gamma_pos": 0.0,
    "clip": 0.05,
    "eps": 1e-8,
    "detach_focal_weight": True,
    "reduction": "mean_over_training_loss_view",
}


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing scheduler-search artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _close(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(right, float):
        return isinstance(left, (int, float)) and math.isclose(
            float(left), right, rel_tol=0.0, abs_tol=tolerance
        )
    return left == right


def expected_multiplier(label: str, step: int) -> float:
    mode, min_ratio, warmup_ratio, _ = SPECS[label]
    bounded = min(max(step, 0), EPOCHS)
    if mode == "constant":
        return 1.0
    if mode == "linear":
        return 1.0 - bounded / EPOCHS
    if mode == "multistep":
        drops = sum(bounded >= milestone for milestone in MILESTONE_EPOCHS)
        return MULTISTEP_GAMMA ** drops
    warmup_epochs = int(math.ceil(EPOCHS * warmup_ratio)) if warmup_ratio else 0
    if warmup_epochs and bounded < warmup_epochs:
        return (bounded + 1) / warmup_epochs
    cosine_steps = EPOCHS - warmup_epochs
    progress = min(max((bounded - warmup_epochs) / cosine_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def _load_complete_run(label: str, root: Path) -> Dict[str, Any]:
    config = _load_json(root / "config.json")
    rows = _load_json(root / "task_metrics.json")
    history = _load_json(root / "training_history.json")
    summary = _load_json(root / "seed_summary.json")
    mismatches = [
        key for key, expected in COMMON_CONFIG.items()
        if not _close(config.get(key), expected)
    ]
    mode, min_ratio, warmup_ratio, scheduler_name = SPECS[label]
    expected_scheduler = {
        "scheduler": scheduler_name,
        "scheduler_mode": mode,
        "scheduler_min_lr_ratio": min_ratio,
        "scheduler_warmup_ratio": warmup_ratio,
        "scheduler_warmup_epochs": (
            int(math.ceil(EPOCHS * warmup_ratio)) if warmup_ratio else 0
        ),
    }
    mismatches.extend(
        key for key, expected in expected_scheduler.items()
        if not _close(config.get(key), expected)
    )
    asl = config.get("asl") or {}
    if any(not _close(asl.get(key), expected) for key, expected in EXPECTED_ASL.items()):
        mismatches.append("asl")
    if mismatches:
        raise ValueError(
            f"Scheduler-search run {label} has protocol mismatches: {sorted(set(mismatches))}"
        )
    if config.get("git", {}).get("dirty") is not False:
        raise ValueError(f"Scheduler-search run {label} did not record clean Git")
    if summary.get("status") != "complete" or summary.get("seed") != 0:
        raise ValueError(f"Scheduler-search run {label} is not a complete seed0 run")
    if len(rows) != NUM_TASKS or [row.get("task_id") for row in rows] != list(range(NUM_TASKS)):
        raise ValueError(f"Scheduler-search run {label} lacks ordered tasks 0--7")
    if list(history) != [str(index) for index in range(NUM_TASKS)]:
        raise ValueError(f"Scheduler-search run {label} has invalid history tasks")
    if any(len(history[str(index)]) != EPOCHS for index in range(NUM_TASKS)):
        raise ValueError(f"Scheduler-search run {label} has incomplete history")
    if summary.get("completed_epochs") != EPOCHS * NUM_TASKS:
        raise ValueError(f"Scheduler-search run {label} did not complete 240 epochs")
    if summary.get("completed_optimizer_updates") != TOTAL_UPDATES:
        raise ValueError(f"Scheduler-search run {label} did not complete {TOTAL_UPDATES} updates")
    updates = sum(
        int(row["optimizer_steps"])
        for task_history in history.values() for row in task_history
    )
    skipped = sum(
        int(row["skipped_optimizer_steps"])
        for task_history in history.values() for row in task_history
    )
    if updates != TOTAL_UPDATES or skipped != 0:
        raise ValueError(f"Scheduler-search run {label} has invalid step accounting")
    for task_history in history.values():
        for step, row in enumerate(task_history):
            factor = expected_multiplier(label, step)
            next_factor = expected_multiplier(label, step + 1)
            expected_values = {
                "learning_rate": 0.0125 * factor,
                "next_learning_rate": 0.0125 * next_factor,
                "adapter_learning_rate": 0.0004 * factor,
                "next_adapter_learning_rate": 0.0004 * next_factor,
            }
            if any(
                not _close(row.get(key), expected, tolerance=1e-10)
                for key, expected in expected_values.items()
            ):
                raise ValueError(
                    f"Scheduler-search run {label} has invalid LR trajectory at step {step}"
                )
    if (root / "checkpoints").exists():
        raise ValueError(f"Scheduler-search run {label} unexpectedly has checkpoints")
    return {
        "label": label,
        "root": str(root.resolve()),
        "config": config,
        "rows": rows,
        "summary": summary,
    }


def _metrics(run: Dict[str, Any]) -> Dict[str, float]:
    values = run["summary"]["metrics"]
    result = {key: float(values[key]) for key in SUMMARY_KEYS}
    result["task6_mAP"] = float(run["rows"][6]["mAP"])
    result["late_task_mAP"] = sum(
        float(run["rows"][index]["mAP"]) for index in (5, 6, 7)
    ) / 3.0
    return result


def summarize_scheduler_search(
    labeled_roots: Sequence[Tuple[str, Path]],
) -> Dict[str, Any]:
    labels = [label for label, _ in labeled_roots]
    if set(labels) != set(SPECS) or len(labels) != len(SPECS):
        raise ValueError("Scheduler search requires the complete eight-run grid")
    runs = {label: _load_complete_run(label, root) for label, root in labeled_roots}
    commits = {
        (run["config"]["git"].get("commit"), run["config"]["git"].get("tree"))
        for run in runs.values()
    }
    if len(commits) != 1:
        raise ValueError("Scheduler-search runs do not share one Git commit and tree")
    anchor_metrics = _metrics(runs[ANCHOR_LABEL])
    ranking = []
    for label, run in runs.items():
        metrics = _metrics(run)
        ranking.append({
            "label": label,
            "root": run["root"],
            "metrics": metrics,
            "delta_vs_anchor": {
                key: metrics[key] - anchor_metrics[key] for key in metrics
            },
        })
    ranking.sort(
        key=lambda item: (item["metrics"]["final_mAP"], item["metrics"]["average_mAP"]),
        reverse=True,
    )
    winner = ranking[0]
    return {
        "schema_version": 1,
        "status": "complete",
        "selection_metric": "final_test_mAP_then_average_test_mAP",
        "exploratory_test_tuning": True,
        "shared_git": {
            "commit": next(iter(commits))[0],
            "tree": next(iter(commits))[1],
        },
        "anchor": next(item for item in ranking if item["label"] == ANCHOR_LABEL),
        "winner": winner,
        "winner_exceeds_anchor": winner["metrics"]["final_mAP"] > anchor_metrics["final_mAP"],
        "next_action": (
            "lock_scheduler_winner_for_confirmation"
            if winner["label"] != ANCHOR_LABEL
            else "stop_single_view_scheduler_search_and_start_dual_view"
        ),
        "ranking": ranking,
    }


def _markdown(result: Dict[str, Any]) -> str:
    lines = [
        "# Image-token Adapter scheduler search",
        "",
        "Exploratory held-out-test tuning; final mAP is the primary ranking metric.",
        "",
        "| Rank | Scheduler | Final mAP | Delta | Average mAP | Late-task mAP | Task6 mAP | cF1 | Forgetting |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, item in enumerate(result["ranking"], start=1):
        metrics = item["metrics"]
        lines.append(
            f"| {index} | {item['label']} | {metrics['final_mAP']:.6f} | "
            f"{item['delta_vs_anchor']['final_mAP']:+.6f} | "
            f"{metrics['average_mAP']:.6f} | {metrics['late_task_mAP']:.6f} | "
            f"{metrics['task6_mAP']:.6f} | {metrics['final_cF1']:.6f} | "
            f"{metrics['forgetting']:.6f} |"
        )
    lines.extend([
        "",
        f"Winner: **{result['winner']['label']}**, final mAP "
        f"**{result['winner']['metrics']['final_mAP']:.6f}**.",
        "",
        f"Next action: `{result['next_action']}`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    roots = [
        (label, args.batch_root / f"{args.batch_id}_{label}") for label in SPECS
    ]
    result = summarize_scheduler_search(roots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scheduler_search_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "scheduler_search_summary.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    print(json.dumps({
        "winner": result["winner"]["label"],
        "winner_final_mAP": result["winner"]["metrics"]["final_mAP"],
        "winner_exceeds_anchor": result["winner_exceeds_anchor"],
        "next_action": result["next_action"],
    }, indent=2))


if __name__ == "__main__":
    main()
