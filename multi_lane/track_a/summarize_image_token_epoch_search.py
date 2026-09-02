"""Validate and rank the strict Image-token Adapter epoch search."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


EPOCHS = (18, 22, 26, 30, 34, 38, 42, 48)
ANCHOR_EPOCHS = 30
NUM_TASKS = 8
UPDATES_PER_EPOCH = 465
SUMMARY_KEYS = (
    "final_mAP",
    "average_mAP",
    "final_cF1",
    "final_oF1",
    "forgetting",
)
COMMON_CONFIG = {
    "seed": 0,
    "reporting_split": "test",
    "max_tasks": 8,
    "training_budget_mode": "epochs",
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
    "scheduler": "CosineAnnealingLR_reset_per_task",
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
        raise FileNotFoundError(f"Missing epoch-search artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _close(left: Any, right: Any) -> bool:
    if isinstance(right, float):
        return isinstance(left, (int, float)) and math.isclose(
            float(left), right, rel_tol=0.0, abs_tol=1e-12
        )
    return left == right


def _load_complete_run(label: str, epochs: int, root: Path) -> Dict[str, Any]:
    config = _load_json(root / "config.json")
    rows = _load_json(root / "task_metrics.json")
    history = _load_json(root / "training_history.json")
    summary = _load_json(root / "seed_summary.json")
    mismatches = [
        key for key, expected in COMMON_CONFIG.items()
        if not _close(config.get(key), expected)
    ]
    if config.get("epochs_per_task") != epochs:
        mismatches.append("epochs_per_task")
    asl = config.get("asl") or {}
    if any(not _close(asl.get(key), expected) for key, expected in EXPECTED_ASL.items()):
        mismatches.append("asl")
    if mismatches:
        raise ValueError(f"Epoch-search run {label} has protocol mismatches: {sorted(set(mismatches))}")
    if config.get("git", {}).get("dirty") is not False:
        raise ValueError(f"Epoch-search run {label} did not record clean Git")
    if summary.get("status") != "complete" or summary.get("seed") != 0:
        raise ValueError(f"Epoch-search run {label} is not a complete seed0 run")
    if len(rows) != NUM_TASKS or [row.get("task_id") for row in rows] != list(range(NUM_TASKS)):
        raise ValueError(f"Epoch-search run {label} lacks ordered tasks 0--7")
    if list(history) != [str(index) for index in range(NUM_TASKS)]:
        raise ValueError(f"Epoch-search run {label} has invalid history tasks")
    if any(len(history[str(index)]) != epochs for index in range(NUM_TASKS)):
        raise ValueError(f"Epoch-search run {label} has incomplete task history")
    completed_epochs = epochs * NUM_TASKS
    completed_updates = epochs * UPDATES_PER_EPOCH
    if summary.get("completed_epochs") != completed_epochs:
        raise ValueError(f"Epoch-search run {label} did not complete {completed_epochs} epochs")
    if summary.get("completed_optimizer_updates") != completed_updates:
        raise ValueError(f"Epoch-search run {label} did not complete {completed_updates} updates")
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
    if updates != completed_updates or skipped != 0:
        raise ValueError(f"Epoch-search run {label} has invalid optimizer-step accounting")
    if (root / "checkpoints").exists():
        raise ValueError(f"Epoch-search run {label} unexpectedly has checkpoints")
    return {
        "label": label,
        "epochs": epochs,
        "root": str(root.resolve()),
        "config": config,
        "rows": rows,
        "summary": summary,
    }


def _metrics(run: Dict[str, Any]) -> Dict[str, float]:
    values = run["summary"]["metrics"]
    result = {key: float(values[key]) for key in SUMMARY_KEYS}
    result["task6_mAP"] = float(run["rows"][6]["mAP"])
    result["late_task_mAP"] = sum(float(run["rows"][index]["mAP"]) for index in (5, 6, 7)) / 3.0
    return result


def _refinement(best_epochs: int) -> Dict[str, Any]:
    if best_epochs == ANCHOR_EPOCHS:
        return {"action": "stop_epoch_search", "epochs": []}
    if best_epochs == max(EPOCHS):
        return {"action": "expand_upper_boundary", "epochs": [54, 60]}
    if best_epochs == min(EPOCHS):
        return {"action": "expand_lower_boundary", "epochs": [10, 14, 16]}
    candidates = [best_epochs + offset for offset in (-3, -2, -1, 1, 2, 3)]
    return {
        "action": "refine_internal_winner",
        "epochs": [value for value in candidates if value > 0 and value not in EPOCHS],
    }


def summarize_epoch_search(
    labeled_roots: Sequence[Tuple[str, int, Path]],
) -> Dict[str, Any]:
    expected = {f"epochs{epochs}": epochs for epochs in EPOCHS}
    received = {label: epochs for label, epochs, _ in labeled_roots}
    if received != expected or len(labeled_roots) != len(expected):
        raise ValueError("Epoch search requires the complete 18/22/26/30/34/38/42/48 grid")
    runs = {
        label: _load_complete_run(label, epochs, root)
        for label, epochs, root in labeled_roots
    }
    commits = {
        (run["config"]["git"].get("commit"), run["config"]["git"].get("tree"))
        for run in runs.values()
    }
    if len(commits) != 1:
        raise ValueError("Epoch-search runs do not share one Git commit and tree")
    anchor = runs[f"epochs{ANCHOR_EPOCHS}"]
    anchor_metrics = _metrics(anchor)
    ranking = []
    for label, run in runs.items():
        metrics = _metrics(run)
        ranking.append({
            "label": label,
            "epochs_per_task": run["epochs"],
            "root": run["root"],
            "metrics": metrics,
            "delta_vs_epochs30": {
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
        "anchor": next(item for item in ranking if item["epochs_per_task"] == ANCHOR_EPOCHS),
        "winner": winner,
        "winner_exceeds_anchor": winner["metrics"]["final_mAP"] > anchor_metrics["final_mAP"],
        "refinement": _refinement(winner["epochs_per_task"]),
        "ranking": ranking,
    }


def _markdown(result: Dict[str, Any]) -> str:
    lines = [
        "# Image-token Adapter epoch search",
        "",
        "Exploratory held-out-test tuning; final mAP is the primary ranking metric.",
        "",
        "| Rank | Epochs/task | Final mAP | Delta vs 30 | Average mAP | Late-task mAP | Task6 mAP | cF1 | Forgetting |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, item in enumerate(result["ranking"], start=1):
        metrics = item["metrics"]
        delta = item["delta_vs_epochs30"]["final_mAP"]
        lines.append(
            f"| {index} | {item['epochs_per_task']} | {metrics['final_mAP']:.6f} | "
            f"{delta:+.6f} | {metrics['average_mAP']:.6f} | "
            f"{metrics['late_task_mAP']:.6f} | {metrics['task6_mAP']:.6f} | "
            f"{metrics['final_cF1']:.6f} | {metrics['forgetting']:.6f} |"
        )
    winner = result["winner"]
    lines.extend([
        "",
        f"Winner: **{winner['epochs_per_task']} epochs/task**, final mAP "
        f"**{winner['metrics']['final_mAP']:.6f}**.",
        "",
        f"Next action: `{result['refinement']['action']}` with "
        f"`{result['refinement']['epochs']}`.",
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
        (f"epochs{epochs}", epochs, args.batch_root / f"{args.batch_id}_epochs{epochs}")
        for epochs in EPOCHS
    ]
    result = summarize_epoch_search(roots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "epoch_search_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "epoch_search_summary.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    print(json.dumps({
        "winner_epochs": result["winner"]["epochs_per_task"],
        "winner_final_mAP": result["winner"]["metrics"]["final_mAP"],
        "winner_exceeds_anchor": result["winner_exceeds_anchor"],
        "refinement": result["refinement"],
    }, indent=2))


if __name__ == "__main__":
    main()
