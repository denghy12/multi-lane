"""Compare paired seed-0 validation runs for the Track-A Adapter study."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


CURVE_METRICS = ("mAP", "cF1", "oF1")
SUMMARY_METRICS = ("final_mAP", "average_mAP", "final_cF1", "final_oF1")
PAIR_FIELDS = (
    "protocol_id",
    "seed",
    "class_order",
    "task_sizes",
    "train_split",
    "validation_split",
    "reporting_split",
    "training_label_scope",
    "training_loss_mode",
    "training_loss_reduction_classes",
    "evaluation_scope",
    "threshold",
    "epochs_per_task",
    "train_batch_size",
    "eval_batch_size",
    "workers",
    "optimizer",
    "learning_rate",
    "scheduler",
    "weight_decay",
    "temperature",
    "input_mode",
    "input_normalization",
    "input_normalization_mean",
    "input_normalization_std",
    "train_crop_scale",
    "amp",
    "tf32",
    "cudnn_benchmark",
    "cudnn_deterministic",
    "num_selectors",
    "num_prompts",
    "num_prompt_layers",
    "normalize",
    "head_mode",
    "max_tasks",
    "clip_checkpoint_sha256",
    "data_root",
)


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing comparison input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run(run_root: Path) -> Dict[str, Any]:
    config = _load_json(run_root / "config.json")
    summary = _load_json(run_root / "seed_summary.json")
    task_metrics = _load_json(run_root / "task_metrics.json")
    if summary.get("status") != "complete":
        raise ValueError(f"Run is not complete: {run_root}")
    if config.get("seed") != 0 or summary.get("seed") != 0:
        raise ValueError("Paired validation comparison only accepts seed0")
    if config.get("reporting_split") != "val":
        raise ValueError("Comparison inputs must use reporting_split=val")
    expected_tasks = int(config.get("max_tasks", 0))
    if expected_tasks != 8 or len(task_metrics) != expected_tasks:
        raise ValueError("Comparison inputs must contain all 8 tasks")
    if summary.get("completed_epochs") != config.get("epochs_per_task") * 8:
        raise ValueError(f"Run has incomplete epochs: {run_root}")
    if config.get("git", {}).get("dirty") is not False:
        raise ValueError(f"Run was not recorded from a clean worktree: {run_root}")
    task_ids = [int(row["task_id"]) for row in task_metrics]
    if task_ids != list(range(8)):
        raise ValueError(f"Unexpected task ids: {task_ids}")
    for row in task_metrics:
        if len(row["per_class_ap"]) != int(row["seen_classes"]):
            raise ValueError(f"Invalid per-class AP length at task {row['task_id']}")
    return {"root": str(run_root), "config": config, "summary": summary,
            "task_metrics": task_metrics}


def _validate_pair(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_task_initialization: str,
) -> None:
    baseline_config = baseline["config"]
    candidate_config = candidate["config"]
    mismatches = [
        field for field in PAIR_FIELDS
        if baseline_config.get(field) != candidate_config.get(field)
    ]
    for git_field in ("commit", "tree"):
        if baseline_config["git"].get(git_field) != candidate_config["git"].get(
            git_field
        ):
            mismatches.append(f"git.{git_field}")
    if mismatches:
        raise ValueError(f"Runs are not paired; mismatched fields: {mismatches}")
    if baseline_config.get("adapter_mode") != "disabled":
        raise ValueError("Baseline run must have adapter_mode=disabled")
    if candidate_config.get("adapter_mode") != "task_lane":
        raise ValueError("Candidate run must have adapter_mode=task_lane")
    if (
        candidate_config.get("adapter_task_initialization")
        != candidate_task_initialization
    ):
        raise ValueError(
            "Candidate Adapter initialization does not match the requested mode"
        )
    if candidate_config.get("adapter_initialization_rng") != "forked_global_state":
        raise ValueError("Candidate run must isolate Adapter initialization RNG")


def _metric_triplet(
    baseline: float, candidate: float, candidate_label: str
) -> Dict[str, float]:
    return {
        "disabled": float(baseline),
        candidate_label: float(candidate),
        "delta": float(candidate) - float(baseline),
    }


def compare_runs(
    baseline_root: Path,
    candidate_root: Path,
    focus_task_id: int = 6,
    candidate_task_initialization: str = "copy_previous",
) -> Dict[str, Any]:
    if candidate_task_initialization not in {"independent", "copy_previous"}:
        raise ValueError("Candidate Adapter initialization mode is invalid")
    baseline = _load_run(baseline_root)
    candidate = _load_run(candidate_root)
    _validate_pair(baseline, candidate, candidate_task_initialization)
    candidate_label = candidate_task_initialization

    baseline_config = baseline["config"]
    class_order = list(baseline_config["class_order"])
    task_sizes = [int(value) for value in baseline_config["task_sizes"]]
    if not 0 <= focus_task_id < len(task_sizes):
        raise ValueError("Focus task id is outside the protocol")
    focus_start = sum(task_sizes[:focus_task_id])
    focus_end = focus_start + task_sizes[focus_task_id]
    focus_classes = class_order[focus_start:focus_end]

    curve = []
    for baseline_row, candidate_row in zip(
        baseline["task_metrics"], candidate["task_metrics"]
    ):
        if (
            baseline_row["task_id"] != candidate_row["task_id"]
            or baseline_row["seen_classes"] != candidate_row["seen_classes"]
            or baseline_row["samples"] != candidate_row["samples"]
        ):
            raise ValueError("Task rows are not aligned")
        curve.append({
            "task_id": int(baseline_row["task_id"]),
            "seen_classes": int(baseline_row["seen_classes"]),
            "samples": int(baseline_row["samples"]),
            "metrics": {
                metric: _metric_triplet(
                    baseline_row[metric], candidate_row[metric], candidate_label
                )
                for metric in CURVE_METRICS
            },
        })

    focus_rows: Dict[str, Any] = {}
    for stage_name, task_id in (("at_introduction", focus_task_id), ("final", 7)):
        baseline_ap = baseline["task_metrics"][task_id]["per_class_ap"]
        candidate_ap = candidate["task_metrics"][task_id]["per_class_ap"]
        classes = {
            class_name: _metric_triplet(
                baseline_ap[class_index], candidate_ap[class_index], candidate_label
            )
            for class_index, class_name in zip(
                range(focus_start, focus_end), focus_classes
            )
        }
        baseline_mean = statistics.mean(
            baseline_ap[focus_start:focus_end]
        )
        candidate_mean = statistics.mean(
            candidate_ap[focus_start:focus_end]
        )
        focus_rows[stage_name] = {
            "task_id": task_id,
            "classes": classes,
            "mean_ap": _metric_triplet(
                baseline_mean, candidate_mean, candidate_label
            ),
        }

    baseline_summary = baseline["summary"]["metrics"]
    candidate_summary = candidate["summary"]["metrics"]
    summary_metrics = {
        metric: _metric_triplet(
            baseline_summary[metric], candidate_summary[metric], candidate_label
        )
        for metric in SUMMARY_METRICS
    }
    forgetting = _metric_triplet(
        baseline_summary["forgetting"],
        candidate_summary["forgetting"],
        candidate_label,
    )
    forgetting["improvement"] = -forgetting["delta"]
    summary_metrics["forgetting"] = forgetting

    criteria = {
        "task6_mAP_improved": curve[focus_task_id]["metrics"]["mAP"]["delta"] > 0,
        "task7_final_mAP_improved": curve[-1]["metrics"]["mAP"]["delta"] > 0,
        "task6_new_class_mean_ap_improved": (
            focus_rows["at_introduction"]["mean_ap"]["delta"] > 0
        ),
    }
    continue_method = all(criteria.values())
    decision = {
        "primary_metric": "mAP",
        "criteria": criteria,
        "continue_task_lane_adapter": continue_method,
        "recommendation": (
            "continue_method_refinement"
            if continue_method
            else "stop_task_lane_adapter_capacity_scaling"
        ),
        "rule": (
            "Continue only when the candidate improves task6 mAP, task7/final mAP, "
            "and the task6 new-class mean AP versus the paired disabled run."
        ),
    }

    return {
        "schema_version": 2,
        "status": "complete",
        "comparison": "seed0_full_8task_validation",
        "protocol_id": baseline_config["protocol_id"],
        "seed": 0,
        "git_commit": baseline_config["git"]["commit"],
        "baseline_run": baseline["root"],
        "candidate_run": candidate["root"],
        "candidate_task_initialization": candidate_task_initialization,
        "task_curve": curve,
        "summary_metrics": summary_metrics,
        "focus_task": {
            "task_id": focus_task_id,
            "classes": focus_classes,
            "stages": focus_rows,
        },
        "decision": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--focus-task-id", type=int, default=6)
    parser.add_argument(
        "--candidate-task-init",
        choices=("independent", "copy_previous"),
        default="copy_previous",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_runs(
        args.baseline_run.expanduser().resolve(),
        args.candidate_run.expanduser().resolve(),
        args.focus_task_id,
        args.candidate_task_init,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
