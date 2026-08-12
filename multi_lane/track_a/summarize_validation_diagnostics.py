"""Summarize multiple complete seed-0 validation diagnostics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


REQUIRED_COMMON_FIELDS = (
    "protocol_id",
    "seed",
    "class_order",
    "task_sizes",
    "reporting_split",
    "training_label_scope",
    "training_loss_mode",
    "training_loss_reduction_classes",
    "training_loss_current_only_gradient_multiplier_vs_legacy",
    "training_loss_optimizer_scale_note",
    "evaluation_scope",
    "epochs_per_task",
    "train_batch_size",
    "eval_batch_size",
    "workers",
    "threshold",
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
    "adapter_mode",
    "adapter_bottleneck_dim",
    "adapter_layer_indices",
    "adapter_residual_scale",
    "adapter_activation",
    "adapter_task_initialization",
    "adapter_initialization_rng",
    "adapter_learning_rate",
    "clip_checkpoint_sha256",
    "data_root",
)


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing diagnostic input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run(label: str, root: Path) -> Dict[str, Any]:
    config = _load_json(root / "config.json")
    summary = _load_json(root / "seed_summary.json")
    rows = _load_json(root / "task_metrics.json")
    if summary.get("status") != "complete" or summary.get("seed") != 0:
        raise ValueError(f"Diagnostic {label} is not a complete seed0 run")
    if config.get("seed") != 0 or config.get("reporting_split") != "val":
        raise ValueError(f"Diagnostic {label} must be seed0 and val-only")
    if config.get("max_tasks") != 8 or len(rows) != 8:
        raise ValueError(f"Diagnostic {label} must contain all 8 tasks")
    if summary.get("completed_epochs") != config.get("epochs_per_task") * 8:
        raise ValueError(f"Diagnostic {label} has incomplete epochs")
    if config.get("git", {}).get("dirty") is not False:
        raise ValueError(f"Diagnostic {label} was not recorded from clean Git")
    if [row.get("task_id") for row in rows] != list(range(8)):
        raise ValueError(f"Diagnostic {label} has invalid task ordering")
    return {
        "label": label,
        "root": str(root),
        "config": config,
        "summary": summary,
        "rows": rows,
    }


def summarize_diagnostics(
    labeled_roots: Sequence[Tuple[str, Path]],
    varying_fields: Sequence[str],
    focus_task_id: int = 6,
) -> Dict[str, Any]:
    if len(labeled_roots) < 2:
        raise ValueError("At least two diagnostics are required")
    labels = [label for label, _ in labeled_roots]
    if len(labels) != len(set(labels)):
        raise ValueError("Diagnostic labels must be unique")
    runs = [_load_run(label, root.resolve()) for label, root in labeled_roots]
    reference = runs[0]["config"]
    allowed = set(varying_fields)
    unknown_varying_fields = allowed.difference(REQUIRED_COMMON_FIELDS)
    if unknown_varying_fields:
        raise ValueError(
            "Unknown varying diagnostic fields: "
            f"{sorted(unknown_varying_fields)}"
        )
    for run in runs[1:]:
        mismatches = [
            field for field in REQUIRED_COMMON_FIELDS
            if field not in allowed and reference.get(field) != run["config"].get(field)
        ]
        if reference["git"].get("commit") != run["config"]["git"].get("commit"):
            mismatches.append("git.commit")
        if reference["git"].get("tree") != run["config"]["git"].get("tree"):
            mismatches.append("git.tree")
        if mismatches:
            raise ValueError(
                f"Diagnostic {run['label']} is not comparable: {mismatches}"
            )

    task_sizes = [int(value) for value in reference["task_sizes"]]
    class_order = list(reference["class_order"])
    if not 0 <= focus_task_id < len(task_sizes):
        raise ValueError("Focus task is outside the protocol")
    focus_start = sum(task_sizes[:focus_task_id])
    focus_end = focus_start + task_sizes[focus_task_id]
    focus_classes = class_order[focus_start:focus_end]

    result_runs = []
    for run in runs:
        rows = run["rows"]
        introduction_ap = rows[focus_task_id]["per_class_ap"][
            focus_start:focus_end
        ]
        final_ap = rows[-1]["per_class_ap"][focus_start:focus_end]
        result_runs.append({
            "label": run["label"],
            "root": run["root"],
            "varying_config": {
                field: run["config"].get(field) for field in varying_fields
            },
            "task_curve": [
                {
                    "task_id": int(row["task_id"]),
                    "mAP": float(row["mAP"]),
                    "cF1": float(row["cF1"]),
                    "oF1": float(row["oF1"]),
                }
                for row in rows
            ],
            "summary_metrics": {
                key: float(run["summary"]["metrics"][key])
                for key in (
                    "final_mAP", "average_mAP", "final_cF1",
                    "final_oF1", "forgetting",
                )
            },
            "focus_task": {
                "task_id": focus_task_id,
                "classes": {
                    name: {
                        "at_introduction_ap": float(introduction),
                        "final_ap": float(final),
                    }
                    for name, introduction, final in zip(
                        focus_classes, introduction_ap, final_ap
                    )
                },
                "at_introduction_mean_ap": statistics.mean(introduction_ap),
                "final_mean_ap": statistics.mean(final_ap),
            },
        })
    return {
        "schema_version": 1,
        "status": "complete",
        "comparison": "seed0_full_8task_validation_diagnostics",
        "git_commit": reference["git"]["commit"],
        "varying_fields": list(varying_fields),
        "selection_policy": (
            "Review the full 8-task curve, focus-task new-class AP, and final mAP; "
            "this report intentionally does not auto-select a winner."
        ),
        "runs": result_runs,
    }


def _labeled_root(value: str) -> Tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("Run must have LABEL=PATH form")
    return label, Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=_labeled_root, action="append", required=True)
    parser.add_argument("--vary-field", action="append", default=[])
    parser.add_argument("--focus-task-id", type=int, default=6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize_diagnostics(
        args.run, args.vary_field, args.focus_task_id
    )
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
