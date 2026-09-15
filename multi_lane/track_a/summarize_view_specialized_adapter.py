"""Summarize the locked shared-plus-view-specific Adapter validation stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


METHODS = ("P0", "P1", "P2")
EXPECTED_DIMS = {
    "P0": (32, 0, 49_952),
    "P1": (45, 0, 69_933),
    "P2": (32, 4, 70_700),
}
AUDIT_EPOCHS = (0, 14, 29)
INDEPENDENT_R1_REFERENCE = 43.5811932162358


def _read(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(runs: Dict[str, Path]) -> Dict[str, object]:
    records: Dict[str, object] = {}
    commits = set()
    for method in METHODS:
        run = runs[method]
        config = _read(run / "config.json")
        summary = _read(run / "seed_summary.json")
        history = _read(run / "training_history.json")
        diagnostics = _read(run / "view_diagnostics.json")
        shared_dim, view_dim, parameter_count = EXPECTED_DIMS[method]
        if summary.get("status") != "complete" or int(summary.get("seed", -1)) != 0:
            raise ValueError(f"{method} is not a complete seed0 run")
        if config.get("reporting_split") != "val" or config.get("save_checkpoints") is not False:
            raise ValueError(f"{method} is not locked validation-only")
        if config.get("view_fusion") != "fixed_three_view":
            raise ValueError(f"{method} does not use fixed three-view fusion")
        if config.get("view_gradient_routing") != "joint":
            raise ValueError(f"{method} does not use ordinary joint gradients")
        if float(config.get("view_auxiliary_loss_weight", -1)) != 0.1:
            raise ValueError(f"{method} auxiliary supervision differs")
        if int(config.get("adapter_bottleneck_dim", -1)) != shared_dim:
            raise ValueError(f"{method} shared Adapter bottleneck differs")
        if int(config.get("adapter_view_bottleneck_dim", -1)) != view_dim:
            raise ValueError(f"{method} view Adapter bottleneck differs")
        if int(config.get("adapter_parameters_per_task", -1)) != parameter_count:
            raise ValueError(f"{method} Adapter parameter count differs")
        if config.get("view_path_gradient_audit_epochs") != list(AUDIT_EPOCHS):
            raise ValueError(f"{method} path-audit epochs differ")
        if int(config.get("view_path_gradient_audit_batches_per_epoch", -1)) != 3:
            raise ValueError(f"{method} path-audit batch count differs")
        if set(history) != {str(task_id) for task_id in range(8)}:
            raise ValueError(f"{method} training history is incomplete")
        if set(diagnostics) != set(history):
            raise ValueError(f"{method} view diagnostics are incomplete")
        audit = {}
        for task_id in range(8):
            rows = history[str(task_id)]
            if len(rows) != 30:
                raise ValueError(f"{method} task{task_id} epoch count differs")
            audit[str(task_id)] = {}
            for epoch in AUDIT_EPOCHS:
                row = rows[epoch]
                if int(row.get("path_gradient_audit_samples", -1)) != 3:
                    raise ValueError(
                        f"{method} task{task_id} epoch{epoch} path audit is incomplete"
                    )
                values = {
                    key: value for key, value in row.items()
                    if key.startswith("path_gradient_")
                }
                if not values:
                    raise ValueError(
                        f"{method} task{task_id} epoch{epoch} path audit is missing"
                    )
                audit[str(task_id)][str(epoch)] = values
        final_diagnostic = diagnostics["7"]
        records[method] = {
            "run": str(run.resolve()),
            "metrics": summary["metrics"],
            "final_task": summary["task_metrics"][-1],
            "final_view_metrics": final_diagnostic["metrics"],
            "final_pairwise_ranking_vs_full": final_diagnostic[
                "pairwise_ranking_vs_full"
            ],
            "path_gradient_audit": audit,
            "adapter_parameters_per_task": parameter_count,
        }
        commits.add(config["git"]["commit"])
    if len(commits) != 1:
        raise ValueError("Adapter comparison runs were not produced by one commit")

    p0_final = float(records["P0"]["metrics"]["final_mAP"])
    p1_final = float(records["P1"]["metrics"]["final_mAP"])
    p2_final = float(records["P2"]["metrics"]["final_mAP"])
    p0_average = float(records["P0"]["metrics"]["average_mAP"])
    p2_average = float(records["P2"]["metrics"]["average_mAP"])
    p0_full = float(records["P0"]["final_view_metrics"]["full"]["mAP"])
    p2_full = float(records["P2"]["final_view_metrics"]["full"]["mAP"])
    checks = {
        "P2_final_at_least_P0_plus_0.10": p2_final >= p0_final + 0.10,
        "P2_final_at_least_P1_plus_0.10": p2_final >= p1_final + 0.10,
        "P2_average_not_lower_than_P0": p2_average >= p0_average,
        "P2_Full_standalone_not_lower_than_P0": p2_full >= p0_full,
        "gap_to_independent_R1_shrinks": (
            abs(INDEPENDENT_R1_REFERENCE - p2_final)
            < abs(INDEPENDENT_R1_REFERENCE - p0_final)
        ),
    }
    ranking = sorted(
        METHODS,
        key=lambda name: (
            float(records[name]["metrics"]["final_mAP"]),
            float(records[name]["metrics"]["average_mAP"]),
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "stage": "seed0_shared_plus_view_specific_adapter_validation_only",
        "selection_uses_test": False,
        "git_commit": commits.pop(),
        "independent_R1_reference_final_mAP": INDEPENDENT_R1_REFERENCE,
        "methods": records,
        "paired_differences": {
            "P1_minus_P0_final_mAP": p1_final - p0_final,
            "P2_minus_P0_final_mAP": p2_final - p0_final,
            "P2_minus_P1_final_mAP": p2_final - p1_final,
            "P2_minus_P0_average_mAP": p2_average - p0_average,
            "P2_minus_P0_Full_standalone_mAP": p2_full - p0_full,
        },
        "advance_checks": checks,
        "advance_P2_to_dynamic_router_stage": all(checks.values()),
        "ranking": ranking,
        "test_permitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0-run", type=Path, required=True)
    parser.add_argument("--p1-run", type=Path, required=True)
    parser.add_argument("--p2-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize({"P0": args.p0_run, "P1": args.p1_run, "P2": args.p2_run})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "ranking": result["ranking"],
        "paired_differences": result["paired_differences"],
        "advance_checks": result["advance_checks"],
        "advance": result["advance_P2_to_dynamic_router_stage"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
