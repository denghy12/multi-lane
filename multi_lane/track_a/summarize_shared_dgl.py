"""Summarize the locked seed0 shared multi-view DGL validation stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


METHODS = ("G0", "G1", "G2")
EXPECTED_ROUTING = {
    "G0": "joint",
    "G1": "fusion_detach",
    "G2": "dgl",
}
EXPECTED_AUXILIARY = {"G0": 0.1, "G1": 0.1, "G2": 0.0}
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
        summary = _read(run / "seed_summary.json")
        config = _read(run / "config.json")
        history = _read(run / "training_history.json")
        diagnostics = _read(run / "view_diagnostics.json")
        if summary.get("status") != "complete" or int(summary.get("seed", -1)) != 0:
            raise ValueError(f"{method} is not a complete seed0 run")
        if config.get("reporting_split") != "val" or config.get("save_checkpoints") is not False:
            raise ValueError(f"{method} is not locked validation-only")
        if config.get("view_fusion") != "fixed_three_view":
            raise ValueError(f"{method} is not fixed three-view fusion")
        if config.get("view_gradient_routing") != EXPECTED_ROUTING[method]:
            raise ValueError(f"{method} gradient routing differs")
        if float(config.get("view_auxiliary_loss_weight", -1)) != EXPECTED_AUXILIARY[method]:
            raise ValueError(f"{method} auxiliary loss weight differs")
        if config.get("view_gradient_audit") is not True or config.get("view_evaluation_diagnostics") is not True:
            raise ValueError(f"{method} diagnostics are incomplete")
        if method == "G2" and float(config.get("view_dgl_unimodal_weight", -1)) != 1.0:
            raise ValueError("G2 DGL alpha differs")
        if set(history) != {str(index) for index in range(8)} or set(diagnostics) != set(history):
            raise ValueError(f"{method} does not contain all eight tasks")
        audits = []
        for task_id in range(8):
            first_epoch = history[str(task_id)][0]
            keys = [key for key in first_epoch if key.startswith("gradient_")]
            if not keys:
                raise ValueError(f"{method} task{task_id} gradient audit is missing")
            audits.append({key: first_epoch[key] for key in keys})
        final_diagnostic = diagnostics["7"]
        records[method] = {
            "run": str(run.resolve()),
            "metrics": summary["metrics"],
            "final_task": summary["task_metrics"][-1],
            "final_view_metrics": final_diagnostic["metrics"],
            "final_pairwise_ranking_vs_full": final_diagnostic[
                "pairwise_ranking_vs_full"
            ],
            "gradient_audit_by_task": audits,
        }
        commits.add(config["git"]["commit"])
    if len(commits) != 1:
        raise ValueError("Shared-DGL runs were not produced by one commit")

    g0_final = float(records["G0"]["metrics"]["final_mAP"])
    g2_final = float(records["G2"]["metrics"]["final_mAP"])
    g0_average = float(records["G0"]["metrics"]["average_mAP"])
    g2_average = float(records["G2"]["metrics"]["average_mAP"])
    g0_full = float(records["G0"]["final_view_metrics"]["full"]["mAP"])
    g2_full = float(records["G2"]["final_view_metrics"]["full"]["mAP"])
    checks = {
        "final_mAP_gain_at_least_0.10": g2_final >= g0_final + 0.10,
        "average_mAP_not_lower": g2_average >= g0_average,
        "Full_standalone_not_lower": g2_full >= g0_full,
        "gap_to_independent_R1_shrinks": (
            abs(INDEPENDENT_R1_REFERENCE - g2_final)
            < abs(INDEPENDENT_R1_REFERENCE - g0_final)
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
        "stage": "seed0_complete_8_task_shared_multiview_DGL_validation_only",
        "selection_uses_test": False,
        "git_commit": commits.pop(),
        "independent_R1_reference_final_mAP": INDEPENDENT_R1_REFERENCE,
        "methods": records,
        "paired_differences": {
            "G1_minus_G0_final_mAP": (
                float(records["G1"]["metrics"]["final_mAP"]) - g0_final
            ),
            "G2_minus_G0_final_mAP": g2_final - g0_final,
            "G2_minus_G1_final_mAP": (
                g2_final - float(records["G1"]["metrics"]["final_mAP"])
            ),
            "G2_minus_G0_average_mAP": g2_average - g0_average,
            "G2_minus_G0_Full_standalone_mAP": g2_full - g0_full,
        },
        "advance_checks": checks,
        "advance_G2_to_dynamic_router_stage": all(checks.values()),
        "ranking": ranking,
        "test_permitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0-run", type=Path, required=True)
    parser.add_argument("--g1-run", type=Path, required=True)
    parser.add_argument("--g2-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize({"G0": args.g0_run, "G1": args.g1_run, "G2": args.g2_run})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "ranking": result["ranking"],
        "paired_differences": result["paired_differences"],
        "advance_checks": result["advance_checks"],
        "advance": result["advance_G2_to_dynamic_router_stage"],
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
