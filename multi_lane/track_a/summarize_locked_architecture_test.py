"""Summarize J0/A2 locked tests beside reusable R1 and C2 test results."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Sequence


METRICS = ("final_mAP", "average_mAP", "final_cF1", "final_oF1", "forgetting")


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    values = [float(value) for value in values]
    return {"values": values, "mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def _audit_runs(paths: Sequence[Path], method: str) -> Dict[str, Any]:
    rows = []
    expected_mode = {"J0": "fixed_three_view", "A2": "residual_three_view"}[method]
    for path in paths:
        config = _read(path / "config.json")
        summary = _read(path / "seed_summary.json")
        if (
            summary.get("status") != "complete"
            or summary.get("config") != config
            or config.get("view_fusion") != expected_mode
            or config.get("reporting_split") != "test"
            or config.get("evaluation_score_purpose") != "fixed_test_fusion"
            or config.get("save_evaluation_scores") is not True
            or config.get("save_checkpoints") is not False
            or config.get("git", {}).get("dirty") is not False
            or len(summary.get("task_metrics", ())) != 8
            or summary.get("completed_epochs") != 240
        ):
            raise ValueError(f"Incomplete or unlocked {method} source: {path}")
        rows.append({"seed": int(config["seed"]), "metrics": summary["metrics"], "source": str(path.resolve())})
    rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in rows] != [0, 1, 2]:
        raise ValueError(f"{method} requires seed0/1/2")
    return {
        "seeds": rows,
        "aggregate": {
            metric: _stats([row["metrics"][metric] for row in rows]) for metric in METRICS
        },
    }


def summarize(
    j0_runs: Sequence[Path], a2_runs: Sequence[Path], r1_path: Path, c2_path: Path
) -> Dict[str, Any]:
    r1 = _read(r1_path)
    c2 = _read(c2_path)
    return {
        "schema_version": 1,
        "evaluation_split": "test",
        "test_selection_or_search": False,
        "scope": "exploratory horizontal architecture comparison",
        "methods": {
            "R1_independent_experts_fixed_fusion": {
                "aggregate": r1["groups"]["R1_fixed_three_view"],
                "source": str(r1_path.resolve()),
            },
            "C2_nonshared_dynamic_class_aware": {
                "aggregate": c2["aggregate"]["C2_class_aware"],
                "source": str(c2_path.resolve()),
            },
            "J0_shared_fixed_feature_fusion": _audit_runs(j0_runs, "J0"),
            "A2_full_anchored_three_view_residual": _audit_runs(a2_runs, "A2"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--j0-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--a2-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--r1-summary", type=Path, required=True)
    parser.add_argument("--c2-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.j0_runs, args.a2_runs, args.r1_summary, args.c2_summary)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        name: data["aggregate"]["final_mAP"]
        for name, data in result["methods"].items()
    }, indent=2), flush=True)
    print("LOCKED_ARCHITECTURE_TEST_SUMMARY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
