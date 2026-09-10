"""Summarize the locked seed0 Full-anchored residual validation stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


METHODS = ("A0", "A1", "A2")
EXPECTED_MODES = {
    "A0": "disabled",
    "A1": "residual_full_person",
    "A2": "residual_three_view",
}


def _read(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(runs: Dict[str, Path]) -> Dict[str, object]:
    records = {}
    commits = set()
    for method in METHODS:
        run = runs[method]
        summary = _read(run / "seed_summary.json")
        config = _read(run / "config.json")
        if summary.get("status") != "complete" or int(summary.get("seed", -1)) != 0:
            raise ValueError(f"{method} is not a complete seed0 run")
        if config.get("reporting_split") != "val":
            raise ValueError(f"{method} is not validation-only")
        if config.get("view_fusion") != EXPECTED_MODES[method]:
            raise ValueError(f"{method} view-fusion mode differs")
        if float(config.get("view_auxiliary_loss_weight", -1)) != 0.0:
            raise ValueError(f"{method} must not backpropagate auxiliary-view losses")
        if config.get("adapter_mode") != "image_token":
            raise ValueError(f"{method} does not use the champion Adapter")
        if method != "A0" and float(config.get("view_residual_scale", -1)) != 0.1:
            raise ValueError(f"{method} residual scale differs")
        commits.add(config["git"]["commit"])
        records[method] = {
            "run": str(run.resolve()),
            "metrics": summary["metrics"],
            "final_task": summary["task_metrics"][-1],
        }
    if len(commits) != 1:
        raise ValueError("Residual runs were not produced by one commit")
    final = {
        name: float(records[name]["metrics"]["final_mAP"])
        for name in METHODS
    }
    advance = final["A2"] > final["A0"] and final["A2"] > final["A1"]
    ranking = sorted(
        METHODS,
        key=lambda name: (
            final[name], float(records[name]["metrics"]["average_mAP"]),
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "stage": "seed0_complete_8_task_validation_only",
        "selection_uses_test": False,
        "git_commit": commits.pop(),
        "methods": records,
        "paired_final_mAP": {
            "A1_minus_A0": final["A1"] - final["A0"],
            "A2_minus_A0": final["A2"] - final["A0"],
            "A2_minus_A1": final["A2"] - final["A1"],
        },
        "ranking": ranking,
        "advance_rule": "A2 final validation mAP must exceed both A0 and A1",
        "advance_A2_to_seed1_seed2_validation": advance,
        "test_permitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for method in METHODS:
        parser.add_argument(f"--{method.lower()}-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize({
        method: getattr(args, f"{method.lower()}_run") for method in METHODS
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "ranking": result["ranking"],
        "paired_final_mAP": result["paired_final_mAP"],
        "advance": result["advance_A2_to_seed1_seed2_validation"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
