"""Summarize the locked seed0 end-to-end view-fusion validation stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


METHODS = ("J0", "J1", "J3")
EXPECTED_MODES = {
    "J0": "fixed_three_view",
    "J1": "soft_three_view",
    "J3": "soft_full_person",
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
        if float(config.get("view_auxiliary_loss_weight", -1)) != 0.1:
            raise ValueError(f"{method} auxiliary loss weight differs")
        if config.get("adapter_mode") != "image_token":
            raise ValueError(f"{method} does not use the champion Adapter")
        commits.add(config["git"]["commit"])
        records[method] = {
            "run": str(run.resolve()),
            "metrics": summary["metrics"],
            "final_task": summary["task_metrics"][-1],
        }
    if len(commits) != 1:
        raise ValueError("End-to-end fusion runs were not produced by one commit")
    j0 = float(records["J0"]["metrics"]["final_mAP"])
    j1 = float(records["J1"]["metrics"]["final_mAP"])
    j3 = float(records["J3"]["metrics"]["final_mAP"])
    advance = j1 > j0 and j1 > j3
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
        "stage": "seed0_complete_8_task_validation_only",
        "selection_uses_test": False,
        "git_commit": commits.pop(),
        "methods": records,
        "paired_final_mAP": {
            "J1_minus_J0": j1 - j0,
            "J1_minus_J3": j1 - j3,
        },
        "ranking": ranking,
        "advance_rule": "J1 final validation mAP must exceed both J0 and J3",
        "advance_J1_to_seed1_seed2_validation": advance,
        "test_permitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--j0-run", type=Path, required=True)
    parser.add_argument("--j1-run", type=Path, required=True)
    parser.add_argument("--j3-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize({"J0": args.j0_run, "J1": args.j1_run, "J3": args.j3_run})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "ranking": result["ranking"],
        "paired_final_mAP": result["paired_final_mAP"],
        "advance": result["advance_J1_to_seed1_seed2_validation"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
