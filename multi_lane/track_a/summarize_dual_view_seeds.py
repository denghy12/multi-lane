"""Summarize paired seed0/1/2 locked dual-view tests without further selection."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .fuse_fixed_test_scores import LOCKED_VALIDATION_SUMMARY_SHA256


METRICS = ("final_mAP", "average_mAP", "final_cF1", "final_oF1", "forgetting")
LOCKED_RULE = {
    "mode": "probability", "full_weight": 0.8, "person_weight": 0.2, "threshold": 0.5,
}


def _stats(values):
    return {"values": values, "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values)}


def aggregate_seed_rows(rows):
    if sorted(seed for seed, _ in rows) != [0, 1, 2]:
        raise ValueError("Exactly one result for each seed0/1/2 is required")
    rows = sorted(rows, key=lambda item: item[0])
    for _, record in rows:
        if record.get("evaluation_split") != "test" or record.get("search_performed_on_test") is not False:
            raise ValueError("Only locked test results without search are allowed")
        if record.get("locked_rule") != LOCKED_RULE:
            raise ValueError("Fusion rule differs across seeds or from the locked rule")
        if record.get("validation_selection", {}).get("sha256") != LOCKED_VALIDATION_SUMMARY_SHA256:
            raise ValueError("Validation selection provenance changed")
    groups = {}
    for name in ("full", "person", "fusion"):
        metrics = [
            (record["fixed_fusion"] if name == "fusion" else record["anchors"][name])["metrics"]
            for _, record in rows
        ]
        groups[name] = {key: _stats([item[key] for item in metrics]) for key in METRICS}
    differences = {
        key: _stats([
            record["fixed_fusion"]["metrics"][key] - record["anchors"]["full"]["metrics"][key]
            for _, record in rows
        ])
        for key in METRICS
    }
    return {
        "schema_version": 1, "seeds": [0, 1, 2], "std_ddof": 1,
        "locked_rule": LOCKED_RULE, "search_performed_on_test": False,
        "groups": groups, "paired_fusion_minus_full": differences,
        "positive_final_mAP_seeds": sum(value > 0 for value in differences["final_mAP"]["values"]),
        "note": "Paired seed confirmation, not a new hyperparameter search or significance test.",
    }


def summarize_files(paths):
    rows = []
    reference_configs = {}
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        seeds = []
        for view in ("full", "person"):
            run = Path(record[f"{view}_run"])
            config = json.loads((run / "config.json").read_text(encoding="utf-8"))
            summary = json.loads((run / "seed_summary.json").read_text(encoding="utf-8"))
            history = json.loads((run / "training_history.json").read_text(encoding="utf-8"))
            if summary.get("status") != "complete" or summary.get("completed_epochs") != 240 or summary.get("completed_optimizer_updates") != 13950:
                raise ValueError(f"Incomplete fixed training budget: {run}")
            if len(history) != 8 or any(len(history.get(str(task), [])) != 30 for task in range(8)):
                raise ValueError(f"Missing task/epoch history: {run}")
            if sum(epoch["skipped_optimizer_steps"] for task in history.values() for epoch in task) != 0:
                raise ValueError(f"Skipped optimizer updates: {run}")
            if config.get("git", {}).get("dirty") is not False:
                raise ValueError(f"Dirty source worktree: {run}")
            comparable = {key: value for key, value in config.items() if key not in ("seed", "git")}
            if view in reference_configs and comparable != reference_configs[view]:
                changed = [key for key in set(comparable) | set(reference_configs[view])
                           if comparable.get(key) != reference_configs[view].get(key)]
                raise ValueError(f"Configuration drift for {view}: {changed}")
            reference_configs[view] = comparable
            seeds.append(config["seed"])
        if seeds[0] != seeds[1] or record.get("seed", seeds[0]) != seeds[0]:
            raise ValueError("Full/person result seeds do not match")
        rows.append((seeds[0], record))
    result = aggregate_seed_rows(rows)
    result["source_summaries"] = [str(path.resolve()) for path in paths]
    result["training_audit"] = "all six runs complete; fixed budgets; skipped=0; same per-view config except seed/git"
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--fusion-summaries", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_files(args.fusion_summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["groups"]["fusion"], indent=2), flush=True)
    print("DUAL_VIEW_SEED012_SUMMARY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
