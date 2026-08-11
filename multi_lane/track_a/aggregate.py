"""Validate and aggregate completed MULTI-LANE Track-A seed runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = ("final_mAP", "final_cF1", "final_oF1", "average_mAP", "forgetting")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--project-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    commits = set()
    for seed in (0, 1, 2):
        path = args.run_root / f"seed{seed}" / "seed_summary.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed summary: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("seed") != seed:
            raise RuntimeError(f"Invalid seed summary: {path}")
        if payload.get("completed_epochs") != 240:
            raise RuntimeError(f"Seed {seed} did not complete 240 epochs")
        commits.add(payload["config"]["git"]["commit"])
        rows.append(payload)
    if len(commits) != 1:
        raise RuntimeError(f"Seed runs used different commits: {sorted(commits)}")
    aggregate = {}
    for metric in METRICS:
        values = [float(row["metrics"][metric]) for row in rows]
        aggregate[metric] = {
            "values": values,
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
        }
    result = {
        "schema_version": 1,
        "status": "complete",
        "method": "MULTI-LANE",
        "protocol_id": "emotic_b5c3_v0.1",
        "track": "A",
        "seeds": [0, 1, 2],
        "git_commit": next(iter(commits)),
        "aggregate": aggregate,
        "seed_summaries": rows,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    formal = args.run_root / "formal_seed_summary.json"
    formal.write_text(text, encoding="utf-8")
    args.project_summary.parent.mkdir(parents=True, exist_ok=True)
    args.project_summary.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
