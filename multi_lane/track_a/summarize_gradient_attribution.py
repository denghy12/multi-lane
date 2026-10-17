"""Summarize existing gradient JSON; never load data, checkpoints or Torch."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

PAIRS = (
    ("view_full", "view_person"),
    ("view_full_reliable", "view_face_reliable"),
    ("view_person_reliable", "view_face_reliable"),
    ("view_full_reliable", "view_person_reliable"),
    ("view_full_unreliable", "view_person_unreliable"),
    ("fused_path_full", "fused_path_person"),
    ("fused_path_full", "fused_path_face"),
    ("fused_path_person", "fused_path_face"),
    ("fused", "auxiliary"), ("fused", "total"),
)


def summarize(root: Path):
    config = json.loads((root / "config.json").read_text())
    result = json.loads((root / "result.json").read_text())
    if result["status"] != "complete" or result["training_updates"] != 0:
        raise ValueError("A complete read-only audit is required")
    rows = []
    for path in sorted(root.glob("*_task*_batch*.json")):
        record = json.loads(path.read_text())
        meta = {k: record[k] for k in ("method", "task", "precision", "batch")}
        names = record["objective_names"]
        for group, geometry in record["geometry"].items():
            for left, right in PAIRS:
                if left not in names or right not in names:
                    continue
                i, j = names.index(left), names.index(right)
                rows.append(dict(meta, group=group, comparison=f"{left}__{right}",
                                 metric="cosine", value=geometry["cosines"][i][j]))
            for name, norm in zip(names, geometry["norms"]):
                rows.append(dict(meta, group=group, comparison=name, metric="norm", value=norm))
        for key in ("auxiliary_vs_fused", "supervised_on_vs_same_parameter_moe_off",
                    "total_on_vs_same_parameter_moe_off"):
            for group, metrics in record.get(key, {}).items():
                for metric, value in metrics.items():
                    rows.append(dict(meta, group=group, comparison=key, metric=metric, value=value))
    snapshots = len(list(root.glob("*_task*_batch*.json")))
    if snapshots != result["snapshots"]:
        raise ValueError("Snapshot count mismatch")
    grouped = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in ("method", "task", "precision", "group", "comparison", "metric"))
        grouped[key].append(row["value"])
    aggregates = []
    for key, values in sorted(grouped.items()):
        valid = [x for x in values if x is not None]
        row = dict(zip(("method", "task", "precision", "group", "comparison", "metric"), key))
        row.update(n=len(valid), missing=len(values)-len(valid),
                   mean=statistics.mean(valid) if valid else None,
                   median=statistics.median(valid) if valid else None,
                   minimum=min(valid) if valid else None, maximum=max(valid) if valid else None,
                   below_zero=sum(x < 0 for x in valid),
                   below_minus_point_one=sum(x < -.1 for x in valid))
        aggregates.append(row)
    stability = defaultdict(dict)
    for row in aggregates:
        if row["metric"] != "cosine" or "__" not in row["comparison"]:
            continue
        key = tuple(row[k] for k in ("method", "group", "comparison", "precision"))
        stability[key][row["task"]] = row["below_minus_point_one"] >= 3 and row["n"] == 4
    signals = []
    for (method, group, comparison, precision), tasks in sorted(stability.items()):
        signals.append(dict(method=method, group=group, comparison=comparison, precision=precision,
                            stable_tasks=[t for t, stable in tasks.items() if stable],
                            at_least_two_tasks=sum(tasks.values()) >= 2))
    output = dict(source_diagnostic_git=config["diagnostic_git"], snapshots=snapshots,
                  aggregates=aggregates, descriptive_stability=signals,
                  notes=["Null/zero gradients excluded, never counted as no conflict",
                         "Precision configurations are sensitivity checks, not independent samples",
                         "Endpoint raw gradients do not reconstruct Adam or establish generalization causality"])
    (root / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    for name, data in (("metrics.csv", rows), ("aggregates.csv", aggregates)):
        with (root / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(f"Summarized {snapshots} snapshots into {root}/summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
