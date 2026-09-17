"""Summarize the FP32 P0--P3 interaction and selective-Prompt arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_summary(batch: Path, method: str) -> dict[str, Any]:
    matches = sorted(batch.glob(f"runs/*_{method}_seed0_val/seed_summary.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one {method} seed summary under {batch}, found {len(matches)}")
    payload = json.loads(matches[0].read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"{method} is not complete: {payload.get('status')}")
    return payload


def metric(payload: dict[str, Any], name: str) -> float:
    return float(payload["metrics"][name])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payloads = {name: read_summary(args.batch, name) for name in ("F0", "F1", "F2", "F3", "S1")}
    metrics = ("final_mAP", "average_mAP", "final_cF1", "final_oF1", "forgetting")
    interaction = {name: metric(payloads["F3"], name) - metric(payloads["F1"], name)
                   - metric(payloads["F2"], name) + metric(payloads["F0"], name)
                   for name in metrics}
    selective_delta = {name: metric(payloads["S1"], name) - metric(payloads["F2"], name)
                       for name in metrics}
    task_values = {
        method: [float(row["mAP"]) for row in payloads[method]["task_metrics"]]
        for method in payloads
    }
    task_interaction = [task_values["F3"][i] - task_values["F1"][i]
                        - task_values["F2"][i] + task_values["F0"][i]
                        for i in range(len(task_values["F0"]))]
    result = {
        "batch": str(args.batch),
        "arms": {method: {name: metric(payloads[method], name) for name in metrics}
                 for method in payloads},
        "interaction_F3_minus_F1_minus_F2_plus_F0": interaction,
        "selective_S1_minus_F2": selective_delta,
        "task_mAP": task_values,
        "task_interaction": task_interaction,
    }
    output = args.output or args.batch / "interaction_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# FP32 interaction and selective Prompt summary", "", "| Arm | Final mAP | Avg. mAP | cF1 | oF1 | Forgetting |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method in ("F0", "F1", "F2", "F3", "S1"):
        values = result["arms"][method]
        lines.append(f"| {method} | {values['final_mAP']:.4f} | {values['average_mAP']:.4f} | {values['final_cF1']:.4f} | {values['final_oF1']:.4f} | {values['forgetting']:.4f} |")
    lines += ["", "Interaction `F3 - F1 - F2 + F0`:", ""]
    lines += [f"- final mAP: `{interaction['final_mAP']:+.4f}`", f"- average mAP: `{interaction['average_mAP']:+.4f}`", f"- task-wise mAP interaction: {[round(v, 4) for v in task_interaction]}", "", "Selective Prompt `S1 - F2`:", "", f"- final mAP: `{selective_delta['final_mAP']:+.4f}`", f"- average mAP: `{selective_delta['average_mAP']:+.4f}`"]
    report = output.with_name("interaction_summary.md")
    report.write_text("\n".join(lines) + "\n")
    print(report)


if __name__ == "__main__":
    main()
