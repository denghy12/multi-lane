"""Summarize matched seeds without mixing the earlier dynamic-loss-scale run."""
import argparse
import json
from pathlib import Path

import numpy as np

from .compare_joint26 import METHODS, TITLES, require


def summarize(result_root: Path, batch_prefix: str):
    results = [json.loads((result_root / f"{batch_prefix}_seed{seed}" / "comparison.json").read_text()) for seed in (0, 1, 2)]
    for seed, result in enumerate(results):
        require(result["seed"] == seed and result["strict_zero_skip_audit_passed"], "Incomplete or non-strict seed")
    methods = {method: {"mean_mAP": float(np.mean([r["runs"][method]["mAP"] for r in results])),
                        "std_mAP": float(np.std([r["runs"][method]["mAP"] for r in results], ddof=1))}
               for method in METHODS}
    differences = {name: {"per_seed": [r["mAP_differences"][name] for r in results],
                          "mean": float(np.mean([r["mAP_differences"][name] for r in results])),
                          "std": float(np.std([r["mAP_differences"][name] for r in results], ddof=1))}
                   for name in results[0]["mAP_differences"]}
    gain = differences["ParaX 在三路中的收益"]["per_seed"]
    passed = float(np.mean(gain)) >= 0.3 and sum(value > 0 for value in gain) >= 2 and min(gain) >= -0.3
    report = {"methods": methods, "paired_differences": differences,
              "parax_development_gate_passed": passed,
              "rule": "Mean three-view ParaX gain >=0.3 mAP points, at least two positive seeds, no seed below -0.3; exploratory development rule, not a significance test."}
    output = result_root / f"{batch_prefix}_summary"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 全量学习三 seed 复核", "", "| 模型修改 | mAP 均值 | seed 间标准差 |", "| --- | ---: | ---: |"]
    lines += [f"| {TITLES[method]} | {row['mean_mAP']:.4f} | {row['std_mAP']:.4f} |" for method, row in methods.items()]
    lines += ["", "| 配对比较 | seed0 | seed1 | seed2 | 均值 | 标准差 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    lines += [f"| {name} | " + " | ".join(f"{v:+.4f}" for v in row["per_seed"]) + f" | {row['mean']:+.4f} | {row['std']:.4f} |" for name, row in differences.items()]
    lines += ["", f"预注册开发门槛通过：{passed}。门槛不是统计显著性检验；全量与增量路径数和训练预算仍不同。", ""]
    (output / "summary.md").write_text("\n".join(lines))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--batch-prefix", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.result_root, args.batch_prefix), ensure_ascii=False, indent=2))
