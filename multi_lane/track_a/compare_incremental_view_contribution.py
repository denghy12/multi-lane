"""Audit matched fixed-view incremental runs and fixed-cohort old-class drift."""
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from .compare_joint26 import require
from .compare_joint26_view_contribution import CONFIG_KEYS, TITLES, MODES
from .runner import average_precision


def fixed_cohort_drift(first, later):
    """Track the same old examples/classes as the seen evaluation set expands."""
    positions = {str(sample): i for i, sample in enumerate(later["sample_ids"])}
    require(all(str(sample) in positions for sample in first["sample_ids"]), "Old anchor samples disappeared")
    rows = [positions[str(sample)] for sample in first["sample_ids"]]
    count = first["targets"].shape[1]
    require(np.array_equal(first["class_indices"], later["class_indices"][:count]), "Old anchor class order changed")
    require(np.array_equal(first["targets"], later["targets"][rows, :count]), "Old anchor labels changed")
    logits = later["logits"][rows, :count]
    delta = np.abs(logits - first["logits"])
    ap = lambda probabilities: float(np.mean([100 * average_precision(probabilities[:, c], first["targets"][:, c]) for c in range(count)]))
    before, after = ap(first["probabilities"]), ap(later["probabilities"][rows, :count])
    return dict(samples=len(rows), classes=count, initial_mAP=before, later_mAP=after,
                mAP_change=after-before, mean_absolute_logit_drift=float(delta.mean()), max_absolute_logit_drift=float(delta.max()))


def compare(root, smoke_updates=0):
    results, reference, scores_reference = {}, None, {}
    for method in TITLES:
        path = root / method
        summary = json.loads((path / "seed_summary.json").read_text())
        config = json.loads((path / "config.json").read_text())
        audit = json.loads((path / "paired_protocol_audit.json").read_text())
        history = json.loads((path / "training_history.json").read_text())
        require(summary["status"] == "complete" and summary["config"] == config, "Incomplete/config mismatch")
        require(config["training_protocol"] == "incremental" and config["task_sizes"] == [5,3,3,3,3,3,3,3]
                and config["task_pathway_count"] == 8 and config["fixed_view_paired_audit"], "Wrong incremental protocol")
        require(config["max_tasks"] == (3 if smoke_updates else 8), "Wrong task budget")
        require(config["reporting_split"] == ("val" if smoke_updates else "test")
                and not config["also_report_test"] and config["skip_validation_eval"], "Wrong reporting protocol")
        require(config["adapter_mode"] == config["parax_mode"] == "disabled" and config["view_fusion"] == MODES[method]
                and config["git"]["dirty"] is False, "Unexpected adapter/views/dirty source")
        require(config["supervised_loss_scale"] == (1.1 if method == "FULL_ONLY" else 1.)
                and config["view_auxiliary_loss_weight"] == (0. if method == "FULL_ONLY" else .1), "Loss mismatch")
        require(audit["frozen_visual_unchanged"] and audit["frozen_visual_sha256_before"] == audit["frozen_visual_sha256_after"], "CLIP changed")
        require(set(history) == set(audit["tasks"]) == {str(t) for t in range(config["max_tasks"])}, "Missing task")
        require(summary["completed_optimizer_updates"] == sum(r["optimizer_steps"] for h in history.values() for r in h), "Update total mismatch")
        if reference is None:
            reference = (config, audit)
        else:
            c0, a0 = reference
            for key in CONFIG_KEYS + ("git", "max_tasks", "fixed_view_sampler_protocol", "total_method_parameters"):
                require(config.get(key) == c0.get(key), f"Config mismatch: {method}/{key}")
            require(audit["initial_selector_prompt_classifier_sha256"] == a0["initial_selector_prompt_classifier_sha256"], "Initialization mismatch")
        anchors, drift = None, {}
        for task in sorted(history, key=int):
            h = history[task]
            a = audit["tasks"][task]
            expected = smoke_updates or 30 * math.ceil(a["train_instances"] / config["train_batch_size"])
            require(sum(r["optimizer_steps"] for r in h) == expected and sum(r["skipped_optimizer_steps"] for r in h) == 0, "Incorrect updates/skips")
            require(all(math.isfinite(r["current_loss"]) and r["amp_loss_scale_end"] == 1024 for r in h), "Nonfinite loss/changed scale")
            require(len(h) == 30 or bool(smoke_updates), "Incomplete epochs")
            require(a["sampler_seed"] == config["seed"] + 1009 * int(task), "Wrong sampler seed")
            a0 = reference[1]["tasks"][task]
            for key in ("train_instances", "train_sample_ids_sha256", "first_train_batch_ids_sha256", "validation_sample_ids_sha256",
                        "reporting_instances", "reporting_sample_ids_sha256", "frozen_visual_sha256_before"):
                require(a[key] == a0[key], f"Task mismatch: {method}/{task}/{key}")
            with np.load(path / f"{config['reporting_split']}_scores/task{task}.npz") as z:
                data = {key: z[key].copy() for key in z.files}
            require(data["targets"].shape == data["logits"].shape and np.isfinite(data["logits"]).all(), "Invalid scores")
            if task not in scores_reference:
                scores_reference[task] = (data["sample_ids"], data["targets"])
            else:
                require(np.array_equal(data["sample_ids"], scores_reference[task][0])
                        and np.array_equal(data["targets"], scores_reference[task][1]), "Test sample/target mismatch")
            if anchors is None:
                anchors = data
            drift[task] = fixed_cohort_drift(anchors, data)
        results[method] = dict(metrics=summary["metrics"], fixed_task0_cohort=drift, updates=summary["completed_optimizer_updates"],
                               elapsed_seconds=summary["elapsed_seconds"], peak_allocated_mib=audit["peak_allocated_mib"])
    report = dict(seed=reference[0]["seed"], paired_audit_passed=True, smoke_updates=smoke_updates, results=results)
    (root / ("smoke_audit.json" if smoke_updates else "comparison.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 增量学习中的辅助视图贡献", "", "所有方法共享冻结CLIP，关闭两种Adapter；固定融合，原任务后冻结其任务参数。", "",
             "| 输入视图 | Final mAP | Average mAP | Forgetting | 固定task0样本logit平均漂移 |", "| --- | ---: | ---: | ---: | ---: |"]
    for method, row in results.items():
        m = row["metrics"]
        lines.append(f"| {TITLES[method]} | {m['final_mAP']:.4f} | {m['average_mAP']:.4f} | {m['forgetting']:.4f} | {row['fixed_task0_cohort'][str(reference[0]['max_tasks']-1)]['mean_absolute_logit_drift']:.6g} |")
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"INCREMENTAL_VIEW_AUDIT_PASSED seed={report['seed']} smoke={smoke_updates}", flush=True)
    return report


def summarize(root):
    rows = [json.loads((root / f"seed{s}/comparison.json").read_text()) for s in (0,1,2)]
    require(all(r["seed"] == s and r["paired_audit_passed"] for s, r in enumerate(rows)), "Incomplete seeds")
    metrics = {m: {key: dict(mean=statistics.mean(r["results"][m]["metrics"][key] for r in rows),
                            std=statistics.stdev(r["results"][m]["metrics"][key] for r in rows))
                   for key in ("final_mAP", "average_mAP", "forgetting")} for m in TITLES}
    differences = {}
    for a,b in (("FULL_PERSON","FULL_ONLY"),("FULL_FACE","FULL_ONLY"),("THREE_VIEW","FULL_PERSON"),("THREE_VIEW","FULL_FACE"),("THREE_VIEW","FULL_ONLY")):
        differences[f"{a}_minus_{b}"] = {key: [r["results"][a]["metrics"][key]-r["results"][b]["metrics"][key] for r in rows] for key in ("final_mAP","average_mAP","forgetting")}
    (root / "summary.json").write_text(json.dumps(dict(metrics=metrics, paired_differences=differences), indent=2, ensure_ascii=False) + "\n")
    lines=["# 八任务增量三seed汇总", "", "| 输入视图 | Final mAP | Average mAP | Forgetting |", "| --- | ---: | ---: | ---: |"]
    for m in TITLES:
        lines.append(f"| {TITLES[m]} | " + " | ".join(f"{metrics[m][k]['mean']:.4f} ± {metrics[m][k]['std']:.4f}" for k in ("final_mAP","average_mAP","forgetting")) + " |")
    (root / "summary.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--smoke-updates",type=int,default=0)
    parser.add_argument("--summarize",action="store_true")
    args=parser.parse_args()
    summarize(args.root) if args.summarize else compare(args.root,args.smoke_updates)
