"""Strict paired audit of original image-stream ParaX transferred to eight tasks."""
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from .compare_joint26 import require
from .compare_joint26_view_contribution import CONFIG_KEYS
from .compare_incremental_view_contribution import fixed_cohort_drift

METHODS = {"THREE_VIEW": "三视图，无 ParaX", "THREE_VIEW_PARAX": "三视图，冻结视觉主干第11个block后加入原始ParaX"}


def compare(root, smoke_updates=0):
    results, reference, score_reference, anchor_ids = {}, None, {}, None
    for method in METHODS:
        path = root / method
        read = lambda name: json.loads((path / name).read_text())
        config, summary = read("config.json"), read("seed_summary.json")
        audit, history = read("paired_protocol_audit.json"), read("training_history.json")
        drift = read("image_stream_transfer_audit.json")
        tasks = 3 if smoke_updates else 8
        require(summary["status"] == "complete" and summary["config"] == config, "Incomplete or config mismatch")
        require(config["training_protocol"] == "incremental" and config["task_sizes"] == [5,3,3,3,3,3,3,3]
                and config["max_tasks"] == tasks and config["image_stream_transfer_audit"], "Wrong task protocol")
        require(config["reporting_split"] == "val" and config["also_report_test"] == (not bool(smoke_updates))
                and config["skip_validation_eval"], "Wrong reporting protocol")
        require(config["adapter_mode"] == "disabled" and config["view_fusion"] == "fixed_three_view"
                and config["parax_mode"] == ("image" if method.endswith("PARAX") else "disabled")
                and not config["git"]["dirty"], "Wrong architecture or dirty runtime")
        require(config["supervised_loss_scale"] == 1 and config["view_auxiliary_loss_weight"] == .1, "Loss mismatch")
        if method.endswith("PARAX"):
            for key, value in dict(parax_layer_indices=[10], parax_rank=32, parax_num_experts=3,
                                   parax_initialization="official", parax_residual_scale=.1,
                                   parax_output_scale_mode="learnable", parax_trainable_components="all",
                                   parax_freeze_center_after_task0=False).items():
                require(config[key] == value, f"Original joint26 ParaX changed: {key}")
        require(audit["frozen_visual_unchanged"] and audit["frozen_visual_sha256_before"] == audit["frozen_visual_sha256_after"], "CLIP changed")
        require(set(history) == set(audit["tasks"]) == set(drift["tasks"]) == {str(t) for t in range(tasks)}, "Missing tasks")
        require(summary["completed_optimizer_updates"] == sum(r["optimizer_steps"] for h in history.values() for r in h), "Update sum mismatch")
        if reference is None:
            reference = config, audit
            anchor_ids = drift["sample_ids"]
        else:
            for key in tuple(k for k in CONFIG_KEYS if k != "parax_mode") + ("git", "max_tasks", "fixed_view_sampler_protocol"):
                require(config.get(key) == reference[0].get(key), f"Paired config differs: {key}")
            require(audit["initial_selector_prompt_classifier_sha256"] == reference[1]["initial_selector_prompt_classifier_sha256"], "Base initialization mismatch")
            require(anchor_ids == drift["sample_ids"], "FP32 anchor samples mismatch")
        cohorts = {split: {} for split in (["val"] if smoke_updates else ["val", "test"])}
        anchors = {}
        for task, h in history.items():
            a = audit["tasks"][task]
            expected = smoke_updates or 30 * math.ceil(a["train_instances"] / config["train_batch_size"])
            require(sum(r["optimizer_steps"] for r in h) == expected and sum(r["skipped_optimizer_steps"] for r in h) == 0, "Wrong updates or AMP skips")
            require(all(math.isfinite(r["current_loss"]) and r["amp_loss_scale_end"] == 1024 for r in h), "Loss/AMP instability")
            require(bool(smoke_updates) or len(h) == 30, "Incomplete epochs")
            if method.endswith("PARAX"):
                require(all(r["parax_grad_finite"] == 1 for r in h), "Nonfinite ParaX gradients")
            require(a["sampler_seed"] == config["seed"]+1009*int(task), "Sampler mismatch")
            require(a["test_loaded"] == (not bool(smoke_updates)), "Test loading policy mismatch")
            for key in ("train_instances", "train_sample_ids_sha256", "first_train_batch_ids_sha256", "validation_sample_ids_sha256",
                        "reporting_instances", "reporting_sample_ids_sha256", "frozen_visual_sha256_before"):
                require(a[key] == reference[1]["tasks"][task][key], f"Sample/CLIP pairing differs: {task}/{key}")
            d = drift["tasks"][task]
            require(d["old_lane_parameters_unchanged"] and d["old_lane_hash_before"] == d["old_lane_hash_after"], "Old task state changed")
            require(d["shared_parax_changed"] == method.endswith("PARAX"), "ParaX did not update as expected")
            for split in cohorts:
                with np.load(path / f"{split}_scores/task{task}.npz") as z:
                    data = {k: z[k].copy() for k in z.files}
                require(np.isfinite(data["logits"]).all(), "Nonfinite logits")
                key = split, task
                if key not in score_reference:
                    score_reference[key] = data
                else:
                    require(all(np.array_equal(data[k], score_reference[key][k]) for k in ("sample_ids", "targets", "class_indices")), "Evaluation cohort mismatch")
                with np.load(path / f"view_{split}_scores/task{task}.npz") as z:
                    require(len(z.files) > 0, "Missing per-view scores")
                anchors.setdefault(split, data)
                cohorts[split][task] = fixed_cohort_drift(anchors[split], data)
        results[method] = dict(validation=summary["metrics"], test=summary.get("test_metrics"),
                               fixed_task0_cohort=cohorts, fp32_anchor=drift,
                               validation_views=read("view_diagnostics.json"),
                               test_views=read("test_view_diagnostics.json") if not smoke_updates else None,
                               updates=summary["completed_optimizer_updates"], peak_allocated_mib=audit["peak_allocated_mib"])
    report = dict(seed=reference[0]["seed"], paired_audit_passed=True, smoke_updates=smoke_updates, results=results)
    (root / ("smoke_audit.json" if smoke_updates else "comparison.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(f"THREE_VIEW_TRANSFER_AUDIT_PASSED seed={report['seed']} smoke={smoke_updates}", flush=True)
    return report


def summarize(root):
    reports = [json.loads((root / f"seed{s}/comparison.json").read_text()) for s in (0,1,2)]
    require(all(r["paired_audit_passed"] and r["seed"] == s for s,r in enumerate(reports)), "Incomplete seeds")
    metrics, differences = {}, {}
    lines = ["# 原三视图全量学习配置迁移至八任务增量学习", "", "同一训练模型分别评估validation和test；test不用于选配置。", "",
             "| 方法 | 数据划分 | Final mAP | Average mAP | Forgetting |", "| --- | --- | ---: | ---: | ---: |"]
    for split in ("validation", "test"):
        metrics[split], differences[split] = {}, {}
        for method, title in METHODS.items():
            metrics[split][method] = {}
            for key in ("final_mAP", "average_mAP", "forgetting"):
                values = [r["results"][method][split][key] for r in reports]
                metrics[split][method][key] = dict(mean=statistics.mean(values), std=statistics.stdev(values), seeds=values)
            lines.append(f"| {title} | {split} | " + " | ".join(f"{metrics[split][method][k]['mean']:.4f} ± {metrics[split][method][k]['std']:.4f}" for k in ("final_mAP","average_mAP","forgetting")) + " |")
        for key in ("final_mAP", "average_mAP", "forgetting"):
            values = [r["results"]["THREE_VIEW_PARAX"][split][key]-r["results"]["THREE_VIEW"][split][key] for r in reports]
            differences[split][key] = dict(mean=statistics.mean(values), std=statistics.stdev(values), seeds=values)
    (root / "summary.json").write_text(json.dumps(dict(metrics=metrics, paired_differences=differences), indent=2, ensure_ascii=False)+"\n")
    (root / "summary.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    summarize(args.root) if args.summarize else compare(args.root, args.smoke_updates)
