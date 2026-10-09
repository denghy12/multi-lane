"""Audit four completed joint26 arms before comparing validation scores."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

METHODS = ("FULL_ONLY", "THREE_VIEW", "FULL_ONLY_PARAX", "THREE_VIEW_PARAX")
TITLES = {
    "FULL_ONLY": "仅 Full，不加入 ParaX",
    "THREE_VIEW": "Full、Person、Face，不加入 ParaX",
    "FULL_ONLY_PARAX": "仅 Full，冻结 ViT 第 11 个 block 后加入 ParaX",
    "THREE_VIEW_PARAX": "Full、Person、Face，冻结 ViT 第 11 个 block 后加入共享 ParaX",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(batch_root: Path, allow_skipped_updates: bool = False):
    runs, reference = {}, None
    shared_config = (
        "seed", "class_order", "task_sizes", "train_batch_size", "eval_batch_size",
        "learning_rate", "scheduler_mode", "epochs_per_task", "optimizer_updates_per_task",
        "training_loss_mode", "parameter_group_loss_routing", "input_normalization",
        "selector_mode", "prompt_mode", "adapter_mode", "git",
        "amp_initial_scale", "amp_growth_interval",
    )
    shared_audit = (
        "train_instances", "validation_instances", "train_sample_ids_sha256",
        "validation_sample_ids_sha256", "first_train_batch_ids_sha256",
        "initial_selector_prompt_classifier_sha256", "frozen_visual_sha256_before",
    )
    for method in METHODS:
        root = batch_root / method
        summary = json.loads((root / "seed_summary.json").read_text())
        config = summary["config"]
        audit = json.loads((root / "joint_protocol_audit.json").read_text())
        history = json.loads((root / "training_history.json").read_text())["0"]
        require(summary["status"] == "complete", f"{method}: incomplete")
        require(config["training_protocol"] == "joint26" and config["task_sizes"] == [26],
                f"{method}: not joint26")
        require(config["max_tasks"] == 1 and config["task_pathway_count"] == 1,
                f"{method}: incorrect pathway count")
        require(config["reporting_split"] == "val" and not config["also_report_test"]
                and not audit["test_loaded"] and summary["test_metrics"] is None,
                f"{method}: test was accessed")
        require(audit["label_dimensions"] == 26 and audit["frozen_visual_unchanged"]
                and audit["frozen_visual_sha256_before"] == audit["frozen_visual_sha256_after"],
                f"{method}: labels/backbone failed audit")
        is_three, is_parax = method.startswith("THREE"), method.endswith("PARAX")
        require(config["adapter_mode"] == "disabled", f"{method}: unexpected Image-token Adapter")
        require(config["view_fusion"] == ("fixed_three_view" if is_three else "disabled"),
                f"{method}: incorrect views")
        require(config["parax_mode"] == ("image" if is_parax else "disabled"),
                f"{method}: incorrect ParaX mode")
        require(config["view_auxiliary_loss_weight"] == (0.1 if is_three else 0.0)
                and config["supervised_loss_scale"] == (1.0 if is_three else 1.1),
                f"{method}: unmatched loss coefficients")
        skipped = sum(int(row["skipped_optimizer_steps"]) for row in history)
        require(history and (skipped == 0 or allow_skipped_updates), f"{method}: skipped steps")
        require(all(math.isfinite(row["current_loss"]) for row in history), f"{method}: nonfinite loss")
        if config["epochs_per_task"] is not None:
            expected = config["epochs_per_task"] * math.ceil(audit["train_instances"] / config["train_batch_size"])
            require(summary["completed_epochs"] == config["epochs_per_task"], f"{method}: incomplete epochs")
        else:
            expected = config["optimizer_updates_per_task"]
        attempted = int(summary["completed_optimizer_updates"]) + skipped
        if config["epochs_per_task"] is not None:
            require(attempted == expected, f"{method}: incomplete training batches")
        else:
            require(summary["completed_optimizer_updates"] == expected, f"{method}: incomplete updates")
        if is_parax:
            for row in history:
                nonfinite_batches = (1.0 - row["parax_grad_finite"]) * (row["optimizer_steps"] + row["skipped_optimizer_steps"])
                require(abs(nonfinite_batches) < 1e-6 or
                        (allow_skipped_updates and nonfinite_batches <= row["skipped_optimizer_steps"] + 1e-6),
                        f"{method}: nonfinite gradients beyond skipped AMP batches")
            for component in ("expert_a", "expert_b", "routers"):
                require(any(value > 0 for row in history for key, value in row.items()
                            if key.startswith(f"parax_grad_{component}") and key.endswith("_norm")),
                        f"{method}: no {component} gradient")
        with np.load(root / "val_scores" / "task0.npz", allow_pickle=False) as scores:
            ids, targets, logits = scores["sample_ids"], scores["targets"], scores["logits"]
            require(targets.shape == logits.shape == (audit["validation_instances"], 26),
                    f"{method}: incorrect score shape")
            require(np.isfinite(logits).all() and np.isfinite(targets).all(), f"{method}: nonfinite scores")
            if reference is None:
                reference = (config, audit, ids.copy(), targets.copy())
            else:
                c0, a0, ids0, targets0 = reference
                for field in shared_config:
                    require(c0.get(field) == config.get(field), f"{method}: config differs: {field}")
                for field in shared_audit:
                    require(a0[field] == audit[field], f"{method}: audit differs: {field}")
                require(summary["completed_optimizer_updates"] == runs[METHODS[0]]["optimizer_updates"],
                        f"{method}: successful update counts differ")
                require(np.array_equal(ids0, ids) and np.array_equal(targets0, targets),
                        f"{method}: evaluation samples/targets differ")
        runs[method] = {
            "description": TITLES[method], "mAP": summary["metrics"]["final_mAP"],
            "cF1": summary["metrics"]["final_cF1"], "oF1": summary["metrics"]["final_oF1"],
            "per_class_ap": summary["task_metrics"][0]["per_class_ap"],
            "optimizer_updates": summary["completed_optimizer_updates"],
            "skipped_optimizer_updates": skipped,
            "skipped_epochs": [int(row["epoch"]) + 1 for row in history if row["skipped_optimizer_steps"]],
            "elapsed_seconds": summary["elapsed_seconds"], "peak_allocated_mib": audit["peak_allocated_mib"],
        }
        if is_three:
            runs[method]["view_diagnostics"] = json.loads((root / "view_diagnostics.json").read_text())["0"]
    m = lambda name: runs[name]["mAP"]
    deltas = {
        "加入两路视图的收益（不使用 ParaX）": m("THREE_VIEW") - m("FULL_ONLY"),
        "ParaX 在 Full 单路中的收益": m("FULL_ONLY_PARAX") - m("FULL_ONLY"),
        "ParaX 在三路中的收益": m("THREE_VIEW_PARAX") - m("THREE_VIEW"),
        "加入两路视图的收益（使用 ParaX）": m("THREE_VIEW_PARAX") - m("FULL_ONLY_PARAX"),
        "ParaX 与三路联合的交互差值": (m("THREE_VIEW_PARAX") - m("THREE_VIEW")) - (m("FULL_ONLY_PARAX") - m("FULL_ONLY")),
    }
    has_skips = any(row["skipped_optimizer_updates"] for row in runs.values())
    result = {"paired_audit_passed": True, "strict_zero_skip_audit_passed": not has_skips,
              "seed": reference[0]["seed"], "split": "validation", "runs": runs, "mAP_differences": deltas,
              "warnings": (["AMP skipped updates were explicitly accepted for exploratory analysis. Equal successful update counts do not imply identical skipped batches or training trajectories."] if has_skips else []),
              "limitations": "Single-seed diagnostic only. One joint26 pathway differs from eight incremental pathways; 30 joint epochs is not 240 incremental epochs. No forgetting/average-task mAP applies."}
    (batch_root / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 全量学习：ParaX 与三路视图对照", "", "四组配对审计通过。以下为最终训练轮次的 validation 结果，未使用 test。", "",
             "| 修改方式 | mAP | cF1 | oF1 | 有效更新 / AMP 跳步 |", "| --- | ---: | ---: | ---: | ---: |"]
    lines += [f"| {row['description']} | {row['mAP']:.4f} | {row['cF1']:.4f} | {row['oF1']:.4f} | {row['optimizer_updates']} / {row['skipped_optimizer_updates']} |" for row in runs.values()]
    if has_skips:
        lines += ["", "注意：本批次未通过原先注册的零跳步要求。仅在显式启用允许跳步的分析选项后汇总；不同组溢出发生的轮次不同，结果用作探索性证据，不宣称严格零跳步复现。"]
    lines += ["", "| 比较 | mAP 差值（百分点） |", "| --- | ---: |"]
    lines += [f"| {name} | {value:+.4f} |" for name, value in deltas.items()]
    lines += ["", "这些实验区分全量学习下的 ParaX 作用、额外视图的整体作用及两者交互。不能凭单 seed 或直接比较全量/增量的绝对 mAP 断定退化完全由增量学习引起。", ""]
    (batch_root / "comparison.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--allow-skipped-updates", action="store_true", help="Explicitly report existing AMP-skipped runs as exploratory; strict default remains unchanged.")
    args = parser.parse_args()
    result = compare(args.batch_root, args.allow_skipped_updates)
    print(json.dumps(result["mAP_differences"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
