"""Compare retrained two-view models against immutable matched-seed references."""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from .compare_joint26 import require
from .evaluate_joint26_checkpoint import audit_source

TITLES = {"FULL_ONLY": "只输入 Full", "FULL_PERSON": "输入 Full+Person",
          "FULL_FACE": "输入 Full+Face", "THREE_VIEW": "输入 Full+Person+Face"}
MODES = {"FULL_ONLY": "disabled", "FULL_PERSON": "fixed_full_person",
         "FULL_FACE": "fixed_full_face", "THREE_VIEW": "fixed_three_view"}
CONFIG_KEYS = ("seed", "class_order", "task_sizes", "train_batch_size", "eval_batch_size", "workers",
               "learning_rate", "weight_decay", "scheduler_mode", "epochs_per_task", "input_normalization",
               "train_crop_scale", "full_crop_mode", "person_crop_margin", "person_transform_mode",
               "person_color_jitter_strength", "person_color_jitter_probability", "selector_mode", "prompt_mode",
               "num_selectors", "num_prompts", "num_prompt_layers", "normalize", "amp", "tf32",
               "amp_initial_scale", "amp_growth_interval", "temperature", "threshold", "training_loss_mode",
               "parameter_group_loss_routing", "adapter_mode", "parax_mode", "view_classifier_mode")
AUDIT_KEYS = ("train_instances", "validation_instances", "train_sample_ids_sha256", "validation_sample_ids_sha256",
              "first_train_batch_ids_sha256", "initial_selector_prompt_classifier_sha256", "frozen_visual_sha256_before")


def compare_seed(root, reference, reference_test, smoke_updates=0, include_test=False):
    roots = {m: (reference if m in ("FULL_ONLY", "THREE_VIEW") else root) / m for m in TITLES}
    base_config = json.loads((roots["THREE_VIEW"] / "config.json").read_text())
    base_audit = json.loads((roots["THREE_VIEW"] / "joint_protocol_audit.json").read_text())
    results, score_reference, test_reference = {}, None, None
    for method, path in roots.items():
        config = json.loads((path / "config.json").read_text())
        summary = json.loads((path / "seed_summary.json").read_text())
        audit = json.loads((path / "joint_protocol_audit.json").read_text())
        history = json.loads((path / "training_history.json").read_text())["0"]
        if not smoke_updates or method in ("FULL_ONLY", "THREE_VIEW"):
            audit_source(path)
        require(config["view_fusion"] == MODES[method] and config["parax_mode"] == config["adapter_mode"] == "disabled",
                f"Incorrect view/adapter configuration: {method}")
        require(summary["status"] == "complete" and not audit["test_loaded"] and audit["frozen_visual_unchanged"], "Invalid training audit")
        require(sum(r["skipped_optimizer_steps"] for r in history) == 0, "Training had skipped updates")
        require(summary["completed_optimizer_updates"] == (smoke_updates if smoke_updates and method in ("FULL_PERSON", "FULL_FACE") else 7530), "Wrong update budget")
        for key in CONFIG_KEYS:
            if smoke_updates and key == "epochs_per_task":
                continue
            require(config.get(key) == base_config.get(key), f"Configuration differs: {method}/{key}")
        for key in AUDIT_KEYS:
            require(audit[key] == base_audit[key], f"Paired audit differs: {method}/{key}")
        require(config["supervised_loss_scale"] == (1.1 if method == "FULL_ONLY" else 1.)
                and config["view_auxiliary_loss_weight"] == (0. if method == "FULL_ONLY" else .1), "Unmatched loss coefficients")
        require(all(np.isfinite(r["current_loss"]) and r["amp_loss_scale_end"] == 1024 for r in history), "Numerical stability audit failed")
        row = dict(source=str(path), source_git=config["git"], validation_mAP=summary["metrics"].get("final_mAP"))
        if not smoke_updates:
            with np.load(path / "val_scores/task0.npz") as z:
                values = (z["sample_ids"].copy(), z["targets"].copy())
                if score_reference is None:
                    score_reference = values
                else:
                    require(all(np.array_equal(a, b) for a, b in zip(values, score_reference)), "Validation sample mismatch")
        if include_test:
            test_path = (reference_test if method in ("FULL_ONLY", "THREE_VIEW") else root / "test") / method
            test = json.loads((test_path / "seed_summary.json").read_text())
            require(test["status"] == "complete" and test["source_unchanged"] and test["clip_unchanged"]
                    and test["method_weights_unchanged"] and test["optimizer_updates"] == 0
                    and test["seed"] == config["seed"] and test["source_config"] == config
                    and test["validation_replay_max_logit_difference"] <= .01, "Test source/evaluation audit failed")
            with np.load(test_path / "test_scores/task0.npz") as z:
                values = (z["sample_ids"].copy(), z["targets"].copy())
                if test_reference is None:
                    test_reference = values
                else:
                    require(all(np.array_equal(a, b) for a, b in zip(values, test_reference)), "Test sample mismatch")
            row["test_mAP"] = test["metrics"]["final_mAP"]
        results[method] = row
    report = dict(seed=base_config["seed"], paired_audit_passed=True, smoke_updates=smoke_updates,
                  reference_models_retrained=False, same_git_head=False, runs=results)
    root.mkdir(parents=True, exist_ok=True)
    name = "smoke_audit" if smoke_updates else "comparison"
    (root / (name + ".json")).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if not smoke_updates:
        lines = ["# 辅助视图独立贡献", "", "各组均无 ParaX、无 Image-token Adapter；最终第30轮，同seed同样本。Full和三路引用已有模型。", "",
                 "| 输入视图 | Validation mAP | Test mAP |", "| --- | ---: | ---: |"]
        for method, row in results.items():
            test_value = f"{row['test_mAP']:.4f}" if "test_mAP" in row else "待完成"
            lines.append(f"| {TITLES[method]} | {row['validation_mAP']:.4f} | {test_value} |")
        (root / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"VIEW_CONTRIBUTION_AUDIT_PASSED seed={base_config['seed']} smoke={smoke_updates}", flush=True)
    return report


def summarize(root):
    reports = [json.loads((root / f"seed{s}" / "comparison.json").read_text()) for s in (0, 1, 2)]
    result = {split: {m: dict(mean=statistics.mean(r["runs"][m][split + "_mAP"] for r in reports),
                             std=statistics.stdev(r["runs"][m][split + "_mAP"] for r in reports))
                      for m in TITLES} for split in ("validation", "test")}
    differences = {}
    for split in ("validation", "test"):
        differences[split] = {}
        for a, b in (("FULL_PERSON", "FULL_ONLY"), ("FULL_FACE", "FULL_ONLY"), ("THREE_VIEW", "FULL_PERSON"), ("THREE_VIEW", "FULL_FACE")):
            values = [r["runs"][a][split + "_mAP"] - r["runs"][b][split + "_mAP"] for r in reports]
            differences[split][f"{a}_minus_{b}"] = dict(per_seed=values, mean=statistics.mean(values), std=statistics.stdev(values))
    (root / "summary.json").write_text(json.dumps(dict(metrics=result, paired_differences=differences), indent=2, ensure_ascii=False) + "\n")
    lines = ["# 两种辅助视图独立贡献：三seed汇总", "", "| 输入视图 | Validation mAP 均值±标准差 | Test mAP 均值±标准差 |", "| --- | ---: | ---: |"]
    for m in TITLES:
        lines.append(f"| {TITLES[m]} | {result['validation'][m]['mean']:.4f} ± {result['validation'][m]['std']:.4f} | {result['test'][m]['mean']:.4f} ± {result['test'][m]['std']:.4f} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-test", type=Path)
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.root)
    else:
        require(args.reference is not None, "Reference required")
        compare_seed(args.root, args.reference, args.reference_test, args.smoke_updates, args.include_test)
