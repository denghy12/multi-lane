"""Read-only full-cohort validation/test of fixed final static MoE checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .compare_joint26 import require
from .evaluate_joint26_checkpoint import make_source
from .evaluate_whole_parax_eight_task import seen_payload, metrics_at_task, filter_seen_people, summarize_curve
from .export_compact_test_scores import _sha256, _torch_load, restore_compact_model_state
from .model import MultiLaneModel
from .openai_clip_loader import load_openai_clip_visual
from .runner import CLASS_ORDER, TASK_SIZES, LabelView, compact_model_state_dict, git_metadata, set_seed, tensor_state_digest, validate_classes
from .snapshot_validation import infer, replay
from .static_moe import ROUTES

METHODS = ("BASELINE", "SHARED", "MOE_UNIFORM", "MOE_OVERLAP")


def read(path):
    return json.loads(path.read_text())


def build_static_model(config, visual):
    return MultiLaneModel(
        visual, task_sizes=config["task_sizes"], num_selectors=config["num_selectors"],
        num_prompts=config["num_prompts"], num_prompt_layers=config["num_prompt_layers"],
        selector_mode=config["selector_mode"], prompt_mode=config["prompt_mode"],
        normalize=config["normalize"], adapter_mode=config["adapter_mode"],
        adapter_bottleneck_dim=config["adapter_bottleneck_dim"],
        adapter_layer_indices=config["adapter_layer_indices"],
        adapter_residual_scale=config["adapter_residual_scale"],
        adapter_residual_bound_ratio=config.get("adapter_residual_bound_ratio", 0.0),
        adapter_activation=config["adapter_activation"],
        adapter_task_initialization=config["adapter_task_initialization"],
        static_moe_mode=config["static_moe_mode"], view_fusion=config["view_fusion"],
        view_classifier_mode=config["view_classifier_mode"], parax_mode="disabled")


def audit_source(source, smoke_updates=0):
    c = read(source / "config.json")
    s = read(source / "seed_summary.json")
    pa = read(source / "paired_protocol_audit.json")
    audit = read(source / "static_moe_audit.json")
    history = read(source / "training_history.json")
    require(s["status"] == "complete" and s["config"] == c, "Source incomplete/config mismatch")
    require(c["static_moe_paired_audit"] and c["training_protocol"] == "incremental"
            and c["task_sizes"] == list(TASK_SIZES) and c["class_order"] == list(CLASS_ORDER)
            and c["max_tasks"] == 8
            and c["epochs_per_task"] == (None if smoke_updates else 30)
            and c["training_budget_mode"] == ("optimizer_updates" if smoke_updates else "epochs"),
            "Wrong source protocol/budget schema")
    require(c["parax_mode"] == "disabled" and c["view_fusion"] == "fixed_three_view"
            and c["selector_mode"] == c["prompt_mode"] == "shared"
            and c["adapter_mode"] in ("disabled", "image_token"), "Wrong source architecture")
    require(c["reporting_split"] == "val" and not c["also_report_test"] and not c["git"]["dirty"]
            and c["save_compact_checkpoints"], "Invalid source provenance")
    require(bool(c["static_moe_smoke_samples"]) == bool(smoke_updates)
            and c["optimizer_updates_per_task"] == (smoke_updates or None), "Smoke/formal source mismatch")
    require(pa["frozen_visual_unchanged"] and pa["frozen_visual_sha256_before"] == pa["frozen_visual_sha256_after"],
            "Frozen CLIP changed")
    require(audit["status"] == "passed" and len(audit["tasks"]) == 8, "Missing expert audit")
    updates = epochs = 0
    for task in range(8):
        rows = history[str(task)]
        expected = smoke_updates or 30 * math.ceil(pa["tasks"][str(task)]["train_instances"] / c["train_batch_size"])
        require(bool(rows) and all(r["skipped_optimizer_steps"] == 0 and r["amp_loss_scale_end"] == 1024
                                  and math.isfinite(r["current_loss"]) for r in rows), "Unstable training")
        count = sum(int(r["optimizer_steps"]) for r in rows)
        require(count == expected and (smoke_updates or len(rows) == 30), "Incomplete update budget")
        if c["adapter_mode"] != "disabled":
            require(all(r["adapter_grad_finite"] == 1 and r["adapter_grad_total_norm"] > 0 for r in rows),
                    "Missing/nonfinite adapter gradients")
            if c["static_moe_mode"] != "disabled":
                require(all(r[f"adapter_grad_layer1_expert{e}_norm"] > 0 for r in rows for e in range(3)),
                        "An expert did not receive gradients")
        row = audit["tasks"][str(task)]
        require(row["old_lane_parameters_unchanged"] and row["adapter_old_state_unchanged"]
                and row["adapter_old_sha256_before"] == row["adapter_old_sha256_after"]
                and row["identity"]["raw_token_max_difference"] == 0., "Task protection/identity failed")
        updates += count
        epochs += len(rows)
    require(s["completed_optimizer_updates"] == updates and s["completed_epochs"] == epochs, "Summary budget mismatch")
    return c, pa


def audit_pairing(batch, smoke_updates=0):
    configs, audits = {}, {}
    for method in METHODS:
        configs[method], audits[method] = audit_source(batch / method, smoke_updates)
    base, pa = configs["BASELINE"], audits["BASELINE"]
    expected_modes = dict(BASELINE="disabled", SHARED="disabled", MOE_UNIFORM="uniform", MOE_OVERLAP="overlap")
    common_keys = ("seed", "class_order", "task_sizes", "max_tasks", "epochs_per_task",
                   "train_batch_size", "eval_batch_size", "adapter_layer_indices", "adapter_residual_scale",
                   "view_fusion", "view_auxiliary_loss_weight", "parameter_group_loss_routing", "training_loss_mode",
                   "learning_rate", "optimizer", "scheduler_mode", "scheduler_min_lr_ratio", "scheduler_warmup_ratio",
                   "input_normalization", "full_crop_mode", "train_crop_scale", "person_crop_margin",
                   "amp_initial_scale", "amp_growth_interval", "clip_checkpoint_sha256", "face_manifest",
                   "static_moe_smoke_samples", "optimizer_updates_per_task", "git")
    task_keys = ("train_sample_ids_sha256", "validation_sample_ids_sha256", "first_train_batch_ids_sha256",
                 "reporting_sample_ids_sha256", "train_instances", "sampler_seed")
    for method, c in configs.items():
        require(c["static_moe_mode"] == expected_modes[method]
                and c["adapter_mode"] == ("disabled" if method == "BASELINE" else "image_token"),
                "Method/source mismatch")
        require(all(c[key] == base[key] for key in common_keys), "Paired configs differ")
        require(audits[method]["initial_selector_prompt_classifier_sha256"] == pa["initial_selector_prompt_classifier_sha256"],
                "Base initialization differs")
        for task in range(8):
            require(all(audits[method]["tasks"][str(task)][key] == pa["tasks"][str(task)][key] for key in task_keys),
                    "Paired samples/samplers differ")
    budgets = [configs[m]["adapter_parameter_counts_per_task"] for m in METHODS[1:]]
    require(all(value == [74544] * 8 for value in budgets), "Adapter budgets differ")
    require(all(configs[m]["adapter_learning_rate"] == .0004 and configs[m]["adapter_weight_decay"] == 0
                for m in METHODS[1:]), "Adapter optimizer settings differ")
    require(configs["MOE_UNIFORM"]["initial_adapter_parameters_sha256"] ==
            configs["MOE_OVERLAP"]["initial_adapter_parameters_sha256"], "Expert initializations differ")
    for method in METHODS[2:]:
        expected = np.asarray(ROUTES[expected_modes[method]], dtype=np.float32)
        require(np.array_equal(np.asarray(configs[method]["static_moe_route_weights"], dtype=np.float32), expected),
                "Routing table differs")
    return configs, audits


def source_hashes(source):
    paths = [source / name for name in ("config.json", "seed_summary.json", "paired_protocol_audit.json",
                                       "static_moe_audit.json", "training_history.json")]
    paths += [source / f"{folder}/task{task}.{suffix}" for task in range(8)
              for folder, suffix in (("compact_checkpoints", "pth"), ("val_scores", "npz"))]
    return {str(p): _sha256(p) for p in paths}


def preceding_validation(batch, split, metadata, smoke_updates=0, preflight=False, parallel_splits=False):
    require(split in ("val", "test"), "Invalid split")
    if split == "val":
        return None
    require(not smoke_updates and not preflight, "No test in preflight")
    if parallel_splits:
        return None
    path = batch / "full_val" / "comparison.json"
    require(path.is_file(), "All methods must complete full validation before test is loaded")
    result = read(path)
    require(result["status"] == "complete" and result["split"] == "val" and result["samples"] == 2397
            and result["evaluator_git"]["commit"] == metadata["commit"]
            and set(result["results"]) == set(METHODS), "Invalid preceding validation")
    return result


def evaluate_batch(batch, output, data_root, clip_checkpoint, face_manifest_root,
                   split="val", preflight=False, smoke_updates=0, parallel_splits=False, source_lock=None):
    metadata = git_metadata(Path(__file__).resolve().parents[2])
    require(not metadata["dirty"] and not output.exists(), "Dirty evaluator or existing output")
    previous = preceding_validation(batch, split, metadata, smoke_updates, preflight, parallel_splits)
    locked = None
    if parallel_splits:
        require(source_lock is not None, "Parallel evaluation requires a shared immutable source lock")
        locked = read(source_lock)
        require(locked["evaluation_mode"] == "parallel_fixed_checkpoints"
                and locked["batch"] == str(batch.resolve())
                and locked["evaluator_git"] == metadata
                and locked["smoke_updates"] == smoke_updates
                and set(locked["source_sha256"]) == set(METHODS), "Invalid parallel source lock")
        lock_hash = _sha256(source_lock)
    configs, audits = audit_pairing(batch, smoke_updates)
    output.mkdir(parents=True)
    results, expected_ids, expected_targets, expected_face = {}, None, None, None
    for method in METHODS:
        source, c = batch / method, configs[method]
        require(_sha256(clip_checkpoint) == c["clip_checkpoint_sha256"], "CLIP checkpoint changed")
        require(Path(c["data_root"]).resolve() == data_root.resolve(), "Dataset root changed")
        hashes = source_hashes(source)
        if locked:
            require(hashes == locked["source_sha256"][method], "Parallel source changed after locking")
        if previous:
            require(previous["results"][method]["source_sha256"] == hashes, "Source changed after validation")
        manifest_paths = {"detector_config": face_manifest_root / "detector_config.json",
                          "train_manifest": face_manifest_root / "manifests/train.jsonl",
                          "val_manifest": face_manifest_root / "manifests/val.jsonl"}
        if split == "test":
            manifest_paths["test_manifest"] = face_manifest_root / "manifests/test.jsonl"
        manifest_hashes = {str(p): _sha256(p) for p in manifest_paths.values()}
        require(all(manifest_hashes[str(p)] == c["face_manifest"]["artifact_sha256"][key]
                    for key, p in manifest_paths.items()), "Face provenance differs")
        set_seed(c["seed"], c["tf32"])
        device = torch.device("cuda")
        model = build_static_model(c, load_openai_clip_visual(clip_checkpoint)).float().to(device)
        visual_hash = tensor_state_digest(model.visual_encoder.state_dict())
        require(visual_hash == audits[method]["frozen_visual_sha256_after"], "Restored CLIP differs")
        validation = make_source(c, data_root, face_manifest_root, "val")
        validate_classes(validation)
        require(len(validation) == 2397, "Incomplete validation population")
        loader, rows = None, {}
        for task in range(8):
            payload = _torch_load(source / f"compact_checkpoints/task{task}.pth")
            restore_compact_model_state(model, payload, task, c["git"])
            model.requires_grad_(False).eval()
            if c["static_moe_mode"] != "disabled":
                model.adapter_bank.validate_routes()
            difference = replay(model, validation, source, c, task, device)
            # All four training sources are audited; parallel splits use the same fixed source lock.
            if loader is None:
                cohort = validation if split == "val" else make_source(c, data_root, face_manifest_root, "test")
                validate_classes(cohort)
                require(len(cohort) == (2397 if split == "val" else 5368), "Incomplete cohort")
                indices = range(min(64, len(cohort))) if preflight else range(len(cohort))
                loader = DataLoader(LabelView(cohort, indices, range(26), include_sample_id=True),
                                    batch_size=64, num_workers=2, shuffle=False)
            before = tensor_state_digest(compact_model_state_dict(model))
            _, full = infer(model, loader, device, c["amp"], task, c["task_sizes"])
            require(tensor_state_digest(compact_model_state_dict(model)) == before
                    and all(p.grad is None for p in model.parameters()), "Inference changed state/gradients")
            data = seen_payload(full, task)
            if expected_ids is None:
                expected_ids, expected_targets, expected_face = full["sample_ids"], full["full_targets"], full["face_reliable"]
            require(np.array_equal(expected_ids, full["sample_ids"])
                    and np.array_equal(expected_targets, full["full_targets"])
                    and np.array_equal(expected_face, full["face_reliable"]), "Paired cohorts differ")
            destination = output / method
            destination.mkdir(exist_ok=True)
            np.savez_compressed(destination / f"task{task}.npz", **data)
            row = dict(samples=len(data["sample_ids"]), validation_replay_max_logit_difference=difference,
                       model_state_sha256_before=before, model_state_sha256_after=before)
            if not preflight:
                metrics = metrics_at_task(data, .5, task)
                start = sum(TASK_SIZES[:task])
                row.update(metrics=metrics, new_class_mAP=float(np.mean(metrics["fused"]["per_class_ap"][start:])),
                           old_class_mAP=float(np.mean(metrics["fused"]["per_class_ap"][:start])) if start else None,
                           filtered_seen_people_metrics=metrics_at_task(filter_seen_people(data), .5, task))
            rows[str(task)] = row
            print(f"STATIC_MOE_{split.upper()} method={method} task={task} replay={difference:.8g} samples={row['samples']}", flush=True)
        require(tensor_state_digest(model.visual_encoder.state_dict()) == visual_hash, "Inference changed CLIP")
        require(all(_sha256(Path(p)) == sha for p, sha in hashes.items()), "Inference changed source")
        require(all(_sha256(Path(p)) == sha for p, sha in manifest_hashes.items()), "Face manifests changed")
        results[method] = dict(tasks=rows, source_sha256=hashes, face_manifest_sha256=manifest_hashes,
                               source_unchanged=True, clip_unchanged=True)
        if not preflight:
            results[method]["curve_summary"] = summarize_curve(rows)
        del model
        torch.cuda.empty_cache()
    result = dict(status="preflight_passed" if preflight else "complete", split=split,
                  seed=configs["BASELINE"]["seed"], samples=len(expected_ids), evaluator_git=metadata,
                  training_performed=False, optimizer_updates=0,
                  checkpoint_selection="fixed_smoke_updates" if smoke_updates else "fixed_final_epoch30",
                  test_loaded=split == "test", test_weight_search=False, results=results,
                  validation_preceded_test=previous is not None, smoke_updates=smoke_updates,
                  evaluation_mode="parallel_fixed_checkpoints" if parallel_splits else "sequential",
                  evaluation_source_lock_sha256=lock_hash if locked else None)
    if locked:
        require(_sha256(source_lock) == lock_hash, "Parallel source lock changed during evaluation")
    (output / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if not preflight:
        lines = [f"# Static MoE v1: seed0 complete {split}", "",
                 "| Task | Classes | Baseline | Shared | Uniform MoE | Overlap MoE | Overlap - shared |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for task in range(8):
            values = [results[m]["tasks"][str(task)]["metrics"]["fused"]["mAP"] for m in METHODS]
            lines.append(f"| {task} | {sum(TASK_SIZES[:task+1])} | " +
                         " | ".join(f"{v:.4f}" for v in values + [values[3] - values[1]]) + " |")
        (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print("STATIC_MOE_EVALUATION_COMPLETE " + result["status"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("batch", "output", "data-root", "clip-checkpoint", "face-manifest-root"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--parallel-splits", action="store_true")
    parser.add_argument("--source-lock", type=Path)
    evaluate_batch(**vars(parser.parse_args()))

