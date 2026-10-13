"""Fixed-checkpoint full-cohort validation/test for bounded static MoE v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .compare_joint26 import require
from .evaluate_static_moe import audit_source, build_static_model, make_source, read
from .evaluate_whole_parax_eight_task import metrics_at_task, seen_payload
from .export_compact_test_scores import _sha256, _torch_load, restore_compact_model_state
from .openai_clip_loader import load_openai_clip_visual
from .runner import LabelView, TASK_SIZES, git_metadata, set_seed, tensor_state_digest, validate_classes
from .snapshot_validation import infer, replay

METHODS = ("BASELINE", "SHARED", "MOE_OVERLAP")


def source_hashes(root: Path, method: str) -> dict[str, str]:
    source = root / method
    paths = [source / "config.json"]
    paths += [source / name for name in ("seed_summary.json", "paired_protocol_audit.json",
                                        "static_moe_audit.json", "training_history.json")]
    paths += [source / "compact_checkpoints" / f"task{t}.pth" for t in range(8)]
    paths += [source / "val_scores" / f"task{t}.npz" for t in range(8)]
    return {str(p): _sha256(p) for p in paths}


def evaluate(batch: Path, output: Path, split: str, data_root: Path,
             clip_checkpoint: Path, face_manifest_root: Path,
             source_lock: Path | None = None, preflight: bool = False,
             smoke_updates: int = 0) -> dict:
    require(split in ("val", "test"), "Invalid split")
    meta = git_metadata(Path(__file__).resolve().parents[2])
    require(not meta["dirty"] and not output.exists(), "Dirty evaluator or existing output")
    require(not (preflight and split == "test") and not (preflight and not smoke_updates),
            "Preflight is smoke-only validation")
    configs, audits = {}, {}
    for method in METHODS:
        configs[method], audits[method] = audit_source(batch / method, smoke_updates)
        c = configs[method]
        require(c["adapter_residual_bound_ratio"] == 0.1,
                "v2 sources must use the locked residual ratio 0.1")
    base = configs["BASELINE"]
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
        require(all(c[k] == base[k] for k in common_keys), "Paired v2 configs differ")
        require(audits[method]["initial_selector_prompt_classifier_sha256"] ==
                audits["BASELINE"]["initial_selector_prompt_classifier_sha256"], "Initial shared model differs")
        for task in range(8):
            require(all(audits[method]["tasks"][str(task)][k] ==
                        audits["BASELINE"]["tasks"][str(task)][k] for k in task_keys),
                    "Paired samples/samplers differ")
        require(c["static_moe_mode"] == ("overlap" if method == "MOE_OVERLAP" else "disabled")
                and c["adapter_mode"] == ("disabled" if method == "BASELINE" else "image_token"),
                "Wrong v2 method mapping")
        require(c["seed"] == 0 and c["task_sizes"] == base["task_sizes"]
                and c["class_order"] == base["class_order"]
                and c["clip_checkpoint_sha256"] == base["clip_checkpoint_sha256"]
                and c["adapter_residual_bound_ratio"] == base["adapter_residual_bound_ratio"],
                "v2 method pairing/config mismatch")
        require(c["data_root"] == base["data_root"] and c["git"] == base["git"],
                "v2 method source provenance mismatch")
    lock = read(source_lock) if source_lock else None
    lock_sha = _sha256(source_lock) if source_lock else None
    if lock:
        require(lock["batch"] == str(batch.resolve()) and lock["evaluation_git"] == meta
                and lock["split_pair"] == ["val", "test"], "Invalid v2 source lock")
    expected_count = 64 if preflight else (2397 if split == "val" else 5368)
    output.mkdir(parents=True)
    results, expected_ids, expected_targets = {}, None, None
    for method in METHODS:
        c, source = configs[method], batch / method
        hashes = source_hashes(batch, method)
        if lock:
            require(hashes == lock["source_sha256"][method], "v2 checkpoint/source changed after locking")
        require(_sha256(clip_checkpoint) == c["clip_checkpoint_sha256"], "CLIP checkpoint changed")
        require(Path(c["data_root"]).resolve() == data_root.resolve(), "Dataset root mismatch")
        manifest_paths = {"detector_config": face_manifest_root / "detector_config.json",
                          "val_manifest": face_manifest_root / "manifests/val.jsonl"}
        if split == "test":
            manifest_paths["test_manifest"] = face_manifest_root / "manifests/test.jsonl"
        manifests = {str(p): _sha256(p) for p in manifest_paths.values()}
        require(all(manifests[str(p)] == c["face_manifest"]["artifact_sha256"][k]
                    for k, p in manifest_paths.items()), "Face source changed")
        set_seed(c["seed"], c["tf32"])
        model = build_static_model(c, load_openai_clip_visual(clip_checkpoint)).float().cuda()
        visual_sha = tensor_state_digest(model.visual_encoder.state_dict())
        validation = make_source(c, data_root, face_manifest_root, "val")
        validate_classes(validation)
        require(len(validation) == 2397, "Incomplete validation cohort")
        cohort = validation if split == "val" else make_source(c, data_root, face_manifest_root, "test")
        validate_classes(cohort)
        require(len(cohort) == (2397 if split == "val" else 5368), "Incomplete evaluation cohort")
        indices = range(min(64, len(cohort))) if preflight else range(len(cohort))
        loader = DataLoader(LabelView(cohort, indices, range(26), include_sample_id=True),
                            batch_size=c["eval_batch_size"], num_workers=c["workers"], shuffle=False)
        rows = {}
        for task in range(8):
            payload = _torch_load(source / "compact_checkpoints" / f"task{task}.pth")
            restore_compact_model_state(model, payload, task, c["git"])
            model.requires_grad_(False).eval()
            difference = replay(model, validation, source, c, task, torch.device("cuda"))
            before = tensor_state_digest(model.state_dict())
            _, full = infer(model, loader, torch.device("cuda"), c["amp"], task, TASK_SIZES)
            require(tensor_state_digest(model.state_dict()) == before, "Read-only inference changed model")
            data = seen_payload(full, task)
            require(len(data["sample_ids"]) == expected_count, "Wrong full-cohort size")
            if expected_ids is None:
                expected_ids, expected_targets = full["sample_ids"], full["full_targets"]
            require(np.array_equal(expected_ids, full["sample_ids"])
                    and np.array_equal(expected_targets, full["full_targets"]),
                    "Paired task/method cohorts differ")
            row = {"samples": int(len(data["sample_ids"])),
                   "validation_replay_max_logit_difference": difference}
            if not preflight:
                row["metrics"] = metrics_at_task(data, c["threshold"], task)
            task_out = output / method
            task_out.mkdir(exist_ok=True)
            np.savez_compressed(task_out / f"task{task}.npz", **data)
            rows[str(task)] = row
            print(f"STATIC_MOE_V2_{split.upper()} {method} task={task} n={row['samples']}", flush=True)
        require(tensor_state_digest(model.visual_encoder.state_dict()) == visual_sha,
                "Frozen CLIP changed during inference")
        require(source_hashes(batch, method) == hashes, "Source files changed during inference")
        require(all(_sha256(Path(p)) == sha for p, sha in manifests.items()), "Face source changed during inference")
        results[method] = {"tasks": rows, "source_sha256": hashes, "source_unchanged": True}
        del model
        torch.cuda.empty_cache()
    result = {"status": "preflight_passed" if preflight else "complete", "split": split,
              "samples": expected_count, "training_performed": False, "optimizer_updates": 0,
              "validation_preceded_test": False, "evaluation_mode": "parallel_fixed_checkpoints",
              "source_lock_sha256": lock_sha,
              "evaluator_git": meta, "results": results}
    if lock:
        require(_sha256(source_lock) == lock_sha, "Source lock changed during inference")
    (output / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("batch", "output", "data-root", "clip-checkpoint", "face-manifest-root"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--source-lock", type=Path)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--smoke-updates", type=int, default=0)
    evaluate(**vars(p.parse_args()))
