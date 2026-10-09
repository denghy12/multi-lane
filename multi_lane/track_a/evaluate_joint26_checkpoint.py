"""Evaluate a final joint26 checkpoint without optimization or test selection."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_lane.continual_datasets.continual_datasets import EMOTIC
from .compare_joint26 import require
from .export_compact_test_scores import _read_json, _sha256, _torch_load, restore_compact_model_state
from .model import MultiLaneModel
from .openai_clip_loader import load_openai_clip_visual
from .paired_transforms import ThreeViewTransform, move_model_inputs
from .runner import (LabelView, build_transforms, compact_model_state_dict, evaluate,
                     evaluate_view_diagnostics, git_metadata, ids_digest, resolve_dataset_parent,
                     set_seed, summarize_tasks, tensor_state_digest, using_training_protocol, validate_classes)


def audit_source(source: Path):
    config, summary = _read_json(source / "config.json"), _read_json(source / "seed_summary.json")
    audit = _read_json(source / "joint_protocol_audit.json")
    require(summary["status"] == "complete" and summary["config"] == config, "Source incomplete/config mismatch")
    require(config["training_protocol"] == "joint26" and config["task_sizes"] == [26]
            and config["task_pathway_count"] == config["max_tasks"] == 1, "Expected one joint26 pathway")
    require(config["reporting_split"] == "val" and not config["also_report_test"] and not audit["test_loaded"],
            "Expected validation-trained source")
    require(config["epochs_per_task"] == summary["completed_epochs"] == 30, "Only final 30-epoch models are eligible")
    require(config["git"]["dirty"] is False and config["save_compact_checkpoints"], "Source provenance invalid")
    require(audit["frozen_visual_unchanged"] and audit["frozen_visual_sha256_before"] == audit["frozen_visual_sha256_after"],
            "Source changed CLIP")
    require(config["adapter_mode"] == "disabled" and config["parax_mode"] in ("disabled", "image")
            and config["view_fusion"] in ("disabled", "fixed_three_view")
            and config["selector_mode"] == config["prompt_mode"] == "shared"
            and config["view_classifier_mode"] == "shared_post_fusion", "Unsupported source architecture")
    history = _read_json(source / "training_history.json")["0"]
    skipped = sum(int(row["skipped_optimizer_steps"]) for row in history)
    require(len(history) == 30 and summary["completed_optimizer_updates"] + skipped
            == 30 * ((audit["train_instances"] + config["train_batch_size"] - 1) // config["train_batch_size"]),
            "Source budget incomplete")
    return config, summary, audit, skipped


def build_model(config, visual):
    options = dict(task_sizes=(26,), num_selectors=config["num_selectors"],
                   num_prompts=config["num_prompts"], num_prompt_layers=config["num_prompt_layers"],
                   selector_mode=config["selector_mode"], prompt_mode=config["prompt_mode"],
                   normalize=config["normalize"], adapter_mode="disabled",
                   view_fusion=config["view_fusion"], view_classifier_mode=config["view_classifier_mode"],
                   parax_mode=config["parax_mode"])
    if config["parax_mode"] == "image":
        keys = ("rank", "num_experts", "layer_indices", "router_hidden", "residual_scale", "level_conditioned",
                "initialization", "trainable_components", "output_scale_mode", "residual_ratio_cap",
                "task_local_gate", "freeze_center_after_task0", "projector_bottleneck_dim", "smooth_ratio_bound")
        for key in keys:
            name = "parax_" + key
            if name in config:
                options[name] = config[name]
    return MultiLaneModel(visual, **options)


def make_source(config, data_root, face_manifest_root, split):
    _, transform = build_transforms(config["input_normalization"], config["train_crop_scale"])
    three = config["view_fusion"] == "fixed_three_view"
    multi = ThreeViewTransform(train=False, normalization=config["input_normalization"],
                              crop_scale=config["train_crop_scale"], margin=config["person_crop_margin"],
                              jitter_strength=config["person_color_jitter_strength"],
                              jitter_probability=config["person_color_jitter_probability"],
                              full_crop_mode=config["full_crop_mode"]) if three else None
    return EMOTIC(str(resolve_dataset_parent(data_root)), train=False, transform=transform,
                  eval_splits=(split,), input_mode="full", person_crop_margin=config["person_crop_margin"],
                  multi_view_transform=multi, face_manifest_root=face_manifest_root if three else None)


def loader_for(source, config):
    return DataLoader(LabelView(source, range(len(source)), range(26), include_sample_id=True),
                      batch_size=config["eval_batch_size"], shuffle=False, num_workers=config["workers"],
                      pin_memory=False, drop_last=False)


def audit_face_manifest(config, root):
    """Allow a test-extended manifest only when training artifacts are identical."""
    original = Path(config["face_manifest"]["root"])
    expected = config["face_manifest"]["artifact_sha256"]
    hashes = {}
    for split in ("train", "val", "test"):
        path = root / "manifests" / (split + ".jsonl")
        hashes[str(path)] = _sha256(path)
        if split != "test":
            require(hashes[str(path)] == expected[split + "_manifest"], f"Face {split} manifest changed")
    original_detector = _read_json(original / "detector_config.json")
    require(_sha256(original / "detector_config.json") == expected["detector_config"], "Source detector config changed")
    detector = _read_json(root / "detector_config.json")
    require("test" in detector.get("splits", []), "Missing test detector provenance")
    require({k: v for k, v in detector.items() if k != "splits"}
            == {k: v for k, v in original_detector.items() if k != "splits"}, "Face detector protocol changed")
    hashes[str(root / "detector_config.json")] = _sha256(root / "detector_config.json")
    return hashes


def evaluate_checkpoint(source, output, data_root, clip_checkpoint, face_manifest_root):
    start = time.time()
    config, source_summary, source_audit, skipped = audit_source(source)
    require(not output.exists(), "Test output already exists")
    require(torch.cuda.is_available(), "CUDA required")
    metadata = git_metadata(Path(__file__).resolve().parents[2])
    require(not metadata["dirty"], "Evaluation worktree must be clean")
    require(Path(config["data_root"]).resolve() == (resolve_dataset_parent(data_root) / "EMOTIC").resolve(),
            "Dataset root changed")
    manifest_hashes = audit_face_manifest(config, face_manifest_root) if config["view_fusion"] == "fixed_three_view" else {}
    source_files = [source / name for name in ("config.json", "seed_summary.json", "joint_protocol_audit.json",
                                              "training_history.json", "compact_checkpoints/task0.pth", "val_scores/task0.npz")]
    hashes = {str(p): _sha256(p) for p in source_files}
    set_seed(config["seed"], config["tf32"])
    device = torch.device("cuda")
    model = build_model(config, load_openai_clip_visual(clip_checkpoint)).float().to(device)
    restore_compact_model_state(model, _torch_load(source / "compact_checkpoints" / "task0.pth"), 0, config["git"])
    model.requires_grad_(False).eval()
    before = tensor_state_digest(compact_model_state_dict(model))
    clip_before = tensor_state_digest(model.visual_encoder.state_dict())
    require(clip_before == source_audit["frozen_visual_sha256_after"], "Restored CLIP differs from training")
    # Replay one saved validation batch before opening test images. This checks
    # architecture, transforms, class order and restored parameters together.
    validation = make_source(config, data_root, face_manifest_root, "val")
    validate_classes(validation)
    images, targets, sample_ids = next(iter(loader_for(validation, config)))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=config["amp"]):
        logits = model.seen_logits(move_model_inputs(images, device)).float().cpu().numpy()
    with np.load(source / "val_scores" / "task0.npz") as reference:
        require(np.array_equal(reference["sample_ids"][:len(sample_ids)], np.asarray(sample_ids))
                and np.array_equal(reference["targets"][:len(sample_ids)], targets.numpy()), "Validation replay samples differ")
        difference = float(np.max(np.abs(reference["logits"][:len(logits)] - logits)))
        require(difference <= 0.01, f"Restored validation logits differ: {difference}")
    print(f"CHECKPOINT_REPLAY_PASSED max_logit_difference={difference:.8g}", flush=True)
    output.mkdir(parents=True)
    test = make_source(config, data_root, face_manifest_root, "test")
    validate_classes(test)
    test_loader = loader_for(test, config)
    with using_training_protocol("joint26"):
        row = evaluate(model, test_loader, device, 0, config["threshold"], config["amp"],
                       score_output_path=output / "test_scores" / "task0.npz")
        metrics = summarize_tasks([row])
    views = evaluate_view_diagnostics(model, test_loader, device, 0, config["threshold"], config["amp"],
                                     score_output_path=output / "view_test_scores" / "task0.npz") if config["view_fusion"] != "disabled" else None
    require(tensor_state_digest(compact_model_state_dict(model)) == before, "Evaluation modified method weights")
    require(tensor_state_digest(model.visual_encoder.state_dict()) == clip_before, "Evaluation modified CLIP")
    require(all(_sha256(Path(p)) == value for p, value in hashes.items()), "Source artifacts changed during evaluation")
    require(all(_sha256(Path(p)) == value for p, value in manifest_hashes.items()), "Face manifests changed during evaluation")
    with np.load(output / "test_scores" / "task0.npz") as z:
        test_ids_hash = ids_digest(z["sample_ids"].tolist())
    result = dict(status="complete", evaluation_split="test", training_performed=False, optimizer_updates=0,
                  test_weight_search=False, checkpoint_selection="fixed_final_epoch_30", source_run=str(source.resolve()),
                  source_validation_mAP=source_summary["metrics"]["final_mAP"], seed=config["seed"],
                  source_amp_initial_scale=config.get("amp_initial_scale", 65536),
                  source_amp_growth_interval=config.get("amp_growth_interval", 2000), source_skipped_updates=skipped,
                  source_config=config, evaluator_git=metadata, source_sha256=hashes, source_unchanged=True,
                  clip_unchanged=True, method_weights_unchanged=True, validation_replay_max_logit_difference=difference,
                  test_sample_ids_sha256=test_ids_hash, metrics=metrics, task_metrics=[asdict(row)],
                  view_diagnostics=views, face_manifest_sha256=manifest_hashes, elapsed_seconds=time.time()-start,
                  test_face_manifest_sha256=_sha256(face_manifest_root / "manifests" / "test.jsonl") if views else None)
    (output / "seed_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"JOINT26_LOCKED_TEST_COMPLETE mAP={row.mAP:.6f}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "output", "data-root", "clip-checkpoint", "face-manifest-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    evaluate_checkpoint(args.source, args.output, args.data_root, args.clip_checkpoint, args.face_manifest_root)
