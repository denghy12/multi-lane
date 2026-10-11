"""Fit a protected frozen pool on existing baseline checkpoints, or test old runs.

No optimizer ever receives original model parameters. Validation/test features
are never passed to fit_cached_routes. Source checkpoints are immutable.
"""
from __future__ import annotations
import argparse
import copy
import inspect
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from multi_lane.continual_datasets.continual_datasets import EMOTIC
from .evaluate_joint26_checkpoint import audit_face_manifest
from .export_compact_test_scores import _read_json, _sha256, _torch_load, restore_compact_model_state
from .model import MultiLaneModel
from .openai_clip_loader import load_openai_clip_visual
from .paired_transforms import ThreeViewTransform
from .post_task_calibration import cache_current_features, fit_cached_routes, base_hash, parameter_hash
from .protected_parax_audit import ProtectedRouteAudit
from .runner import (build_transforms, compact_model_state_dict, dataset_view, evaluate,
                     evaluate_view_diagnostics, git_metadata, resolve_dataset_parent,
                     set_seed, seen_indices, summarize_tasks, task_indices, validate_classes)


def build_model(config, visual, frozen_pool=False):
    keys = set(inspect.signature(MultiLaneModel.__init__).parameters) - {"self", "visual_encoder"}
    # Serialized configs use null for inactive options (e.g. fixed fusion's
    # hidden dimension). Constructor defaults, not None, apply to those fields.
    options = {k: v for k, v in config.items() if k in keys and v is not None}
    if frozen_pool:
        options.update(parax_mode="post_task_protected_frozen", parax_rank=32,
                       parax_num_experts=16, parax_layer_indices=(0,), parax_router_hidden=16,
                       parax_residual_scale=1., parax_initialization="zero_output",
                       parax_trainable_components="all", parax_output_scale_mode="fixed",
                       parax_smooth_ratio_bound=.02, parax_level_conditioned=False,
                       parax_task_local_gate=False, parax_residual_ratio_cap=0.,
                       parax_projector_bottleneck_dim=0, adapter_mode="disabled")
    return MultiLaneModel(visual, **options)


def restore_base(model, payload, task, source_git):
    if payload.get("schema_version") != 1 or payload.get("task_id") != task or payload.get("source_git") != source_git:
        raise ValueError("Baseline checkpoint provenance mismatch")
    state = payload["model"]
    expected = {k for k in model.state_dict() if not k.startswith(("visual_encoder.", "parax_bank."))}
    if set(state) != expected:
        raise ValueError("Baseline keys do not exactly match original model")
    model.load_state_dict(state, strict=False)
    if any(not torch.equal(model.state_dict()[k].cpu(), v) for k, v in state.items()):
        raise RuntimeError("Baseline tensor recovery failed")
    model.restore_task(task)  # activate_task would overwrite trained prompts/selectors!
    model.requires_grad_(False)
    return parameter_hash(sorted(state.items()))


def source_dataset(config, data_root, face_root, split):
    _, transform = build_transforms(config["input_normalization"], config["train_crop_scale"])
    multi = ThreeViewTransform(train=False, normalization=config["input_normalization"],
                              crop_scale=config["train_crop_scale"], margin=config["person_crop_margin"],
                              jitter_strength=config["person_color_jitter_strength"],
                              jitter_probability=config["person_color_jitter_probability"],
                              full_crop_mode=config["full_crop_mode"])
    return EMOTIC(str(resolve_dataset_parent(data_root)), train=split == "train", transform=transform,
                  eval_splits=(split,) if split != "train" else ("val",), input_mode="full",
                  person_crop_margin=config["person_crop_margin"], multi_view_transform=multi,
                  face_manifest_root=face_root, face_crop_margin=config.get("face_crop_margin"),
                  face_min_training_short_side=config.get("face_min_training_short_side"),
                  face_min_training_score=config.get("face_min_training_score"))


def loader(source, classes, config, smoke=False):
    view = dataset_view(source, classes, include_sample_id=True)
    if smoke:
        view = Subset(view, range(min(64, len(view))))
    return DataLoader(view,
                      batch_size=config["eval_batch_size"], shuffle=False,
                      num_workers=config["workers"], pin_memory=False, drop_last=False)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def validate_source(config, summary, fit):
    if summary.get("status") != "complete" or summary.get("config") != config:
        raise ValueError("Source run incomplete or config mismatch")
    if config["training_protocol"] != "incremental" or config["view_fusion"] != "fixed_three_view":
        raise ValueError("Expected fixed three-view incremental source")
    if fit and (config["parax_mode"] != "disabled" or config["adapter_mode"] != "disabled"
                or config.get("calibration_fraction", 0) != 0):
        raise ValueError("Cached fit requires the actual unmodified baseline and full train split")


def cached_identity(model, cache):
    device = next(model.parameters()).device
    features = {k: v[:64].to(device) for k, v in cache["features"].items()}
    reliable = cache["face_reliable"][:64].to(device)
    task = model.current_task_id
    routed = {}
    with torch.no_grad():
        for name, feature in features.items():
            after, _ = model.parax_bank(0, feature, name, task_id=task)
            routed[name] = model.normalize_routed_task_features(feature, after)
            if not torch.equal(feature, routed[name]):
                raise RuntimeError("Cached zero residual must be strict identity")
        a, _ = model.view_fusion_module(features, [task], reliable)
        b, _ = model.view_fusion_module(routed, [task], reliable)
        if not torch.equal(model.head(a[:, 0]), model.head(b[:, 0])):
            raise RuntimeError("Cached initial logits mismatch")
    return {"cached_feature_max_difference": 0., "cached_logit_max_difference": 0.}


def compare_source_scores(actual, source):
    a, b = np.load(actual), np.load(source)
    if not np.array_equal(a["sample_ids"], b["sample_ids"]) or not np.array_equal(a["targets"], b["targets"]):
        raise RuntimeError("Recovered baseline evaluation cohort changed")
    delta = float(np.max(np.abs(a["logits"] - b["logits"])))
    # Report AMP reconstruction error separately from strict cached identity.
    pa = 1. / (1. + np.exp(-a["logits"]))
    pb = 1. / (1. + np.exp(-b["logits"]))
    probability_delta = float(np.max(np.abs(pa - pb)))
    if probability_delta > .002:
        raise RuntimeError("Recovered full-forward baseline predictions mismatch: " + str(probability_delta))
    return {"amp_logit_max_difference": delta, "amp_probability_max_difference": probability_delta,
            "probability_recovery_tolerance": .002}


def run(args):
    source, output = Path(args.source), Path(args.output)
    config, summary = _read_json(source / "config.json"), _read_json(source / "seed_summary.json")
    fit = args.mode == "fit"
    validate_source(config, summary, fit)
    if output.exists():
        raise FileExistsError(output)
    metadata = git_metadata(Path(__file__).resolve().parents[2])
    if metadata["dirty"]:
        raise RuntimeError("Clean worktree required")
    clip = Path(args.clip_checkpoint)
    if _sha256(clip) != config["clip_checkpoint_sha256"]:
        raise ValueError("Pretrained CLIP changed")
    if Path(config["data_root"]).resolve() != (resolve_dataset_parent(Path(args.data_root)) / "EMOTIC").resolve():
        raise ValueError("Dataset root changed")
    face_hashes = audit_face_manifest(config, Path(args.face_manifest_root))
    device = torch.device("cuda:0")
    set_seed(config["seed"], config["tf32"])
    visual = load_openai_clip_visual(clip)
    model = build_model(config, visual, frozen_pool=fit).float().to(device)
    model.requires_grad_(False)
    model.assert_visual_frozen()
    start_task = args.smoke_start_task if args.smoke else 0
    if not args.smoke and args.smoke_start_task:
        raise ValueError("Nonzero smoke start is forbidden for formal experiments")
    count = start_task + 1 if args.smoke else (args.max_tasks or config["max_tasks"])
    if not 1 <= count <= config["max_tasks"]:
        raise ValueError("Invalid task range")
    datasets = {split: source_dataset(config, args.data_root, args.face_manifest_root, split)
                for split in (("train", "val") if fit else ("test",))}
    validate_classes(*datasets.values())
    banks = {name: copy.deepcopy(model.parax_bank) for name in ("dynamic", "static")} if fit else {}
    if fit:
        banks["static"].router_input_mode = "constant"
    audits = {name: ProtectedRouteAudit(output / name, device) for name in banks}
    for name in banks:
        (output / name).mkdir(parents=True)
    previous_base = None
    records = {name: [] for name in (("baseline", "dynamic", "static") if fit else ("test",))}
    write(output / "config.json", {"protocol": "fixed_checkpoint_train_cache_v1" if fit else "old_checkpoint_test_only_v1",
          "source_config": config, "source": str(source), "git": metadata, "face_manifest_hashes": face_hashes,
          "cache_split": "train" if fit else None, "cache_epochs": 5 if fit else 0,
          "optimizer": "Adam", "learning_rate": .0004, "weight_decay": 0., "scheduler": "cosine",
          "batch_size": 64, "gradient_clip_norm": 1., "consistency_weight": 0., "auxiliary_weight": .1,
          "test_accessed": not fit, "max_tasks": count,
          "router_modes": {"dynamic": "image", "static": "constant_ones"} if fit else {},
          "baseline_retrained": False, "cache_precision": "AMP forward converted to FP32; FP32 cached fit",
          "residual_bound": .02, "rank": 32, "active_experts": 2, "smoke": args.smoke,
          "smoke_start_task": start_task})
    torch.cuda.reset_peak_memory_stats(device)
    for task in range(start_task, count):
        checkpoint = source / "compact_checkpoints" / f"task{task}.pth"
        payload = _torch_load(checkpoint)
        if not fit:
            restore_compact_model_state(model, payload, task, config["git"])
            eval_loader = loader(datasets["test"], seen_indices(task), config, args.smoke)
            score = output / "test_scores" / f"task{task}.npz"
            score.parent.mkdir(exist_ok=True)
            row = evaluate(model, eval_loader, device, task, config["threshold"], config["amp"], score)
            records["test"].append(row)
            views = evaluate_view_diagnostics(model, eval_loader, device, task, config["threshold"], config["amp"])
            write(output / f"task{task}_views.json", views)
            write(output / f"task{task}_checkpoint.json", {"sha256": _sha256(checkpoint)})
        else:
            state_hash = restore_base(model, payload, task, config["git"])
            if previous_base is not None:
                current = audits["dynamic"]._protected_state(model, task)
                original = {k: v for k, v in current.items() if not k.startswith("parax.")}
                if any(not torch.equal(original[k], v) for k, v in previous_base.items()):
                    raise RuntimeError("Source baseline checkpoints changed previously learned task parameters")
            model.set_parax_runtime_enabled(False)
            val_loader = loader(datasets["val"], seen_indices(task), config, args.smoke)
            score = output / "baseline_scores" / f"task{task}.npz"
            score.parent.mkdir(exist_ok=True)
            records["baseline"].append(evaluate(model, val_loader, device, task, config["threshold"], config["amp"], score))
            recovery = ({"full_cohort_recovery_skipped_for_smoke": True} if args.smoke else
                        compare_source_scores(score, source / "val_scores" / f"task{task}.npz"))
            train_loader = loader(datasets["train"], task_indices(task), config, args.smoke)
            cache = cache_current_features(model, train_loader, device, config["amp"])
            cache["source_checkpoint_sha256"] = _sha256(checkpoint)
            (output / "train_cache").mkdir(exist_ok=True)
            torch.save(cache, output / "train_cache" / f"task{task}.pth")
            for name, bank in banks.items():
                model.parax_bank = bank
                model.restore_task(task)
                model.requires_grad_(False)
                identity = cached_identity(model, cache)
                audits[name].begin(model, val_loader, task)
                before = base_hash(model)
                set_seed(config["seed"] + 1009 * task, config["tf32"])
                report = fit_cached_routes(model, cache, task_indices(task), device, epochs=5,
                                           learning_rate=.0004, batch_size=64, consistency_weight=0., auxiliary_weight=.1)
                if base_hash(model) != before or any(p.grad is not None for n, p in model.named_parameters()
                                                     if not n.startswith("parax_bank.")):
                    raise RuntimeError("Original baseline changed during fitting")
                if report["initial_output_max_difference"] != 0.:
                    raise RuntimeError("Initial training batch is not identity")
                model.set_parax_runtime_enabled(True)
                audits[name].finish(model, task)
                arm_score = output / name / "val_scores" / f"task{task}.npz"
                arm_score.parent.mkdir(parents=True, exist_ok=True)
                records[name].append(evaluate(model, val_loader, device, task, config["threshold"], config["amp"], arm_score))
                views = evaluate_view_diagnostics(model, val_loader, device, task, config["threshold"], config["amp"])
                write(output / name / f"task{task}_views.json", views)
                write(output / name / f"task{task}_audit.json", {**report, **identity, **recovery,
                      "source_base_tensor_sha256": state_hash, "source_checkpoint_sha256": _sha256(checkpoint),
                      "policy": bank.policy_summary(task)})
                path = output / name / "compact_checkpoints" / f"task{task}.pth"
                path.parent.mkdir(exist_ok=True)
                torch.save({"schema_version": 1, "task_id": task, "source_git": metadata,
                            "router_input_mode": bank.router_input_mode,
                            "model": compact_model_state_dict(model)}, path)
            previous_base = {k: v for k, v in audits["dynamic"]._protected_state(model, task + 1).items()
                             if not k.startswith("parax.")}
        write(output / "summary.json", {"status": "running", "completed_tasks": task + 1,
              "arms": {k: {"task_metrics": [asdict(r) for r in rows], **summarize_tasks(rows)} for k, rows in records.items()}})
        if task == 2:
            write(output / "first_three_tasks_precheck.json", {"status": "passed", "audit_only_no_metric_selection": True,
                  "continue_to_task7": count == 8, "scope": "recovery, identity, frozen base and old bank, finite fit"})
        print(f"completed task={task} mode={args.mode}", flush=True)
    write(output / "summary.json", {"status": "complete", "completed_tasks": count,
          "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
          "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
          "arms": {k: {"task_metrics": [asdict(r) for r in rows], **summarize_tasks(rows)} for k, rows in records.items()}})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("fit", "evaluate-old"), required=True)
    for key in ("source", "output", "data-root", "clip-checkpoint", "face-manifest-root"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="64 train/val/test samples; never a formal result")
    p.add_argument("--smoke-start-task", type=int, default=0)
    run(p.parse_args())

if __name__ == "__main__":
    main()
