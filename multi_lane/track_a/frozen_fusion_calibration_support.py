"""Restore completed B0 checkpoints and broaden excluded TRAIN calibration.

This is a calibration-memory diagnostic: historical train images can be read
again, but source model parameters are never optimized. No test data is opened.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_lane.continual_datasets.continual_datasets import EMOTIC
from .export_compact_test_scores import build_model, restore_compact_model_state, _torch_load, _sha256
from .openai_clip_loader import load_openai_clip_visual
from .paired_transforms import ThreeViewTransform
from .post_task_calibration import base_hash
from .frozen_view_ranking_fusion import (read_scores, aligned_baseline, check_provenance,
                                         digest_files, run as fit_and_evaluate)
from .runner import (LabelView, build_transforms, fit_calibration_indices, seen_indices,
                     resolve_dataset_parent, evaluate, evaluate_view_diagnostics,
                     write_evaluation_scores, set_seed, git_metadata)


def support(targets, task_sizes, task):
    y = np.asarray(targets)[:, sum(task_sizes[:task]):sum(task_sizes[:task + 1])]
    return {"samples": len(y), "positive_counts": y.sum(0).tolist(),
            "negative_counts": (1 - y).sum(0).tolist(),
            "all_current_negative": int((~y.any(1)).sum()),
            "all_current_negative_fraction": float((~y.any(1)).mean())}


def combine_cached(old, extra, ordered_ids):
    """Keep existing current-positive logits exactly; export only new rows."""
    old_ids = old["sample_ids"].tolist()
    extra_ids = extra["sample_ids"].tolist() if extra is not None else []
    if set(old_ids) & set(extra_ids) or set(old_ids + extra_ids) != set(ordered_ids):
        raise ValueError("Calibration merge has missing, duplicate or overlapping IDs")
    position = {value: i for i, value in enumerate(old_ids + extra_ids)}
    ix = np.asarray([position[value] for value in ordered_ids])
    result = dict(old)
    for key in ("targets", "face_reliable", "fused_logits", "fused_probabilities",
                "full_logits", "person_logits", "face_logits",
                "full_probabilities", "person_probabilities", "face_probabilities"):
        data = np.concatenate([old[key], extra[key]], axis=0) if extra is not None else old[key]
        result[key] = data[ix]
    result["sample_ids"] = np.asarray(ordered_ids)
    n = len(ordered_ids)
    result["batch_lengths"] = np.asarray([min(64, n - i) for i in range(0, n, 64)], dtype=np.int64)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if args.tasks <= 0 or args.epochs <= 0:
        parser.error("Positive task/epoch counts required")
    if args.output.exists():
        raise ValueError("Refuse to overwrite output")
    cfg = json.loads((args.source / "config.json").read_text())
    summary = json.loads((args.source / "seed_summary.json").read_text())
    if (summary.get("status") != "complete" or len(summary["task_metrics"]) < args.tasks
        or cfg["calibration_fraction"] != .2 or not cfg["calibration_training_exclusion"]
        or cfg["reporting_split"] != "val" or cfg["also_report_test"]
        or cfg["parax_mode"] != "disabled" or cfg["adapter_mode"] != "disabled"
        or cfg["view_fusion"] != "fixed_three_view" or cfg["git"]["dirty"]):
        raise ValueError("Requires completed clean heldout B0 validation source")
    if _sha256(args.clip_checkpoint) != cfg["clip_checkpoint_sha256"]:
        raise ValueError("CLIP checkpoint changed")
    parent = resolve_dataset_parent(args.data_root)
    if (parent / "EMOTIC").resolve() != Path(cfg["data_root"]).resolve():
        raise ValueError("Dataset differs from source")
    if args.face_manifest_root.resolve() != Path(cfg["face_manifest_root"]).resolve():
        raise ValueError("Face manifest differs from source")
    metadata = git_metadata(Path(__file__).resolve().parents[2])
    if metadata["dirty"]:
        raise ValueError("Export requires clean source code")
    original_files = [p for p in args.source.rglob("*") if p.is_file()]
    original_hash = digest_files(original_files)
    args.output.mkdir(parents=True)
    set_seed(cfg["seed"], cfg["tf32"])
    device = torch.device("cuda:0")
    visual = load_openai_clip_visual(args.clip_checkpoint)
    model = build_model(cfg, visual).float().to(device)
    _, transform = build_transforms(cfg["input_normalization"], cfg["train_crop_scale"])
    paired = ThreeViewTransform(train=False, normalization=cfg["input_normalization"],
                               crop_scale=cfg["train_crop_scale"], margin=cfg["person_crop_margin"],
                               jitter_strength=cfg["person_color_jitter_strength"],
                               jitter_probability=cfg["person_color_jitter_probability"],
                               full_crop_mode=cfg["full_crop_mode"])
    dataset = EMOTIC(str(parent), train=True, transform=transform, input_mode=cfg["input_mode"],
                     person_crop_margin=cfg["person_crop_margin"], multi_view_transform=paired,
                     face_manifest_root=args.face_manifest_root, face_crop_margin=cfg["face_crop_margin"],
                     face_min_training_short_side=cfg["face_min_training_short_side"],
                     face_min_training_score=cfg["face_min_training_score"])
    exported = args.output / "augmented_source"
    exported.mkdir()
    for folder in ("calibration_scores", "calibration_view_scores", "calibration_split_provenance"):
        (exported / folder).mkdir()
    for folder in ("val_scores", "view_val_scores"):
        os.symlink(os.path.relpath((args.source / folder).resolve(), exported.resolve()), exported / folder)
    records, audit, budgets = [], {}, []
    for task in range(args.tasks):
        record = json.loads((args.source / "calibration_split_provenance" / f"task{task}.json").read_text())
        _, held = fit_calibration_indices(dataset, seen_indices(task), .2)
        ordered = [str(dataset.sample_ids[i]) for i in held]
        previous = aligned_baseline(args.source / "calibration_scores" / f"task{task}.npz",
                                    read_scores(args.source / "calibration_view_scores" / f"task{task}.npz"))
        known = set(previous["sample_ids"].tolist())
        missing = [i for i in held if str(dataset.sample_ids[i]) not in known]
        extra = None
        payload = _torch_load(args.source / "compact_checkpoints" / f"task{task}.pth")
        restore_compact_model_state(model, payload, task, cfg["git"])
        model.eval().requires_grad_(False)
        model.assert_visual_frozen()
        before = base_hash(model)
        if missing:
            view = LabelView(dataset, missing, seen_indices(task), include_sample_id=True)
            loader = DataLoader(view, batch_size=64, shuffle=False, num_workers=2)
            extra_root = args.output / "added_background_scores"
            evaluate(model, loader, device, task, .5, cfg["amp"], extra_root / f"task{task}_fixed.npz")
            evaluate_view_diagnostics(model, loader, device, task, .5, cfg["amp"], extra_root / f"task{task}_views.npz")
            extra = aligned_baseline(extra_root / f"task{task}_fixed.npz",
                                     read_scores(extra_root / f"task{task}_views.npz"))
        after = base_hash(model)
        if before != after:
            raise RuntimeError("Export changed frozen model")
        merged = combine_cached(previous, extra, ordered)
        # Audit labels against the train dataset using visible classes only.
        expected = np.zeros_like(merged["targets"])
        for row, i in enumerate(held):
            expected[row, [c for c in dataset.targets[i] if c in seen_indices(task)]] = 1
        if not np.array_equal(expected, merged["targets"]):
            raise ValueError("Calibration labels changed during merge")
        np.savez_compressed(exported / "calibration_view_scores" / f"task{task}.npz", **merged)
        write_evaluation_scores(exported / "calibration_scores" / f"task{task}.npz", task,
                                ordered, torch.from_numpy(merged["fused_logits"]), torch.from_numpy(merged["targets"]),
                                torch.from_numpy(merged["fused_probabilities"]), merged["batch_lengths"].tolist())
        record["calibration_sample_ids"] = ordered
        record["calibration_memory"] = True
        records.append(record)
        (exported / "calibration_split_provenance" / f"task{task}.json").write_text(json.dumps(record, indent=2) + "\n")
        validation = read_scores(args.source / "view_val_scores" / f"task{task}.npz")
        budgets.append(len(previous["targets"]))
        audit[str(task)] = {"previous": support(previous["targets"], cfg["task_sizes"], task),
                            "augmented": support(merged["targets"], cfg["task_sizes"], task),
                            "validation": support(validation["targets"], cfg["task_sizes"], task),
                            "added_samples": len(missing), "epoch_samples": budgets[-1],
                            "frozen_model_hash_before": before, "frozen_model_hash_after": after}
        print(f"export task={task} old={len(known)} new={len(ordered)} added={len(missing)} epoch_budget={budgets[-1]}", flush=True)
    provenance = check_provenance(records)
    del model, visual
    torch.cuda.empty_cache()
    export_cfg = dict(cfg, calibration_pool="seen_class_heldout_train",
                      calibration_memory=True, calibration_epoch_samples=budgets)
    (exported / "config.json").write_text(json.dumps(export_cfg, indent=2) + "\n")
    (exported / "seed_summary.json").write_text(json.dumps({"status": "complete", "source_model": str(args.source),
                                                            "reused_weights": True, "source_git": cfg["git"]}, indent=2) + "\n")
    rows = summary["task_metrics"][:args.tasks]
    (exported / "task_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    fit_and_evaluate(SimpleNamespace(source=exported, output=args.output / "fusion",
                                    tasks=args.tasks, epochs=args.epochs, batch_size=256, device="cuda:0"))
    if digest_files(original_files) != original_hash:
        raise RuntimeError("Calibration experiment changed original source")
    (args.output / "support_audit.json").write_text(json.dumps({"tasks": audit, "provenance": provenance,
                  "source_hash_before": original_hash, "source_hash_after": original_hash,
                  "git": metadata, "no_base_training": True, "calibration_memory": True,
                  "test_accessed": False, "matched_epoch_samples": budgets}, indent=2) + "\n")
    (args.output / "complete.txt").write_text("CALIBRATION_SUPPORT_COMPLETE\n")
    print("CALIBRATION_SUPPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
