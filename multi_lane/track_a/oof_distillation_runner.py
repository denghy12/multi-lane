"""Validation-only Full-lane training with leakage-free cross-view OOF teachers."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from multi_lane.continual_datasets.continual_datasets import EMOTIC

from .model import MultiLaneModel
from .oof_distillation import DISTILLATION_MODES, OOFTeacherBank, OOFTeacherLabelView
from .openai_clip_loader import OPENAI_VIT_B16_SHA256, load_openai_clip_visual
from .runner import (
    CLASS_ORDER,
    PROTOCOL_ID,
    TASK_SIZES,
    LabelView,
    TaskMetrics,
    _intersects,
    build_transforms,
    dataset_view,
    evaluate,
    git_metadata,
    resolve_dataset_parent,
    seen_indices,
    set_seed,
    summarize_tasks,
    task_indices,
    train_task,
    validate_classes,
)


DISTILLATION_MIX = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(0,))
    parser.add_argument("--mode", choices=DISTILLATION_MODES, required=True)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0125)
    parser.add_argument("--adapter-learning-rate", type=float, default=4e-4)
    parser.add_argument("--distillation-mix", type=float, default=DISTILLATION_MIX)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (
        args.epochs == 30
        and args.train_batch_size == 64
        and args.learning_rate == 0.0125
        and args.adapter_learning_rate == 4e-4
        and args.distillation_mix == DISTILLATION_MIX
        and args.threshold == 0.5
        and args.max_tasks == 8
    ):
        raise ValueError("OOF distillation validation configuration is locked")
    if not torch.cuda.is_available():
        raise RuntimeError("OOF distillation validation requires CUDA")
    amp, tf32 = not args.no_amp, not args.no_tf32
    set_seed(args.seed, tf32)
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "val_scores").mkdir()
    root = Path(__file__).resolve().parents[2]
    metadata = git_metadata(root)
    if metadata["dirty"]:
        raise RuntimeError("OOF distillation requires a clean Git worktree")

    visual = load_openai_clip_visual(args.clip_checkpoint)
    model = MultiLaneModel(
        visual_encoder=visual,
        task_sizes=TASK_SIZES,
        num_selectors=10,
        num_prompts=10,
        num_prompt_layers=5,
        normalize="pre-head",
        adapter_mode="image_token",
        adapter_bottleneck_dim=32,
        adapter_layer_indices=(1,),
        adapter_residual_scale=0.1,
        adapter_activation="relu",
        adapter_task_initialization="independent",
        adapter_residual_gate_mode="fixed",
    ).float().cuda()
    model.visual_encoder.requires_grad_(False)
    model.assert_visual_frozen()

    train_transform, eval_transform = build_transforms(
        "clip", (0.05, 1.0), input_mode="full"
    )
    dataset_parent = resolve_dataset_parent(args.data_root)
    train_source = EMOTIC(
        str(dataset_parent), train=True, input_mode="full", transform=train_transform
    )
    val_source = EMOTIC(
        str(dataset_parent), train=False, eval_splits=("val",),
        input_mode="full", transform=eval_transform,
    )
    validate_classes(train_source, val_source)
    teacher_bank = OOFTeacherBank(
        args.oof_root.expanduser().resolve(),
        args.face_manifest_root.expanduser().resolve(),
        train_source,
        args.mode,
    )

    asl = {
        "source": "CODE_DDP MaskedAsymmetricLoss",
        "gamma_neg": 9.8,
        "gamma_pos": 0.0,
        "clip": 0.05,
        "eps": 1e-8,
        "detach_focal_weight": True,
        "reduction": "mean_over_training_loss_view",
    }
    config = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "track": "A",
        "method": "MULTI-LANE+OOFCrossViewDistillation",
        "seed": args.seed,
        "class_order": list(CLASS_ORDER),
        "task_sizes": list(TASK_SIZES),
        "train_split": "train",
        "validation_split": "val",
        "reporting_split": "val",
        "test_accessed": False,
        "max_tasks": args.max_tasks,
        "save_checkpoints": False,
        "save_evaluation_scores": True,
        "evaluation_score_purpose": "validation_search",
        "training_budget_mode": "epochs",
        "epochs_per_task": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "workers": args.workers,
        "threshold": args.threshold,
        "training_label_scope": "current_classes_only",
        "training_loss_mode": "legacy_full_zero",
        "parameter_group_loss_routing": "adapter_asl",
        "model_parameter_objective": "bce",
        "adapter_parameter_objective": "asl",
        "asl": asl,
        "learning_rate": args.learning_rate,
        "optimizer": "Adam_reset_per_task",
        "weight_decay": 0.0,
        "scheduler": "CosineAnnealingLR_reset_per_task",
        "scheduler_mode": "cosine",
        "scheduler_min_lr_ratio": 0.0,
        "scheduler_warmup_ratio": 0.0,
        "scheduler_step_unit": "epoch",
        "input_mode": "full",
        "input_normalization": "clip",
        "train_crop_scale": [0.05, 1.0],
        "full_crop_mode": "legacy",
        "amp": amp,
        "tf32": tf32,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "num_selectors": 10,
        "num_prompts": 10,
        "num_prompt_layers": 5,
        "normalize": "pre-head",
        "head_mode": "concat",
        "adapter_mode": "image_token",
        "adapter_bottleneck_dim": 32,
        "adapter_layer_indices": [1],
        "adapter_residual_scale": 0.1,
        "adapter_residual_gate_mode": "fixed",
        "adapter_activation": "relu",
        "adapter_task_initialization": "independent",
        "adapter_learning_rate": args.adapter_learning_rate,
        "adapter_weight_decay": 0.0,
        "adapter_regularization": "none",
        "clip_checkpoint": str(args.clip_checkpoint.expanduser().resolve()),
        "clip_checkpoint_sha256": OPENAI_VIT_B16_SHA256,
        "oof_distillation": {
            "mode": args.mode,
            "mix": args.distillation_mix,
            "supervised_bce_weight": 1.0 - args.distillation_mix,
            "teacher_bce_weight": args.distillation_mix,
            "loss_view": "legacy_full_zero_current_soft_targets_hidden_logits_zero",
            "parameter_scope": "base_model_only_adapter_remains_hard_label_asl",
            "class_scope": "current_task_new_classes_only",
            "old_task_modules_frozen": True,
            "teacher": teacher_bank.summary(),
            "oof_root": str(args.oof_root.expanduser().resolve()),
        },
        "git": metadata,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    task_rows: List[TaskMetrics] = []
    histories: Dict[str, List[Dict[str, float]]] = {}
    started = time.time()
    for task_id in range(args.max_tasks):
        model.activate_task(task_id)
        current = task_indices(task_id)
        fit_indices = [
            index for index, target in enumerate(train_source.targets)
            if _intersects(target, current)
        ]
        train_view = OOFTeacherLabelView(
            train_source, fit_indices, current, teacher_bank.tasks[task_id]
        )
        train_loader = DataLoader(
            train_view, batch_size=args.train_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=False,
        )
        current_val_loader = DataLoader(
            dataset_view(val_source, current), batch_size=args.eval_batch_size,
            shuffle=False, num_workers=args.workers,
        )
        reporting_loader = DataLoader(
            dataset_view(val_source, seen_indices(task_id), include_sample_id=True),
            batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers,
        )
        histories[str(task_id)] = train_task(
            model, train_loader, current_val_loader, torch.device("cuda"),
            task_id, args.epochs, args.learning_rate, 0.0, 1.0, amp,
            adapter_learning_rate=args.adapter_learning_rate,
            adapter_weight_decay=0.0,
            loss_mode="legacy_full_zero",
            loss_routing="adapter_asl",
            asl_gamma_neg=9.8,
            asl_gamma_pos=0.0,
            asl_clip=0.05,
            asl_eps=1e-8,
            scheduler_mode="cosine",
            scheduler_min_lr_ratio=0.0,
            scheduler_warmup_ratio=0.0,
            oof_distillation_mix=args.distillation_mix,
        )
        row = evaluate(
            model, reporting_loader, torch.device("cuda"), task_id,
            args.threshold, amp, output / "val_scores" / f"task{task_id}.npz",
        )
        task_rows.append(row)
        (output / "task_metrics.json").write_text(
            json.dumps([asdict(value) for value in task_rows], indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "training_history.json").write_text(
            json.dumps(histories, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"task={task_id} val_mAP={row.mAP:.6f} val_cF1={row.cF1:.6f} "
            f"val_oF1={row.oF1:.6f}", flush=True,
        )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "method": config["method"],
        "protocol_id": PROTOCOL_ID,
        "track": "A",
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
        "completed_epochs": sum(len(rows) for rows in histories.values()),
        "completed_optimizer_updates": int(sum(
            row["optimizer_steps"] for rows in histories.values() for row in rows
        )),
        "metrics": summarize_tasks(task_rows),
        "task_metrics": [asdict(value) for value in task_rows],
        "config": config,
    }
    (output / "seed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["metrics"], indent=2), flush=True)
    print("OOF_CROSS_VIEW_DISTILLATION_VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
