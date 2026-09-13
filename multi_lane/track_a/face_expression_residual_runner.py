"""Validation-only CLIP Face expert with frozen expression residuals."""

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

from .face_dual_transform import DualFaceTransform
from .face_expression_model import load_frozen_emotieff_encoder_with_classifier
from .face_expression_residual import FaceExpressionResidualModel
from .face_expression_runner import MODEL_SOURCE_URL, landmark_counts, sha256_file
from .runner import (
    CLASS_ORDER, PROTOCOL_ID, TASK_SIZES, LabelView, TaskMetrics, _intersects,
    dataset_view, evaluate, filter_face_training_indices, git_metadata,
    load_face_manifest_provenance, load_openai_clip_visual, resolve_dataset_parent,
    seen_indices, set_seed, summarize_tasks, task_indices, train_task,
    validate_classes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(0,))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--expression-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-expression-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0125)
    parser.add_argument("--adapter-learning-rate", type=float, default=4e-4)
    parser.add_argument("--expression-residual-learning-rate", type=float, default=4e-4)
    parser.add_argument("--expression-residual-rank", type=int, default=32)
    parser.add_argument("--expression-residual-scale", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    locked = (
        args.epochs == 30 and args.train_batch_size == 64
        and args.learning_rate == 0.0125
        and args.adapter_learning_rate == 4e-4
        and args.expression_residual_learning_rate == 4e-4
        and args.expression_residual_rank == 32
        and args.expression_residual_scale == 0.1
        and args.threshold == 0.5 and args.max_tasks == 8
    )
    if not locked:
        raise ValueError("Face expression residual validation configuration is locked")
    if not torch.cuda.is_available():
        raise RuntimeError("Face expression residual validation requires CUDA")
    expression_checkpoint = args.expression_checkpoint.expanduser().resolve()
    expression_sha = sha256_file(expression_checkpoint)
    if expression_sha != args.expected_expression_sha256:
        raise ValueError("Expression checkpoint SHA-256 differs from the locked artifact")
    amp, tf32 = not args.no_amp, not args.no_tf32
    set_seed(args.seed, tf32)
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "val_scores").mkdir()
    root = Path(__file__).resolve().parents[2]

    expression_encoder, expression_classifier, feature_dim, expression_classes = (
        load_frozen_emotieff_encoder_with_classifier(expression_checkpoint)
    )
    visual = load_openai_clip_visual(args.clip_checkpoint)
    model = FaceExpressionResidualModel(
        expression_encoder=expression_encoder,
        expression_classifier=expression_classifier,
        expression_feature_dim=feature_dim,
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
        residual_rank=args.expression_residual_rank,
        residual_scale=args.expression_residual_scale,
    ).float().cuda()
    model.visual_encoder.requires_grad_(False)
    model.assert_visual_frozen()
    model.assert_expression_frozen()

    dataset_parent = resolve_dataset_parent(args.data_root)
    train_source = EMOTIC(
        str(dataset_parent), train=True, input_mode="face_crop", transform=None,
        face_manifest_root=args.face_manifest_root, face_crop_margin=0.15,
        face_min_training_short_side=0, face_min_training_score=0,
        face_dual_transform=DualFaceTransform(True),
    )
    val_source = EMOTIC(
        str(dataset_parent), train=False, eval_splits=("val",),
        input_mode="face_crop", transform=None,
        face_manifest_root=args.face_manifest_root, face_crop_margin=0.15,
        face_min_training_short_side=0, face_min_training_score=0,
        face_dual_transform=DualFaceTransform(False),
    )
    validate_classes(train_source, val_source)
    manifest = load_face_manifest_provenance(args.face_manifest_root)
    config = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "track": "A",
        "method": "CLIPFaceWithFrozenExpressionResidual",
        "seed": args.seed,
        "class_order": list(CLASS_ORDER),
        "task_sizes": list(TASK_SIZES),
        "train_split": "train",
        "validation_split": "val",
        "reporting_split": "val",
        "max_tasks": args.max_tasks,
        "save_checkpoints": False,
        "save_evaluation_scores": True,
        "evaluation_score_purpose": "validation_search",
        "training_budget_mode": "epochs",
        "epochs_per_task": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "threshold": args.threshold,
        "training_loss_mode": "legacy_full_zero",
        "parameter_group_loss_routing": "adapter_asl",
        "model_parameter_objective": "bce",
        "adapter_parameter_objective": "asl",
        "asl": {"gamma_neg": 9.8, "gamma_pos": 0.0, "clip": 0.05, "eps": 1e-8},
        "learning_rate": args.learning_rate,
        "optimizer": "Adam_reset_per_task",
        "weight_decay": 0.0,
        "scheduler": "CosineAnnealingLR_reset_per_task",
        "scheduler_mode": "cosine",
        "scheduler_min_lr_ratio": 0.0,
        "scheduler_warmup_ratio": 0.0,
        "amp": amp,
        "tf32": tf32,
        "input_mode": "face_crop",
        "input_normalization": "clip",
        "face_manifest": manifest,
        "face_training_policy": "valid_face_and_not_ambiguous",
        "face_alignment": "clip_legacy_margin15_plus_expression_five_point_margin15",
        "face_alignment_counts": {
            "train": landmark_counts(train_source), "val": landmark_counts(val_source)
        },
        "face_encoder": "CLIP_ViT_B16_plus_EmotiEffLib_enet_b0_8_best_afew",
        "face_encoder_source": MODEL_SOURCE_URL,
        "expression_checkpoint": str(expression_checkpoint),
        "expression_checkpoint_sha256": expression_sha,
        "expression_feature_dim": feature_dim,
        "expression_class_count": expression_classes,
        "expression_descriptor": "layernorm_embedding_concat_layernorm_8class_logits",
        "expression_encoder_frozen": True,
        "expression_residual": "task_specific_zero_init_low_rank",
        "expression_residual_rank": args.expression_residual_rank,
        "expression_residual_scale": args.expression_residual_scale,
        "expression_residual_learning_rate": args.expression_residual_learning_rate,
        "expression_residual_objective": "bce",
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
        "git": git_metadata(root),
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
        fit_indices = filter_face_training_indices(train_source, [
            index for index, target in enumerate(train_source.targets)
            if _intersects(target, current)
        ])
        current_val_indices = [
            index for index, target in enumerate(val_source.targets)
            if _intersects(target, current) and val_source.face_training_valid[index]
        ]
        train_loader = DataLoader(
            LabelView(train_source, fit_indices, current),
            batch_size=args.train_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )
        current_val_loader = DataLoader(
            LabelView(val_source, current_val_indices, current),
            batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers,
        )
        reporting_loader = DataLoader(
            dataset_view(val_source, seen_indices(task_id), include_sample_id=True),
            batch_size=args.eval_batch_size, shuffle=False, num_workers=args.workers,
        )
        histories[str(task_id)] = train_task(
            model=model, loader=train_loader, validation_loader=current_val_loader,
            device=torch.device("cuda"), task_id=task_id, epochs=args.epochs,
            learning_rate=args.learning_rate, weight_decay=0.0, temperature=1.0,
            amp=amp, adapter_learning_rate=args.adapter_learning_rate,
            adapter_weight_decay=0.0, loss_mode="legacy_full_zero",
            loss_routing="adapter_asl", asl_gamma_neg=9.8, asl_gamma_pos=0.0,
            asl_clip=0.05, asl_eps=1e-8, scheduler_mode="cosine",
            scheduler_min_lr_ratio=0.0, scheduler_warmup_ratio=0.0,
            view_fusion_learning_rate=args.expression_residual_learning_rate,
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
    print("FACE_EXPRESSION_RESIDUAL_VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
