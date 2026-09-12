"""Validation-only incremental Face expert with a frozen emotion encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from multi_lane.continual_datasets.continual_datasets import EMOTIC

from .face_alignment import valid_five_point_landmarks
from .face_expression_model import (
    FaceExpressionIncrementalModel,
    load_frozen_emotieff_encoder,
)
from .runner import (
    CLASS_ORDER,
    PROTOCOL_ID,
    TASK_SIZES,
    LabelView,
    PadToSquare,
    TaskMetrics,
    _intersects,
    backward_routed_training_losses,
    compute_asymmetric_training_loss,
    compute_training_loss,
    current_validation_map,
    dataset_view,
    evaluate,
    filter_face_training_indices,
    git_metadata,
    load_face_manifest_provenance,
    resolve_dataset_parent,
    set_seed,
    summarize_tasks,
    task_indices,
    seen_indices,
    validate_classes,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_FILL = tuple(int(round(value * 255)) for value in IMAGENET_MEAN)
MODEL_SOURCE_URL = (
    "https://github.com/sb-ai-lab/EmotiEffLib/blob/main/"
    "models/affectnet_emotions/enet_b0_8_best_afew.pt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_expression_transform(train: bool) -> transforms.Compose:
    operations = [
        PadToSquare(fill=IMAGENET_FILL),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    ]
    if train:
        operations.append(transforms.RandomHorizontalFlip())
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return transforms.Compose(operations)


def landmark_counts(source: EMOTIC) -> Dict[str, int]:
    eligible = [
        (record, source.face_training_valid[index])
        for index, record in enumerate(source.face_records)
        if bool(record.get("valid_face")) and not bool(record.get("ambiguous_match"))
    ]
    valid = sum(
        valid_five_point_landmarks(
            record.get("face_keypoints"),
            (int(record["image_width"]), int(record["image_height"])),
        )
        for record, _ in eligible
    )
    return {
        "valid_nonambiguous_faces": len(eligible),
        "valid_five_point_landmarks": int(valid),
        "fallback_letterbox_faces": len(eligible) - int(valid),
    }


def train_task(
    model: FaceExpressionIncrementalModel,
    loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    task_id: int,
    epochs: int,
    learning_rate: float,
    adapter_learning_rate: float,
    amp: bool,
) -> List[Dict[str, float]]:
    head_parameters = list(model.base_optimizer_parameters())
    adapter_parameters = list(model.adapter_optimizer_parameters())
    groups = [{"params": head_parameters, "lr": learning_rate, "weight_decay": 0.0}]
    if adapter_parameters:
        groups.append(
            {
                "params": adapter_parameters,
                "lr": adapter_learning_rate,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    history = []
    current = task_indices(task_id)
    for epoch in range(epochs):
        model.train()
        loss_total = adapter_loss_total = 0.0
        optimizer_steps = skipped_steps = batches = 0
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        epoch_adapter_lr = (
            float(optimizer.param_groups[1]["lr"]) if adapter_parameters else None
        )
        started = time.time()
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model.current_all_logits(images)
                bce_loss = compute_training_loss(
                    logits, targets, current, 1.0, "legacy_full_zero"
                )
                adapter_loss = (
                    compute_asymmetric_training_loss(
                        logits,
                        targets,
                        current,
                        1.0,
                        "legacy_full_zero",
                        gamma_neg=9.8,
                        gamma_pos=0.0,
                        clip=0.05,
                        eps=1e-8,
                    )
                    if adapter_parameters
                    else bce_loss
                )
            scale_before = float(scaler.get_scale())
            backward_routed_training_losses(
                bce_loss,
                adapter_loss,
                head_parameters,
                adapter_parameters,
                scaler,
                same_objective=not adapter_parameters,
            )
            scaler.step(optimizer)
            scaler.update()
            if float(scaler.get_scale()) >= scale_before:
                optimizer_steps += 1
            else:
                skipped_steps += 1
            loss_total += float(bce_loss.detach().cpu())
            adapter_loss_total += float(adapter_loss.detach().cpu())
            batches += 1
        if not batches:
            raise RuntimeError("Face expression training loader produced no batches")
        if optimizer_steps:
            scheduler.step()
        row = {
            "epoch": float(epoch),
            "current_loss": loss_total / batches,
            "model_objective_loss": loss_total / batches,
            "adapter_objective_loss": adapter_loss_total / batches,
            "optimizer_steps": float(optimizer_steps),
            "skipped_optimizer_steps": float(skipped_steps),
            "learning_rate": epoch_lr,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - started,
        }
        if adapter_parameters:
            row["adapter_learning_rate"] = epoch_adapter_lr
            row["next_adapter_learning_rate"] = float(optimizer.param_groups[1]["lr"])
        history.append(row)
        print(
            f"task={task_id} epoch={epoch + 1}/{epochs} "
            f"loss={row['current_loss']:.8f} "
            f"adapter_loss={row['adapter_objective_loss']:.8f} "
            f"steps={optimizer_steps} skipped={skipped_steps} "
            f"seconds={row['elapsed_seconds']:.1f}",
            flush=True,
        )
    history[-1]["validation_current_mAP"] = current_validation_map(
        model, validation_loader, device, amp
    )
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--expression-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=FaceExpressionIncrementalModel.VARIANTS, required=True
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0125)
    parser.add_argument("--adapter-learning-rate", type=float, default=4e-4)
    parser.add_argument("--adapter-bottleneck-dim", type=int, default=32)
    parser.add_argument("--adapter-residual-scale", type=float, default=0.1)
    parser.add_argument("--adapter-activation", choices=("relu", "gelu"), default="relu")
    parser.add_argument("--face-crop-margin", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-tasks", type=int, default=len(TASK_SIZES))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs != 30 or args.train_batch_size != 64 or args.threshold != 0.5:
        raise ValueError("Face expression validation locks 30 epochs, batch64 and threshold0.5")
    if not 1 <= args.max_tasks <= len(TASK_SIZES):
        raise ValueError("max-tasks is outside the EMOTIC protocol")
    if args.face_crop_margin != 0.15:
        raise ValueError("Face expression validation locks margin0.15")
    if args.learning_rate != 0.0125 or args.adapter_learning_rate != 4e-4:
        raise ValueError("Face expression validation locks head/Adapter learning rates")
    if not torch.cuda.is_available():
        raise RuntimeError("Face expression validation requires CUDA")
    checkpoint = args.expression_checkpoint.expanduser().resolve()
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("Expression checkpoint SHA-256 differs from the locked artifact")
    amp, tf32 = not args.no_amp, not args.no_tf32
    set_seed(args.seed, tf32)
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "val_scores").mkdir()
    root = Path(__file__).resolve().parents[2]
    encoder, feature_dim = load_frozen_emotieff_encoder(checkpoint)
    model = FaceExpressionIncrementalModel(
        encoder,
        feature_dim,
        TASK_SIZES,
        args.variant,
        args.adapter_bottleneck_dim,
        args.adapter_residual_scale,
        args.adapter_activation,
    ).float().cuda()
    model.assert_encoder_frozen()

    dataset_parent = resolve_dataset_parent(args.data_root)
    train_source = EMOTIC(
        str(dataset_parent),
        train=True,
        transform=build_expression_transform(True),
        input_mode="face_crop",
        face_manifest_root=args.face_manifest_root,
        face_crop_margin=args.face_crop_margin,
        face_min_training_short_side=0,
        face_min_training_score=0,
        face_alignment="five_point",
    )
    val_source = EMOTIC(
        str(dataset_parent),
        train=False,
        eval_splits=("val",),
        transform=build_expression_transform(False),
        input_mode="face_crop",
        face_manifest_root=args.face_manifest_root,
        face_crop_margin=args.face_crop_margin,
        face_min_training_short_side=0,
        face_min_training_score=0,
        face_alignment="five_point",
    )
    validate_classes(train_source, val_source)
    manifest = load_face_manifest_provenance(args.face_manifest_root)
    alignment_counts = {
        "train": landmark_counts(train_source),
        "val": landmark_counts(val_source),
    }
    config = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "track": "A",
        "method": "FaceExpressionIncrementalExpert",
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
        "parameter_group_loss_routing": (
            "adapter_asl" if args.variant == "bottleneck_adapter" else "joint_bce"
        ),
        "model_parameter_objective": "bce",
        "adapter_parameter_objective": (
            "asl" if args.variant == "bottleneck_adapter" else None
        ),
        "asl": (
            {"gamma_neg": 9.8, "gamma_pos": 0.0, "clip": 0.05, "eps": 1e-8}
            if args.variant == "bottleneck_adapter"
            else None
        ),
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
        "input_normalization": "imagenet",
        "face_manifest": manifest,
        "face_training_policy": "valid_face_and_not_ambiguous",
        "face_alignment": "five_point_similarity_arcface_template",
        "face_alignment_margin": args.face_crop_margin,
        "face_alignment_counts": alignment_counts,
        "face_alignment_fallback": "margin0.15_pad_square_letterbox",
        "face_encoder": "EmotiEffLib_enet_b0_8_best_afew",
        "face_encoder_source": MODEL_SOURCE_URL,
        "face_encoder_checkpoint": str(checkpoint),
        "face_encoder_checkpoint_sha256": checkpoint_sha,
        "face_encoder_feature_dim": feature_dim,
        "face_encoder_frozen": True,
        "face_head": "task_specific_linear_projection",
        "adapter_mode": (
            "face_feature_bottleneck" if args.variant == "bottleneck_adapter" else "disabled"
        ),
        "adapter_bottleneck_dim": (
            args.adapter_bottleneck_dim if args.variant == "bottleneck_adapter" else None
        ),
        "adapter_layer_indices": [],
        "adapter_residual_scale": (
            args.adapter_residual_scale if args.variant == "bottleneck_adapter" else None
        ),
        "adapter_residual_gate_mode": "fixed",
        "adapter_activation": (
            args.adapter_activation if args.variant == "bottleneck_adapter" else None
        ),
        "adapter_task_initialization": "independent",
        "adapter_learning_rate": (
            args.adapter_learning_rate if args.variant == "bottleneck_adapter" else None
        ),
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
        fit_indices = [
            index
            for index, target in enumerate(train_source.targets)
            if _intersects(target, task_indices(task_id))
        ]
        fit_indices = filter_face_training_indices(train_source, fit_indices)
        val_indices = [
            index
            for index, target in enumerate(val_source.targets)
            if _intersects(target, task_indices(task_id))
            and val_source.face_training_valid[index]
        ]
        train_loader = DataLoader(
            LabelView(train_source, fit_indices, task_indices(task_id)),
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            LabelView(val_source, val_indices, task_indices(task_id)),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        reporting_loader = DataLoader(
            dataset_view(
                val_source, seen_indices(task_id), include_sample_id=True
            ),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        histories[str(task_id)] = train_task(
            model,
            train_loader,
            val_loader,
            torch.device("cuda"),
            task_id,
            args.epochs,
            args.learning_rate,
            args.adapter_learning_rate,
            amp,
        )
        row = evaluate(
            model,
            reporting_loader,
            torch.device("cuda"),
            task_id,
            args.threshold,
            amp,
            output / "val_scores" / f"task{task_id}.npz",
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
            f"task={task_id} val_mAP={row.mAP:.6f} "
            f"val_cF1={row.cF1:.6f} val_oF1={row.oF1:.6f}",
            flush=True,
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
        "completed_optimizer_updates": int(
            sum(
                row["optimizer_steps"]
                for rows in histories.values()
                for row in rows
            )
        ),
        "metrics": summarize_tasks(task_rows),
        "task_metrics": [asdict(value) for value in task_rows],
        "config": config,
    }
    (output / "seed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["metrics"], indent=2), flush=True)
    print("FACE_EXPRESSION_VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
