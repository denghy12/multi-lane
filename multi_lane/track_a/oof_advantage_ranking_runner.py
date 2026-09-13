"""Validation-only Full training with OOF-advantage pairwise ranking supervision."""

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
from .oof_advantage_ranking import CorrectedPairDataset, OOFAdvantagePairBank
from .openai_clip_loader import OPENAI_VIT_B16_SHA256, load_openai_clip_visual
from .runner import (
    CLASS_ORDER, PROTOCOL_ID, TASK_SIZES, LabelView, TaskMetrics, _intersects,
    build_transforms, dataset_view, evaluate, git_metadata, resolve_dataset_parent,
    seen_indices, set_seed, summarize_tasks, task_indices, train_task,
    validate_classes,
)


RANKING_BATCH_SIZE = 16
RANKING_LOSS_WEIGHT = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=(0,))
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--ranking-batch-size", type=int, default=RANKING_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0125)
    parser.add_argument("--adapter-learning-rate", type=float, default=4e-4)
    parser.add_argument("--ranking-loss-weight", type=float, default=RANKING_LOSS_WEIGHT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-tasks", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (
        args.epochs == 30
        and args.train_batch_size == 64
        and args.ranking_batch_size == RANKING_BATCH_SIZE
        and args.eval_batch_size == 64
        and args.learning_rate == 0.0125
        and args.adapter_learning_rate == 4e-4
        and args.ranking_loss_weight == RANKING_LOSS_WEIGHT
        and args.threshold == 0.5
        and args.max_tasks == 8
    ):
        raise ValueError("OOF advantage-ranking validation configuration is locked")
    if not torch.cuda.is_available():
        raise RuntimeError("OOF advantage-ranking validation requires CUDA")
    set_seed(args.seed, True)
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "val_scores").mkdir()
    root = Path(__file__).resolve().parents[2]
    metadata = git_metadata(root)
    if metadata["dirty"]:
        raise RuntimeError("OOF advantage-ranking requires a clean Git worktree")

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

    random_transform, deterministic_transform = build_transforms(
        "clip", (0.05, 1.0), input_mode="full"
    )
    dataset_parent = resolve_dataset_parent(args.data_root)
    train_source = EMOTIC(
        str(dataset_parent), train=True, input_mode="full",
        transform=random_transform,
    )
    deterministic_source = EMOTIC(
        str(dataset_parent), train=True, input_mode="full",
        transform=deterministic_transform,
    )
    val_source = EMOTIC(
        str(dataset_parent), train=False, eval_splits=("val",),
        input_mode="full", transform=deterministic_transform,
    )
    validate_classes(train_source, val_source)
    if (
        list(train_source.sample_ids) != list(deterministic_source.sample_ids)
        or list(train_source.targets) != list(deterministic_source.targets)
    ):
        raise ValueError("Random and deterministic Full training sources differ")
    pair_bank = OOFAdvantagePairBank(
        args.oof_root.expanduser().resolve(),
        args.face_manifest_root.expanduser().resolve(),
        deterministic_source,
    )

    config = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "track": "A",
        "method": "MULTI-LANE+OOFAdvantageRankingDistillation",
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
        "ranking_batch_size": args.ranking_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "workers": args.workers,
        "threshold": args.threshold,
        "training_label_scope": "current_classes_only",
        "training_loss_mode": "legacy_full_zero",
        "parameter_group_loss_routing": "adapter_asl",
        "model_parameter_objective": "hard_bce_plus_oof_pairwise_ranking",
        "adapter_parameter_objective": "hard_label_asl",
        "asl": {
            "source": "CODE_DDP MaskedAsymmetricLoss",
            "gamma_neg": 9.8,
            "gamma_pos": 0.0,
            "clip": 0.05,
            "eps": 1e-8,
            "detach_focal_weight": True,
            "reduction": "mean_over_training_loss_view",
        },
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
        "ranking_view": "deterministic_resize256_center_crop224",
        "amp": True,
        "tf32": True,
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
        "oof_advantage_ranking": {
            "loss": "softplus(negative_logit-positive_logit)",
            "loss_weight": args.ranking_loss_weight,
            "hard_bce_weight": 1.0,
            "adapter_receives_ranking_gradient": False,
            "old_task_modules_frozen": True,
            "pairs_per_hard_update": args.ranking_batch_size,
            "pair_bank": pair_bank.summary(),
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
        train_loader = DataLoader(
            LabelView(train_source, fit_indices, current),
            batch_size=args.train_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=False,
        )
        table = pair_bank.tasks[task_id]
        ranking_loader = None
        ranking_weight = 0.0
        if table.class_indices:
            ranking_samples = args.epochs * len(train_loader) * args.ranking_batch_size
            ranking_dataset = CorrectedPairDataset(
                deterministic_source, table, ranking_samples,
                seed=1000 + task_id,
            )
            ranking_loader = DataLoader(
                ranking_dataset, batch_size=args.ranking_batch_size,
                shuffle=False, num_workers=args.workers, pin_memory=True,
                drop_last=False,
                generator=torch.Generator().manual_seed(2000 + task_id),
            )
            ranking_weight = args.ranking_loss_weight
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
            task_id, args.epochs, args.learning_rate, 0.0, 1.0, True,
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
            ranking_loader=ranking_loader,
            ranking_loss_weight=ranking_weight,
        )
        row = evaluate(
            model, reporting_loader, torch.device("cuda"), task_id,
            args.threshold, True, output / "val_scores" / f"task{task_id}.npz",
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
            f"val_oF1={row.oF1:.6f} ranking_classes={len(table.class_indices)}",
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
    print("OOF_ADVANTAGE_RANKING_VALIDATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
