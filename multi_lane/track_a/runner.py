"""Strict three-seed-capable EMOTIC Track-A runner for MULTI-LANE.

This entry point intentionally lives in the original multi-lane repository.  It
keeps the historical ``main.py`` path unchanged while reproducing the frozen
benchmark protocol used for the registered MULTI-LANE Track-A result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from multi_lane.continual_datasets.continual_datasets import EMOTIC

from .model import MultiLaneModel
from .openai_clip_loader import OPENAI_VIT_B16_SHA256, load_openai_clip_visual


CLASS_ORDER: Tuple[str, ...] = (
    "Affection", "Anger", "Annoyance", "Anticipation", "Aversion",
    "Confidence", "Disapproval", "Disconnection", "Disquietment",
    "Doubt/Confusion", "Embarrassment", "Engagement", "Esteem",
    "Excitement", "Fatigue", "Fear", "Happiness", "Pain", "Peace",
    "Pleasure", "Sadness", "Sensitivity", "Suffering", "Surprise",
    "Sympathy", "Yearning",
)
TASK_SIZES: Tuple[int, ...] = (5, 3, 3, 3, 3, 3, 3, 3)
PROTOCOL_ID = "emotic_b5c3_v0.1"
METHOD_NAME = "MULTI-LANE"
TRACK = "A"
CLIP_IMAGE_MEAN: Tuple[float, ...] = (
    0.48145466,
    0.4578275,
    0.40821073,
)
CLIP_IMAGE_STD: Tuple[float, ...] = (
    0.26862954,
    0.26130258,
    0.27577711,
)


def task_indices(task_id: int) -> Tuple[int, ...]:
    start = sum(TASK_SIZES[:task_id])
    return tuple(range(start, start + TASK_SIZES[task_id]))


def seen_indices(task_id: int) -> Tuple[int, ...]:
    return tuple(range(sum(TASK_SIZES[: task_id + 1])))


class LabelView(Dataset):
    def __init__(
        self,
        source: EMOTIC,
        indices: Sequence[int],
        class_indices: Sequence[int],
    ) -> None:
        self.source = source
        self.indices = tuple(int(value) for value in indices)
        self.class_indices = tuple(int(value) for value in class_indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, target = self.source[self.indices[index]]
        return image, target[list(self.class_indices)].float()


def _intersects(target: Sequence[int], class_indices: Sequence[int]) -> bool:
    return bool(set(int(value) for value in target).intersection(class_indices))


def dataset_view(
    source: EMOTIC, class_indices: Sequence[int]
) -> LabelView:
    indices = [
        index
        for index, target in enumerate(source.targets)
        if _intersects(target, class_indices)
    ]
    if not indices:
        raise RuntimeError("EMOTIC protocol view contains no samples")
    return LabelView(source, indices, class_indices)


def build_transforms(
    input_normalization: str = "none",
    train_crop_scale: Sequence[float] = (0.05, 1.0),
):
    if input_normalization not in {"none", "clip"}:
        raise ValueError("Input normalization must be none or clip")
    if len(train_crop_scale) != 2:
        raise ValueError("Train crop scale must contain minimum and maximum")
    crop_scale = tuple(float(value) for value in train_crop_scale)
    if not 0 < crop_scale[0] <= crop_scale[1] <= 1:
        raise ValueError("Train crop scale must satisfy 0 < min <= max <= 1")
    normalize = (
        [transforms.Normalize(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD)]
        if input_normalization == "clip" else []
    )
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224, scale=crop_scale, ratio=(3.0 / 4.0, 4.0 / 3.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            *normalize,
        ]
    )
    evaluate = transforms.Compose(
        [
            transforms.Resize(
                256, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            *normalize,
        ]
    )
    return train, evaluate


def compute_training_loss(
    logits: torch.Tensor,
    current_targets: torch.Tensor,
    current_class_indices: Sequence[int],
    temperature: float,
    loss_mode: str,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("Training temperature must be positive")
    loss_logits, loss_targets = training_loss_view(
        logits, current_targets, current_class_indices, loss_mode
    )
    return F.binary_cross_entropy_with_logits(
        loss_logits.float() / temperature, loss_targets
    )


def training_loss_view(
    logits: torch.Tensor,
    current_targets: torch.Tensor,
    current_class_indices: Sequence[int],
    loss_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the historically supervised logit/target view once per objective."""

    if loss_mode not in {"legacy_full_zero", "current_only"}:
        raise ValueError("Training loss mode must be legacy_full_zero or current_only")
    current = tuple(int(index) for index in current_class_indices)
    if not current or len(set(current)) != len(current):
        raise ValueError("Current class indices must be non-empty and unique")
    if current_targets.ndim != 2 or current_targets.shape[1] != len(current):
        raise ValueError("Current targets do not match current class indices")
    if logits.ndim != 2 or logits.shape[0] != current_targets.shape[0]:
        raise ValueError("Training logits and targets have incompatible shapes")
    if min(current) < 0 or max(current) >= logits.shape[1]:
        raise ValueError("Current class index is outside the logits")
    if loss_mode == "current_only":
        loss_logits = logits[:, list(current)]
        loss_targets = current_targets
    else:
        hidden = [index for index in range(logits.shape[1]) if index not in current]
        hidden_tensor = torch.tensor(hidden, device=logits.device, dtype=torch.long)
        loss_targets = torch.zeros_like(logits, dtype=torch.float32)
        loss_targets[:, list(current)] = current_targets
        loss_logits = logits.index_fill(1, hidden_tensor, 0.0)
    return loss_logits, loss_targets


def compute_asymmetric_training_loss(
    logits: torch.Tensor,
    current_targets: torch.Tensor,
    current_class_indices: Sequence[int],
    temperature: float,
    loss_mode: str,
    gamma_neg: float = 9.8,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """CODE_DDP-compatible ASL over MULTI-LANE's supervised logit view."""

    if temperature <= 0:
        raise ValueError("Training temperature must be positive")
    if gamma_neg < 0 or gamma_pos < 0:
        raise ValueError("ASL gamma values must be non-negative")
    if not 0 <= clip < 1:
        raise ValueError("ASL clip must be in [0, 1)")
    if eps <= 0:
        raise ValueError("ASL epsilon must be positive")
    loss_logits, loss_targets = training_loss_view(
        logits, current_targets, current_class_indices, loss_mode
    )
    if not torch.isfinite(loss_logits).all():
        raise FloatingPointError("ASL received non-finite training logits")
    if not torch.isfinite(loss_targets).all():
        raise FloatingPointError("ASL received non-finite training targets")
    probabilities = torch.sigmoid(loss_logits.float() / temperature)
    negative_probabilities = 1.0 - probabilities
    if clip > 0:
        negative_probabilities = (negative_probabilities + clip).clamp(max=1.0)
    targets = loss_targets.float()
    anti_targets = 1.0 - targets
    positive_loss = targets * torch.log(probabilities.clamp_min(eps))
    negative_loss = anti_targets * torch.log(
        negative_probabilities.clamp_min(eps)
    )
    positive_weight = (1.0 - probabilities).pow(gamma_pos)
    negative_weight = (1.0 - negative_probabilities).pow(gamma_neg)
    focal_weight = (
        targets * positive_weight + anti_targets * negative_weight
    ).detach()
    loss = (-(positive_loss + negative_loss) * focal_weight).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("ASL produced a non-finite loss")
    return loss


def backward_routed_training_losses(
    model_loss: torch.Tensor,
    adapter_loss: torch.Tensor,
    model_parameters: Sequence[torch.nn.Parameter],
    adapter_parameters: Sequence[torch.nn.Parameter],
    scaler: torch.cuda.amp.GradScaler,
    same_objective: bool,
) -> None:
    """Route distinct objectives to model and Adapter parameter groups.

    A normal backward is retained when both groups use one objective.  With
    mixed objectives, autograd computes gradients for each disjoint parameter
    group explicitly, preventing either loss from leaking into the other group.
    """

    model_parameters = tuple(model_parameters)
    adapter_parameters = tuple(adapter_parameters)
    if not adapter_parameters or same_objective:
        scaler.scale(model_loss).backward()
        return
    overlap = {id(parameter) for parameter in model_parameters}.intersection(
        id(parameter) for parameter in adapter_parameters
    )
    if overlap:
        raise ValueError("Model and Adapter parameter groups must be disjoint")
    model_gradients = torch.autograd.grad(
        scaler.scale(model_loss), model_parameters, retain_graph=True
    )
    adapter_gradients = torch.autograd.grad(
        scaler.scale(adapter_loss), adapter_parameters
    )
    for parameter, gradient in zip(model_parameters, model_gradients):
        parameter.grad = gradient
    for parameter, gradient in zip(adapter_parameters, adapter_gradients):
        parameter.grad = gradient


def resolve_dataset_parent(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "EMOTIC" and (path / "CVPR17_Annotations.mat").is_file():
        return path.parent
    if (path / "EMOTIC" / "CVPR17_Annotations.mat").is_file():
        return path
    raise FileNotFoundError(
        "Expected CVPR17_Annotations.mat under DATA_ROOT/EMOTIC or DATA_ROOT"
    )


def validate_classes(*datasets: EMOTIC) -> None:
    for source in datasets:
        if tuple(source.classes) != CLASS_ORDER:
            raise RuntimeError(
                "EMOTIC class order differs from the frozen protocol: "
                f"{tuple(source.classes)}"
            )


def set_seed(seed: int, tf32: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    order = scores.argsort()[::-1]
    sorted_targets = targets[order]
    positives = sorted_targets == 1
    positive_count = np.cumsum(positives)
    total = int(positive_count[-1]) if len(positive_count) else 0
    if total == 0:
        return 0.0
    positive_count[~positives] = 0
    rank = np.arange(1, len(scores) + 1)
    return float(np.sum(positive_count / rank) / (total + 1e-8))


def binary_counts(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    predictions = predictions.astype(bool)
    targets = targets.astype(bool)
    tp = int(np.logical_and(predictions, targets).sum())
    fp = int(np.logical_and(predictions, ~targets).sum())
    fn = int(np.logical_and(~predictions, targets).sum())
    tn = int(np.logical_and(~predictions, ~targets).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "support": tp + fn, "predicted_positive": tp + fp,
        "precision": precision, "recall": recall, "f1": f1,
    }


@dataclass
class TaskMetrics:
    task_id: int
    seen_classes: int
    samples: int
    threshold: float
    mAP: float
    cPrecision: float
    cRecall: float
    cF1: float
    oPrecision: float
    oRecall: float
    oF1: float
    per_class_ap: List[float]


def compute_metrics(
    task_id: int, scores: torch.Tensor, targets: torch.Tensor, threshold: float
) -> TaskMetrics:
    scores_np = scores.float().cpu().numpy()
    targets_np = targets.float().cpu().numpy()
    per_class_ap = [
        100.0 * average_precision(scores_np[:, index], targets_np[:, index])
        for index in range(scores_np.shape[1])
    ]
    predictions = scores_np > threshold
    class_rows = [
        binary_counts(predictions[:, index], targets_np[:, index])
        for index in range(scores_np.shape[1])
    ]
    overall = binary_counts(predictions, targets_np)
    class_mean = lambda key: 100.0 * float(np.mean([row[key] for row in class_rows]))
    return TaskMetrics(
        task_id=task_id,
        seen_classes=scores_np.shape[1],
        samples=scores_np.shape[0],
        threshold=threshold,
        mAP=float(np.mean(per_class_ap)),
        cPrecision=class_mean("precision"),
        cRecall=class_mean("recall"),
        cF1=class_mean("f1"),
        oPrecision=100.0 * overall["precision"],
        oRecall=100.0 * overall["recall"],
        oF1=100.0 * overall["f1"],
        per_class_ap=per_class_ap,
    )


def evaluate(
    model: MultiLaneModel,
    loader: Iterable,
    device: torch.device,
    task_id: int,
    threshold: float,
    amp: bool,
) -> TaskMetrics:
    model.eval()
    scores: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model.seen_logits(images)
            scores.append(torch.sigmoid(logits.float()).cpu())
            targets.append(target.float().cpu())
    if not scores:
        raise RuntimeError("Evaluation loader produced no samples")
    return compute_metrics(
        task_id, torch.cat(scores), torch.cat(targets), threshold
    )


def current_validation_map(
    model: MultiLaneModel, loader: Iterable, device: torch.device, amp: bool
) -> float:
    model.eval()
    scores: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model.current_logits(images)
            scores.append(torch.sigmoid(logits.float()).cpu())
            targets.append(target.float().cpu())
    all_scores = torch.cat(scores).numpy()
    all_targets = torch.cat(targets).numpy()
    return 100.0 * float(np.mean([
        average_precision(all_scores[:, index], all_targets[:, index])
        for index in range(all_targets.shape[1])
    ]))


def build_optimizer_groups(
    model: MultiLaneModel,
    weight_decay: float,
    adapter_learning_rate: Optional[float] = None,
    adapter_weight_decay: Optional[float] = None,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[Dict[str, object]]]:
    if weight_decay < 0:
        raise ValueError("Weight decay must be non-negative")
    model_parameters = list(model.base_optimizer_parameters())
    optimizer_groups: List[Dict[str, object]] = [
        {
            "params": [model.selectors, *list(model.prompts)],
            "weight_decay": weight_decay,
        },
        {"params": list(model.head.parameters()), "weight_decay": 0.0},
    ]
    adapter_parameters = list(model.adapter_optimizer_parameters())
    if adapter_parameters:
        if adapter_learning_rate is None or adapter_learning_rate <= 0:
            raise ValueError("Enabled adapters require a positive learning rate")
        resolved_adapter_weight_decay = (
            weight_decay
            if adapter_weight_decay is None
            else float(adapter_weight_decay)
        )
        if resolved_adapter_weight_decay < 0:
            raise ValueError("Adapter weight decay must be non-negative")
        optimizer_groups.append(
            {
                "params": adapter_parameters,
                "weight_decay": resolved_adapter_weight_decay,
                "lr": adapter_learning_rate,
            }
        )
    return model_parameters, adapter_parameters, optimizer_groups


def calibrated_adapter_regularization_weight(
    metric_total: float,
    reference_loss_total: float,
    samples: int,
    target_fraction: float,
    eps: float = 1e-12,
) -> float:
    """Derive one fixed per-task weight from a shared warm-up rule.

    Every task calibrates on the same number of successful updates.  The
    resulting fixed weight makes the warm-up metric approximately the requested
    fraction of the Adapter objective without consulting task ids, validation,
    test labels, or future-task statistics.
    """
    if not 0 <= target_fraction <= 1:
        raise ValueError("Adapter regularization fraction must be between 0 and 1")
    if samples <= 0:
        raise ValueError("Adapter regularization calibration requires samples")
    if metric_total <= eps:
        return 0.0
    return float(target_fraction) * float(reference_loss_total) / float(metric_total)


def train_task(
    model: MultiLaneModel,
    loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    task_id: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    amp: bool,
    adapter_learning_rate: Optional[float] = None,
    adapter_weight_decay: Optional[float] = None,
    loss_mode: str = "legacy_full_zero",
    loss_routing: str = "joint_bce",
    asl_gamma_neg: float = 9.8,
    asl_gamma_pos: float = 0.0,
    asl_clip: float = 0.05,
    asl_eps: float = 1e-8,
    optimizer_updates_per_task: Optional[int] = None,
    adapter_regularization: str = "none",
    adapter_regularization_fraction: float = 0.0,
    adapter_regularization_calibration_updates: int = 30,
) -> List[Dict[str, float]]:
    if loss_routing not in {
        "joint_bce", "model_asl", "adapter_asl", "both_asl"
    }:
        raise ValueError("Unknown model/Adapter loss routing")
    if optimizer_updates_per_task is not None and optimizer_updates_per_task <= 0:
        raise ValueError("Optimizer updates per task must be positive")
    if adapter_regularization not in {
        "none", "residual_ratio", "feature_cosine"
    }:
        raise ValueError("Unknown Adapter regularization mode")
    if adapter_regularization == "none" and adapter_regularization_fraction != 0:
        raise ValueError("Disabled Adapter regularization requires fraction zero")
    if adapter_regularization != "none" and not 0 < adapter_regularization_fraction <= 1:
        raise ValueError("Enabled Adapter regularization requires fraction in (0, 1]")
    if adapter_regularization_calibration_updates <= 0:
        raise ValueError("Adapter regularization calibration updates must be positive")
    model_parameters, adapter_parameters, optimizer_groups = build_optimizer_groups(
        model=model,
        weight_decay=weight_decay,
        adapter_learning_rate=adapter_learning_rate,
        adapter_weight_decay=adapter_weight_decay,
    )
    if loss_routing in {"adapter_asl", "both_asl"} and not adapter_parameters:
        raise ValueError("Adapter ASL routing requires an enabled Adapter")
    optimizer = torch.optim.Adam(optimizer_groups, lr=learning_rate)
    scheduler_steps = (
        int(optimizer_updates_per_task)
        if optimizer_updates_per_task is not None else epochs
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    current = list(task_indices(task_id))
    history: List[Dict[str, float]] = []
    completed_task_updates = 0
    regularization_metric_calibration_total = 0.0
    regularization_reference_calibration_total = 0.0
    regularization_calibration_samples = 0
    regularization_weight: Optional[float] = None
    epoch = 0
    while (
        completed_task_updates < optimizer_updates_per_task
        if optimizer_updates_per_task is not None
        else epoch < epochs
    ):
        model.train()
        loss_total = 0.0
        adapter_loss_total = 0.0
        adapter_base_loss_total = 0.0
        adapter_regularization_total = 0.0
        adapter_regularization_metric_total = 0.0
        batches = 0
        optimizer_steps = 0
        skipped_steps = 0
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        epoch_adapter_lr = (
            float(optimizer.param_groups[-1]["lr"])
            if adapter_parameters else None
        )
        epoch_start = time.time()
        for images, current_targets in loader:
            if (
                optimizer_updates_per_task is not None
                and completed_task_updates >= optimizer_updates_per_task
            ):
                break
            images = images.to(device, non_blocking=True).float()
            current_targets = current_targets.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model.current_all_logits(images)
                bce_loss = compute_training_loss(
                    logits,
                    current_targets,
                    current,
                    temperature,
                    loss_mode,
                )
                asl_loss = None
                if loss_routing != "joint_bce":
                    asl_loss = compute_asymmetric_training_loss(
                        logits,
                        current_targets,
                        current,
                        temperature,
                        loss_mode,
                        gamma_neg=asl_gamma_neg,
                        gamma_pos=asl_gamma_pos,
                        clip=asl_clip,
                        eps=asl_eps,
                    )
                model_loss = (
                    asl_loss
                    if loss_routing in {"model_asl", "both_asl"}
                    else bce_loss
                )
                adapter_base_loss = (
                    asl_loss
                    if loss_routing in {"adapter_asl", "both_asl"}
                    else bce_loss
                )
                regularization_metric = logits.new_zeros(())
                regularization_loss = logits.new_zeros(())
                if adapter_regularization != "none":
                    regularization_metric = model.adapter_auxiliary_metric(
                        adapter_regularization
                    )
                    if (
                        regularization_weight is None
                        and completed_task_updates
                        >= adapter_regularization_calibration_updates
                    ):
                        regularization_weight = calibrated_adapter_regularization_weight(
                            regularization_metric_calibration_total,
                            regularization_reference_calibration_total,
                            regularization_calibration_samples,
                            adapter_regularization_fraction,
                        )
                    if regularization_weight is not None:
                        regularization_loss = (
                            regularization_metric * regularization_weight
                        )
                adapter_loss = adapter_base_loss + regularization_loss
            scale_before = float(scaler.get_scale())
            backward_routed_training_losses(
                model_loss=model_loss,
                adapter_loss=adapter_loss,
                model_parameters=model_parameters,
                adapter_parameters=adapter_parameters,
                scaler=scaler,
                same_objective=(
                    loss_routing in {"joint_bce", "both_asl"}
                    and adapter_regularization == "none"
                ),
            )
            scaler.step(optimizer)
            scaler.update()
            if float(scaler.get_scale()) >= scale_before:
                optimizer_steps += 1
                completed_task_updates += 1
                if (
                    adapter_regularization != "none"
                    and regularization_weight is None
                    and completed_task_updates
                    <= adapter_regularization_calibration_updates
                ):
                    regularization_metric_calibration_total += float(
                        regularization_metric.detach().cpu()
                    )
                    regularization_reference_calibration_total += float(
                        adapter_base_loss.detach().cpu()
                    )
                    regularization_calibration_samples += 1
                if optimizer_updates_per_task is not None:
                    scheduler.step()
            else:
                skipped_steps += 1
            loss_total += float(model_loss.detach().cpu())
            adapter_loss_total += float(adapter_loss.detach().cpu())
            adapter_base_loss_total += float(adapter_base_loss.detach().cpu())
            adapter_regularization_total += float(
                regularization_loss.detach().cpu()
            )
            adapter_regularization_metric_total += float(
                regularization_metric.detach().cpu()
            )
            batches += 1
        if not batches:
            raise RuntimeError("Training loader produced no batches")
        if optimizer_steps and optimizer_updates_per_task is None:
            scheduler.step()
        row = {
            "epoch": float(epoch),
            "current_loss": loss_total / batches,
            "model_objective_loss": loss_total / batches,
            "adapter_objective_loss": adapter_loss_total / batches,
            "adapter_base_objective_loss": adapter_base_loss_total / batches,
            "adapter_regularization_loss": adapter_regularization_total / batches,
            "adapter_regularization_metric": (
                adapter_regularization_metric_total / batches
            ),
            "adapter_regularization_weight": (
                regularization_weight if regularization_weight is not None else 0.0
            ),
            "optimizer_steps": float(optimizer_steps),
            "completed_task_optimizer_updates": float(completed_task_updates),
            "skipped_optimizer_steps": float(skipped_steps),
            "learning_rate": epoch_lr,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.time() - epoch_start,
        }
        if adapter_parameters:
            row["adapter_learning_rate"] = epoch_adapter_lr
            row["next_adapter_learning_rate"] = float(
                optimizer.param_groups[-1]["lr"]
            )
        history.append(row)
        print(
            f"task={task_id} cycle={epoch + 1} "
            f"budget={optimizer_updates_per_task or epochs} "
            f"budget_mode={'updates' if optimizer_updates_per_task else 'epochs'} "
            f"loss={row['current_loss']:.8f} lr={epoch_lr:.8f} "
            f"adapter_loss={row['adapter_objective_loss']:.8f} "
            f"adapter_reg={row['adapter_regularization_loss']:.8f} "
            f"adapter_reg_weight={row['adapter_regularization_weight']:.8f} "
            f"steps={optimizer_steps} skipped={skipped_steps} "
            f"seconds={row['elapsed_seconds']:.1f}",
            flush=True,
        )
        epoch += 1
    history[-1]["validation_current_mAP"] = current_validation_map(
        model, validation_loader, device, amp
    )
    return history


def git_metadata(root: Path) -> Dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()
    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "dirty": bool(status),
    }


def summarize_tasks(rows: Sequence[TaskMetrics]) -> Dict[str, object]:
    final = rows[-1]
    forgetting_by_class: Dict[str, float] = {}
    final_task = len(rows) - 1
    for class_id, class_name in enumerate(CLASS_ORDER):
        introduction = next(
            task_id for task_id in range(len(TASK_SIZES))
            if class_id in task_indices(task_id)
        )
        if introduction >= final_task:
            continue
        history = [
            row.per_class_ap[class_id]
            for row in rows
            if row.task_id >= introduction and class_id < row.seen_classes
        ]
        forgetting_by_class[class_name] = max(history) - history[-1]
    return {
        "final_mAP": final.mAP,
        "average_mAP": float(np.mean([row.mAP for row in rows])),
        "final_cF1": final.cF1,
        "final_oF1": final.oF1,
        "forgetting": (
            float(np.mean(list(forgetting_by_class.values())))
            if forgetting_by_class else 0.0
        ),
        "per_class_forgetting": forgetting_by_class,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MULTI-LANE EMOTIC Track-A reproduction")
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--optimizer-updates-per-task",
        type=int,
        default=None,
        help=(
            "Use one exact successful-optimizer-update budget for every task. "
            "When set, --epochs is ignored for stopping and cosine scheduling."
        ),
    )
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--source-learning-rate", type=float, default=0.05)
    parser.add_argument("--source-reference-batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--training-loss-mode",
        choices=("legacy_full_zero", "current_only"),
        default="legacy_full_zero",
    )
    parser.add_argument(
        "--loss-routing",
        choices=("joint_bce", "model_asl", "adapter_asl", "both_asl"),
        default="joint_bce",
    )
    parser.add_argument("--asl-gamma-neg", type=float, default=9.8)
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--asl-eps", type=float, default=1e-8)
    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help="Do not write per-task checkpoints (intended for validation sweeps).",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument("--input-mode", choices=("full", "person_crop"), default="full")
    parser.add_argument(
        "--input-normalization", choices=("none", "clip"), default="none"
    )
    parser.add_argument(
        "--train-crop-scale", type=float, nargs=2, default=(0.05, 1.0),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--adapter-mode",
        choices=("disabled", "task_lane", "image_token"),
        default="disabled",
    )
    parser.add_argument("--adapter-bottleneck-dim", type=int, default=64)
    parser.add_argument(
        "--adapter-bottleneck-dims-per-task",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional bottleneck dimensions for every protocol task. "
            "When omitted, --adapter-bottleneck-dim is shared by all tasks."
        ),
    )
    parser.add_argument(
        "--adapter-layer-indices", type=int, nargs="+", default=[11]
    )
    parser.add_argument("--adapter-residual-scale", type=float, default=0.1)
    parser.add_argument(
        "--adapter-residual-gate-mode",
        choices=("fixed", "learnable"),
        default="fixed",
    )
    parser.add_argument(
        "--adapter-activation", choices=("relu", "gelu"), default="relu"
    )
    parser.add_argument("--adapter-learning-rate", type=float, default=4e-4)
    parser.add_argument(
        "--adapter-weight-decay",
        type=float,
        default=None,
        help=(
            "Adapter-only weight decay. Defaults to the shared --weight-decay "
            "for backward compatibility."
        ),
    )
    parser.add_argument(
        "--adapter-task-init",
        choices=("independent", "copy_previous"),
        default="independent",
    )
    parser.add_argument(
        "--adapter-regularization",
        choices=("none", "residual_ratio", "feature_cosine"),
        default="none",
    )
    parser.add_argument(
        "--adapter-regularization-fraction", type=float, default=0.0
    )
    parser.add_argument(
        "--adapter-regularization-calibration-updates", type=int, default=30
    )
    parser.add_argument("--max-tasks", type=int, default=len(TASK_SIZES))
    parser.add_argument(
        "--reporting-split", choices=("val", "test"), default="test"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_batch_size != 64:
        raise ValueError("Frozen Track-A protocol requires train batch size 64")
    if args.threshold != 0.5:
        raise ValueError("Frozen Track-A protocol requires threshold 0.5")
    if not 1 <= args.max_tasks <= len(TASK_SIZES):
        raise ValueError(f"max-tasks must be between 1 and {len(TASK_SIZES)}")
    if args.loss_routing in {"adapter_asl", "both_asl"} and args.adapter_mode == "disabled":
        raise ValueError("Adapter ASL routing requires an enabled Adapter")
    if args.adapter_bottleneck_dims_per_task is not None:
        if len(args.adapter_bottleneck_dims_per_task) != len(TASK_SIZES):
            raise ValueError(
                "adapter-bottleneck-dims-per-task must provide one value for "
                f"each of the {len(TASK_SIZES)} protocol tasks"
            )
        if any(value <= 0 for value in args.adapter_bottleneck_dims_per_task):
            raise ValueError("Per-task Adapter bottleneck dimensions must be positive")
    if args.adapter_weight_decay is not None and args.adapter_weight_decay < 0:
        raise ValueError("Adapter weight decay must be non-negative")
    if args.optimizer_updates_per_task is not None and args.optimizer_updates_per_task <= 0:
        raise ValueError("optimizer-updates-per-task must be positive")
    if args.adapter_residual_gate_mode == "learnable":
        if args.adapter_mode != "image_token":
            raise ValueError("Learnable residual gates require Image-token Adapter mode")
        if not 0 < args.adapter_residual_scale < 1:
            raise ValueError("Learnable residual gate initialization must be in (0, 1)")
    if args.adapter_regularization == "none":
        if args.adapter_regularization_fraction != 0:
            raise ValueError(
                "adapter-regularization-fraction must be zero when regularization is disabled"
            )
    else:
        if args.adapter_mode != "image_token":
            raise ValueError("Adapter output regularization requires Image-token mode")
        if not 0 < args.adapter_regularization_fraction <= 1:
            raise ValueError(
                "Enabled Adapter regularization requires fraction in (0, 1]"
            )
    if args.adapter_regularization_calibration_updates <= 0:
        raise ValueError(
            "adapter-regularization-calibration-updates must be positive"
        )
    if not torch.cuda.is_available() or args.device != "cuda":
        raise RuntimeError("Formal Track-A reproduction requires CUDA")
    amp = not args.no_amp
    tf32 = not args.no_tf32
    set_seed(args.seed, tf32)
    device = torch.device("cuda")
    root = Path(__file__).resolve().parents[2]
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_checkpoints = not args.no_save_checkpoints
    if save_checkpoints:
        (output / "checkpoints").mkdir()

    metadata = git_metadata(root)
    visual = load_openai_clip_visual(args.clip_checkpoint)
    model = MultiLaneModel(
        visual_encoder=visual,
        task_sizes=TASK_SIZES,
        num_selectors=10,
        num_prompts=10,
        num_prompt_layers=5,
        normalize="pre-head",
        adapter_mode=args.adapter_mode,
        adapter_bottleneck_dim=args.adapter_bottleneck_dim,
        adapter_layer_indices=args.adapter_layer_indices,
        adapter_residual_scale=args.adapter_residual_scale,
        adapter_activation=args.adapter_activation,
        adapter_task_initialization=args.adapter_task_init,
        adapter_bottleneck_dims_per_task=args.adapter_bottleneck_dims_per_task,
        adapter_residual_gate_mode=args.adapter_residual_gate_mode,
        adapter_auxiliary_metric_mode=args.adapter_regularization,
    ).float().to(device)
    model.visual_encoder.requires_grad_(False)
    model.assert_visual_frozen()
    lane_parameters = model.selectors.numel() + sum(p.numel() for p in model.prompts)
    classifier_parameters = sum(p.numel() for p in model.head.parameters())
    adapter_parameters = (
        model.adapter_bank.total_parameter_count()
        if model.adapter_bank is not None else 0
    )
    adapter_parameters_per_task = (
        model.adapter_bank.per_task_parameter_count()
        if model.adapter_bank is not None else 0
    )
    adapter_parameter_counts_per_task = (
        list(model.adapter_bank.per_task_parameter_counts())
        if model.adapter_bank is not None else []
    )
    print(
        f"trainable_parameters={lane_parameters + classifier_parameters + adapter_parameters_per_task} "
        f"task_lane={lane_parameters} classifier={classifier_parameters} "
        f"adapter_total={adapter_parameters} "
        f"adapter_per_task={adapter_parameter_counts_per_task}",
        flush=True,
    )

    train_transform, eval_transform = build_transforms(
        args.input_normalization, args.train_crop_scale
    )
    dataset_parent = resolve_dataset_parent(args.data_root)
    train_source = EMOTIC(
        str(dataset_parent), train=True, transform=train_transform,
        input_mode=args.input_mode,
    )
    val_source = EMOTIC(
        str(dataset_parent), train=False, transform=eval_transform,
        eval_splits=("val",), input_mode=args.input_mode,
    )
    if args.reporting_split == "test":
        reporting_source = EMOTIC(
            str(dataset_parent), train=False, transform=eval_transform,
            eval_splits=("test",), input_mode=args.input_mode,
        )
        validate_classes(train_source, val_source, reporting_source)
    else:
        reporting_source = val_source
        validate_classes(train_source, val_source)

    learning_rate = args.source_learning_rate * (
        args.train_batch_size / args.source_reference_batch_size
    )
    config = {
        "protocol_id": PROTOCOL_ID,
        "track": TRACK,
        "method": METHOD_NAME,
        "seed": args.seed,
        "class_order": list(CLASS_ORDER),
        "task_sizes": list(TASK_SIZES),
        "train_split": "train",
        "validation_split": "val",
        "reporting_split": args.reporting_split,
        "training_label_scope": "current_classes_only",
        "training_loss_mode": args.training_loss_mode,
        "parameter_group_loss_routing": args.loss_routing,
        "model_parameter_objective": (
            "asl"
            if args.loss_routing in {"model_asl", "both_asl"} else "bce"
        ),
        "adapter_parameter_objective": (
            "asl"
            if args.loss_routing in {"adapter_asl", "both_asl"} else "bce"
        ),
        "asl": (
            {
                "source": "CODE_DDP MaskedAsymmetricLoss",
                "gamma_neg": args.asl_gamma_neg,
                "gamma_pos": args.asl_gamma_pos,
                "clip": args.asl_clip,
                "eps": args.asl_eps,
                "detach_focal_weight": True,
                "reduction": "mean_over_training_loss_view",
            }
            if args.loss_routing != "joint_bce" else None
        ),
        "training_loss_reduction_classes": (
            "current_task_only"
            if args.training_loss_mode == "current_only"
            else "all_26_with_zeroed_hidden_logits"
        ),
        "training_loss_current_only_gradient_multiplier_vs_legacy": (
            [len(CLASS_ORDER) / size for size in TASK_SIZES]
            if args.training_loss_mode == "current_only" else None
        ),
        "training_loss_optimizer_scale_note": (
            "Adam moment normalization can largely cancel constant gradient scaling"
        ),
        "evaluation_scope": "samples_intersect_seen_classes",
        "save_checkpoints": save_checkpoints,
        "threshold": args.threshold,
        "training_budget_mode": (
            "optimizer_updates"
            if args.optimizer_updates_per_task is not None else "epochs"
        ),
        "epochs_per_task": (
            None if args.optimizer_updates_per_task is not None else args.epochs
        ),
        "optimizer_updates_per_task": args.optimizer_updates_per_task,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "workers": args.workers,
        "optimizer": "Adam_reset_per_task",
        "learning_rate": learning_rate,
        "scheduler": (
            "CosineAnnealingLR_reset_per_task_by_optimizer_step"
            if args.optimizer_updates_per_task is not None
            else "CosineAnnealingLR_reset_per_task"
        ),
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "input_mode": args.input_mode,
        "input_normalization": args.input_normalization,
        "input_normalization_mean": (
            list(CLIP_IMAGE_MEAN) if args.input_normalization == "clip" else None
        ),
        "input_normalization_std": (
            list(CLIP_IMAGE_STD) if args.input_normalization == "clip" else None
        ),
        "train_crop_scale": [float(value) for value in args.train_crop_scale],
        "amp": amp,
        "tf32": tf32,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "num_selectors": 10,
        "num_prompts": 10,
        "num_prompt_layers": 5,
        "normalize": "pre-head",
        "head_mode": "concat",
        "max_tasks": args.max_tasks,
        "adapter_mode": args.adapter_mode,
        "adapter_bottleneck_dim": args.adapter_bottleneck_dim,
        "adapter_bottleneck_dims_per_task": (
            list(args.adapter_bottleneck_dims_per_task)
            if args.adapter_bottleneck_dims_per_task is not None
            else [args.adapter_bottleneck_dim] * len(TASK_SIZES)
        ),
        "adapter_layer_indices": list(args.adapter_layer_indices),
        "adapter_residual_scale": args.adapter_residual_scale,
        "adapter_residual_gate_mode": args.adapter_residual_gate_mode,
        "adapter_residual_gate_initial_value": args.adapter_residual_scale,
        "adapter_activation": args.adapter_activation,
        "adapter_task_initialization": args.adapter_task_init,
        "adapter_target": (
            "frozen_image_tokens_for_selector"
            if args.adapter_mode == "image_token"
            else "task_lane_tokens" if args.adapter_mode == "task_lane" else None
        ),
        "adapter_image_token_scope": (
            "block_ln1_cls_plus_patch_tokens"
            if args.adapter_mode == "image_token" else None
        ),
        "adapter_writes_back_to_frozen_visual_stream": False,
        "adapter_initialization_rng": (
            "forked_global_state" if args.adapter_mode != "disabled" else None
        ),
        "adapter_learning_rate": (
            args.adapter_learning_rate if args.adapter_mode != "disabled" else None
        ),
        "adapter_weight_decay": (
            (
                args.weight_decay
                if args.adapter_weight_decay is None
                else args.adapter_weight_decay
            )
            if args.adapter_mode != "disabled" else None
        ),
        "adapter_regularization": args.adapter_regularization,
        "adapter_regularization_fraction": args.adapter_regularization_fraction,
        "adapter_regularization_calibration_updates": (
            args.adapter_regularization_calibration_updates
            if args.adapter_regularization != "none" else None
        ),
        "adapter_regularization_scaling": (
            "fixed_per_task_weight_calibrated_on_successful_updates"
            if args.adapter_regularization != "none" else None
        ),
        "clip_checkpoint": str(args.clip_checkpoint.resolve()),
        "clip_checkpoint_sha256": OPENAI_VIT_B16_SHA256,
        "data_root": str(dataset_parent / "EMOTIC"),
        "git": metadata,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "trainable_parameters": (
            lane_parameters + classifier_parameters + adapter_parameters_per_task
        ),
        "total_method_parameters": (
            lane_parameters + classifier_parameters + adapter_parameters
        ),
        "task_lane_parameters": lane_parameters,
        "classifier_parameters": classifier_parameters,
        "adapter_parameters": adapter_parameters,
        "adapter_parameters_per_task": adapter_parameters_per_task,
        "adapter_parameter_counts_per_task": adapter_parameter_counts_per_task,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    task_rows: List[TaskMetrics] = []
    training_history: Dict[str, object] = {}
    start = time.time()
    for task_id in range(args.max_tasks):
        print(f"begin_task={task_id}", flush=True)
        model.activate_task(task_id)
        train_view = dataset_view(train_source, task_indices(task_id))
        val_view = dataset_view(val_source, task_indices(task_id))
        reporting_view = dataset_view(reporting_source, seen_indices(task_id))
        train_loader = DataLoader(
            train_view, batch_size=args.train_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=False,
        )
        val_loader = DataLoader(
            val_view, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False, drop_last=False,
        )
        reporting_loader = DataLoader(
            reporting_view, batch_size=args.eval_batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False, drop_last=False,
        )
        history = train_task(
            model, train_loader, val_loader, device, task_id,
            args.epochs, learning_rate, args.weight_decay,
            args.temperature, amp,
            adapter_learning_rate=args.adapter_learning_rate,
            adapter_weight_decay=args.adapter_weight_decay,
            loss_mode=args.training_loss_mode,
            loss_routing=args.loss_routing,
            asl_gamma_neg=args.asl_gamma_neg,
            asl_gamma_pos=args.asl_gamma_pos,
            asl_clip=args.asl_clip,
            asl_eps=args.asl_eps,
            optimizer_updates_per_task=args.optimizer_updates_per_task,
            adapter_regularization=args.adapter_regularization,
            adapter_regularization_fraction=args.adapter_regularization_fraction,
            adapter_regularization_calibration_updates=(
                args.adapter_regularization_calibration_updates
            ),
        )
        row = evaluate(
            model, reporting_loader, device, task_id, args.threshold, amp
        )
        task_rows.append(row)
        training_history[str(task_id)] = history
        if save_checkpoints:
            torch.save(
                {
                    "model": model.state_dict(), "task_id": task_id,
                    "config": config,
                },
                output / "checkpoints" / f"task{task_id}.pth",
            )
        (output / "task_metrics.json").write_text(
            json.dumps([asdict(item) for item in task_rows], indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "training_history.json").write_text(
            json.dumps(training_history, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"task={task_id} {args.reporting_split}_mAP={row.mAP:.6f} "
            f"{args.reporting_split}_cF1={row.cF1:.6f} "
            f"{args.reporting_split}_oF1={row.oF1:.6f}",
            flush=True,
        )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "method": METHOD_NAME,
        "protocol_id": PROTOCOL_ID,
        "track": TRACK,
        "seed": args.seed,
        "elapsed_seconds": time.time() - start,
        "completed_epochs": sum(
            len(history) for history in training_history.values()
        ),
        "completed_optimizer_updates": int(sum(
            row["optimizer_steps"]
            for history in training_history.values()
            for row in history
        )),
        "metrics": summarize_tasks(task_rows),
        "task_metrics": [asdict(item) for item in task_rows],
        "adapter_residual_gate_final_values": (
            list(model.adapter_bank.gate_values())
            if model.adapter_bank is not None else None
        ),
        "config": config,
    }
    (output / "seed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["metrics"], indent=2), flush=True)
    print("MULTI_LANE_TRACK_A_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
