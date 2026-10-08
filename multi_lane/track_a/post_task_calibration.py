"""Fit task-local routes on cached B0 features without updating B0 weights."""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .paired_transforms import move_model_inputs


@contextmanager
def isolated_rng(device: torch.device):
    """Extra caching/calibration must not change the next task's training RNG."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def parameter_hash(parameters: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, value in parameters:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def base_hash(model) -> str:
    return parameter_hash((name, value) for name, value in model.named_parameters()
                          if not name.startswith("parax_bank."))


def center_hash(bank) -> str:
    return parameter_hash((name, value) for name, value in bank.named_parameters()
                          if not name.startswith(("routers.", "task_projections.")))


def cache_current_features(model, loader: Iterable, device: torch.device, amp: bool) -> Dict:
    """Cache train-split features only, using deterministic evaluation crops."""
    runtime, training = model.parax_runtime_enabled, model.training
    names = model.view_fusion_module.view_names
    rows = {name: [] for name in names}
    targets, reliable, sample_ids = [], [], []
    model.eval()
    model.set_parax_runtime_enabled(False)
    try:
        with torch.no_grad():
            for images, labels, identifiers in loader:
                reliable.append(images["face_reliable"].bool().cpu())
                images = move_model_inputs(images, device)
                with torch.cuda.amp.autocast(enabled=amp):
                    _, features = model.encode_lanes_with_views(images, all_seen_lanes=False)
                for name in names:
                    rows[name].append(features[name].float().cpu())
                targets.append(labels.float().cpu())
                sample_ids.extend(str(value) for value in identifiers)
    finally:
        model.set_parax_runtime_enabled(runtime)
        model.train(training)
    return {"features": {name: torch.cat(values) for name, values in rows.items()},
            "targets": torch.cat(targets), "face_reliable": torch.cat(reliable),
            "sample_ids": sample_ids, "source_split": "train", "task_id": model.current_task_id}


def fit_cached_routes(model, cache: Dict, current_classes: Sequence[int], device: torch.device,
                      epochs: int, learning_rate: float, batch_size: int = 64,
                      consistency_weight: float = 0.1, auxiliary_weight: float = 0.1) -> Dict:
    """Only current task projection/Router learns; the shared center is fixed."""
    if cache.get("source_split") != "train" or cache["task_id"] != model.current_task_id:
        raise ValueError("Route calibration requires current-task train features")
    bank, task = model.parax_bank, model.current_task_id
    if not model.parax_staged_mode or not bank.freeze_center_from_start:
        raise ValueError("Cached fitting requires staged frozen-center ParaX")
    classes = tuple(int(value) for value in current_classes)
    if cache["targets"].shape[1] != len(classes):
        raise ValueError("Cached targets must contain only current task classes")
    base_before, center_before = base_hash(model), center_hash(bank)
    old_before = parameter_hash((name, value) for name, value in bank.named_parameters()
                                if name.startswith(tuple(f"routers.{t}_" for t in range(task)) +
                                                   tuple(f"task_projections.{t}." for t in range(task))))
    bank.restore_task(task)
    parameters = list(bank.active_parameters())
    if any(value.requires_grad for value in (bank.expert_a, bank.expert_b)):
        raise RuntimeError("Shared expert center must remain frozen")
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    names = model.view_fusion_module.view_names
    weight = model.head.weight.detach()[list(classes)]
    bias = model.head.bias.detach()[list(classes)] if model.head.bias is not None else None
    history = []
    max_initial_difference = 0.0
    for epoch in range(epochs):
        totals = {"loss": 0.0, "bce": 0.0, "consistency": 0.0}
        ratios = {name: 0.0 for name in names}
        gate_sum = {name: torch.zeros(bank.num_experts) for name in names}
        gate_entropy = {name: 0.0 for name in names}
        gate_square_sum = {name: torch.zeros(bank.num_experts) for name in names}
        grad_sq, steps, samples = 0.0, 0, 0
        for indices in torch.randperm(len(cache["targets"])).split(batch_size):
            features = {name: cache["features"][name][indices].to(device) for name in names}
            labels = cache["targets"][indices].to(device)
            reliable = cache["face_reliable"][indices].to(device)
            bank.reset_forward_diagnostics()
            routed, gates = {}, {}
            for name in names:
                value, gate = bank(bank.layer_indices[0], features[name], name, task_id=task)
                if epoch == 0 and steps == 0:
                    max_initial_difference = max(max_initial_difference, float((value - features[name]).abs().max()))
                ratios[name] += bank.last_raw_residual_ratio * len(indices)
                gates[name] = gate
                routed[name] = F.normalize(value, dim=-1)
            fused, _ = model.view_fusion_module(routed, [task], reliable)
            reference, _ = model.view_fusion_module(features, [task], reliable)
            logits = F.linear(fused[:, 0], weight, bias)
            reference_logits = F.linear(reference[:, 0], weight, bias)
            bce = F.binary_cross_entropy_with_logits(logits, labels)
            consistency = F.mse_loss(logits, reference_logits)
            # Face auxiliary loss and consistency use reliable Face only.
            auxiliary, view_consistency = [], []
            for name in names:
                valid = reliable if name == "face" else torch.ones_like(reliable)
                if bool(valid.any()):
                    view_logits = F.linear(routed[name][valid, 0], weight, bias)
                    original_logits = F.linear(features[name][valid, 0], weight, bias)
                    auxiliary.append(F.binary_cross_entropy_with_logits(view_logits, labels[valid]))
                    view_consistency.append(F.mse_loss(view_logits, original_logits))
            bce = bce + auxiliary_weight * torch.stack(auxiliary).mean()
            consistency = 0.5 * (consistency + torch.stack(view_consistency).mean())
            loss = bce + consistency_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for parameter in parameters:
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError("Non-finite route calibration gradient")
                    grad_sq += float(parameter.grad.float().square().sum())
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            for key, value in (("loss", loss), ("bce", bce), ("consistency", consistency)):
                totals[key] += float(value.detach()) * len(indices)
            for name in names:
                gate = gates[name].detach().cpu()
                gate_sum[name] += gate.sum(dim=0)
                gate_square_sum[name] += gate.square().sum(dim=0)
                gate_entropy[name] += float(-(gate.clamp_min(1e-12) * gate.clamp_min(1e-12).log()).sum())
            samples += len(indices)
            steps += 1
        row = {"epoch": epoch, "optimizer_steps": steps,
               "learning_rate": optimizer.param_groups[0]["lr"],
               **{key: value / samples for key, value in totals.items()},
               "gradient_rms_norm_before_clip": (grad_sq / steps) ** 0.5,
               "views": {name: {"residual_ratio": ratios[name] / samples,
                                "gate_mean": (gate_sum[name] / samples).tolist(),
                                "gate_std": (gate_square_sum[name] / samples - (gate_sum[name] / samples).square()).clamp_min(0).sqrt().tolist(),
                                "gate_entropy": gate_entropy[name] / samples} for name in names}}
        history.append(row)
        print(f"route_calibration task={task} epoch={epoch + 1} loss={row['loss']:.6f} steps={steps}", flush=True)
        scheduler.step()
    base_after, center_after = base_hash(model), center_hash(bank)
    old_after = parameter_hash((name, value) for name, value in bank.named_parameters()
                               if name.startswith(tuple(f"routers.{t}_" for t in range(task)) +
                                                  tuple(f"task_projections.{t}." for t in range(task))))
    if base_before != base_after or center_before != center_after or old_before != old_after:
        raise RuntimeError("Route calibration changed base, center, or old-task parameters")
    bank.requires_grad_(False)
    for parameter in bank.parameters():
        parameter.grad = None
    return {"history": history, "samples": len(cache["targets"]),
            "initial_output_max_difference": max_initial_difference,
            "base_hash_before": base_before, "base_hash_after": base_after,
            "center_hash_before": center_before, "center_hash_after": center_after,
            "old_route_hash_before": old_before, "old_route_hash_after": old_after}
