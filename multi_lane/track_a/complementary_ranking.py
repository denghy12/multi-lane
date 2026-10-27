"""Teacher-free ranking supervision with view-specific backward paths.

The prediction values use the existing feature fusion and shared linear head.
Only the Person or reliable-Face feature edge receives *additional* gradients;
the caller retains the original BCE/Adapter-ASL objectives independently.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


VIEW_NAMES = ("full", "person", "face")


def _one_lane_features(
    features: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if set(features) != set(VIEW_NAMES):
        raise ValueError("Complementary ranking requires Full/Person/Face features")
    canonical = {}
    for name in VIEW_NAMES:
        value = features[name]
        if value.ndim == 2:
            value = value.unsqueeze(1)
        if value.ndim != 3 or value.shape[1] != 1:
            raise ValueError("Complementary ranking requires one current task lane")
        canonical[name] = value
    reference = canonical["full"]
    if any(value.shape != reference.shape or value.device != reference.device
           or value.dtype != reference.dtype for value in canonical.values()):
        raise ValueError("View feature shapes, devices and dtypes must agree")
    return canonical


def _view_gradient_fused_logits(
    features: Mapping[str, torch.Tensor],
    head: nn.Linear,
    weights: torch.Tensor,
    differentiable_view: str,
) -> torch.Tensor:
    # Match TaskwiseViewFusion.forward exactly: stack B x lane x view x D,
    # multiply weights, then sum.  Do not rewrite as weighted view logits.
    stacked = torch.stack([
        features[name] if name == differentiable_view else features[name].detach()
        for name in VIEW_NAMES
    ], dim=2)
    fused = torch.sum(
        stacked * weights.detach().unsqueeze(-1).to(stacked.dtype), dim=2
    )
    bias = None if head.bias is None else head.bias.detach()
    return F.linear(fused, head.weight.detach(), bias)[:, 0]


def complementary_ranking_objective(
    *,
    view_features: Mapping[str, torch.Tensor],
    head: nn.Linear,
    fusion_weights: torch.Tensor,
    view_logits: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    current_class_indices: Sequence[int],
    face_reliable: torch.Tensor,
    full_conditioned: bool = True,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Rank natural positive/negative people for each current emotion class.

    ``targets`` is B x current_classes; feature inputs are B x 1 x D (or B x D).
    Both terms use the number of *all* natural class pairs as denominator;
    Face contributes only when both endpoints are reliable.  The two terms
    are averaged, including the zero Face term when no Face pair is usable.
    Full scores only set detached, bounded difficulty weights in [1, 2].

    The returned metrics contain ordinary Python numbers and are suitable
    for JSON logs.  Neither RNG nor parameters/optimizer state are mutated.
    """
    if not isinstance(head, nn.Linear):
        raise TypeError("The strong-B0 shared classification head must be linear")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Ranking temperature must be finite and positive")
    current = tuple(int(index) for index in current_class_indices)
    if not current or len(set(current)) != len(current):
        raise ValueError("Current class indices must be nonempty and distinct")
    if min(current) < 0 or max(current) >= head.out_features:
        raise ValueError("Current class index is outside the shared head")
    features = _one_lane_features(view_features)
    batch, _, width = features["full"].shape
    if head.in_features != width:
        raise ValueError("Feature width does not match the classification head")
    if targets.shape != (batch, len(current)):
        raise ValueError("Targets must contain the current task classes only")
    if targets.device != features["full"].device:
        raise ValueError("Targets and features must be on the same device")
    if face_reliable.shape != (batch,) or face_reliable.dtype != torch.bool:
        raise ValueError("Face reliability must be a boolean vector with one entry per person")
    if face_reliable.device != features["full"].device:
        raise ValueError("Face reliability and features must be on the same device")
    weights = fusion_weights.unsqueeze(1) if fusion_weights.ndim == 2 else fusion_weights
    if weights.shape != (batch, 1, 3) or weights.device != features["full"].device:
        raise ValueError("Fusion weights must be B x 1 x 3 on the feature device")
    if "full" not in view_logits or view_logits["full"].shape != (batch, head.out_features):
        raise ValueError("Current Full logits do not match the classification head")
    if not bool(((targets == 0) | (targets == 1)).all()):
        raise ValueError("Ranking requires binary hard-label targets")

    person_logits = _view_gradient_fused_logits(features, head, weights, "person")
    face_logits = _view_gradient_fused_logits(features, head, weights, "face")
    # Compute logistic margins in FP32 under AMP, as for existing ranking/BCE.
    person_scores = person_logits[:, list(current)].float()
    face_scores = face_logits[:, list(current)].float()
    full_scores = view_logits["full"][:, list(current)].detach().float()
    positive = targets == 1
    negative = targets == 0
    pairs = positive[:, None, :] & negative[None, :, :]
    reliable_pairs = pairs & face_reliable[:, None, None] & face_reliable[None, :, None]
    pair_count_tensor = pairs.sum()
    face_pair_count_tensor = reliable_pairs.sum()
    denominator = pair_count_tensor.clamp_min(1).to(dtype=person_scores.dtype)
    person_margin = person_scores[:, None, :] - person_scores[None, :, :]
    face_margin = face_scores[:, None, :] - face_scores[None, :, :]
    if full_conditioned:
        full_margin = full_scores[:, None, :] - full_scores[None, :, :]
        difficulty = 1.0 + torch.sigmoid(-full_margin / temperature)
    else:
        difficulty = torch.ones_like(person_margin)
    person_total = (
        difficulty * F.softplus(-person_margin / temperature) * pairs
    ).sum()
    face_total = (
        difficulty * F.softplus(-face_margin / temperature) * reliable_pairs
    ).sum()
    person_loss = person_total / denominator
    face_loss = face_total / denominator
    loss = (person_loss + face_loss) * 0.5

    # One transfer for diagnostics rather than synchronizing every metric.
    values = torch.stack((
        pair_count_tensor.to(dtype=person_scores.dtype),
        face_pair_count_tensor.to(dtype=person_scores.dtype),
        pairs.any(dim=(0, 1)).sum().to(dtype=person_scores.dtype),
        face_reliable.sum().to(dtype=person_scores.dtype),
        person_loss.detach(), face_loss.detach(), loss.detach(),
        (difficulty * pairs).sum().detach() / denominator,
        (person_logits.detach().float() - face_logits.detach().float()).abs().max()
        if batch else loss.detach() * 0,
    )).detach().cpu().tolist()
    metrics = dict(
        pair_count=int(values[0]), person_pair_count=int(values[0]),
        face_pair_count=int(values[1]), active_class_count=int(values[2]),
        reliable_face_people=int(values[3]),
        person_ranking_loss=float(values[4]), face_ranking_loss=float(values[5]),
        complementary_ranking_loss=float(values[6]),
        mean_pair_difficulty=float(values[7]),
        extra_branch_logits_max_abs_difference=float(values[8]),
        full_conditioned=bool(full_conditioned), temperature=float(temperature),
        face_denominator="all_natural_class_pairs",
    )
    if not math.isfinite(metrics["complementary_ranking_loss"]):
        raise FloatingPointError("Non-finite complementary ranking loss")
    return loss, metrics
