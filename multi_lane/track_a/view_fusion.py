"""Task-isolated feature-level fusion for Full/Person/Face views."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
from torch import nn


class TaskwiseViewFusion(nn.Module):
    """Fuse matching MULTI-LANE task features without rewriting old lanes.

    A router belongs to exactly one incremental task.  During task ``t`` only
    router ``t`` is trainable; inference applies router ``k`` only to lane
    ``k``.  This mirrors selectors/prompts/Adapters and prevents a later task
    from changing fusion for previously introduced classes.
    """

    MODES = (
        "disabled",
        "fixed_three_view",
        "soft_three_view",
        "soft_full_person",
    )
    THREE_VIEW_PRIOR = (0.64, 0.16, 0.20)
    INVALID_FACE_PRIOR = (0.80, 0.20, 0.0)
    TWO_VIEW_PRIOR = (0.80, 0.20)

    def __init__(
        self,
        num_tasks: int,
        feature_dim: int,
        mode: str,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("Unknown taskwise view-fusion mode")
        if min(int(num_tasks), int(feature_dim), int(hidden_dim)) <= 0:
            raise ValueError("View-fusion dimensions must be positive")
        self.mode = mode
        self.num_tasks = int(num_tasks)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.current_task_id = -1
        self.view_names = (
            ("full", "person")
            if mode == "soft_full_person"
            else ("full", "person", "face")
        )
        self.task_routers = nn.ModuleList()
        if mode.startswith("soft_"):
            view_count = len(self.view_names)
            input_dim = self.feature_dim * view_count
            prior = (
                self.TWO_VIEW_PRIOR
                if view_count == 2 else self.THREE_VIEW_PRIOR
            )
            with torch.random.fork_rng(devices=[]):
                for task_id in range(self.num_tasks):
                    # Deterministic task-local initialization without moving
                    # the global data/augmentation RNG stream.
                    torch.manual_seed(41_003 + task_id)
                    router = nn.Sequential(
                        nn.Linear(input_dim, self.hidden_dim),
                        nn.GELU(),
                        nn.Linear(self.hidden_dim, view_count),
                    )
                    nn.init.xavier_uniform_(router[0].weight)
                    nn.init.zeros_(router[0].bias)
                    nn.init.zeros_(router[2].weight)
                    with torch.no_grad():
                        router[2].bias.copy_(
                            torch.tensor([math.log(value) for value in prior])
                        )
                    self.task_routers.append(router)
        self.requires_grad_(False)

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    @property
    def learned(self) -> bool:
        return self.mode.startswith("soft_")

    def restore_task(self, task_id: int) -> None:
        if not -1 <= int(task_id) < self.num_tasks:
            raise ValueError("Invalid view-fusion task")
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        if self.learned and task_id >= 0:
            self.task_routers[int(task_id)].requires_grad_(True)
        self.current_task_id = int(task_id)

    def active_parameters(self) -> Iterable[nn.Parameter]:
        if self.learned and self.current_task_id >= 0:
            yield from self.task_routers[self.current_task_id].parameters()

    def parameter_count_per_task(self) -> int:
        if not self.learned:
            return 0
        return sum(parameter.numel() for parameter in self.task_routers[0].parameters())

    def _validate(
        self,
        features: Mapping[str, torch.Tensor],
        lane_ids: Sequence[int],
        face_reliable: torch.Tensor | None,
    ) -> Tuple[int, int, int]:
        if not self.enabled:
            raise RuntimeError("Disabled view fusion cannot fuse features")
        if set(features) != set(self.view_names):
            raise ValueError("View-fusion feature names do not match the mode")
        shapes = {tuple(value.shape) for value in features.values()}
        if len(shapes) != 1:
            raise ValueError("View-fusion feature shapes differ")
        shape = next(iter(shapes))
        if len(shape) != 3 or shape[1] != len(lane_ids) or shape[2] != self.feature_dim:
            raise ValueError("Invalid view-fusion lane feature shape")
        if len(set(int(value) for value in lane_ids)) != len(lane_ids):
            raise ValueError("View-fusion lane IDs must be unique")
        if any(not 0 <= int(value) < self.num_tasks for value in lane_ids):
            raise ValueError("View-fusion lane ID is outside the protocol")
        if len(self.view_names) == 3:
            if face_reliable is None or face_reliable.shape != (shape[0],):
                raise ValueError("Three-view fusion requires a per-sample Face mask")
        return shape

    def weights(
        self,
        features: Mapping[str, torch.Tensor],
        lane_ids: Sequence[int],
        face_reliable: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, lane_count, _ = self._validate(features, lane_ids, face_reliable)
        reference = features["full"]
        if self.mode == "fixed_three_view":
            valid_prior = reference.new_tensor(self.THREE_VIEW_PRIOR)
            invalid_prior = reference.new_tensor(self.INVALID_FACE_PRIOR)
            return torch.where(
                face_reliable.to(device=reference.device, dtype=torch.bool)[:, None, None],
                valid_prior[None, None, :],
                invalid_prior[None, None, :],
            ).expand(batch, lane_count, -1)

        rows = []
        for lane_position, task_id in enumerate(lane_ids):
            router_parts = []
            for name in self.view_names:
                part = features[name][:, lane_position]
                if name == "face":
                    part = part * face_reliable.to(
                        device=part.device, dtype=part.dtype
                    )[:, None]
                router_parts.append(part)
            router_input = torch.cat(
                router_parts, dim=-1,
            )
            logits = self.task_routers[int(task_id)](router_input.float())
            if len(self.view_names) == 3:
                invalid = ~face_reliable.to(device=logits.device, dtype=torch.bool)
                logits = logits.clone()
                logits[invalid, 2] = torch.finfo(logits.dtype).min
            lane_weights = torch.softmax(logits, dim=-1)
            if not torch.isfinite(lane_weights).all():
                raise FloatingPointError("Non-finite taskwise fusion weights")
            rows.append(lane_weights)
        return torch.stack(rows, dim=1)

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        lane_ids: Sequence[int],
        face_reliable: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = self.weights(features, lane_ids, face_reliable)
        stacked = torch.stack([features[name] for name in self.view_names], dim=2)
        fused = torch.sum(stacked * weights.unsqueeze(-1).to(stacked.dtype), dim=2)
        return fused, weights

    def prior(self, face_reliable: bool = True) -> Dict[str, float]:
        if len(self.view_names) == 2:
            values = self.TWO_VIEW_PRIOR
        elif face_reliable:
            values = self.THREE_VIEW_PRIOR
        else:
            values = self.INVALID_FACE_PRIOR
        return dict(zip(self.view_names, values))
