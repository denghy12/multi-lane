"""CLIP Face expert complemented by frozen facial-expression residuals."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .model import MultiLaneModel, ModelInputs


class TaskExpressionResidualBank(nn.Module):
    def __init__(
        self, task_count: int, input_dim: int, output_dim: int,
        rank: int = 32, scale: float = 0.1,
    ) -> None:
        super().__init__()
        if min(task_count, input_dim, output_dim, rank) <= 0:
            raise ValueError("Expression residual dimensions must be positive")
        if not 0 < scale <= 1:
            raise ValueError("Expression residual scale must be in (0, 1]")
        self.task_modules = nn.ModuleList()
        for _ in range(task_count):
            module = nn.Sequential(
                nn.Linear(input_dim, rank), nn.ReLU(), nn.Linear(rank, output_dim)
            )
            nn.init.xavier_uniform_(module[0].weight)
            nn.init.zeros_(module[0].bias)
            nn.init.zeros_(module[2].weight)
            nn.init.zeros_(module[2].bias)
            self.task_modules.append(module)
        self.scale = float(scale)
        self._current_task_id = -1
        self.requires_grad_(False)

    def activate_task(self, task_id: int) -> None:
        if not 0 <= task_id < len(self.task_modules):
            raise ValueError("Expression residual task id is outside the protocol")
        self.requires_grad_(False)
        self.task_modules[task_id].requires_grad_(True)
        self._current_task_id = int(task_id)

    def forward(
        self, descriptor: torch.Tensor, task_ids: Sequence[int]
    ) -> torch.Tensor:
        return torch.stack(
            [self.scale * self.task_modules[int(task_id)](descriptor)
             for task_id in task_ids],
            dim=1,
        )

    def active_parameters(self) -> Iterable[nn.Parameter]:
        if self._current_task_id < 0:
            return iter(())
        return iter(self.task_modules[self._current_task_id].parameters())

    def per_task_parameter_count(self) -> int:
        return sum(value.numel() for value in self.task_modules[0].parameters())


class FaceExpressionResidualModel(MultiLaneModel):
    """Preserve the CLIP lane and add a task-local expression feature residual."""

    def __init__(
        self,
        expression_encoder: nn.Module,
        expression_classifier: nn.Linear,
        expression_feature_dim: int,
        task_sizes: Sequence[int],
        residual_rank: int = 32,
        residual_scale: float = 0.1,
        **multi_lane_kwargs,
    ) -> None:
        super().__init__(task_sizes=task_sizes, **multi_lane_kwargs)
        self.expression_encoder = expression_encoder
        self.expression_classifier = expression_classifier
        self.expression_encoder.requires_grad_(False)
        self.expression_classifier.requires_grad_(False)
        self.expression_feature_dim = int(expression_feature_dim)
        self.expression_class_count = int(expression_classifier.out_features)
        descriptor_dim = self.expression_feature_dim + self.expression_class_count
        with torch.random.fork_rng(devices=[]):
            self.expression_residual_bank = TaskExpressionResidualBank(
                len(task_sizes), descriptor_dim, self.output_dim,
                residual_rank, residual_scale,
            )

    def activate_task(self, task_id: int) -> None:
        super().activate_task(task_id)
        self.expression_residual_bank.activate_task(task_id)

    def train(self, mode: bool = True):
        super().train(mode)
        self.expression_encoder.eval()
        self.expression_classifier.eval()
        return self

    def _expression_descriptor(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.expression_encoder(images)
            if embedding.ndim > 2:
                embedding = torch.flatten(embedding, 1)
            embedding = embedding.float()
            logits = self.expression_classifier(embedding)
            embedding = F.layer_norm(embedding, (self.expression_feature_dim,))
            logits = F.layer_norm(logits.float(), (self.expression_class_count,))
            descriptor = torch.cat([embedding, logits], dim=1)
        if not torch.isfinite(descriptor).all():
            raise RuntimeError("Expression descriptor contains non-finite values")
        return descriptor

    def encode_lanes(
        self, images: ModelInputs, all_seen_lanes: bool
    ) -> torch.Tensor:
        if not isinstance(images, dict) or set(images) != {"clip", "expression"}:
            raise ValueError("Expression residual model requires clip/expression inputs")
        clip_features = self._encode_single_lanes(images["clip"], all_seen_lanes)
        descriptor = self._expression_descriptor(images["expression"])
        lane_ids = self._lane_ids(all_seen_lanes)
        residual = self.expression_residual_bank(descriptor, lane_ids)
        # The zero-initialized branch is mathematically identical to the old
        # normalized CLIP Face lane at initialization.  Its fixed 0.1 scale
        # bounds the later departure without renormalizing the anchor.
        return clip_features + residual

    def encode_lanes_with_views(
        self, images: ModelInputs, all_seen_lanes: bool
    ) -> Tuple[torch.Tensor, dict]:
        return self.encode_lanes(images, all_seen_lanes), {}

    def fusion_optimizer_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.expression_residual_bank.active_parameters()

    def assert_expression_frozen(self) -> None:
        names = [
            name for name, value in [
                *self.expression_encoder.named_parameters(),
                *self.expression_classifier.named_parameters(),
            ] if value.requires_grad
        ]
        if names:
            raise RuntimeError("Expression representation unexpectedly trainable")
