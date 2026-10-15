"""Task-Forward MoE adapters parallel to the frozen FFN.

Each continual task owns one expert pool. Full, Person and Face share those
experts, but have separate content-conditioned noisy Top-2 routers. Completed
tasks are frozen as whole pools, including routers and expert parameters.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class _Expert(nn.Module):
    def __init__(self, width: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        self.dropout = float(dropout)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.up(F.dropout(F.relu(self.down(tokens)), p=self.dropout,
                                 training=self.training))


class _LayerPool(nn.Module):
    def __init__(self, width: int, bottleneck: int, experts: int,
                 dropout: float) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            _Expert(width, bottleneck, dropout) for _ in range(experts)
        )
        self.gates = nn.ParameterList(
            nn.Parameter(torch.zeros(width, experts)) for _ in range(3)
        )
        self.noise = nn.ParameterList(
            nn.Parameter(torch.zeros(width, experts)) for _ in range(3)
        )


class TaskForwardMoEBank(nn.Module):
    """Sparse image-level routing over complete-output token adapters."""

    view_names = ("full", "person", "face")

    def __init__(self, num_tasks: int, hidden_dim: int, bottleneck_dim: int,
                 layer_indices: Sequence[int], residual_scale: float,
                 num_experts: int = 22, top_k: int = 2,
                 dropout: float = 0.1, noisy_gating: bool = True) -> None:
        super().__init__()
        if num_tasks <= 0 or hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("MoE tasks, width and bottleneck must be positive")
        if num_experts < 2 or not 1 <= top_k <= num_experts:
            raise ValueError("MoE expert count or top-k is invalid")
        if not layer_indices or len(set(layer_indices)) != len(layer_indices):
            raise ValueError("MoE layer indices must be nonempty and unique")
        if not 0 <= dropout < 1 or residual_scale < 0:
            raise ValueError("MoE dropout or residual scale is invalid")
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.residual_scale = float(residual_scale)
        self.noisy_gating = bool(noisy_gating)
        self.tasks = nn.ModuleList([
            nn.ModuleList([
                _LayerPool(hidden_dim, bottleneck_dim, num_experts, dropout)
                for _ in self.layer_indices
            ]) for _ in range(num_tasks)
        ])
        self._current_task_id = -1
        self.register_buffer(
            "route_counts",
            torch.zeros(num_tasks, len(self.layer_indices), 3, num_experts),
            persistent=False,
        )
        self.register_buffer(
            "route_samples",
            torch.zeros(num_tasks, len(self.layer_indices), 3),
            persistent=False,
        )
        self.requires_grad_(False)

    def train(self, mode: bool = True) -> "TaskForwardMoEBank":
        super().train(mode)
        # The parent model trains while older task lanes may still be read.
        # Their dropout and noisy routers must remain in inference mode.
        for task_id, task in enumerate(self.tasks):
            if task_id != self._current_task_id:
                task.eval()
        return self

    def activate_task(self, task_id: int) -> None:
        if task_id != self._current_task_id + 1 or task_id >= len(self.tasks):
            raise ValueError("MoE tasks must activate sequentially")
        if self._current_task_id >= 0:
            self.tasks[self._current_task_id].requires_grad_(False)
            self.tasks[self._current_task_id].eval()
        self.tasks[task_id].requires_grad_(True)
        self.tasks[task_id].train(self.training)
        self._current_task_id = int(task_id)

    def restore_task(self, task_id: int) -> None:
        if not -1 <= task_id < len(self.tasks):
            raise ValueError("Invalid restored MoE task")
        self.requires_grad_(False)
        self._current_task_id = int(task_id)

    def active_parameters(self) -> Iterable[nn.Parameter]:
        if self._current_task_id >= 0:
            yield from self.tasks[self._current_task_id].parameters()

    def parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.named_parameters())

    def gate_values(self) -> tuple[float, ...]:
        return (self.residual_scale,) * len(self.tasks)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def per_task_parameter_counts(self) -> tuple[int, ...]:
        return tuple(sum(parameter.numel() for parameter in task.parameters())
                     for task in self.tasks)

    def per_task_parameter_count(self) -> int:
        return self.per_task_parameter_counts()[0]

    def reset_route_statistics(self) -> None:
        self.route_counts.zero_()
        self.route_samples.zero_()

    def reset_auxiliary_metrics(self) -> None:
        """The MoE uses only the existing multi-label training objective."""

    def route_statistics(self) -> Dict[str, object]:
        counts = self.route_counts.detach().cpu()
        samples = self.route_samples.detach().cpu()
        return {
            str(task): {
                str(layer): {
                    view: {
                        "samples": int(samples[task, layer_index, view_index]),
                        "expert_topk_count": counts[task, layer_index, view_index].tolist(),
                    }
                    for view_index, view in enumerate(self.view_names)
                }
                for layer_index, layer in enumerate(self.layer_indices)
            }
            for task in range(self._current_task_id + 1)
        }

    def _route(self, cls: torch.Tensor, pool: _LayerPool,
               view_index: int) -> torch.Tensor:
        # Router logits use the post-Drop-and-Replace task CLS. Noise is
        # training-only, matching the official noisy Top-k convention.
        clean = cls.float() @ pool.gates[view_index].float()
        if pool.training and self.noisy_gating:
            std = F.softplus(cls.float() @ pool.noise[view_index].float()) + 1e-2
            logits = clean + torch.randn_like(clean) * std
        else:
            logits = clean
        values, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(values, dim=-1)
        return torch.zeros_like(logits).scatter(-1, indices, weights)

    def delta_for_layer(self, layer_id: int, task_lane_tokens: torch.Tensor,
                        lane_ids: Sequence[int], image_view: str) -> torch.Tensor:
        if layer_id not in self.layer_indices or self.residual_scale == 0:
            return torch.zeros_like(task_lane_tokens)
        if image_view not in self.view_names:
            raise ValueError(f"Unknown MoE image view: {image_view}")
        if task_lane_tokens.ndim != 4 or task_lane_tokens.shape[0] != len(lane_ids):
            raise ValueError("MoE lane tensor must have shape [lane,batch,token,width]")
        layer_index = self.layer_indices.index(layer_id)
        view_index = self.view_names.index(image_view)
        lane_outputs = []
        for lane_index, task_id in enumerate(lane_ids):
            if not 0 <= task_id <= self._current_task_id:
                raise ValueError("MoE cannot access a future task")
            tokens = task_lane_tokens[lane_index]
            pool = self.tasks[task_id][layer_index]
            gates = self._route(tokens[:, 0], pool, view_index)
            if not self.training:
                with torch.no_grad():
                    self.route_counts[task_id, layer_index, view_index].add_(
                        (gates > 0).sum(0).to(self.route_counts.dtype)
                    )
                    self.route_samples[task_id, layer_index, view_index] += len(tokens)
            combined = torch.zeros_like(tokens)
            for expert_id, expert in enumerate(pool.experts):
                selected = torch.nonzero(gates[:, expert_id] > 0,
                                         as_tuple=True)[0]
                if selected.numel() == 0:
                    continue
                result = expert(tokens.index_select(0, selected))
                weight = gates.index_select(0, selected)[:, expert_id]
                combined = combined.index_add(
                    0, selected,
                    result * weight.to(result.dtype).view(-1, 1, 1),
                )
            lane_outputs.append(combined)
        return self.residual_scale * torch.stack(lane_outputs, dim=0)
