"""Task-routed transformer adapters for MULTI-LANE lane tokens."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
from torch import nn


class TransformerBlockAdapter(nn.Module):
    """A zero-initialized bottleneck residual branch."""

    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("Adapter dimensions must be positive")
        if activation not in {"relu", "gelu"}:
            raise ValueError("Adapter activation must be relu or gelu")
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(value)))


class TaskLaneTransformerAdapterBank(nn.Module):
    """Preallocated task-specific adapters, routed by MULTI-LANE lane id."""

    def __init__(
        self,
        num_tasks: int,
        hidden_dim: int,
        bottleneck_dim: int,
        layer_indices: Sequence[int],
        residual_scale: float = 0.1,
        activation: str = "relu",
        task_initialization: str = "independent",
    ) -> None:
        super().__init__()
        layers = tuple(int(index) for index in layer_indices)
        if num_tasks <= 0:
            raise ValueError("Adapter task count must be positive")
        if not layers or any(index < 0 for index in layers):
            raise ValueError("Adapter layer indices must be non-negative")
        if len(set(layers)) != len(layers):
            raise ValueError("Adapter layer indices must be unique")
        if residual_scale < 0:
            raise ValueError("Adapter residual scale must be non-negative")
        if task_initialization not in {"independent", "copy_previous"}:
            raise ValueError(
                "Adapter task initialization must be independent or copy_previous"
            )
        self.num_tasks = int(num_tasks)
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.layer_indices = tuple(sorted(layers))
        self.residual_scale = float(residual_scale)
        self.activation_name = activation
        self.task_initialization = task_initialization
        self._current_task_id = -1

        self.task_adapters = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        str(layer_id): TransformerBlockAdapter(
                            hidden_dim=self.hidden_dim,
                            bottleneck_dim=self.bottleneck_dim,
                            activation=activation,
                        )
                        for layer_id in self.layer_indices
                    }
                )
                for _ in range(self.num_tasks)
            ]
        )
        self.requires_grad_(False)

    @property
    def current_task_id(self) -> int:
        return self._current_task_id

    def activate_task(self, task_id: int) -> None:
        if not 0 <= task_id < self.num_tasks:
            raise ValueError("Adapter task id is outside the protocol")
        self.requires_grad_(False)
        if task_id > 0 and self.task_initialization == "copy_previous":
            self.task_adapters[task_id].load_state_dict(
                self.task_adapters[task_id - 1].state_dict()
            )
        self.task_adapters[task_id].requires_grad_(True)
        self._current_task_id = int(task_id)

    def restore_task(self, task_id: int) -> None:
        if not -1 <= task_id < self.num_tasks:
            raise ValueError("Restored adapter task id is invalid")
        self.requires_grad_(False)
        if task_id >= 0:
            self.task_adapters[task_id].requires_grad_(True)
        self._current_task_id = int(task_id)

    def delta_for_layer(
        self,
        layer_id: int,
        normalized_lane_tokens: torch.Tensor,
        lane_ids: Sequence[int],
    ) -> torch.Tensor:
        if layer_id not in self.layer_indices:
            return torch.zeros_like(normalized_lane_tokens)
        if normalized_lane_tokens.ndim != 4:
            raise ValueError("Lane tokens must have shape [lanes, batch, tokens, width]")
        if normalized_lane_tokens.shape[0] != len(lane_ids):
            raise ValueError("Lane ids and lane-token count differ")
        if normalized_lane_tokens.shape[-1] != self.hidden_dim:
            raise ValueError("Lane-token width differs from adapter width")
        deltas = []
        for lane_position, task_id in enumerate(lane_ids):
            if not 0 <= int(task_id) < self.num_tasks:
                raise ValueError("Lane id is outside the adapter bank")
            adapter = self.task_adapters[int(task_id)][str(layer_id)]
            deltas.append(adapter(normalized_lane_tokens[lane_position]))
        return torch.stack(deltas, dim=0) * self.residual_scale

    def active_parameters(self) -> Iterable[nn.Parameter]:
        if self._current_task_id < 0:
            return iter(())
        return self.task_adapters[self._current_task_id].parameters()

    def parameter_names(self) -> Tuple[str, ...]:
        return tuple(name for name, _ in self.named_parameters())

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def per_task_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.task_adapters[0].parameters())
