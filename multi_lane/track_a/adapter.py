"""Task-routed transformer adapters for MULTI-LANE token streams."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
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


class ParaXImageAdapterBank(nn.Module):
    """Token-only ParaX-style image-stream adapter for frozen CLIP blocks.

    The expert matrices are shared by all selected layers and views. Each layer
    owns a small router, so the effective low-rank projection is input-dependent.
    """

    def __init__(
        self, hidden_dim: int, rank: int, num_experts: int, layer_indices: Sequence[int],
        router_hidden: int = 16, residual_scale: float = 0.1,
        level_conditioned: bool = False, static: bool = False,
        initialization: str = "official",
        trainable_components: str = "all",
        output_scale_mode: str = "learnable",
        residual_ratio_cap: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or rank <= 0 or num_experts <= 0:
            raise ValueError("ParaX dimensions must be positive")
        layers = tuple(sorted(set(int(v) for v in layer_indices)))
        if not layers or any(v < 0 for v in layers):
            raise ValueError("ParaX layer indices must be non-negative")
        if residual_scale < 0:
            raise ValueError("ParaX residual scale must be non-negative")
        if initialization not in {"official", "small"}:
            raise ValueError("ParaX initialization must be official or small")
        if trainable_components not in {"all", "router", "experts"}:
            raise ValueError("ParaX trainable components must be all, router, or experts")
        if output_scale_mode not in {"learnable", "fixed"}:
            raise ValueError("ParaX output scale mode must be learnable or fixed")
        if residual_ratio_cap < 0:
            raise ValueError("ParaX residual ratio cap must be non-negative")
        self.hidden_dim = int(hidden_dim)
        self.rank = int(rank)
        self.num_experts = int(num_experts)
        self.layer_indices = layers
        self.residual_scale = float(residual_scale)
        self.level_conditioned = bool(level_conditioned)
        self.static = bool(static)
        self.initialization = initialization
        self.trainable_components = trainable_components
        self.output_scale_mode = output_scale_mode
        self.residual_ratio_cap = float(residual_ratio_cap)
        self.level_names = ("full", "person", "face")
        self.expert_a = nn.Parameter(torch.empty(num_experts, rank, hidden_dim))
        self.expert_b = nn.Parameter(torch.empty(num_experts, hidden_dim, rank))
        nn.init.trunc_normal_(self.expert_a, std=0.02)
        nn.init.trunc_normal_(self.expert_b, std=0.02)
        self.routers = nn.ModuleDict()
        input_dim = hidden_dim + (hidden_dim if self.level_conditioned else 0)
        for layer in self.layer_indices:
            self.routers[str(layer)] = nn.Sequential(
                nn.Linear(input_dim, router_hidden), nn.GELU(),
                nn.Linear(router_hidden, num_experts)
            )
        self.level_embeddings = (
            nn.Parameter(torch.zeros(len(self.level_names), hidden_dim))
            if self.level_conditioned else None
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(rank, rank)
        initial_scale = float(residual_scale) if initialization == "official" else min(float(residual_scale), 1e-3)
        self.output_scale = nn.Parameter(
            torch.tensor(initial_scale), requires_grad=output_scale_mode == "learnable"
        )
        self.requires_grad_(False)

    def activate_task(self, task_id: int) -> None:
        self._set_trainability()

    def restore_task(self, task_id: int) -> None:
        self._set_trainability()

    def _set_trainability(self) -> None:
        self.requires_grad_(False)
        if self.trainable_components in {"all", "experts"}:
            for name in ("expert_a", "expert_b", "norm", "proj"):
                getattr(self, name).requires_grad_(True)
        if self.trainable_components in {"all", "router"} and not self.static:
            self.routers.requires_grad_(True)
            if self.level_embeddings is not None:
                self.level_embeddings.requires_grad_(True)
        self.output_scale.requires_grad_(self.output_scale_mode == "learnable")

    def active_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, layer_id: int, tokens: torch.Tensor, view_name: str = "full",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_id not in self.layer_indices:
            return tokens, tokens.new_zeros((tokens.shape[0], self.num_experts))
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_dim:
            raise ValueError("ParaX tokens must have shape [batch, tokens, hidden_dim]")
        pooled = tokens.mean(dim=1)
        router_input = pooled
        if self.level_conditioned:
            if view_name not in self.level_names:
                raise ValueError(f"Unknown ParaX view: {view_name}")
            level = self.level_embeddings[self.level_names.index(view_name)]
            router_input = torch.cat([pooled, level.expand_as(pooled)], dim=-1)
        if self.static:
            gates = tokens.new_full((tokens.shape[0], self.num_experts), 1.0 / self.num_experts)
        else:
            gates = torch.softmax(self.routers[str(layer_id)](router_input.float()), dim=-1)
            gates = gates.to(dtype=tokens.dtype)
        a = torch.einsum("be,erk->brk", gates, self.expert_a.to(tokens))
        b = torch.einsum("be,ekr->bkr", gates, self.expert_b.to(tokens))
        x = self.norm(tokens)
        low = torch.einsum("blk,brk->blr", x, a)
        low = torch.nn.functional.gelu(self.proj(low))
        delta = torch.einsum("blr,bkr->blk", low, b)
        scaled_delta = self.output_scale.to(tokens) * delta
        if self.residual_ratio_cap > 0:
            token_norm = tokens.float().norm(dim=-1).mean(dim=1).clamp_min(1e-12)
            delta_norm = scaled_delta.float().norm(dim=-1).mean(dim=1)
            scale = (
                self.residual_ratio_cap * token_norm / delta_norm.clamp_min(1e-12)
            ).clamp(max=1.0).to(dtype=delta.dtype)
            scaled_delta = scaled_delta * scale[:, None, None]
        output = tokens + scaled_delta
        return output, gates


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
        bottleneck_dims_per_task: Optional[Sequence[int]] = None,
        residual_gate_mode: str = "fixed",
        auxiliary_metric_mode: str = "none",
    ) -> None:
        super().__init__()
        layers = tuple(int(index) for index in layer_indices)
        if num_tasks <= 0:
            raise ValueError("Adapter task count must be positive")
        if bottleneck_dim <= 0:
            raise ValueError("Adapter reference bottleneck dimension must be positive")
        if not layers or any(index < 0 for index in layers):
            raise ValueError("Adapter layer indices must be non-negative")
        if len(set(layers)) != len(layers):
            raise ValueError("Adapter layer indices must be unique")
        if residual_scale < 0:
            raise ValueError("Adapter residual scale must be non-negative")
        if residual_gate_mode not in {"fixed", "learnable"}:
            raise ValueError("Adapter residual gate mode must be fixed or learnable")
        if residual_gate_mode == "learnable" and not 0 < residual_scale < 1:
            raise ValueError(
                "Learnable Adapter residual gate initialization must be between 0 and 1"
            )
        if auxiliary_metric_mode not in {
            "none", "residual_ratio", "feature_cosine"
        }:
            raise ValueError("Unknown Adapter auxiliary metric mode")
        if task_initialization not in {"independent", "copy_previous"}:
            raise ValueError(
                "Adapter task initialization must be independent or copy_previous"
            )
        if bottleneck_dims_per_task is None:
            bottleneck_dims = (int(bottleneck_dim),) * int(num_tasks)
        else:
            bottleneck_dims = tuple(
                int(value) for value in bottleneck_dims_per_task
            )
            if len(bottleneck_dims) != int(num_tasks):
                raise ValueError(
                    "Per-task Adapter bottleneck dimensions must match task count"
                )
        if any(value <= 0 for value in bottleneck_dims):
            raise ValueError("Per-task Adapter bottleneck dimensions must be positive")
        if task_initialization == "copy_previous" and len(set(bottleneck_dims)) > 1:
            raise ValueError(
                "copy_previous Adapter initialization requires uniform bottleneck dimensions"
            )
        self.num_tasks = int(num_tasks)
        self.hidden_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.bottleneck_dims_per_task = bottleneck_dims
        self.layer_indices = tuple(sorted(layers))
        self.residual_scale = float(residual_scale)
        self.residual_gate_mode = residual_gate_mode
        self.auxiliary_metric_mode = auxiliary_metric_mode
        self.activation_name = activation
        self.task_initialization = task_initialization
        self._current_task_id = -1

        def build_task_adapters(task_bottleneck_dim: int) -> nn.ModuleDict:
            return nn.ModuleDict(
                {
                    str(layer_id): TransformerBlockAdapter(
                        hidden_dim=self.hidden_dim,
                        bottleneck_dim=task_bottleneck_dim,
                        activation=activation,
                    )
                    for layer_id in self.layer_indices
                }
            )

        task_adapters = []
        for task_bottleneck_dim in self.bottleneck_dims_per_task:
            state_before_task = torch.get_rng_state()
            task_adapters.append(build_task_adapters(task_bottleneck_dim))
            if task_bottleneck_dim != self.bottleneck_dim:
                # Keep later task initializations identical to the uniform
                # reference-bottleneck control.  Otherwise changing task 0's
                # capacity would shift the initialization RNG of tasks 1--N
                # and confound a task-dependent capacity experiment.
                torch.set_rng_state(state_before_task)
                build_task_adapters(self.bottleneck_dim)
        self.task_adapters = nn.ModuleList(task_adapters)
        if self.residual_gate_mode == "learnable":
            gate_logit = math.log(self.residual_scale / (1.0 - self.residual_scale))
            self.task_gates = nn.ParameterList(
                [
                    nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))
                    for _ in range(self.num_tasks)
                ]
            )
        else:
            self.task_gates = nn.ParameterList()
        self._auxiliary_residual_ratios = []
        self._auxiliary_cosine_distances = []
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
            if self.residual_gate_mode == "learnable":
                with torch.no_grad():
                    self.task_gates[task_id].copy_(self.task_gates[task_id - 1])
        self.task_adapters[task_id].requires_grad_(True)
        if self.residual_gate_mode == "learnable":
            self.task_gates[task_id].requires_grad_(True)
        self._current_task_id = int(task_id)

    def restore_task(self, task_id: int) -> None:
        if not -1 <= task_id < self.num_tasks:
            raise ValueError("Restored adapter task id is invalid")
        self.requires_grad_(False)
        if task_id >= 0:
            self.task_adapters[task_id].requires_grad_(True)
            if self.residual_gate_mode == "learnable":
                self.task_gates[task_id].requires_grad_(True)
        self._current_task_id = int(task_id)

    def residual_multiplier(self, task_id: int) -> torch.Tensor | float:
        if not 0 <= int(task_id) < self.num_tasks:
            raise ValueError("Adapter task id is outside the protocol")
        if self.residual_gate_mode == "learnable":
            return torch.sigmoid(self.task_gates[int(task_id)])
        return self.residual_scale

    def gate_values(self) -> Tuple[float, ...]:
        if self.residual_gate_mode == "learnable":
            return tuple(
                float(torch.sigmoid(gate.detach()).cpu()) for gate in self.task_gates
            )
        return (self.residual_scale,) * self.num_tasks

    def reset_auxiliary_metrics(self) -> None:
        self._auxiliary_residual_ratios = []
        self._auxiliary_cosine_distances = []

    def auxiliary_metric(self, mode: str) -> torch.Tensor:
        if mode != self.auxiliary_metric_mode:
            raise RuntimeError(
                "Requested Adapter auxiliary metric was not enabled at construction"
            )
        if mode == "residual_ratio":
            values = self._auxiliary_residual_ratios
        elif mode == "feature_cosine":
            values = self._auxiliary_cosine_distances
        else:
            raise ValueError("Unknown Adapter auxiliary metric")
        if not values:
            raise RuntimeError("Adapter auxiliary metric is unavailable before a forward pass")
        return torch.stack(values).mean()

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
            deltas.append(
                self.residual_multiplier(int(task_id))
                * adapter(normalized_lane_tokens[lane_position])
            )
        return torch.stack(deltas, dim=0)

    def active_parameters(self) -> Iterable[nn.Parameter]:
        if self._current_task_id < 0:
            return iter(())
        parameters = list(self.task_adapters[self._current_task_id].parameters())
        if self.residual_gate_mode == "learnable":
            parameters.append(self.task_gates[self._current_task_id])
        return iter(parameters)

    def parameter_names(self) -> Tuple[str, ...]:
        return tuple(name for name, _ in self.named_parameters())

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def per_task_parameter_count(self, task_id: int = 0) -> int:
        if not 0 <= int(task_id) < self.num_tasks:
            raise ValueError("Adapter task id is outside the protocol")
        return sum(
            parameter.numel()
            for parameter in self.task_adapters[int(task_id)].parameters()
        ) + (1 if self.residual_gate_mode == "learnable" else 0)

    def per_task_parameter_counts(self) -> Tuple[int, ...]:
        return tuple(
            self.per_task_parameter_count(task_id)
            for task_id in range(self.num_tasks)
        )


class TaskImageTokenAdapterBank(TaskLaneTransformerAdapterBank):
    """Task-specific residual adapters over frozen CLIP CLS + patch tokens.

    Each lane receives the frozen image-token stream plus the residual produced
    by its own task adapter.  The caller decides where those adapted tokens are
    consumed; Track A uses them only for selector matching and aggregation, so
    they never replace the frozen CLIP residual stream.
    """

    def __init__(
        self,
        *args,
        view_bottleneck_dim: int = 0,
        view_names: Sequence[str] = ("full", "person", "face"),
        view_mode: str = "shared",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(view_bottleneck_dim) < 0:
            raise ValueError("View-specific Adapter bottleneck must be non-negative")
        names = tuple(str(name) for name in view_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("View-specific Adapter names must be unique and non-empty")
        self.view_bottleneck_dim = int(view_bottleneck_dim)
        if view_mode not in {"shared", "independent"}:
            raise ValueError("Image-token Adapter view mode must be shared or independent")
        if view_mode == "independent" and not self.view_bottleneck_dim:
            raise ValueError("Independent Adapter views require a positive bottleneck")
        self.view_mode = view_mode
        self.view_names = names
        self.view_task_adapters = nn.ModuleList()
        if self.view_bottleneck_dim:
            for _ in range(self.num_tasks):
                self.view_task_adapters.append(nn.ModuleDict({
                    name: nn.ModuleDict({
                        str(layer_id): TransformerBlockAdapter(
                            hidden_dim=self.hidden_dim,
                            bottleneck_dim=self.view_bottleneck_dim,
                            activation=self.activation_name,
                        )
                        for layer_id in self.layer_indices
                    })
                    for name in self.view_names
                }))
            if self.view_mode == "independent":
                # All view branches start from the same weights.  This keeps
                # the first optimization step free of view-specific RNG
                # differences; specialization then comes only from gradients.
                for task_adapters in self.view_task_adapters:
                    source = task_adapters[self.view_names[0]].state_dict()
                    for name in self.view_names[1:]:
                        task_adapters[name].load_state_dict(source)
        self.requires_grad_(False)

    @property
    def view_specialized(self) -> bool:
        return self.view_bottleneck_dim > 0

    def activate_task(self, task_id: int) -> None:
        super().activate_task(task_id)
        if self.view_mode == "independent":
            self.task_adapters[task_id].requires_grad_(False)
        if self.view_specialized:
            if task_id > 0 and self.task_initialization == "copy_previous":
                self.view_task_adapters[task_id].load_state_dict(
                    self.view_task_adapters[task_id - 1].state_dict()
                )
            self.view_task_adapters[task_id].requires_grad_(True)

    def restore_task(self, task_id: int) -> None:
        super().restore_task(task_id)
        if self.view_mode == "independent" and task_id >= 0:
            self.task_adapters[task_id].requires_grad_(False)
        if self.view_specialized and task_id >= 0:
            self.view_task_adapters[task_id].requires_grad_(True)

    def active_parameters(self) -> Iterable[nn.Parameter]:
        parameters = (
            [] if self.view_mode == "independent"
            else list(super().active_parameters())
        )
        if self.view_specialized and self.current_task_id >= 0:
            parameters.extend(
                self.view_task_adapters[self.current_task_id].parameters()
            )
        return iter(parameters)

    def per_task_parameter_count(self, task_id: int = 0) -> int:
        count = (
            0 if self.view_mode == "independent"
            else super().per_task_parameter_count(task_id)
        )
        if self.view_specialized:
            count += sum(
                parameter.numel()
                for parameter in self.view_task_adapters[int(task_id)].parameters()
            )
        return count

    def total_parameter_count(self) -> int:
        if self.view_mode == "independent":
            return sum(
                parameter.numel()
                for parameter in self.view_task_adapters.parameters()
            )
        return super().total_parameter_count()

    def shared_active_parameters(self) -> Iterable[nn.Parameter]:
        if self.view_mode == "independent":
            return iter(())
        return TaskLaneTransformerAdapterBank.active_parameters(self)

    def view_active_parameters(self, view_name: str) -> Iterable[nn.Parameter]:
        if view_name not in self.view_names:
            raise ValueError("Unknown Image-token Adapter view")
        if not self.view_specialized or self.current_task_id < 0:
            return iter(())
        return iter(
            self.view_task_adapters[self.current_task_id][view_name].parameters()
        )

    def adapted_tokens_for_layer(
        self,
        layer_id: int,
        frozen_image_tokens: torch.Tensor,
        lane_ids: Sequence[int],
        view_name: str = "full",
    ) -> torch.Tensor:
        if frozen_image_tokens.ndim != 3:
            raise ValueError("Image tokens must have shape [batch, tokens, width]")
        if frozen_image_tokens.shape[-1] != self.hidden_dim:
            raise ValueError("Image-token width differs from adapter width")
        if not lane_ids:
            raise ValueError("At least one lane id is required")
        if view_name not in self.view_names:
            raise ValueError("Unknown Image-token Adapter view")
        expanded = frozen_image_tokens.unsqueeze(0).expand(
            len(lane_ids), -1, -1, -1
        )
        if layer_id not in self.layer_indices:
            return expanded
        adapted = []
        for task_id in lane_ids:
            if not 0 <= int(task_id) < self.num_tasks:
                raise ValueError("Lane id is outside the adapter bank")
            adapter = self.task_adapters[int(task_id)][str(layer_id)]
            if self.view_mode == "independent":
                delta = self.view_task_adapters[int(task_id)][view_name][
                    str(layer_id)
                ](frozen_image_tokens)
            else:
                delta = adapter(frozen_image_tokens)
                if self.view_specialized:
                    delta = delta + self.view_task_adapters[int(task_id)][view_name][
                        str(layer_id)
                    ](frozen_image_tokens)
            delta = self.residual_multiplier(int(task_id)) * delta
            adapted_tokens = frozen_image_tokens + delta
            if self.auxiliary_metric_mode == "residual_ratio":
                frozen_float = frozen_image_tokens.float()
                delta_float = delta.float()
                self._auxiliary_residual_ratios.append(
                    (
                        delta_float.square().sum(dim=-1)
                        / frozen_float.square().sum(dim=-1).clamp_min(1e-8)
                    ).mean()
                )
            elif self.auxiliary_metric_mode == "feature_cosine":
                frozen_float = frozen_image_tokens.float()
                adapted_float = adapted_tokens.float()
                self._auxiliary_cosine_distances.append(
                    1.0
                    - F.cosine_similarity(
                        adapted_float, frozen_float, dim=-1, eps=1e-8
                    ).mean()
                )
            adapted.append(adapted_tokens)
        return torch.stack(adapted, dim=0)
