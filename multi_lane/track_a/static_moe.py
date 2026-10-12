"""Static, view-routed output experts on the task-local image evidence path."""
from __future__ import annotations

import math
import torch
from torch import nn

from .adapter import TaskImageTokenAdapterBank


ROUTES = {
    "uniform": ((1 / 3, 1 / 3, 1 / 3),) * 3,
    "overlap": ((.5, .5, 0.), (.5, 0., .5), (0., .5, .5)),
}


class OutputExpert(nn.Module):
    def __init__(self, width, rank, activation):
        super().__init__()
        self.down = nn.Linear(width, rank)
        self.up = nn.Linear(rank, width, bias=False)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        return self.up(self.activation(self.down(value)))


class OutputExpertPool(nn.Module):
    """One shared output bias makes the budget equal to one width-3r adapter."""
    def __init__(self, width, rank, activation):
        super().__init__()
        self.experts = nn.ModuleList([OutputExpert(width, rank, activation) for _ in range(3)])
        self.output_bias = nn.Parameter(torch.zeros(width))
        self.last_output_correlation = 0.0

    def forward(self, value, weights):
        # Evaluate all three in v1: uniform/overlap have the same forward graph
        # and FLOPs. A zero route coefficient still gives that expert zero gradient.
        outputs = torch.stack([expert(value) for expert in self.experts])
        flat = outputs.detach().float().reshape(3, -1)
        centered = flat - flat.mean(dim=1, keepdim=True)
        normalized = centered / centered.norm(dim=1, keepdim=True).clamp_min(1e-12)
        corr = normalized @ normalized.transpose(0, 1)
        self.last_output_correlation = float(
            torch.stack((corr[0, 1], corr[0, 2], corr[1, 2])).mean().cpu()
        )
        weights = weights.to(outputs).view(3, *([1] * value.ndim))
        return (outputs * weights).sum(dim=0) + self.output_bias


class StaticMoEImageTokenAdapterBank(TaskImageTokenAdapterBank):
    """A separate expert pool per task/layer, shared across three crop views."""
    def __init__(self, num_tasks, hidden_dim, bottleneck_dim, layer_indices,
                 residual_scale=.03, activation="relu", task_initialization="independent",
                 bottleneck_dims_per_task=None, residual_gate_mode="fixed",
                 auxiliary_metric_mode="none", view_bottleneck_dim=0,
                 view_mode="shared", routing="overlap"):
        # Reuse the task lifecycle/interface without allocating an unused adapter bank.
        nn.Module.__init__(self)
        if routing not in ROUTES:
            raise ValueError("Static MoE routing must be uniform or overlap")
        if num_tasks <= 0 or hidden_dim <= 0 or bottleneck_dim <= 0 or bottleneck_dim % 3:
            raise ValueError("Static MoE needs positive dimensions and total bottleneck divisible by three")
        layers = tuple(int(i) for i in layer_indices)
        if not layers or len(set(layers)) != len(layers) or min(layers) < 0:
            raise ValueError("Invalid static MoE layers")
        if activation not in ("relu", "gelu") or not math.isfinite(residual_scale) or residual_scale < 0:
            raise ValueError("Invalid static MoE activation/scale")
        if (task_initialization != "independent" or bottleneck_dims_per_task is not None
                or residual_gate_mode != "fixed" or auxiliary_metric_mode != "none"
                or view_bottleneck_dim != 0 or view_mode != "shared"):
            raise ValueError("Static MoE v1 requires independent task pools, fixed scale and no extra view modules")
        self.num_tasks, self.hidden_dim = int(num_tasks), int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.bottleneck_dims_per_task = (self.bottleneck_dim,) * self.num_tasks
        self.layer_indices = tuple(sorted(layers))
        self.residual_scale, self.activation_name = float(residual_scale), activation
        self.task_initialization, self.residual_gate_mode = task_initialization, residual_gate_mode
        self.auxiliary_metric_mode = auxiliary_metric_mode
        self.view_names = ("full", "person", "face")
        self.view_bottleneck_dim, self.view_mode = 0, "shared"
        self.view_task_adapters = nn.ModuleList()
        self.task_gates = nn.ParameterList()
        self.routing = routing
        self.register_buffer("route_weights", torch.tensor(ROUTES[routing], dtype=torch.float32))
        self.task_adapters = nn.ModuleList([
            nn.ModuleDict({str(layer): OutputExpertPool(hidden_dim, bottleneck_dim // 3, activation)
                           for layer in self.layer_indices})
            for _ in range(self.num_tasks)
        ])
        self._current_task_id = -1
        self.reset_auxiliary_metrics()
        self.reset_view_diagnostics()
        self._runtime_residual_scope = "all"
        self._runtime_residual_multiplier = 1.0
        self._cumulative_expert_correlations = []
        self.requires_grad_(False)

    def validate_routes(self):
        expected = self.route_weights.new_tensor(ROUTES[self.routing])
        if not torch.equal(self.route_weights, expected):
            raise RuntimeError("Static expert routing table changed")

    def set_residual_ablation(self, scope: str = "all", multiplier: float = 1.0) -> None:
        if scope not in {"all", "full", "auxiliary"}:
            raise ValueError("Residual ablation scope must be all, full, or auxiliary")
        if not math.isfinite(float(multiplier)) or float(multiplier) < 0:
            raise ValueError("Residual ablation multiplier must be finite and non-negative")
        self._runtime_residual_scope = str(scope)
        self._runtime_residual_multiplier = float(multiplier)

    def reset_ablation_diagnostics(self) -> None:
        self._cumulative_expert_correlations = []

    def expert_output_correlation(self) -> float | None:
        if not self._cumulative_expert_correlations:
            return None
        return float(sum(self._cumulative_expert_correlations) / len(self._cumulative_expert_correlations))

    def adapted_tokens_for_layer(self, layer_id, frozen_image_tokens, lane_ids, view_name="full"):
        if (frozen_image_tokens.ndim != 3 or frozen_image_tokens.shape[-1] != self.hidden_dim
                or not lane_ids or len(set(lane_ids)) != len(lane_ids)):
            raise ValueError("Invalid static MoE tokens/lanes")
        if view_name not in self.view_names:
            raise ValueError("Unknown static MoE view")
        if any(not 0 <= int(task) <= self.current_task_id for task in lane_ids):
            raise ValueError("Static MoE cannot access future/unactivated tasks")
        if layer_id not in self.layer_indices:
            return frozen_image_tokens.unsqueeze(0).expand(len(lane_ids), -1, -1, -1)
        weights = self.route_weights[self.view_names.index(view_name)]
        adapted, ratios = [], []
        for task in lane_ids:
            active = (
                self._runtime_residual_scope == "all"
                or (self._runtime_residual_scope == "full" and view_name == "full")
                or (self._runtime_residual_scope == "auxiliary" and view_name != "full")
            )
            multiplier = self._runtime_residual_multiplier if active else 1.0
            pool = self.task_adapters[int(task)][str(layer_id)]
            delta = self.residual_scale * multiplier * pool(frozen_image_tokens, weights)
            self._cumulative_expert_correlations.append(pool.last_output_correlation)
            adapted.append(frozen_image_tokens + delta)
            ratios.append((delta.detach().float().square().sum(-1)
                           / frozen_image_tokens.detach().float().square().sum(-1).clamp_min(1e-8)).mean())
        self._last_view_residual_ratios[view_name] = torch.stack(ratios).mean()
        return torch.stack(adapted)

    def expert_gradient_diagnostics(self, gradient_scale=1.):
        result = {}
        if self.current_task_id < 0:
            return result
        for layer, pool in self.task_adapters[self.current_task_id].items():
            for index, expert in enumerate(pool.experts):
                squared = sum(float((p.grad.detach().float() / gradient_scale).square().sum())
                              for p in expert.parameters() if p.grad is not None)
                result[f"adapter_grad_layer{layer}_expert{index}_norm"] = squared ** .5
        return result

