"""Task-local three-view MoE with an always-used shared expert.

The adapter stays parallel to the Task Forward FFN.  Its router reads the
image evidence gathered by the selectors in the same block, while the frozen
image stream and completed task pools remain untouched.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .moe_task_ffn import _Expert


class _SharedLayerPool(nn.Module):
    def __init__(self, width: int, bottleneck: int, experts: int,
                 dropout: float) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            _Expert(width, bottleneck, dropout) for _ in range(experts)
        )
        self.gates = nn.ParameterList([
            nn.Parameter(torch.empty(width, experts - 1)) for _ in range(3)
        ])
        for gate in self.gates:
            nn.init.normal_(gate, mean=0.0, std=width ** -0.5)


class TaskForwardSharedMoEBank(nn.Module):
    """One common expert plus one view/content-selected specialist per task."""

    view_names = ("full", "person", "face")
    route_modes = ("dynamic", "shared_fixed", "view_fixed")

    def __init__(
        self, num_tasks: int, hidden_dim: int, bottleneck_dim: int,
        layer_indices: Sequence[int], residual_scale: float,
        num_experts: int = 7, dropout: float = 0.1,
        route_mode: str = "dynamic", shared_fraction: float = 0.5,
        router_logit_scale: float = 2.0, router_temperature: float = 1.0,
        noise_std: float = 0.5,
    ) -> None:
        super().__init__()
        if num_tasks <= 0 or hidden_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("MoE tasks, width and bottleneck must be positive")
        if num_experts < 4 or route_mode not in self.route_modes:
            raise ValueError("The three-view MoE needs at least four experts and a valid route mode")
        if not layer_indices or len(set(layer_indices)) != len(layer_indices):
            raise ValueError("MoE layer indices must be nonempty and unique")
        if not 0 <= dropout < 1 or residual_scale < 0:
            raise ValueError("Invalid dropout or residual scale")
        if not 0 < shared_fraction < 1 or router_logit_scale <= 0:
            raise ValueError("Invalid shared fraction or router logit scale")
        if router_temperature <= 0 or noise_std < 0:
            raise ValueError("Invalid router temperature or noise standard deviation")
        self.hidden_dim = int(hidden_dim)
        self.layer_indices = tuple(int(index) for index in layer_indices)
        self.residual_scale = float(residual_scale)
        self.num_experts = int(num_experts)
        self.route_mode = route_mode
        self.shared_fraction = float(shared_fraction)
        self.router_logit_scale = float(router_logit_scale)
        self.router_temperature = float(router_temperature)
        self.noise_std = float(noise_std)
        self.noise_progress = 0.0
        self.tasks = nn.ModuleList([
            nn.ModuleList([
                _SharedLayerPool(hidden_dim, bottleneck_dim, num_experts, dropout)
                for _ in self.layer_indices
            ]) for _ in range(num_tasks)
        ])
        self._current_task_id = -1
        self._balance_terms: list[torch.Tensor] = []
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

    def train(self, mode: bool = True) -> "TaskForwardSharedMoEBank":
        super().train(mode)
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
        return tuple(sum(p.numel() for p in task.parameters()) for task in self.tasks)

    def per_task_parameter_count(self) -> int:
        return self.per_task_parameter_counts()[0]

    def set_training_progress(self, fraction: float) -> None:
        if not 0 <= fraction <= 1:
            raise ValueError("Training progress must be in [0, 1]")
        self.noise_progress = float(fraction)

    def reset_auxiliary_metrics(self) -> None:
        self._balance_terms.clear()

    def balance_loss(self) -> torch.Tensor:
        if not self._balance_terms:
            return self.route_counts.new_zeros(())
        return torch.stack(self._balance_terms).mean()

    def reset_route_statistics(self) -> None:
        self.route_counts.zero_()
        self.route_samples.zero_()

    def route_statistics(self) -> Dict[str, object]:
        counts = self.route_counts.detach().cpu()
        samples = self.route_samples.detach().cpu()
        return {
            str(task): {
                str(layer): {
                    view: {
                        "samples": int(samples[task, layer_index, view_index]),
                        "expert_topk_count": counts[task, layer_index, view_index].tolist(),
                        "route_mode": self.route_mode,
                    }
                    for view_index, view in enumerate(self.view_names)
                }
                for layer_index, layer in enumerate(self.layer_indices)
            }
            for task in range(self._current_task_id + 1)
        }

    def _specialist_gate(
        self, evidence: torch.Tensor, pool: _SharedLayerPool,
        view_index: int,
    ) -> torch.Tensor:
        batch = evidence.shape[0]
        if self.route_mode == "shared_fixed":
            return evidence.new_zeros(batch, self.num_experts - 1, dtype=torch.float32)
        if self.route_mode == "view_fixed":
            index = torch.full((batch,), view_index, device=evidence.device)
            return F.one_hot(index, self.num_experts - 1).float()
        with torch.autocast(device_type=evidence.device.type, enabled=False):
            context = F.layer_norm(evidence.float(), (self.hidden_dim,))
            clean = self.router_logit_scale * (
                F.normalize(context, dim=-1)
                @ F.normalize(pool.gates[view_index].float(), dim=0)
            )
            soft = F.softmax(clean / self.router_temperature, dim=-1)
            if pool.training:
                importance = soft.mean(dim=0)
                uniform = 1.0 / (self.num_experts - 1)
                self._balance_terms.append(
                    (self.num_experts - 1) * (importance - uniform).square().sum()
                )
                std = self.noise_std * (1.0 - self.noise_progress)
                selected_logits = clean + torch.randn_like(clean) * std if std else clean
            else:
                selected_logits = clean
            index = selected_logits.argmax(dim=-1)
            hard = F.one_hot(index, self.num_experts - 1).float()
            return hard - soft.detach() + soft if pool.training else hard

    def delta_for_layer(
        self, layer_id: int, task_lane_tokens: torch.Tensor,
        lane_ids: Sequence[int], image_view: str,
        router_evidence: torch.Tensor,
    ) -> torch.Tensor:
        if layer_id not in self.layer_indices or self.residual_scale == 0:
            return torch.zeros_like(task_lane_tokens)
        if image_view not in self.view_names:
            raise ValueError(f"Unknown MoE image view: {image_view}")
        if (task_lane_tokens.ndim != 4
            or task_lane_tokens.shape[0] != len(lane_ids)
            or router_evidence.shape != task_lane_tokens.shape[:2] + (self.hidden_dim,)):
            raise ValueError("MoE lane tokens/evidence have invalid shapes")
        layer_index = self.layer_indices.index(layer_id)
        view_index = self.view_names.index(image_view)
        lane_outputs = []
        for lane_index, task_id in enumerate(lane_ids):
            if not 0 <= task_id <= self._current_task_id:
                raise ValueError("MoE cannot access a future task")
            tokens = task_lane_tokens[lane_index]
            pool = self.tasks[task_id][layer_index]
            specialist_gate = self._specialist_gate(
                router_evidence[lane_index], pool, view_index
            )
            shared_output = pool.experts[0](tokens)
            if self.route_mode == "shared_fixed":
                combined = shared_output
                active = self.route_counts.new_zeros(self.num_experts)
                active[0] = len(tokens)
            else:
                specialist_output = torch.zeros_like(tokens)
                active = self.route_counts.new_zeros(self.num_experts)
                active[0] = len(tokens)
                for expert_id in range(1, self.num_experts):
                    selected = torch.nonzero(
                        specialist_gate[:, expert_id - 1] > 0,
                        as_tuple=True,
                    )[0]
                    if selected.numel() == 0:
                        continue
                    result = pool.experts[expert_id](tokens.index_select(0, selected))
                    weight = specialist_gate.index_select(0, selected)[:, expert_id - 1]
                    specialist_output = specialist_output.index_add(
                        0, selected,
                        (result * weight.to(result.dtype).view(-1, 1, 1))
                        .to(specialist_output.dtype),
                    )
                    active[expert_id] = len(selected)
                combined = (
                    self.shared_fraction * shared_output
                    + (1.0 - self.shared_fraction) * specialist_output
                )
            if not self.training:
                with torch.no_grad():
                    self.route_counts[task_id, layer_index, view_index].add_(active)
                    self.route_samples[task_id, layer_index, view_index] += len(tokens)
            lane_outputs.append(combined)
        return self.residual_scale * torch.stack(lane_outputs, dim=0)
