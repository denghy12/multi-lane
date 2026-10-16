"""Matched shared controls and train/eval-consistent soft Top-2 specialists."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from .moe_task_ffn_v2 import TaskForwardSharedMoEBank, _SharedLayerPool


class TaskForwardSoftMoEBank(TaskForwardSharedMoEBank):
    """A half-weight shared expert plus two weighted specialists from three."""

    diagnostic_names = (
        "shared_norm", "specialist_norm", "combined_norm",
        "weighted_specialist_shared_ratio", "shared_specialist_cosine",
        "shared_cls_norm", "specialist_cls_norm",
        "weighted_specialist_shared_cls_ratio", "gate_entropy",
    )

    def __init__(self, num_tasks: int, hidden_dim: int, bottleneck_dim: int,
                 layer_indices: Sequence[int], residual_scale: float,
                 num_experts: int = 4, dropout: float = 0.1,
                 route_mode: str = "dynamic", shared_fraction: float = 0.5,
                 router_logit_scale: float = 2.0,
                 router_temperature: float = 1.0, noise_std: float = 0.0) -> None:
        if num_experts != 4 or shared_fraction != 0.5 or noise_std != 0:
            raise ValueError("V3 requires four experts, shared fraction 0.5 and no routing noise")
        super().__init__(
            num_tasks, hidden_dim, bottleneck_dim, layer_indices, residual_scale,
            num_experts, dropout, route_mode, shared_fraction,
            router_logit_scale, router_temperature, noise_std,
        )
        self.specialist_runtime_enabled = True
        self._forward_records: list[tuple[str, torch.Tensor]] = []
        self.register_buffer("route_soft_weights", torch.zeros_like(self.route_counts),
                             persistent=False)
        self.register_buffer(
            "route_output_sums",
            torch.zeros(*self.route_samples.shape, len(self.diagnostic_names)),
            persistent=False,
        )

    def reset_auxiliary_metrics(self) -> None:
        super().reset_auxiliary_metrics()
        self._forward_records.clear()

    def reset_route_statistics(self) -> None:
        super().reset_route_statistics()
        self.route_soft_weights.zero_()
        self.route_output_sums.zero_()

    def _specialist_gate(self, evidence: torch.Tensor, pool: _SharedLayerPool,
                         view_index: int) -> torch.Tensor:
        batch = evidence.shape[0]
        if self.route_mode == "shared_fixed" or not self.specialist_runtime_enabled:
            return evidence.new_zeros(batch, 3, dtype=torch.float32)
        if self.route_mode == "view_fixed":
            # Full: 1/2; Person: 2/3; Face: 3/1. Every view activates two.
            gate = evidence.new_zeros(batch, 3, dtype=torch.float32)
            gate[:, view_index] = 0.5
            gate[:, (view_index + 1) % 3] = 0.5
            return gate
        with torch.autocast(device_type=evidence.device.type, enabled=False):
            context = F.layer_norm(evidence.float(), (self.hidden_dim,))
            logits = self.router_logit_scale * (
                F.normalize(context, dim=-1)
                @ F.normalize(pool.gates[view_index].float(), dim=0)
            )
            values, indices = logits.topk(2, dim=-1)
            weights = F.softmax(values / self.router_temperature, dim=-1)
            gate = torch.zeros_like(logits).scatter(1, indices, weights)
            if pool.training:
                self._balance_terms.append(3 * (gate.mean(0) - 1 / 3).square().sum())
            return gate

    def _record(self, task_id: int, layer_index: int, view_index: int,
                tokens: torch.Tensor, shared: torch.Tensor,
                specialist: torch.Tensor, combined: torch.Tensor,
                gate: torch.Tensor) -> None:
        with torch.no_grad():
            s, p, c = shared.float(), specialist.float(), combined.float()
            sn = s.norm(dim=-1).mean(dim=-1)
            pn = p.norm(dim=-1).mean(dim=-1)
            cn = c.norm(dim=-1).mean(dim=-1)
            sc = s[:, 0].norm(dim=-1)
            pc = p[:, 0].norm(dim=-1)
            ratio = pn / sn.clamp_min(1e-8)
            cls_ratio = pc / sc.clamp_min(1e-8)
            cosine = F.cosine_similarity(s.flatten(1), p.flatten(1), dim=-1)
            entropy = -(gate * gate.clamp_min(1e-8).log()).sum(dim=-1)
            metrics = torch.stack((sn, pn, cn, ratio, cosine, sc, pc,
                                   cls_ratio, entropy), dim=-1)
            hard = (gate > 0).float()
            # Records are detached and do not retain the optimization graph.
            vector = torch.cat((metrics.mean(0), hard.mean(0), gate.mean(0)))
            self._forward_records.append((self.view_names[view_index], vector))
            if not self.training:
                index = (task_id, layer_index, view_index)
                self.route_counts[index][0] += len(tokens)
                self.route_counts[index][1:] += hard.sum(0)
                self.route_samples[index] += len(tokens)
                self.route_soft_weights[index][0] += 0.5 * len(tokens)
                self.route_soft_weights[index][1:] += 0.5 * gate.sum(0)
                self.route_output_sums[index] += metrics.sum(0)

    def forward_diagnostics(self) -> Dict[str, float]:
        result = {}
        names = self.diagnostic_names + tuple(f"expert{i}_hard_fraction" for i in range(1, 4)) + tuple(
            f"expert{i}_soft_weight" for i in range(1, 4)
        )
        for view in self.view_names:
            rows = [row for name, row in self._forward_records if name == view]
            if rows:
                means = torch.stack(rows).mean(0).cpu().tolist()
                result.update({f"moe_{view}_{name}": value for name, value in zip(names, means)})
        return result

    def route_statistics(self) -> Dict[str, object]:
        result = super().route_statistics()
        soft = self.route_soft_weights.detach().cpu()
        output = self.route_output_sums.detach().cpu()
        for task_id in range(self._current_task_id + 1):
            for layer_index, layer in enumerate(self.layer_indices):
                for view_index, view in enumerate(self.view_names):
                    row = result[str(task_id)][str(layer)][view]
                    n = row["samples"]
                    row["expert_soft_weight_sum"] = soft[task_id, layer_index, view_index].tolist()
                    row["output_mean"] = dict(zip(
                        self.diagnostic_names,
                        (output[task_id, layer_index, view_index] / max(n, 1)).tolist(),
                    ))
                    row["specialist_runtime_enabled"] = self.specialist_runtime_enabled
        return result

    def delta_for_layer(self, layer_id: int, task_lane_tokens: torch.Tensor,
                        lane_ids: Sequence[int], image_view: str,
                        router_evidence: torch.Tensor) -> torch.Tensor:
        if layer_id not in self.layer_indices or self.residual_scale == 0:
            return torch.zeros_like(task_lane_tokens)
        if image_view not in self.view_names:
            raise ValueError(f"Unknown MoE image view: {image_view}")
        if (task_lane_tokens.ndim != 4 or task_lane_tokens.shape[0] != len(lane_ids)
                or router_evidence.shape != task_lane_tokens.shape[:2] + (self.hidden_dim,)):
            raise ValueError("MoE lane tokens/evidence have invalid shapes")
        layer_index = self.layer_indices.index(layer_id)
        view_index = self.view_names.index(image_view)
        outputs = []
        for lane_index, task_id in enumerate(lane_ids):
            if not 0 <= task_id <= self._current_task_id:
                raise ValueError("MoE cannot access a future task")
            tokens = task_lane_tokens[lane_index]
            pool = self.tasks[task_id][layer_index]
            gate = self._specialist_gate(router_evidence[lane_index], pool, view_index)
            shared = pool.experts[0](tokens)
            specialist = torch.zeros_like(tokens)
            for expert_id in range(1, 4):
                selected = torch.nonzero(gate[:, expert_id - 1] > 0, as_tuple=True)[0]
                if selected.numel():
                    value = pool.experts[expert_id](tokens.index_select(0, selected))
                    weight = gate.index_select(0, selected)[:, expert_id - 1]
                    specialist = specialist.index_add(
                        0, selected,
                        (value * weight.to(value.dtype).view(-1, 1, 1)).to(specialist.dtype),
                    )
            combined = 0.5 * shared + 0.5 * specialist
            self._record(task_id, layer_index, view_index, tokens, shared,
                         specialist, combined, gate)
            outputs.append(combined)
        return self.residual_scale * torch.stack(outputs)
