"""Image-evidence experts whose FFN-parallel residual writes only task CLS.

Unlike v3, restored Selector tokens are neither expert inputs nor residual
targets. Both the router and experts consume the same parameter-free normalized
image summary; task-local pools and soft Top-2 routing are retained.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .moe_task_ffn_v3 import TaskForwardSoftMoEBank


class TaskForwardEvidenceMoEBank(TaskForwardSoftMoEBank):
    diagnostic_names = TaskForwardSoftMoEBank.diagnostic_names + (
        "delta_cls_input_cls_ratio", "evidence_centered_l2",
        "delta_cls_input_cosine",
    )

    def _record_cls(self, task_id, layer_index, view_index, tokens,
                    evidence, shared, specialist, combined, gate):
        with torch.no_grad():
            s, p, c = (x[:, 0].float() for x in (shared, specialist, combined))
            sn, pn, cn = (x.norm(dim=-1) for x in (s, p, c))
            ratio = pn / sn.clamp_min(1e-8)
            cosine = F.cosine_similarity(s, p, dim=-1)
            entropy = -(gate * gate.clamp_min(1e-8).log()).sum(dim=-1)
            input_cls = tokens[:, 0].float()
            delta = self.residual_scale * c
            delta_ratio = delta.norm(dim=-1) / input_cls.norm(dim=-1).clamp_min(1e-8)
            centered = (evidence.float() - evidence.float().mean(0)).norm(dim=-1)
            delta_cosine = F.cosine_similarity(delta, input_cls, dim=-1)
            # Existing norm fields now refer strictly to the CLS residual.
            metrics = torch.stack((sn, pn, cn, ratio, cosine, sn, pn, ratio,
                                   entropy, delta_ratio, centered, delta_cosine), -1)
            hard = (gate > 0).float()
            self._forward_records.append((
                self.view_names[view_index],
                torch.cat((metrics.mean(0), hard.mean(0), gate.mean(0))),
            ))
            if not self.training:
                index = (task_id, layer_index, view_index)
                self.route_counts[index][0] += len(tokens)
                self.route_counts[index][1:] += hard.sum(0)
                self.route_samples[index] += len(tokens)
                self.route_soft_weights[index][0] += 0.5 * len(tokens)
                self.route_soft_weights[index][1:] += 0.5 * gate.sum(0)
                self.route_output_sums[index] += metrics.sum(0)

    def delta_for_layer(self, layer_id: int, task_lane_tokens: torch.Tensor,
                        lane_ids: Sequence[int], image_view: str,
                        router_evidence: torch.Tensor) -> torch.Tensor:
        if layer_id not in self.layer_indices or self.residual_scale == 0:
            return torch.zeros_like(task_lane_tokens)
        if image_view not in self.view_names:
            raise ValueError(f"Unknown MoE image view: {image_view}")
        if (task_lane_tokens.ndim != 4 or task_lane_tokens.shape[2] < 1
                or task_lane_tokens.shape[0] != len(lane_ids)
                or task_lane_tokens.shape[-1] != self.hidden_dim
                or router_evidence.shape != task_lane_tokens.shape[:2] + (self.hidden_dim,)):
            raise ValueError("MoE lane tokens/evidence have invalid shapes")
        layer_index = self.layer_indices.index(layer_id)
        view_index = self.view_names.index(image_view)
        # FP32 normalization, no affine parameters. Keep the evidence gradient
        # to current selectors; the original CLIP image stream remains frozen.
        with torch.autocast(device_type=router_evidence.device.type, enabled=False):
            evidence = F.layer_norm(router_evidence.float(), (self.hidden_dim,))
        outputs = []
        for lane_index, task_id in enumerate(lane_ids):
            if not 0 <= task_id <= self._current_task_id:
                raise ValueError("MoE cannot access a future task")
            tokens = task_lane_tokens[lane_index]
            context = evidence[lane_index].to(tokens.dtype)
            expert_input = context.unsqueeze(1)
            pool = self.tasks[task_id][layer_index]
            gate = self._specialist_gate(context, pool, view_index)
            shared = pool.experts[0](expert_input)
            specialist = torch.zeros_like(expert_input)
            for expert_id in range(1, 4):
                selected = torch.nonzero(gate[:, expert_id - 1] > 0, as_tuple=True)[0]
                if selected.numel():
                    value = pool.experts[expert_id](expert_input.index_select(0, selected))
                    weight = gate.index_select(0, selected)[:, expert_id - 1]
                    specialist = specialist.index_add(
                        0, selected,
                        (value * weight.to(value.dtype).view(-1, 1, 1)).to(specialist.dtype),
                    )
            combined = 0.5 * shared + 0.5 * specialist
            self._record_cls(task_id, layer_index, view_index, tokens, context,
                             shared, specialist, combined, gate)
            # Concatenation keeps autograd attached without modifying task tokens.
            outputs.append(torch.cat((
                (self.residual_scale * combined).to(tokens.dtype),
                torch.zeros_like(tokens[:, 1:]),
            ), dim=1))
        return torch.stack(outputs)
