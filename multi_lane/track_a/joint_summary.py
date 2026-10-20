"""Task-local fusion of three Selector summaries before task attention.

Each expert sees the JOINT evidence, rather than adapting one image stream.
The frozen image tower is never modified. No expert or router is shared across
tasks; the complete fusion dependency of an old task stays immutable.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class JointResidual(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.down = nn.Linear(3 * width + 3, hidden)
        self.up = nn.Linear(hidden, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, evidence):
        return self.up(F.gelu(self.down(evidence)))


class JointSummaryLayer(nn.Module):
    def __init__(self, width, mode, bottleneck, num_experts, top_k, scale):
        super().__init__()
        self.width, self.mode, self.top_k, self.scale = width, mode, top_k, scale
        # Dense has the same bottleneck compute budget as the active MoE experts.
        self.experts = nn.ModuleList([
            JointResidual(width, bottleneck if mode == 'moe' else bottleneck * top_k)
            for _ in range(num_experts if mode == 'moe' else int(mode == 'dense'))
        ])
        self.router = None
        if mode == 'moe':
            self.router = nn.Sequential(nn.Linear(3 * width + 3, 16), nn.GELU(),
                                        nn.Linear(16, num_experts))
            nn.init.normal_(self.router[-1].weight, std=0.01)
            nn.init.zeros_(self.router[-1].bias)

    def forward(self, summaries, valid):
        # [B,M,3,D]. Mask BEFORE normalization, the router, and expert MLPs.
        clean = summaries.masked_fill(~valid[:, None, :, None], 0)
        evidence = F.layer_norm(clean.float(), (self.width,)).to(clean.dtype)
        mask_features = valid[:, None].expand(-1, clean.shape[1], -1).to(clean.dtype)
        evidence = torch.cat((evidence.flatten(-2), mask_features), dim=-1)
        prior = clean.new_tensor((0.64, 0.16, 0.20)) * valid.to(clean.dtype)
        weights = prior / prior.sum(-1, keepdim=True)
        base = (clean * weights[:, None, :, None]).sum(-2)
        gates = None
        if self.mode == 'fixed':
            return base, gates, base, torch.zeros_like(base)
        if self.mode == 'dense':
            delta = self.experts[0](evidence)
        else:
            logits = self.router(evidence.mean(1)).float()
            values, indices = logits.topk(self.top_k, dim=-1)
            selected_weights = values.softmax(-1).to(clean.dtype)
            gates = torch.zeros_like(logits).scatter(-1, indices, selected_weights.float())
            delta = torch.zeros_like(base)
            # Sparse dispatch: compute only the selected image/expert pairs.
            for expert_id, expert in enumerate(self.experts):
                rows, slots = torch.where(indices == expert_id)
                if rows.numel():
                    contribution = expert(evidence[rows]) * selected_weights[rows, slots, None, None]
                    delta = delta.index_add(0, rows, contribution)
        return base + self.scale * delta, gates, base, self.scale * delta


class TaskJointSummaryBank(nn.Module):
    MODES = ('disabled', 'fixed', 'dense', 'moe')

    def __init__(self, num_tasks, num_layers, width, mode, bottleneck=16,
                 num_experts=4, top_k=2, scale=0.1):
        super().__init__()
        if (mode not in self.MODES[1:] or min(num_tasks, num_layers, width, bottleneck) <= 0
                or not 1 <= top_k <= num_experts or not math.isfinite(scale) or not 0 < scale <= 1):
            raise ValueError('Invalid joint summary configuration')
        self.mode, self.num_tasks, self.num_layers = mode, num_tasks, num_layers
        self.num_experts, self.top_k, self.scale = num_experts, top_k, scale
        self.current_task_id = -1
        self.tasks = nn.ModuleList()
        # Independent reproducible initialization leaves base/sampler RNG intact.
        with torch.random.fork_rng(devices=[]):
            for task in range(num_tasks):
                torch.manual_seed(91_003 + task)
                self.tasks.append(nn.ModuleList([
                    JointSummaryLayer(width, mode, bottleneck, num_experts, top_k, scale)
                    for _ in range(num_layers)
                ]))
        self.requires_grad_(False)
        self.reset_statistics()

    def restore_task(self, task_id):
        if not -1 <= task_id < self.num_tasks:
            raise ValueError('Invalid joint summary task')
        self.requires_grad_(False)
        for p in self.parameters():
            p.grad = None
        if task_id >= 0:
            self.tasks[task_id].requires_grad_(True)
        self.current_task_id = task_id

    def active_parameters(self):
        if self.current_task_id >= 0:
            yield from self.tasks[self.current_task_id].parameters()

    def parameter_count_per_task(self):
        return sum(p.numel() for p in self.tasks[0].parameters())

    def reset_statistics(self):
        self._statistics = {}

    def statistics(self):
        result = {}
        for key, row in self._statistics.items():
            n = float(row['samples'].cpu())
            result[key] = {name: value.cpu().tolist() if value.ndim else float(value.cpu())
                           for name, value in row.items()}
            result[key]['base_norm_mean'] = result[key]['base_norm_sum'] / n
            result[key]['residual_norm_mean'] = result[key]['residual_norm_sum'] / n
        return result

    def forward(self, layer_id: int, summaries: torch.Tensor,
                lane_ids: Sequence[int], valid: torch.Tensor):
        # [T,B,M,3,D] -> [T,B,M,D]. Same Selector index, NOT same patch index.
        if (summaries.ndim != 5 or summaries.shape[0] != len(lane_ids)
                or summaries.shape[3] != 3 or valid.shape != (summaries.shape[1], 3)
                or valid.dtype != torch.bool or not bool(valid[:, 0].all())):
            raise ValueError('Invalid joint summary evidence or Full validity')
        outputs = []
        for position, task in enumerate(lane_ids):
            out, gates, base, residual = self.tasks[task][layer_id](summaries[position], valid)
            outputs.append(out)
            with torch.no_grad():
                key = f'task{task}/layer{layer_id}'
                row = dict(samples=base.new_tensor(len(base), dtype=torch.float64),
                           base_norm_sum=base.float().norm(dim=-1).mean(-1).sum().double(),
                           residual_norm_sum=residual.float().norm(dim=-1).mean(-1).sum().double())
                if gates is not None:
                    row['soft_mass'] = gates.double().sum(0)
                    row['hard_load'] = (gates > 0).double().sum(0)
                if key not in self._statistics:
                    self._statistics[key] = row
                else:
                    for name, value in row.items():
                        self._statistics[key][name].add_(value)
        return torch.stack(outputs)
