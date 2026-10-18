"""Shared experts with task/view Routers and cumulative frequency Top-2 freeze.

This is the v1 Task-Forward interface, with expert ownership changed from
task-local pools to one pool per layer. Old Routers can still call every expert.
Only expert weights are frozen; gradients through their inputs are preserved.
"""
from __future__ import annotations

import torch
from torch import nn

from .moe_task_ffn import TaskForwardMoEBank, _Expert


class _Routers(nn.Module):
    def __init__(self, width, experts):
        super().__init__()
        self.gates = nn.ParameterList([nn.Parameter(torch.zeros(width, experts)) for _ in range(3)])
        self.noise = nn.ParameterList([nn.Parameter(torch.zeros(width, experts)) for _ in range(3)])


class TaskForwardActivateFreezeBank(TaskForwardMoEBank):
    def __init__(self, num_tasks, hidden_dim, bottleneck_dim, layer_indices,
                 residual_scale, num_experts=22, top_k=2, dropout=.1, noisy_gating=True):
        nn.Module.__init__(self)
        if num_tasks < 1 or num_experts < 2 or top_k != 2:
            raise ValueError("Activate-freeze requires positive tasks and Top-2")
        if not layer_indices or len(set(layer_indices)) != len(layer_indices):
            raise ValueError("Unique nonempty layer indices required")
        if hidden_dim < 1 or bottleneck_dim < 1 or not 0 <= dropout < 1 or residual_scale <= 0:
            raise ValueError("Invalid expert configuration")
        self.layer_indices = tuple(layer_indices)
        self.num_experts, self.top_k = num_experts, top_k
        self.residual_scale, self.noisy_gating = residual_scale, noisy_gating
        self.experts = nn.ModuleList([nn.ModuleList([
            _Expert(hidden_dim, bottleneck_dim, dropout) for _ in range(num_experts)
        ]) for _ in self.layer_indices])
        self.tasks = nn.ModuleList([nn.ModuleList([
            _Routers(hidden_dim, num_experts) for _ in self.layer_indices
        ]) for _ in range(num_tasks)])
        shape = (num_tasks, len(self.layer_indices), 3)
        self.register_buffer("frozen_experts", torch.zeros(len(self.layer_indices), num_experts, dtype=torch.bool))
        self.register_buffer("finalized_tasks", torch.zeros(num_tasks, dtype=torch.bool))
        self.register_buffer("training_counts", torch.zeros(*shape, num_experts, dtype=torch.long))
        self.register_buffer("training_samples", torch.zeros(*shape, dtype=torch.long))
        self.register_buffer("freeze_top2", torch.full((num_tasks, len(self.layer_indices), 2), -1, dtype=torch.long))
        self.register_buffer("frozen_history", torch.zeros(num_tasks, len(self.layer_indices), num_experts, dtype=torch.bool))
        self.register_buffer("route_counts", torch.zeros(*shape, num_experts), persistent=False)
        self.register_buffer("route_samples", torch.zeros(*shape), persistent=False)
        self._current_task_id = -1
        self._face_reliable = None
        self.requires_grad_(False)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        for t, routers in enumerate(self.tasks):
            if t != self._current_task_id or bool(self.finalized_tasks[t]):
                routers.eval()
        return self

    def activate_task(self, task_id):
        if task_id != self._current_task_id + 1 or not 0 <= task_id < len(self.tasks):
            raise ValueError("Tasks must activate sequentially")
        if task_id and not bool(self.finalized_tasks[task_id - 1]):
            raise RuntimeError("Finalize the previous task before activating the next")
        self.requires_grad_(False)
        for p in self.parameters():
            p.grad = None
        self._current_task_id = task_id
        self.tasks[task_id].requires_grad_(True)
        for layer, pool in enumerate(self.experts):
            for i, expert in enumerate(pool):
                expert.requires_grad_(not bool(self.frozen_experts[layer, i]))
        self.train(self.training)

    def set_face_reliability(self, mask):
        self._face_reliable = mask.detach().bool()

    def active_parameters(self):
        yield from (p for p in self.parameters() if p.requires_grad)

    def per_task_parameter_counts(self):
        # Allocation accounting: shared expert storage charged once to task0.
        counts = [sum(p.numel() for p in t.parameters()) for t in self.tasks]
        counts[0] += sum(p.numel() for p in self.experts.parameters())
        return tuple(counts)

    @torch.no_grad()
    def finish_task(self, task_id):
        if task_id != self._current_task_id or bool(self.finalized_tasks[task_id]):
            raise RuntimeError("Task must be active and unfinished")
        records = {}
        for layer, pool in enumerate(self.experts):
            per_view = self.training_counts[task_id, layer]
            counts = per_view.sum(0)
            if int(counts.sum()) == 0:
                raise RuntimeError("Cannot freeze from empty training activation counts")
            # Rank ALL experts, including those frozen earlier; do not top up.
            # Equal counts deterministically prefer the smaller expert ID.
            selected = sorted(range(self.num_experts), key=lambda i: (-int(counts[i]), i))[:2]
            before = self.frozen_experts[layer].clone()
            self.frozen_experts[layer, selected] = True
            self.freeze_top2[task_id, layer] = torch.tensor(selected, device=counts.device)
            self.frozen_history[task_id, layer].copy_(self.frozen_experts[layer])
            for i, expert in enumerate(pool):
                if bool(self.frozen_experts[layer, i]):
                    expert.requires_grad_(False)
                    for p in expert.parameters():
                        p.grad = None
            records[str(self.layer_indices[layer])] = dict(
                counts=counts.tolist(), per_view_counts=per_view.tolist(),
                per_view_samples=self.training_samples[task_id, layer].tolist(),
                top2=selected, newly_frozen=[i for i in selected if not bool(before[i])],
                cumulative_frozen=torch.nonzero(self.frozen_experts[layer]).flatten().tolist(),
                remaining_trainable=int((~self.frozen_experts[layer]).sum()),
            )
        self.finalized_tasks[task_id] = True
        self.tasks[task_id].requires_grad_(False)
        self.tasks[task_id].eval()
        for p in self.tasks[task_id].parameters():
            p.grad = None
        return records

    def delta_for_layer(self, layer_id, task_lane_tokens, lane_ids, image_view):
        if layer_id not in self.layer_indices:
            return torch.zeros_like(task_lane_tokens)
        if image_view not in self.view_names:
            raise ValueError("Unknown view")
        if task_lane_tokens.ndim != 4 or len(lane_ids) != task_lane_tokens.shape[0]:
            raise ValueError("Expected [lane,batch,token,width]")
        layer = self.layer_indices.index(layer_id)
        view = self.view_names.index(image_view)
        outputs = []
        for lane, task in enumerate(lane_ids):
            if not 0 <= task <= self._current_task_id:
                raise ValueError("Future task access forbidden")
            tokens = task_lane_tokens[lane]
            gates = self._route(tokens[:, 0], self.tasks[task][layer], view)
            with torch.no_grad():
                hard = gates > 0
            count_training = (self.training and torch.is_grad_enabled()
                              and task == self._current_task_id
                              and not bool(self.finalized_tasks[task]))
            with torch.no_grad():
                if count_training:
                    mask = torch.ones(len(tokens), device=tokens.device, dtype=torch.bool)
                    if image_view == "face":
                        if self._face_reliable is None or self._face_reliable.shape != mask.shape:
                            raise RuntimeError("Training Face activation counts require reliability mask")
                        mask = self._face_reliable.to(tokens.device)
                    self.training_counts[task, layer, view] += hard[mask].sum(0)
                    self.training_samples[task, layer, view] += mask.sum()
                if not self.training:
                    self.route_counts[task, layer, view] += hard.sum(0)
                    self.route_samples[task, layer, view] += len(tokens)
            combined = torch.zeros_like(tokens)
            for expert_id, expert in enumerate(self.experts[layer]):
                ids = torch.nonzero(hard[:, expert_id], as_tuple=True)[0]
                if not ids.numel():
                    continue
                # Never detach frozen experts: their input and Router gradients
                # must remain live, including when both selected experts freeze.
                value = expert(tokens.index_select(0, ids))
                weight = gates.index_select(0, ids)[:, expert_id]
                combined = combined.index_add(0, ids, (value * weight[:, None, None].to(value.dtype)).to(combined.dtype))
            outputs.append(combined)
        return self.residual_scale * torch.stack(outputs)
