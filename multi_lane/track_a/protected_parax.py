"""Task-protected ParaX parameter mixing after Task Forward.

An expert is a pair of parameter matrices, not an independent output branch.
Old tasks retain their Router, projection and exact eligible expert set.
"""
from __future__ import annotations

import torch
from torch import nn

from .adapter import ParaXImageAdapterBank

PROTECTED_MODES = {
    "post_task_protected_frozen": "frozen_pool",
    "post_task_protected_fresh": "fresh_only",
    "post_task_protected_reuse": "reuse_old",
}


class ProtectedParaXBank(ParaXImageAdapterBank):
    def __init__(self, *args, policy: str, **kwargs):
        if policy not in PROTECTED_MODES.values():
            raise ValueError("Unknown protected expert policy")
        super().__init__(*args, **kwargs)
        if (not self.task_local_router or not self.task_local_projection
                or self.initialization != "zero_output" or self.output_scale_mode != "fixed"
                or self.smooth_ratio_bound <= 0 or self.level_conditioned
                or self.task_local_gate or self.projector is not None or self.static
                or self.trainable_components != "all" or self.residual_ratio_cap
                or self.residual_scale <= 0 or len(self.layer_indices) != 1):
            raise ValueError("Protected routing requires zero task projection, fixed scale, smooth bound and all current components")
        if self.num_experts < 2:
            raise ValueError("Task0 needs two initial experts")
        self.policy = policy
        # Independent Parameters prevent optimizer momentum/weight decay from
        # touching frozen slices of an otherwise trainable stacked tensor.
        a, b = self.expert_a.detach().clone(), self.expert_b.detach().clone()
        del self.expert_a, self.expert_b
        self.expert_a = nn.ParameterList(nn.Parameter(value) for value in a)
        self.expert_b = nn.ParameterList(nn.Parameter(value) for value in b)
        access = torch.zeros(self.num_tasks, self.num_experts, dtype=torch.bool)
        supported = self.num_tasks if policy == "frozen_pool" else min(self.num_tasks, self.num_experts // 2)
        for task in range(supported):
            if task == 0 or policy == "frozen_pool":
                access[task, :2] = True
            elif policy == "fresh_only":
                access[task, 2 * task:2 * task + 2] = True
            else:
                access[task, :2 * task + 2] = True
        self.register_buffer("expert_access", access)
        self.register_buffer("sealed_tasks", torch.zeros(self.num_tasks, dtype=torch.bool))
        self.requires_grad_(False)

    def new_expert_ids(self, task_id: int):
        if task_id == 0:
            return (0, 1)
        return () if self.policy == "frozen_pool" else (2 * task_id, 2 * task_id + 1)

    def _check_task(self, task_id):
        if not 0 <= task_id < self.num_tasks or not bool(self.expert_access[task_id].any()):
            raise ValueError("Task exceeds protected expert budget")

    def activate_task(self, task_id):
        self._check_task(task_id)
        if bool(self.sealed_tasks[task_id]) or not bool(self.sealed_tasks[:task_id].all()):
            raise RuntimeError("Seal previous tasks before activating a new task")
        super().activate_task(task_id)

    def restore_task(self, task_id):
        self._check_task(task_id)
        super().restore_task(task_id)

    def seal_task(self, task_id):
        if task_id != self._current_task_id:
            raise ValueError("Can only seal the active task")
        self.sealed_tasks[task_id] = True
        self._set_trainability()

    def _set_trainability(self):
        self.requires_grad_(False)
        task = self._current_task_id
        if bool(self.sealed_tasks[task]):
            return
        for index in self.new_expert_ids(task):
            self.expert_a[index].requires_grad_(True)
            self.expert_b[index].requires_grad_(True)
        self.task_projections[task].requires_grad_(True)
        for layer in self.layer_indices:
            self.routers[f"{task}_{layer}"].requires_grad_(True)
        # LayerNorm, unused shared projection, and scale remain immutable.

    def _expert_matrices(self, tokens):
        return torch.stack(tuple(self.expert_a)).to(tokens), torch.stack(tuple(self.expert_b)).to(tokens)

    def _router_gates(self, key, inputs, task_id):
        self._check_task(task_id)
        logits = self.routers[key](inputs)
        return torch.softmax(logits.masked_fill(~self.expert_access[task_id], -torch.inf), dim=-1)

    def protected_state(self, before_task: int):
        """Exact tensors consumed by old paths (excluding task counters)."""
        used = self.expert_access[:before_task].any(dim=0)
        names = {f"expert_{side}.{i}" for side in ("a", "b")
                 for i in range(self.num_experts) if bool(used[i])}
        prefixes = tuple(f"routers.{t}_" for t in range(before_task)) + tuple(
            f"task_projections.{t}." for t in range(before_task))
        result = {name: value for name, value in self.named_parameters()
                  if name in names or name.startswith(prefixes)
                  or name.startswith(("norm.", "proj.")) or name == "output_scale"}
        result["old_expert_access"] = self.expert_access[:before_task]
        result["old_sealed_tasks"] = self.sealed_tasks[:before_task]
        return result

    def policy_summary(self, task_id):
        return {"policy": self.policy, "eligible_experts": self.expert_access[task_id].nonzero().flatten().tolist(),
                "new_experts": list(self.new_expert_ids(task_id)),
                "trainable_parameters": sum(p.numel() for p in self.active_parameters()),
                "total_parameters": self.parameter_count(), "sealed": bool(self.sealed_tasks[task_id])}
