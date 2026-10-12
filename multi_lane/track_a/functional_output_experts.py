"""Protected complete residual functions; mix outputs, never A/B matrices.

The two experts born in task s share its P_s, which becomes immutable with
them. LayerNorm and the fixed scale are immutable across all tasks. Routers
only combine complete functions; no new-task projection is applied to old
expert outputs. This is a functional MoE-style control, not original ParaX.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .protected_parax import ProtectedParaXBank
from .protected_parax_audit import ProtectedRouteAudit

COMBINATION = "complete_residual_outputs_v1"


class FunctionalOutputExpertBank(ProtectedParaXBank):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_experts != 2 * self.num_tasks:
            raise ValueError("Complete output experts require two slots per task")
        self.register_buffer("expert_owners", torch.arange(self.num_experts) // 2)
        # A signed bounded old-contribution coefficient has a nonzero
        # derivative at zero. A nonnegative smooth interior minimum would
        # have zero derivative and could never enable old functions.
        self.reuse_gains = nn.ParameterList(nn.Parameter(torch.zeros(()), requires_grad=False)
                                            for _ in range(self.num_tasks))
        self.last_output_combination_diagnostics = {}

    def _set_trainability(self):
        super()._set_trainability()
        # Parent construction/restore can run before these extra parameters
        # exist. Never activate an old scalar through a stacked Parameter.
        if hasattr(self, "reuse_gains") and not bool(self.sealed_tasks[self._current_task_id]):
            self.reuse_gains[self._current_task_id].requires_grad_(True)

    def protected_state(self, before_task):
        state = super().protected_state(before_task)
        state.update({f"reuse_gains.{t}": self.reuse_gains[t] for t in range(before_task)})
        state["expert_owners"] = self.expert_owners
        return state

    def expert_output(self, expert_id, tokens):
        """An independently evaluable raw residual with its ORIGINAL P_s."""
        if not 0 <= expert_id < self.num_experts:
            raise ValueError("Expert outside bank")
        owner = int(self.expert_owners[expert_id])
        if owner > self._current_task_id:
            raise ValueError("Cannot evaluate a future expert")
        low = F.linear(self.norm(tokens), self.expert_a[expert_id])
        low = F.gelu(self.task_projections[owner](low))
        return self.output_scale.to(tokens) * F.linear(low, self.expert_b[expert_id])

    def forward(self, layer_id, tokens, view_name="full", task_id=None):
        if layer_id not in self.layer_indices:
            return tokens, tokens.new_zeros((tokens.shape[0], self.num_experts))
        if tokens.ndim != 3 or tokens.shape[-1] != self.hidden_dim:
            raise ValueError("Expected [batch,tokens,hidden_dim]")
        task = self._current_task_id if task_id is None else int(task_id)
        self._check_task(task)
        if task > self._current_task_id:
            raise ValueError("Router cannot access a future task")
        pooled = tokens.mean(dim=1).float()
        if self.router_input_mode == "constant":
            pooled = torch.ones_like(pooled)
        elif self.router_input_mode != "image":
            raise ValueError("Unknown Router input mode")
        logits = self.routers[f"{task}_{layer_id}"](pooled)
        selection = logits.masked_fill(~self.expert_access[task], -torch.inf).softmax(-1)
        new = list(self.new_expert_ids(task))
        old = [i for i in self.expert_access[task].nonzero().flatten().tolist() if i not in new]

        def combine(ids):
            # Group normalization keeps the new-expert contribution from
            # shrinking merely because more frozen experts became eligible.
            weights = logits[:, ids].softmax(-1).to(tokens)
            outputs = torch.stack([self.expert_output(i, tokens) for i in ids], dim=1)
            return (weights[:, :, None, None] * outputs).sum(dim=1)

        delta = combine(new)
        gain = self.reuse_gains[task].tanh().to(tokens)
        if old:
            delta = delta + gain * combine(old)
        radius = self.smooth_ratio_bound * tokens.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        factor = radius / (radius.square() + delta.float().square().sum(-1, keepdim=True)).sqrt()
        delta = delta * factor.to(delta)
        ratio = delta.float().norm(dim=-1).mean(dim=1) / tokens.float().norm(dim=-1).mean(dim=1).clamp_min(1e-12)
        self._last_residual_penalties.append(ratio.square().mean())
        self.last_raw_residual_ratio = float(ratio.detach().mean().cpu())
        self.last_projector_correction_ratio = 0.
        self.last_output_combination_diagnostics = {
            "task": task, "old_gain": float(gain.detach().cpu()),
            "new_group_mass": 1., "old_group_mass": float(gain.detach().cpu()) if old else 0.,
            "selection_semantics": "global eligible softmax BEFORE conditional group normalization and signed old gain",
        }
        return tokens + delta, selection.to(tokens)

    def policy_summary(self, task_id):
        return {**super().policy_summary(task_id), "combination": COMBINATION,
                "expert_owners": self.expert_owners.tolist(),
                "projection_policy": "each expert uses its birth-task P; two newborns share P",
                "old_gain": float(self.reuse_gains[task_id].detach().tanh().cpu()),
                "old_gain_range": [-1., 1.], "new_group_mass": 1.,
                "selection_gate_semantics": "eligible softmax, not effective signed output coefficients",
                "group_normalization": "separate old/new softmax"}


class FunctionalRouteAudit(ProtectedRouteAudit):
    def _identity_check(self, model, images, task):
        bank = model.parax_bank
        if not isinstance(bank, FunctionalOutputExpertBank):
            raise ValueError("Expected functional output experts")
        if bool(torch.count_nonzero(bank.reuse_gains[task])):
            raise RuntimeError("New task old-contribution gate must start at zero")
        return {**super()._identity_check(model, images, task), "old_gain_strictly_zero": True,
                "complete_expert_combination": COMBINATION}
