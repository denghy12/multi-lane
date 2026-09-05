"""Small, task-isolated residual queries conditioned on the target person."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
from torch import nn


class TaskSelectorConditioner(nn.Module):
    """Task-isolated residual query conditioning for Full selectors.

    Global descriptor modes broadcast one delta over selectors. Person-patch
    mode produces one delta per selector from same-depth frozen tokens. Only
    the MLPs are trained. Each new task starts independently with zero output;
    old task modules are excluded from subsequent optimizers.
    """

    MODES = ("disabled", "bbox", "person", "bbox_person", "person_patches")

    def __init__(self, num_tasks: int, width: int, mode: str,
                 layer_indices: Sequence[int], hidden_dim: int = 32,
                 residual_scale: float = 0.1) -> None:
        super().__init__()
        if mode not in self.MODES[1:]:
            raise ValueError("Unknown enabled selector conditioning mode")
        layers = tuple(int(i) for i in layer_indices)
        if (not layers or len(set(layers)) != len(layers)
                or min(layers) < 0 or min(num_tasks, width, hidden_dim) <= 0):
            raise ValueError("Invalid selector conditioning dimensions/layers")
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("Selector conditioning scale must be finite and positive")
        self.mode = mode
        self.layer_indices = tuple(sorted(layers))
        self.residual_scale = float(residual_scale)
        self.current_task_id = -1
        input_dim = (
            width if mode == "person_patches"
            else (width if "person" in mode else 0) + (6 if "bbox" in mode else 0)
        )
        self.task_modules = nn.ModuleList()
        for _ in range(num_tasks):
            modules = nn.ModuleDict()
            for layer in self.layer_indices:
                mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, width))
                nn.init.xavier_uniform_(mlp[0].weight)
                nn.init.zeros_(mlp[0].bias)
                nn.init.zeros_(mlp[2].weight)
                nn.init.zeros_(mlp[2].bias)
                modules[str(layer)] = mlp
            self.task_modules.append(modules)
        self.requires_grad_(False)

    def restore_task(self, task_id: int) -> None:
        if not -1 <= task_id < len(self.task_modules):
            raise ValueError("Invalid selector conditioning task")
        self.requires_grad_(False)
        # Clear stale gradients as well as disabling future autograd.
        for parameter in self.parameters():
            parameter.grad = None
        if task_id >= 0:
            self.task_modules[task_id].requires_grad_(True)
        self.current_task_id = task_id

    def active_parameters(self) -> Iterable[nn.Parameter]:
        if self.current_task_id >= 0:
            yield from self.task_modules[self.current_task_id].parameters()

    def query_delta(self, layer_id: int, lane_ids: Sequence[int],
                    person: torch.Tensor | None, bbox: torch.Tensor,
                    valid: torch.Tensor) -> torch.Tensor:
        if self.mode == "person_patches":
            raise RuntimeError("Person-patch conditioning requires selector queries")
        parts = []
        if "person" in self.mode:
            if person is None:
                raise ValueError("Person conditioning requires a person descriptor")
            parts.append(person.detach())
        if "bbox" in self.mode:
            parts.append(bbox.detach())
        features = torch.cat(parts, dim=-1)
        deltas = torch.stack([
            self.task_modules[t][str(layer_id)](features) for t in lane_ids
        ])
        # Missing/fully invisible target: exactly the original query pathway.
        return (deltas * valid.to(deltas.dtype)[None, :, None]
                * self.residual_scale).unsqueeze(2)

    def patch_query_delta(
        self,
        layer_id: int,
        lane_ids: Sequence[int],
        selectors: torch.Tensor,
        person_patches: torch.Tensor,
        patch_mask: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Produce one residual per Selector from same-depth Person patches.

        ``selectors`` is [tasks,batch,selectors,width], while frozen Person
        patches are [batch,patches,width]. Padding is excluded before softmax.
        The task-specific zero-output MLP preserves the original query path at
        initialization and is the only trainable part of this interaction.
        """
        if self.mode != "person_patches":
            raise RuntimeError("Patch query deltas require person_patches mode")
        if (selectors.ndim != 4 or person_patches.ndim != 3
                or patch_mask.shape != person_patches.shape[:2]
                or valid.shape != person_patches.shape[:1]
                or selectors.shape[1] != person_patches.shape[0]
                or selectors.shape[-1] != person_patches.shape[-1]):
            raise ValueError("Invalid selector/Person patch conditioning shapes")
        mask = patch_mask.to(device=person_patches.device, dtype=torch.bool)
        # Fail closed for a malformed all-padding row, then mask that sample's
        # output with `valid` below.
        safe_mask = mask | (~mask.any(dim=1, keepdim=True))
        similarity = torch.einsum(
            "tbsd,bnd->tbsn", selectors, person_patches.detach()
        ) * (selectors.shape[-1] ** -0.5)
        similarity = similarity.masked_fill(~safe_mask[None, :, None, :], -torch.inf)
        summaries = torch.einsum(
            "bnd,tbsn->tbsd",
            person_patches.detach(),
            torch.softmax(similarity, dim=-1),
        )
        deltas = torch.stack([
            self.task_modules[task_id][str(layer_id)](summaries[index])
            for index, task_id in enumerate(lane_ids)
        ])
        return (deltas * valid.to(deltas.dtype)[None, :, None, None]
                * self.residual_scale)
