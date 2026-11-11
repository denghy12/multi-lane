"""Task-private, reliability-conditioned fusion of three classifier outputs."""
import math

import torch
from torch import nn


class ReliabilityLogitFusion(nn.Module):
    """Two learned shares; unavailable Face contributes exactly zero."""

    def __init__(self):
        super().__init__()
        # Reliable: 1/3 each. Otherwise: 1/2 Full and 1/2 Person.
        self.raw_shares = nn.Parameter(torch.tensor([0., -math.log(2.)]))

    def weight_matrix(self):
        full, face = self.raw_shares.sigmoid().unbind()
        fallback = torch.stack((full, 1 - full, torch.zeros_like(full)))
        return torch.stack((fallback, torch.stack(((1-face)*full, (1-face)*(1-full), face))))

    def forward(self, logits, reliable):
        if logits.ndim != 3 or logits.shape[1] != 3 or reliable.shape != logits.shape[:1]:
            raise ValueError("Expected [person, three views, class] and [person] reliability")
        if reliable.dtype != torch.bool:
            raise ValueError("Face reliability must be boolean")
        weights = self.weight_matrix()[reliable.long()]
        return (weights[:, 0, None] * logits[:, 0]
                + weights[:, 1, None] * logits[:, 1]
                + weights[:, 2, None] * logits[:, 2])


class TaskLogitFusion(nn.Module):
    """A growing bank of fusion parameters sealed at each task boundary."""

    def __init__(self, task_sizes):
        super().__init__()
        self.task_sizes = tuple(task_sizes)
        self.tasks = nn.ModuleList()

    def add_task(self, task):
        if task != len(self.tasks) or task >= len(self.task_sizes):
            raise ValueError("Fusion tasks must be introduced in order")
        self.tasks.requires_grad_(False)
        gate = ReliabilityLogitFusion()
        if self.tasks:
            gate.to(self.tasks[0].raw_shares.device)
        self.tasks.append(gate)
        return gate

    def seal(self):
        self.requires_grad_(False)
        return self.eval()

    def forward(self, logits, reliable, stage):
        if not 0 <= stage < len(self.tasks):
            raise ValueError("Untrained fusion task")
        if logits.shape[-1] != sum(self.task_sizes[:stage+1]):
            raise ValueError("Expected exactly the seen task class columns")
        start, blocks = 0, []
        for task in range(stage + 1):
            end = start + self.task_sizes[task]
            blocks.append(self.tasks[task](logits[:, :, start:end], reliable))
            start = end
        return torch.cat(blocks, dim=-1)
