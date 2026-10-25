"""Task-local person/patch geometry in the original Full summarization logits."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class TaskSpatialSummary(nn.Module):
    MODES = ("disabled", "absolute", "target")

    def __init__(self, num_tasks: int, num_selectors: int, mode: str, bias_mode: str = 'raw'):
        super().__init__()
        if mode not in self.MODES[1:] or min(num_tasks, num_selectors) <= 0:
            raise ValueError("Invalid spatial summary configuration")
        self.mode = mode
        if bias_mode not in {'raw', 'content_relative'}:
            raise ValueError('Invalid spatial bias calibration')
        self.bias_mode = bias_mode
        self.tasks = nn.ParameterList([
            nn.Parameter(torch.zeros(num_selectors, 6)) for _ in range(num_tasks)
        ])
        self.current_task_id = -1
        self.requires_grad_(False)
        self.reset_statistics()

    def restore_task(self, task):
        if not -1 <= task < len(self.tasks):
            raise ValueError("Invalid spatial summary task")
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        if task >= 0:
            self.tasks[task].requires_grad_(True)
        self.current_task_id = task

    def active_parameters(self):
        if self.current_task_id >= 0:
            yield self.tasks[self.current_task_id]

    def parameter_count_per_task(self):
        return self.tasks[0].numel()

    def geometry_features(self, boxes, visibility, token_count):
        if boxes.ndim != 2 or boxes.shape[1] != 4 or visibility.shape != boxes.shape[:1]:
            raise ValueError("Spatial geometry must be [batch,4] plus [batch] visibility")
        grid = math.isqrt(token_count - 1)
        if grid <= 0 or grid * grid != token_count - 1:
            raise ValueError("Spatial summary requires a square patch grid and one CLS")
        boxes = boxes.detach().float()
        visibility = visibility.detach().float()
        valid = torch.isfinite(boxes).all(-1) & torch.isfinite(visibility)
        valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        safe = boxes.new_tensor([0., 0., 1., 1.]).expand_as(boxes)
        boxes = torch.where(valid[:, None], boxes, safe)
        v = torch.where(valid, visibility.clamp(0, 1), torch.zeros_like(visibility))
        real_boxes = boxes
        # The control receives the SAME real visibility, but no target location.
        if self.mode == "absolute":
            boxes = safe
        axis = (torch.arange(grid, device=boxes.device, dtype=torch.float32) + .5) / grid
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        x, y = x.reshape(1, -1), y.reshape(1, -1)
        width = (boxes[:, 2] - boxes[:, 0]).clamp_min(1 / grid)[:, None]
        height = (boxes[:, 3] - boxes[:, 1]).clamp_min(1 / grid)[:, None]
        dx = ((x - (boxes[:, 0] + boxes[:, 2])[:, None] / 2) / width).clamp(-2, 2) / 2
        dy = ((y - (boxes[:, 1] + boxes[:, 3])[:, None] / 2) / height).clamp(-2, 2) / 2
        half = .5 / grid
        overlap_x = (torch.minimum(x + half, boxes[:, 2, None])
                     - torch.maximum(x - half, boxes[:, 0, None])).clamp_min(0)
        overlap_y = (torch.minimum(y + half, boxes[:, 3, None])
                     - torch.maximum(y - half, boxes[:, 1, None])).clamp_min(0)
        overlap = (overlap_x * overlap_y * grid ** 2).clamp(0, 1)
        phi = torch.stack((dx, dy, dx.abs(), dy.abs(), overlap, torch.ones_like(dx)), -1)
        real_x = (torch.minimum(x + half, real_boxes[:, 2, None])
                  - torch.maximum(x - half, real_boxes[:, 0, None])).clamp_min(0)
        real_y = (torch.minimum(y + half, real_boxes[:, 3, None])
                  - torch.maximum(y - half, real_boxes[:, 1, None])).clamp_min(0)
        actual_overlap = ((real_x * real_y * grid ** 2).clamp(0, 1)
                          * (valid & (v > 0))[:, None])
        return F.pad(phi, (0, 0, 1, 0)), v, F.pad(actual_overlap, (1, 0))

    def bias(self, boxes, visibility, token_count, lane_ids):
        phi, v, actual_overlap = self.geometry_features(boxes, visibility, token_count)
        weights = torch.stack([self.tasks[t] for t in lane_ids])
        # A zero parameter matrix exactly preserves original similarity logits.
        with torch.autocast(device_type=boxes.device.type, enabled=False):
            bias = torch.einsum("tmf,bnf->tbmn", weights.float(), phi).tanh() * v[None, :, None, None]
        return bias, actual_overlap, v

    def attention(self, similarity, bias):
        if self.bias_mode == 'raw':
            return (similarity + bias.to(similarity.dtype)).softmax(-1), bias
        # Match each layer/selector's evidence scale without changing the
        # zero-bias model or backpropagating through the calibration statistic.
        with torch.autocast(device_type=similarity.device.type, enabled=False):
            scale = similarity.detach().float()[..., 1:].std(-1, keepdim=True, unbiased=False).clamp(.25, 4.)
            effective = bias.float() * scale
            attention = (similarity.float() + effective).softmax(-1)
        return attention.to(similarity.dtype), effective

    def reset_statistics(self):
        self._statistics = {}

    @torch.no_grad()
    def observe(self, layer, lane_ids, bias, overlap, visibility, similarity, attention):
        original = similarity.detach().float().softmax(-1)
        attention = attention.detach().float()
        for index, task in enumerate(lane_ids):
            a, b, p = attention[index], bias[index].detach().float(), original[index]
            n = len(a)
            row = dict(samples=a.new_tensor(n),
                       bias_abs_sum=b.abs().mean((1, 2)).sum(),
                       saturation_sum=(b.abs() > .95 * visibility[:, None, None].clamp_min(1e-6)).float().mean((1, 2)).sum(),
                       visible_sum=(visibility > 0).float().sum(),
                       cls_mass_sum=a[..., 0].mean(-1).sum(),
                       target_overlap_mass_sum=(a * overlap[:, None]).sum(-1).mean(-1).sum(),
                       attention_kl_sum=(a * (a.clamp_min(1e-30).log() - p.clamp_min(1e-30).log())).sum(-1).mean(-1).sum())
            content_std = similarity[index].detach().float()[..., 1:].std(-1, unbiased=False).clamp_min(1e-8)
            row['content_logit_std_sum'] = content_std.mean(-1).sum()
            row['bias_to_content_std_sum'] = (b.abs().mean(-1) / content_std).mean(-1).sum()
            key = f"task{task}/layer{layer}"
            if key not in self._statistics:
                self._statistics[key] = row
            else:
                for name, value in row.items():
                    self._statistics[key][name].add_(value)

    def statistics(self):
        return {key: {"samples": int(row["samples"].item()), **{
            name.removesuffix("_sum") + "_mean": float((value / row["samples"]).item())
            for name, value in row.items() if name != "samples"
        }} for key, row in self._statistics.items()}
