"""Task-local attention between summarized views, before task MSA.

Full -> Person denotes information flow: Person supplies Q, Full supplies K/V.
The two controls have identical parameters and write to the same residual.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class SummaryAttention(nn.Module):
    def __init__(self, width, inner_dim, heads, scale):
        super().__init__()
        self.width, self.heads, self.scale = width, heads, scale
        self.head_dim = inner_dim // heads
        self.q = nn.Linear(width, inner_dim, bias=False)
        self.k = nn.Linear(width, inner_dim, bias=False)
        self.v = nn.Linear(width, inner_dim, bias=False)
        self.out = nn.Linear(inner_dim, width, bias=False)
        for layer in (self.q, self.k, self.v):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(self.out.weight)

    def forward(self, query, source):
        batch, count, _ = query.shape
        norm_q = F.layer_norm(query.float(), (self.width,)).to(query.dtype)
        norm_s = F.layer_norm(source.float(), (self.width,)).to(source.dtype)
        def split(x):
            return x.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q(norm_q)), split(self.k(norm_s)), split(self.v(norm_s))
        # Tiny 10x10 attention stays FP32 under AMP; projections use autocast.
        with torch.autocast(device_type=query.device.type, enabled=False):
            weights = (torch.matmul(q.float(), k.float().transpose(-2, -1))
                       * self.head_dim ** -0.5).softmax(-1)
            read = torch.matmul(weights, v.float()).transpose(1, 2).reshape(batch, count, -1)
        delta = self.scale * self.out(read.to(query.dtype))
        return delta, weights, norm_q


class TaskCrossViewAttentionBank(nn.Module):
    MODES = ('disabled', 'person_self', 'task_full', 'person_full')

    def __init__(self, num_tasks, width, mode, layer_index=5, inner_dim=64,
                 heads=4, scale=.1):
        super().__init__()
        if (mode not in self.MODES[1:] or min(num_tasks, width, inner_dim, heads) <= 0
                or inner_dim % heads or layer_index < 0
                or not math.isfinite(scale) or not 0 < scale <= 1):
            raise ValueError('Invalid cross-view attention configuration')
        self.mode, self.layer_index = mode, layer_index
        self.current_task_id = -1
        self.tasks = nn.ModuleList()
        with torch.random.fork_rng(devices=[]):
            for task in range(num_tasks):
                # Seed CPU weights only; torch.manual_seed would also reset CUDA RNG.
                torch.default_generator.manual_seed(101_003 + task)
                self.tasks.append(SummaryAttention(width, inner_dim, heads, scale))
        self.requires_grad_(False)
        self.reset_statistics()

    def restore_task(self, task):
        if not -1 <= task < len(self.tasks):
            raise ValueError('Invalid cross-view task')
        self.requires_grad_(False)
        for p in self.parameters():
            p.grad = None
        if task >= 0:
            self.tasks[task].requires_grad_(True)
        self.current_task_id = task

    def active_parameters(self):
        if self.current_task_id >= 0:
            yield from self.tasks[self.current_task_id].parameters()

    def parameter_count_per_task(self):
        return sum(p.numel() for p in self.tasks[0].parameters())

    def reset_statistics(self):
        self._statistics = {}

    def statistics(self):
        out = {}
        for key, row in self._statistics.items():
            n = row['samples']
            result = {k: v.cpu().tolist() for k, v in row.items()}
            for name in ('base_norm', 'residual_norm', 'ratio', 'cosine', 'entropy'):
                result[name+'_mean'] = float((row[name+'_sum']/n).cpu())
            result['query_std_across_images'] = (
                row['query_square_sum']/n-(row['query_sum']/n).square()).clamp_min(0).sqrt().cpu().tolist()
            result['attention_mean'] = (row['attention_sum']/n).cpu().tolist()
            result['attention_std_across_images'] = (
                row['attention_square_sum']/n-(row['attention_sum']/n).square()).clamp_min(0).sqrt().cpu().tolist()
            out[key] = result
        return out

    def forward(self, person, full, selectors, lane_ids):
        if (person.ndim != 4 or person.shape != full.shape or selectors.shape != person.shape
                or len(lane_ids) != person.shape[0]):
            raise ValueError('Cross-view summary shapes/tasks do not match')
        result = []
        for index, task in enumerate(lane_ids):
            p, f, s = person[index], full[index], selectors[index]
            query = s if self.mode == 'task_full' else p
            source = p if self.mode == 'person_self' else f
            delta, weights, normalized_query = self.tasks[task](query, source)
            delta = delta.to(p.dtype)
            result.append(p + delta)
            with torch.no_grad():
                bnorm = p.float().norm(dim=-1).mean(-1)
                dnorm = delta.float().norm(dim=-1).mean(-1)
                attention = weights.double().mean(1)
                qmean = normalized_query.double().mean(1)
                row = dict(samples=bnorm.new_tensor(len(p), dtype=torch.float64),
                    base_norm_sum=bnorm.double().sum(), residual_norm_sum=dnorm.double().sum(),
                    ratio_sum=(dnorm/bnorm.clamp_min(1e-12)).double().sum(),
                    cosine_sum=F.cosine_similarity(p.float(), delta.float(), dim=-1).mean(-1).double().sum(),
                    entropy_sum=(-(weights.double()*weights.double().clamp_min(1e-30).log()).sum(-1)).mean((1,2)).sum(),
                    query_sum=qmean.sum(0), query_square_sum=qmean.square().sum(0),
                    attention_sum=attention.sum(0), attention_square_sum=attention.square().sum(0))
                key = f'task{task}/layer{self.layer_index}'
                if key not in self._statistics:
                    self._statistics[key] = row
                else:
                    for name, value in row.items():
                        self._statistics[key][name].add_(value)
        return torch.stack(result)
