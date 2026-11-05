"""Task-private Face CLS communication inside the shared historical B0."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .cross_view_attention import bounded_summary_delta

MODES = ('disabled', 'pool', 'attention')


class FaceCLSReadout(nn.Module):
    def __init__(self, width, dim, heads, mode, cap):
        super().__init__()
        if mode not in MODES[1:] or dim % heads or dim // heads < 2 or not 0 < cap <= 1:
            raise ValueError('Invalid shared Face readout dimensions or bound')
        self.mode, self.dim, self.heads, self.cap = mode, dim, heads, cap
        # All common parameters have identical initial values in both readers.
        self.types = nn.Parameter(torch.empty(2, width))
        nn.init.normal_(self.types, std=.02)
        self.v = nn.Linear(width, dim, bias=False)
        if mode == 'pool':
            hidden = max(1, round(3 * width * dim / (2 * width + 2 * dim)))
            self.mlp = nn.Linear(width + 2 * dim, hidden, bias=False)
            self.out = nn.Linear(hidden, width, bias=False)
        else:
            self.q = nn.Linear(width, dim, bias=False)
            self.k = nn.Linear(width, dim, bias=False)
            self.out = nn.Linear(dim, width, bias=False)
        nn.init.zeros_(self.out.weight)
        self.last = {}

    def forward(self, cls, memories):
        # Full/Person Frozen Forward tokens are evidence only. Receiver CLS
        # remains differentiable, as do the later frozen-weight task blocks.
        with torch.autocast(device_type=cls.device.type, enabled=False):
            query = F.layer_norm(cls.float(), (cls.shape[-1],))
            sources = [F.layer_norm(x.detach().float() + self.types[i].float(), (x.shape[-1],))
                       for i, x in enumerate(memories)]
            if self.mode == 'pool':
                pools = [self.v(x).mean(1) for x in sources]
                delta = self.out(F.gelu(self.mlp(torch.cat([query[:, 0], *pools], -1))))[:, None]
                distribution = None
            else:
                memory = torch.cat([query, *sources], 1)
                b, _, _ = cls.shape
                d = self.dim // self.heads
                q = self.q(query).reshape(b, 1, self.heads, d).transpose(1, 2)
                k = self.k(memory).reshape(b, -1, self.heads, d).transpose(1, 2)
                v = self.v(memory).reshape(b, -1, self.heads, d).transpose(1, 2)
                # Input LN alone does not constrain learned Q/K projection scale.
                q = F.layer_norm(q, (d,))
                k = F.layer_norm(k, (d,))
                weights = ((q @ k.transpose(-2, -1)) / math.sqrt(d)).softmax(-1)
                delta = self.out((weights @ v).transpose(1, 2).reshape(b, 1, self.dim))
                n = sources[0].shape[1]
                distribution = torch.stack([weights[..., :1].sum(-1).mean(),
                    weights[..., 1:1+n].sum(-1).mean(), weights[..., 1+n:].sum(-1).mean()])
            bounded = bounded_summary_delta(delta, cls.float(), self.cap)
            self.last = dict(raw_norm=delta.detach().norm(dim=-1).mean(),
                residual_ratio=(bounded.detach().norm(dim=-1) / cls.detach().float().norm(dim=-1).clamp_min(1e-8)).mean(),
                source_weights=None if distribution is None else distribution.detach())
        return cls + bounded.to(cls.dtype)


class TaskFaceCLSBank(nn.Module):
    def __init__(self, tasks, width, mode, layer=5, dim=64, heads=4, cap=.05):
        super().__init__()
        self.mode, self.layer = mode, layer
        self.runtime_enabled = True
        # Never consume the initialization/augmentation/sampler RNG of B0.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261105)
            self.tasks = nn.ModuleList()
            for task in range(tasks):
                torch.random.default_generator.manual_seed(20261105 + task)
                self.tasks.append(FaceCLSReadout(width, dim, heads, mode, cap))
        self.current = -1
        self.requires_grad_(False)

    def restore_task(self, task):
        self.current = task
        self.requires_grad_(False)
        if task >= 0:
            self.tasks[task].requires_grad_(True)

    def active_parameters(self):
        if self.current >= 0:
            yield from self.tasks[self.current].parameters()

    def parameter_count_per_task(self):
        return sum(p.numel() for p in self.tasks[0].parameters())

    def apply_face(self, lanes, memories, reliable, lane_ids):
        if not self.runtime_enabled or not bool(reliable.any()):
            return lanes
        results = []
        for i, task in enumerate(lane_ids):
            cls = lanes[i, :, :1]
            new = cls.clone()
            new[reliable] = self.tasks[task](cls[reliable], [x[reliable] for x in memories])
            results.append(torch.cat([new, lanes[i, :, 1:]], 1))
        return torch.stack(results)

    def statistics(self):
        return {str(t): {k: None if v is None else v.detach().float().cpu().tolist()
                        for k, v in reader.last.items()} for t, reader in enumerate(self.tasks) if reader.last}
