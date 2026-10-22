"""Task-local context keys/values; never overwrite Person summarized patches."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class SummaryContextReader(nn.Module):
    """Learn where to read, retaining original CLIP-space source values.

    Head attention maps are averaged before reading the unprojected values.
    The receiving frozen task MSA subsequently supplies its own K/V projections.
    """

    def __init__(self, width, inner_dim, heads, max_mass):
        super().__init__()
        if not math.isfinite(max_mass) or not 0 < max_mass <= .5:
            raise ValueError('Context mass cap must be finite and in (0, .5]')
        self.width, self.heads = width, heads
        self.head_dim, self.max_mass = inner_dim // heads, max_mass
        self.q = nn.Linear(width, inner_dim, bias=False)
        self.k = nn.Linear(width, inner_dim, bias=False)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        # At max_mass=.1, initially 5% of receiving attention goes to context.
        # This is deliberately near-identity, not exact zero initialization.
        self.gate_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, query, source, query_override=None, query_mask=None):
        batch, count, _ = query.shape
        normalized = F.layer_norm(query.float(), (self.width,)).to(query.dtype)
        if query_override is not None:
            alternative = F.layer_norm(query_override.float(), (self.width,)).to(query.dtype)
            normalized = torch.where(query_mask[:, None, None], alternative, normalized)
        source_norm = F.layer_norm(source.float(), (self.width,)).to(source.dtype)
        def split(value):
            return value.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)
        q, k = split(self.q(normalized)), split(self.k(source_norm))
        with torch.autocast(device_type=query.device.type, enabled=False):
            weights = (q.float() @ k.float().transpose(-2, -1)
                       * self.head_dim ** -.5).softmax(-1)
            context = weights.mean(1) @ source.float()
            mass = self.max_mass * self.gate_logit.float().sigmoid()
        return context.to(query.dtype), weights, normalized, mass


def mix_context_attention(query, base_output, context, attention, heads, head_dim, mass):
    """Grouped attention: original keys retain >=1-max_mass probability.

    Equivalent to attention over original and context key groups with separately
    normalized probabilities and explicit group mass. Prefix keys stay in the
    original group. Only original task queries are evaluated.
    """
    tasks, batch, count, width = context.shape
    projected = F.linear(context, attention.in_proj_weight, attention.in_proj_bias)
    projected = projected.reshape(tasks, batch, count, 3, heads, head_dim)
    projected = projected.permute(3, 0, 1, 4, 2, 5)
    _, key, value = projected.unbind(0)
    with torch.autocast(device_type=query.device.type, enabled=False):
        weights = (query.float() @ key.float().transpose(-2, -1)
                   * head_dim ** -.5).softmax(-1)
        read = weights @ value.float()
        gate = mass.float()[:, None, None, None, None]
        output = (1 - gate) * base_output.float() + gate * read
    return output.to(base_output.dtype)
