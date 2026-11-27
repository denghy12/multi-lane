"""Read every frozen spatial patch to learn image-level, bounded logit weights."""
import math

import torch
from torch import nn
from torch.nn import functional as F

VIEWS = ('full', 'person', 'face')
METHODS = ('fixed', 'initial', 'shared', 'cls', 'cls_mean', 'cls_regions', 'cls_patch', 'cls_patch_full')
LEARNED = METHODS[2:]
FEATURE_METHODS = METHODS[3:]
NAMES = dict(zip(METHODS, ('原固定比例', '初始分数不学习', '当前任务共用比例', '最终Task CLS',
    'CLS加全部patch等权平均', 'CLS加四分区patch', 'CLS加可学习patch汇总', '辅助patch读取再参考Full CLS')))
PROJECTION_SEED = 20261128
WIDTH = 32
HIDDEN = 8


@torch.no_grad()
def project_tokens(value, view, kind, width=WIDTH):
    """Fixed token-wise channel compression, never fitted to labels or a split."""
    if view not in VIEWS or kind not in ('cls', 'patch') or value.shape[-1] < width:
        raise ValueError('Invalid frozen token projection')
    seed = PROJECTION_SEED + 10*VIEWS.index(view) + (kind == 'patch')
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(value.shape[-1], width, generator=generator)
    matrix = F.normalize(matrix, dim=0).to(value.device)
    with torch.autocast(device_type=value.device.type, enabled=False):
        return F.layer_norm(value.detach().float(), (value.shape[-1],)) @ matrix


def roster_mean(value, counts, valid=None):
    """Contiguous rosters; CPU prefix sums retain gradients on PyTorch 2.0 CUDA.

    Only already-pooled descriptors cross devices, never the spatial patch array.
    CUDA floating cumsum lacks a deterministic implementation in the run environment.
    """
    device = value.device
    if value.is_cuda:
        value, counts = value.cpu(), counts.cpu()
        valid = valid.cpu() if valid is not None else None
    if value.ndim != 2 or counts.ndim != 1 or len(value) != int(counts.sum()) or (counts < 1).any():
        raise ValueError('Invalid complete image roster')
    ends = counts.cumsum(0)
    def summed(x):
        prefix = torch.cat((x.new_zeros((1, x.shape[1])), x.cumsum(0)), 0)
        return prefix[ends] - prefix[ends-counts]
    if valid is None:
        return (summed(value) / counts[:, None]).to(device)
    if valid.shape != (len(value),) or valid.dtype != torch.bool:
        raise ValueError('Invalid roster Face mask')
    result = summed(value.masked_fill(~valid[:, None], 0)) / summed(valid[:, None].to(value.dtype)).clamp_min(1)
    return result.to(device)


class PatchReader(nn.Module):
    def __init__(self, width, hidden, full_condition=False):
        super().__init__()
        self.patch = nn.Linear(width, hidden, bias=False)
        self.cls = nn.Linear(width, hidden, bias=False)
        self.score = nn.Linear(hidden, 1, bias=False)
        self.full = nn.Linear(width, hidden, bias=False) if full_condition else None

    def forward(self, cls, patch, full):
        content = self.patch(patch) + self.cls(cls)[:, None]
        if self.full is not None:
            content = content + self.full(full)[:, None]
        alpha = self.score(content.tanh()).squeeze(-1).softmax(1)
        return (alpha[:, :, None]*patch).sum(1), alpha


class FinalPatchFusion(nn.Module):
    """Only features enter this module. Original class logits are mixed outside it."""
    def __init__(self, method, amplitude=.25, width=WIDTH, hidden=HIDDEN, seed=1):
        super().__init__()
        if method not in METHODS[1:] or not 0 < amplitude < 1 or min(width, hidden) < 1:
            raise ValueError('Invalid feature-only fusion specification')
        self.method, self.amplitude, self.width, self.hidden, self.seed = method, float(amplitude), width, hidden, seed
        multiplier = {'cls': 1, 'cls_mean': 2, 'cls_regions': 9, 'cls_patch': 2, 'cls_patch_full': 2}
        dim = 3*width*multiplier[method]+2 if method in FEATURE_METHODS else 0
        self.register_buffer('center', torch.zeros(dim))
        self.readers = nn.ModuleDict()
        if method in ('cls_patch', 'cls_patch_full'):
            # Corresponding readers start identically across methods/folds/sources.
            for i, v in enumerate(VIEWS):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed+100+i)
                    self.readers[v] = PatchReader(width, hidden, method == 'cls_patch_full' and v != 'full')
        if method == 'shared':
            self.bias = nn.Parameter(torch.zeros(2))
        elif method in FEATURE_METHODS:
            self.matrix = nn.Parameter(torch.zeros(dim, 2))

    def describe(self, cls, patch, counts, reliable, *, return_attention=False):
        if cls.shape != (len(reliable), 3, self.width) or counts.sum() != len(reliable):
            raise ValueError('Invalid CLS/roster dimensions')
        if self.method not in FEATURE_METHODS:
            return cls.new_empty((len(counts), 0)), {}
        cls = cls.detach().float().clone()
        cls[:, 2] = cls[:, 2].masked_fill(~reliable[:, None], 0)
        if not torch.isfinite(cls).all():
            raise ValueError('Nonfinite available CLS')
        if self.method != 'cls':
            if patch.ndim != 4 or patch.shape[:2] != cls.shape[:2] or patch.shape[-1] != self.width:
                raise ValueError('Expected all spatial patches [roster, view, position, width]')
        parts, attention = [], {}
        for i, v in enumerate(VIEWS):
            c = cls[:, i]
            features = [c]
            valid = reliable if v == 'face' else None
            if self.method != 'cls':
                g = patch[:, i].detach().float()
                if valid is not None:
                    g = g.masked_fill(~valid[:, None, None], 0)
                if not torch.isfinite(g).all():
                    raise ValueError('Nonfinite available patches')
                if self.method == 'cls_mean':
                    features.append(g.mean(1))
                elif self.method == 'cls_regions':
                    side = math.isqrt(g.shape[1])
                    if side*side != g.shape[1] or side % 2:
                        raise ValueError('Four regions require an even square patch grid')
                    grid = g.reshape(len(g), side, side, self.width)
                    regions = [grid[:, y:y+side//2, x:x+side//2].reshape(len(g), -1, self.width)
                               for y in (0, side//2) for x in (0, side//2)]
                    features += [r.mean(1) for r in regions] + [r.std(1, unbiased=False) for r in regions]
                else:
                    result, weights = self.readers[v](c, g, cls[:, 0])
                    features.append(result)
                    if return_attention:
                        attention[v] = weights.masked_fill(~valid[:, None], 0) if valid is not None else weights
            parts.append(roster_mean(torch.cat(features, 1), counts, valid))
        fraction = roster_mean(reliable[:, None].float(), counts)
        parts.extend((fraction, (counts.float()/(counts+1))[:, None]))
        return torch.cat(parts, 1), attention

    def scores(self, descriptor, *, train_center=False):
        if descriptor.shape[1] != len(self.center) or not torch.isfinite(descriptor).all():
            raise ValueError('Invalid image descriptor')
        if self.method == 'initial':
            aux = descriptor.new_zeros((len(descriptor), 2))
        elif self.method == 'shared':
            aux = self.amplitude*self.bias.tanh().expand(len(descriptor), -1)
        else:
            # The TRAIN mean remains in the graph while learning the patch reader.
            center = descriptor.mean(0) if train_center else self.center
            aux = self.amplitude*((descriptor-center) @ self.matrix).tanh()
        return torch.cat((descriptor.new_ones((len(descriptor), 1)), aux), 1)

    def forward(self, cls, patch, counts, reliable, *, train_center=False):
        descriptor, _ = self.describe(cls, patch, counts, reliable)
        return self.scores(descriptor, train_center=train_center)

    def export(self):
        return dict(method=self.method, amplitude=self.amplitude, width=self.width, hidden=self.hidden, seed=self.seed,
                    state={k:v.detach().cpu().tolist() for k,v in self.state_dict().items()})

    @classmethod
    def restore(cls, payload):
        obj = cls(**{k:payload[k] for k in ('method','amplitude','width','hidden','seed')})
        expected = obj.state_dict()
        if set(payload['state']) != set(expected):
            raise ValueError('Unexpected fusion parameters')
        values = {k:torch.tensor(payload['state'][k], dtype=v.dtype) for k,v in expected.items()}
        if any(values[k].shape != v.shape or not torch.isfinite(values[k]).all() for k,v in expected.items()):
            raise ValueError('Invalid fusion state')
        obj.load_state_dict(values, strict=True)
        return obj
