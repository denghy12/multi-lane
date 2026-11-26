"""Read detached Summarized Patches; learn only the final logit mixing scores."""
import math

import torch
from torch import nn
from torch.nn import functional as F


METHODS = ('fixed', 'shared', 'summary_mean', 'summary_correspondence')
NAMES = dict(zip(METHODS, ('原固定比例', '当前任务学习一套共用比例',
    '根据Summarized Patches平均相似度学习比例', '根据Summarized Patches对应情况学习比例')))
DIMENSIONS = dict(fixed=0, shared=0, summary_mean=6, summary_correspondence=26)


@torch.no_grad()
def pair_statistics(left, right, temperature=.1):
    """All-pairs cosine statistics, invariant to either selector numbering."""
    if (left.ndim != 3 or right.ndim != 3 or left.shape[0] != right.shape[0]
            or left.shape[2] != right.shape[2] or min(left.shape[1], right.shape[1]) < 1
            or temperature <= 0):
        raise ValueError('Expected two nonempty [person, selector, width] tensors')
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError('Nonfinite available Summarized Patches')
    similarity = F.normalize(left.float(), dim=-1) @ F.normalize(right.float(), dim=-1).transpose(1, 2)
    rows, columns = similarity.max(2).values, similarity.max(1).values
    logp = (similarity / temperature).log_softmax(2)
    entropy = -(logp.exp() * logp).sum(2).mean(1)
    entropy = entropy / math.log(right.shape[1]) if right.shape[1] > 1 else entropy * 0
    return torch.stack((similarity.mean((1, 2)), rows.mean(1), rows.std(1, unbiased=False),
                        columns.mean(1), columns.std(1, unbiased=False), entropy), dim=1)


@torch.no_grad()
def summary_statistics(capture, reliable, task):
    """Equal mean over seen task lanes; no final features, classifiers or labels."""
    expected = set(range(task + 1))
    if set(capture) != {'full', 'person', 'face'} or any(set(v) != expected for v in capture.values()):
        raise ValueError('Capture must contain exactly the seen task lanes in all three views')
    if reliable.dtype != torch.bool or reliable.ndim != 1:
        raise ValueError('Expected a person-level boolean face mask')
    rows = []
    for lane in range(task + 1):
        full, person, face = (capture[v][lane] for v in ('full', 'person', 'face'))
        if any(len(x) != len(reliable) for x in (full, person, face)):
            raise ValueError('Capture/mask population mismatch')
        p = pair_statistics(full, person)
        f = p.new_zeros(p.shape)
        if reliable.any():
            f[reliable] = pair_statistics(full[reliable], face[reliable])
        rows.append(torch.cat((p, f), 1))
    return torch.stack(rows).mean(0)


@torch.no_grad()
def image_descriptors(statistics, reliable, sizes):
    """Pool the complete target roster, including people without current labels."""
    if (statistics.shape != (len(reliable), 12) or sum(sizes) != len(reliable)
            or not sizes or min(sizes) < 1 or reliable.dtype != torch.bool
            or not torch.isfinite(statistics).all()):
        raise ValueError('Invalid complete image roster/statistics')
    means, relations = [], []
    for values, flags in zip(statistics.split(sizes), reliable.split(sizes)):
        p, f = values[:, :6], values[flags, 6:]
        fm = f.mean(0) if len(f) else p.new_zeros(6)
        fs = f.std(0, unbiased=False) if len(f) else p.new_zeros(6)
        mean = torch.cat((p.mean(0), fm))
        std = torch.cat((p.std(0, unbiased=False), fs))
        metadata = p.new_tensor([flags.float().mean().item(), len(flags)/(len(flags)+1)])
        means.append(torch.cat((mean[[0, 6]], std[[0, 6]], metadata)))
        relations.append(torch.cat((mean, std, metadata)))
    return dict(summary_mean=torch.stack(means), summary_correspondence=torch.stack(relations))


def masked_weights(scores, reliable):
    if scores.shape != (len(reliable), 3) or reliable.dtype != torch.bool or reliable.ndim != 1:
        raise ValueError('Expected one base score triplet and boolean mask per person')
    face = scores[:, 2].masked_fill(~reliable, -torch.inf)
    return torch.stack((scores[:, 0], scores[:, 1], face), dim=1).softmax(1)


class SummaryLogitGate(nn.Module):
    """One base triplet per image. Its only input is detached intermediate statistics."""
    def __init__(self, method, amplitude=.25):
        super().__init__()
        if method not in METHODS or amplitude <= 0:
            raise ValueError('Unknown method or invalid amplitude')
        self.method, self.amplitude = method, amplitude
        self.register_buffer('prior', torch.tensor([1., 1.+math.log(.16/.64), 1.+math.log(.20/.64)]))
        self.register_buffer('center', torch.zeros(DIMENSIONS[method]))
        self.bias = nn.Parameter(torch.zeros(2), requires_grad=method != 'fixed')
        self.matrix = nn.Parameter(torch.zeros(DIMENSIONS[method], 2), requires_grad=DIMENSIONS[method] > 0)

    def forward(self, descriptor):
        if (descriptor.ndim != 2 or descriptor.shape[1] != len(self.center)
                or not torch.isfinite(descriptor).all()):
            raise ValueError('Invalid intermediate descriptor')
        x = descriptor.detach() - self.center
        adjustment = self.bias + self.amplitude * torch.tanh(x @ self.matrix)
        aux = self.prior[1:] + adjustment
        return torch.cat((descriptor.new_ones((len(descriptor), 1)), aux), dim=1)

    @torch.no_grad()
    def set_center(self, train_descriptors):
        if not len(train_descriptors) or train_descriptors.shape[1] != len(self.center):
            raise ValueError('TRAIN-only centering requires a nonempty fitting population')
        self.center.copy_(train_descriptors.detach().mean(0))

    def export(self):
        return dict(method=self.method, amplitude=self.amplitude,
                    state={k: v.detach().cpu().tolist() for k, v in self.state_dict().items()})

    @classmethod
    def restore(cls, payload):
        gate = cls(payload['method'], payload['amplitude'])
        state = gate.state_dict()
        if set(state) != set(payload['state']):
            raise ValueError('Invalid gate archive')
        gate.load_state_dict({k: torch.tensor(payload['state'][k], dtype=v.dtype).reshape(v.shape)
                              for k, v in state.items()}, strict=True)
        if not all(torch.isfinite(v).all() for v in gate.state_dict().values()):
            raise ValueError('Nonfinite saved gate')
        return gate


def fused_logits(branch_logits, image_scores, inverse, reliable):
    """Logits are values to mix, never gate inputs; no branch gradient is retained."""
    if branch_logits.ndim != 3 or branch_logits.shape[0] != len(inverse) or branch_logits.shape[2] != 3:
        raise ValueError('Expected [person, current class, view] logits')
    weights = masked_weights(image_scores[inverse], reliable)
    return (branch_logits.detach() * weights[:, None, :]).sum(2)


def image_bce(logits, targets, inverse, image_count):
    observed = torch.bincount(inverse, minlength=image_count)
    if len(observed) != image_count or (observed == 0).any():
        raise ValueError('Every supervised image must have a labeled person')
    loss = F.binary_cross_entropy_with_logits(logits, targets.detach(), reduction='none').mean(1)
    return (loss / observed[inverse]).sum() / image_count
