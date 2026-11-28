"""Frozen Multi-LANE features, with explicitly bounded common and image terms."""
import torch
from torch import nn
from multi_lane.track_a.final_patch_fusion import FinalPatchFusion

READERS = ('cls', 'cls_regions', 'cls_patch')
LEGACY = ('shared', *READERS)


def candidates():
    rows = [dict(key=m, reader=m, common=m == 'shared', lower=.25, variation=0., reuse=True)
            for m in LEGACY]
    rows += [dict(key=m+'__common', reader=m, common=True, lower=.25, variation=0., reuse=False)
             for m in READERS]
    rows += [dict(key=m+'__lower050', reader=m, common=True, lower=.50, variation=0., reuse=False)
             for m in ('shared', *READERS)]
    rows += [dict(key=m+'__variation', reader=m, common=True, lower=.25, variation=.01, reuse=False)
             for m in READERS]
    return rows


def bounded_score(raw, lower, upper=.25):
    """Same value and slope at zero for both ranges; bounds are score units."""
    if lower not in (.25, .50) or upper != .25:
        raise ValueError('Undeclared score interval')
    # Negative half has slope upper at the origin, just like the positive half.
    return torch.where(raw >= 0, upper*raw.tanh(), lower*(raw*(upper/lower)).tanh())


class TaskPatchFusion(FinalPatchFusion):
    def __init__(self, spec):
        if spec not in candidates():
            raise ValueError('Unknown candidate')
        super().__init__(spec['reader'])
        self.spec = dict(spec)
        if spec['common'] and self.method != 'shared':
            self.bias = nn.Parameter(torch.zeros(2))

    def image_term(self, descriptor, train_center=False):
        if self.method == 'shared':
            return descriptor.new_zeros((len(descriptor), 2))
        center = descriptor.mean(0) if train_center else self.center
        return (descriptor-center) @ self.matrix

    def scores(self, descriptor, *, train_center=False):
        if descriptor.shape[1] != len(self.center) or not torch.isfinite(descriptor).all():
            raise ValueError('Invalid image descriptor')
        raw = self.image_term(descriptor, train_center)
        if self.spec['common']:
            raw = raw + self.bias
        return torch.cat((raw.new_ones((len(raw), 1)), bounded_score(raw, self.spec['lower'])), 1)

    def export(self):
        return dict(spec=self.spec, state={k:v.detach().cpu().tolist() for k,v in self.state_dict().items()})

    @classmethod
    def restore(cls, payload):
        obj = cls(payload['spec'])
        expected = obj.state_dict()
        if set(expected) != set(payload['state']):
            raise ValueError('Fusion state schema differs')
        state = {k:torch.tensor(payload['state'][k], dtype=v.dtype) for k,v in expected.items()}
        if any(state[k].shape != v.shape or not torch.isfinite(state[k]).all() for k,v in expected.items()):
            raise ValueError('Invalid fusion state')
        obj.load_state_dict(state)
        return obj


def restore_gate(payload):
    return TaskPatchFusion.restore(payload) if 'spec' in payload else FinalPatchFusion.restore(payload)


def prediction_key(candidate):
    return 'prediction__'+candidate


def control_key(candidate, control):
    return 'control__'+control+'__'+candidate

