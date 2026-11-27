"""Feature-only auxiliary scores, initialized equally at zero; Full score stays one."""
import math
import torch
from torch import nn
from multi_lane.track_a.summary_logit_fusion import DIMENSIONS


def full_bounds(amplitude, face=True):
    auxiliaries = 2 if face else 1
    return (1/(1+auxiliaries*math.exp(amplitude-1)),
            1/(1+auxiliaries*math.exp(-amplitude-1)))


class FeatureBoundFusion(nn.Module):
    def __init__(self, feature, amplitude):
        super().__init__()
        if feature not in ('summary_mean', 'summary_correspondence') or not 0 < amplitude < 1:
            raise ValueError('Expected summary features and a bound below the Full score')
        self.feature, self.amplitude = feature, float(amplitude)
        self.register_buffer('center', torch.zeros(DIMENSIONS[feature]))
        self.matrix = nn.Parameter(torch.zeros(DIMENSIONS[feature], 2))

    @torch.no_grad()
    def set_center(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.center) or not len(x) or not torch.isfinite(x).all():
            raise ValueError('Invalid TRAIN feature population')
        self.center.copy_(x.detach().mean(0))

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.center) or not torch.isfinite(x).all():
            raise ValueError('Invalid feature input')
        aux = self.amplitude * torch.tanh((x.detach()-self.center) @ self.matrix)
        return torch.cat((x.new_ones((len(x), 1)), aux), 1)

    def export(self):
        return dict(feature=self.feature, amplitude=self.amplitude,
                    state={k:v.detach().cpu().tolist() for k,v in self.state_dict().items()})

    @classmethod
    def restore(cls, payload):
        obj = cls(payload['feature'], payload['amplitude'])
        state = obj.state_dict()
        if set(payload['state']) != set(state):
            raise ValueError('Unexpected parameters: no bias or prior is allowed')
        obj.load_state_dict({k:torch.tensor(payload['state'][k],dtype=v.dtype).reshape(v.shape)
                             for k,v in state.items()}, strict=True)
        if not all(torch.isfinite(v).all() for v in obj.state_dict().values()):
            raise ValueError('Nonfinite saved model')
        return obj
