"""Task-static complete functions with a common, fit-only frozen donor pool."""
from __future__ import annotations
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from .functional_output_experts import FunctionalOutputExpertBank, FunctionalRouteAudit

ARMS = ('fresh_only', 'old_only', 'old_then_new')
COMBINATION = 'staged_complete_functions_common_donors_v1'


def image_group(identifier):
    value = str(identifier)
    if '#person=' not in value or not value.rsplit('#person=', 1)[0]:
        raise ValueError('Expected canonical original-image#person=... sample ID')
    return value.rsplit('#person=', 1)[0]


def held_out(identifier):
    # Global assignment, independent of task, labels, and order of cache rows.
    digest = hashlib.sha256(('module-holdout-seed0-v1:' + image_group(identifier)).encode()).digest()
    return int.from_bytes(digest[:8], 'big') % 5 == 0


def split_cache(cache, smoke=False):
    if cache.get('source_split') != 'train':
        raise ValueError('Only TRAIN may be partitioned for module fitting')
    parts = {}
    for name, held in [('fit', False), ('holdout', True)]:
        ids = [i for i, sid in enumerate(cache['sample_ids']) if held_out(sid) == held]
        if not ids:
            raise ValueError('Empty image-group split')
        if smoke:
            ids = ids[:64]
        index = torch.tensor(ids)
        parts[name] = {**cache, 'module_split': name,
            'sample_ids': [cache['sample_ids'][i] for i in ids],
            'targets': cache['targets'][index], 'face_reliable': cache['face_reliable'][index],
            'features': {v: x[index] for v, x in cache['features'].items()}}
    assert not ({image_group(s) for s in parts['fit']['sample_ids']} &
                {image_group(s) for s in parts['holdout']['sample_ids']})
    return parts


class StagedFunctionBank(FunctionalOutputExpertBank):
    def __init__(self, *args, arm, **kwargs):
        if arm not in ARMS:
            raise ValueError('Unknown staged arm')
        self.arm, self.phase = arm, 'off'
        self.signed_output_coefficients = True
        super().__init__(*args, **kwargs)
        self.router_input_mode = 'constant'
        self.old_coefficients = nn.ParameterList(nn.Parameter(torch.zeros(2*t), requires_grad=False)
                                                 for t in range(self.num_tasks))
        self.new_logits = nn.ParameterList(nn.Parameter(torch.zeros(2), requires_grad=False)
                                           for _ in range(self.num_tasks))
        self.register_buffer('donor_a', torch.zeros_like(torch.stack(tuple(self.expert_a))))
        self.register_buffer('donor_b', torch.zeros_like(torch.stack(tuple(self.expert_b))))
        self.register_buffer('donor_pw', torch.zeros_like(torch.stack([p.weight for p in self.task_projections])))
        self.register_buffer('donor_pb', torch.zeros_like(torch.stack([p.bias for p in self.task_projections])))
        self.register_buffer('donor_loaded', torch.zeros(self.num_tasks, dtype=torch.bool))
        self.requires_grad_(False)

    def _set_trainability(self):
        self.requires_grad_(False)
        if not hasattr(self, 'old_coefficients') or bool(self.sealed_tasks[self._current_task_id]):
            return
        t = self._current_task_id
        if self.phase == 'old' and t > 0 and self.arm != 'fresh_only':
            self.old_coefficients[t].requires_grad_(True)
        elif self.phase == 'new' and self.arm != 'old_only':
            for i in self.new_expert_ids(t):
                self.expert_a[i].requires_grad_(True)
                self.expert_b[i].requires_grad_(True)
            self.task_projections[t].requires_grad_(True)
            self.new_logits[t].requires_grad_(True)

    def set_phase(self, phase):
        if phase not in ('off', 'old', 'new'):
            raise ValueError('Invalid phase')
        self.phase = phase
        self._set_trainability()

    @torch.no_grad()
    def install_donor(self, source, birth):
        if source.arm != 'fresh_only' or not bool(source.sealed_tasks[birth]):
            raise ValueError('Donor must be sealed fit-only fresh-control expert')
        if bool(self.donor_loaded[birth]) or not bool(self.donor_loaded[:birth].all()):
            raise ValueError('Donor pool is append-only and ordered')
        for i in range(2*birth, 2*birth+2):
            self.donor_a[i].copy_(source.expert_a[i])
            self.donor_b[i].copy_(source.expert_b[i])
        self.donor_pw[birth].copy_(source.task_projections[birth].weight)
        self.donor_pb[birth].copy_(source.task_projections[birth].bias)
        self.donor_loaded[birth] = True

    def donor_output(self, i, x, task):
        if not 0 <= i < 2*task or not bool(self.donor_loaded[i//2]):
            raise ValueError('Missing or future donor')
        low = F.linear(self.norm(x), self.donor_a[i])
        return self.output_scale.to(x)*F.linear(F.gelu(F.linear(low, self.donor_pw[i//2], self.donor_pb[i//2])), self.donor_b[i])

    def protected_state(self, before_task):
        state = super().protected_state(before_task)
        state.update({f'old_coefficients.{t}': self.old_coefficients[t] for t in range(before_task)})
        state.update({f'new_logits.{t}': self.new_logits[t] for t in range(before_task)})
        # Task s reads donors born strictly before s. Installing donor s
        # after sealing s therefore cannot alter any existing task function.
        n = max(0, before_task-1)
        for name in ('donor_a', 'donor_b'):
            state[name] = getattr(self, name)[:2*n]
        for name in ('donor_pw', 'donor_pb', 'donor_loaded'):
            state[name] = getattr(self, name)[:n]
        return state

    def forward(self, layer_id, tokens, view_name='full', task_id=None):
        t = self._current_task_id if task_id is None else int(task_id)
        self._check_task(t)
        if t > self._current_task_id or layer_id not in self.layer_indices:
            raise ValueError('Invalid staged route')
        weights = tokens.new_zeros(tokens.shape[0], self.num_experts)
        delta = torch.zeros_like(tokens)
        if self.arm != 'fresh_only' and t:
            old = self.old_coefficients[t].tanh().to(tokens)
            delta = sum(old[i]*self.donor_output(i, tokens, t) for i in range(2*t))
            weights[:, :2*t] = old
        if self.arm != 'old_only':
            new = self.new_logits[t].softmax(-1).to(tokens)
            delta = delta + sum(new[j]*self.expert_output(i, tokens) for j,i in enumerate(self.new_expert_ids(t)))
            weights[:, 2*t:2*t+2] = new
        radius = self.smooth_ratio_bound*tokens.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
        delta = delta*(radius/(radius.square()+delta.float().square().sum(-1,keepdim=True)).sqrt()).to(delta)
        ratio = delta.float().norm(dim=-1)/tokens.float().norm(dim=-1).clamp_min(1e-12)
        self.last_raw_residual_ratio = float(ratio.detach().mean().cpu())
        self.last_projector_correction_ratio = 0.
        self._last_residual_penalties.append(ratio.square().mean())
        return tokens+delta, weights

    def policy_summary(self, task_id):
        return {**super().policy_summary(task_id), 'combination': COMBINATION, 'arm': self.arm,
                'phase': self.phase, 'donor_source': 'fresh_only fit partition, sealed birth functions',
                'old_coefficients': self.old_coefficients[task_id].detach().tanh().cpu().tolist(),
                'new_weights': self.new_logits[task_id].detach().softmax(-1).cpu().tolist(),
                'gate_records': 'actual signed coefficients, not a probability distribution'}


class StagedRouteAudit(FunctionalRouteAudit):
    def _identity_check(self, model, images, task):
        if bool(torch.count_nonzero(model.parax_bank.old_coefficients[task])):
            raise RuntimeError('Current old coefficients must start at zero')
        return super()._identity_check(model, images, task)
