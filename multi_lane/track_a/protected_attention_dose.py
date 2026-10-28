"""Four self-trained controls with one common final feature correction bound.

No target-region mask or new Summarize queries are introduced in this experiment.
"""
import torch
from torch import nn

from .protected_joint_attention import InteractionReadout, ProtectedJointModel, bounded_delta

DOSE_ARMS = ('DOSE_MLP', 'DOSE_FIXED', 'DOSE_TASK', 'DOSE_CONTENT')
FINAL_CAP = .01


class DoseMatchedReadout(InteractionReadout):
    def __init__(self, width, output_dim, policy, dim=64, heads=4):
        if policy not in DOSE_ARMS:
            raise ValueError('Unknown dose-matched control')
        super().__init__(width, output_dim, 'C1' if policy == 'DOSE_MLP' else 'C2', dim, heads)
        self.policy = policy
        if policy == 'DOSE_TASK':
            self.task_gate_logit = nn.Parameter(torch.zeros(()))
        if policy == 'DOSE_CONTENT':
            self.gate = nn.Sequential(nn.Linear(3*output_dim+2, 16), nn.GELU(), nn.Linear(16, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def _gate_value(self, base, gate_base, person, face, use_face, full_visible):
        if self.policy == 'DOSE_TASK':
            gain = (2*self.task_gate_logit.sigmoid()).expand(len(base), 1)
        elif self.policy == 'DOSE_CONTENT':
            gain = 2*super()._gate_value(base, gate_base, person, face, use_face, full_visible)
        else:
            gain = base.new_ones((len(base), 1))
        self._last_gain = gain.detach()
        return gain

    def _correction(self, raw, base, gate):
        # The last operation before h0+delta enforces the SAME cap for all arms.
        # Putting an unconstrained gate after this bound would break that control.
        return bounded_delta(gate*raw, base, FINAL_CAP)

    def forward(self, base, *args, **kwargs):
        delta = super().forward(base, *args, **kwargs)
        ratio = delta.detach().float().square().mean(-1).sqrt()/base.detach().float().square().mean(-1).sqrt().clamp_min(1e-12)
        if not bool(torch.isfinite(ratio).all()) or bool((ratio > FINAL_CAP+1e-6).any()):
            raise FloatingPointError('Dose-matched final feature cap violated')
        gain = self._last_gain.float()
        self.last_metrics.update(gate_std=float(gain.std(unbiased=False)),
            gate_min=float(gain.min()), gate_max=float(gain.max()),
            delta_to_base_rms_max=float(ratio.max()),
            delta_near_cap_fraction=float((ratio > .95*FINAL_CAP).float().mean()))
        return delta


class DoseMatchedModel(ProtectedJointModel):
    def initialize_interaction(self, arm, layer=5, dim=64, heads=4):
        if arm not in DOSE_ARMS or not 0 <= layer < len(self.visual_encoder.transformer.resblocks):
            raise ValueError('Invalid dose-matched model configuration')
        # Parent cache logic uses C2's global Full summaries; never region masks.
        self.interaction_arm = 'C1' if arm == 'DOSE_MLP' else 'C2'
        self.dose_policy, self.interaction_layer, self.interaction_enabled = arm, layer, True
        with torch.random.fork_rng(devices=[]):
            modules = []
            for task in range(self.num_tasks):
                # Extra content-gate initialization cannot change later Q/K/V.
                torch.random.default_generator.manual_seed(928100+task)
                modules.append(DoseMatchedReadout(self.width, self.head.in_features, arm, dim, heads))
            self.interaction_bank = nn.ModuleList(modules)
        self.interaction_bank.to(self.head.weight.device).requires_grad_(False)
        self._interaction_cache = {}


def dose_configuration(arm):
    if arm not in DOSE_ARMS:
        raise ValueError(arm)
    return dict(protocol='dose_matched_final_bound_v1', layer=5, dim=64, heads=4,
        final_residual_cap=FINAL_CAP, cap_placement='after_gain_times_raw_before_h0_addition',
        gain_policy=arm.removeprefix('DOSE_').lower(), initial_gain=1.,
        gain_formula='2*sigmoid(logit)' if arm in ('DOSE_TASK','DOSE_CONTENT') else 'constant_one',
        raw_output_zero_initialized=True, attention_task_initialization_seed='928100+task',
        source_only='three_views', global_full_summaries=True, target_region_constraint=False,
        new_private_summarize_queries=False, learning_rate=.0004,
        independent_optimizer_scaler_rng=True)
