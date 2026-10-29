"""Protected subject/context summarization: constant/person/grounded x pool/attention."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .private_summarize_attention import PrivateSummaryModel, norm_matched_query
from .protected_joint_attention import VIEWS, bounded_delta

GROUNDED_ARMS = tuple('PERSON_'+condition+'_'+op
    for condition in ('CONST', 'CONTENT', 'GROUND') for op in ('POOL', 'ATTN'))
MEMORIES = ('global', 'subject', 'person', 'face')


def patch_coverage(boxes, visibility, count):
    """Overlap relative to patch area, using already transformed Full coordinates."""
    grid = math.isqrt(count)
    if grid*grid != count or boxes.shape != (len(visibility), 4):
        raise ValueError('Expected square patch grid and [batch,4] mapped boxes')
    with torch.autocast(device_type=boxes.device.type, enabled=False):
        b = boxes.detach().float(); vis = visibility.detach().float()
        valid = torch.isfinite(b).all(-1) & torch.isfinite(vis)
        valid &= (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1]) & (vis > 0)
        b = torch.where(valid[:, None], b, torch.zeros_like(b))
        axis = torch.arange(grid, device=b.device, dtype=torch.float32)/grid
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        x, y = x.flatten()[None], y.flatten()[None]
        dx = (torch.minimum(x+1/grid, b[:, 2, None])-torch.maximum(x, b[:, 0, None])).clamp_min(0)
        dy = (torch.minimum(y+1/grid, b[:, 3, None])-torch.maximum(y, b[:, 1, None])).clamp_min(0)
        rho = (dx*dy*count).clamp(0, 1)
        valid &= (rho > 0).any(-1)
        rho = torch.where(valid[:, None], rho, torch.zeros_like(rho))
        eligible = valid & (vis >= .9)
    return rho, valid, eligible


class GroundedReadout(nn.Module):
    def __init__(self, width, output_dim, selectors, arm, dim=64, heads=4, rank=16):
        super().__init__()
        if arm not in GROUNDED_ARMS or selectors < 3 or dim % heads:
            raise ValueError('Invalid grounded summary configuration')
        self.arm, self.dim, self.heads = arm, dim, heads
        self.constant = '_CONST_' in arm
        self.grounded = '_GROUND_' in arm
        # All common modules/initializations precede operator-specific modules.
        self.query = nn.Linear(2*output_dim, dim, bias=False)
        self.value = nn.Linear(width, dim, bias=False)
        self.out = nn.Linear(dim, output_dim, bias=False)
        self.view_types = nn.Parameter(torch.empty(4, width))
        nn.init.normal_(self.view_types, std=.01); nn.init.zeros_(self.out.weight)
        self.subject_offset = nn.Parameter(torch.zeros(2, width))
        self.condition_slots = nn.Parameter(torch.empty(2, rank))
        self.condition_input = nn.Linear(output_dim, rank, bias=False)
        self.condition_output = nn.Parameter(torch.zeros(rank, width))
        nn.init.normal_(self.condition_slots, std=.02)
        nn.init.xavier_uniform_(self.condition_input.weight)
        self.register_buffer('constant_person', torch.randn(output_dim))
        if arm.endswith('ATTN'):
            self.key = nn.Linear(width, dim, bias=False)
        else:
            self.pool = nn.Sequential(nn.Linear(5*dim+2, 127), nn.GELU(), nn.Linear(127, dim))
        self.last_metrics = {}

    def summarize(self, tokens, anchors, person, boxes, visibility, zero_condition=False):
        with torch.autocast(device_type=person.device.type, enabled=False):
            g = {v: tokens[v].detach().float() for v in VIEWS}
            s = {v: anchors[v].detach().float() for v in VIEWS}
            source = self.constant_person[None].expand(len(person), -1) if self.constant else person.detach().float()
            condition = torch.tanh(self.condition_input(F.layer_norm(source, (source.shape[-1],))))
            shift = (self.condition_slots[None]*condition[:, None]) @ self.condition_output
            if zero_condition: shift = torch.zeros_like(shift)
            anchor = s['full'][:, -2:]
            q = norm_matched_query(anchor, self.subject_offset+shift)
            alpha = (q @ g['full'][:, 1:].transpose(-1, -2)/math.sqrt(q.shape[-1])).softmax(-1)
            rho, valid, eligible = patch_coverage(boxes, visibility, alpha.shape[-1])
            result = {'subject': alpha @ g['full'][:, 1:]}
            for view, name in (('full', 'global'), ('person', 'person'), ('face', 'face')):
                query = s[view][:, :-2] if view == 'full' else s[view]
                result[name] = (query @ g[view].transpose(-1, -2)/math.sqrt(query.shape[-1])).softmax(-1) @ g[view]
            mass = (alpha*rho[:, None]).sum(-1)
            self.location_loss = -(mass[eligible]+1e-8).log().mean() if bool(eligible.any()) else alpha.sum()*0.
            self.subject_valid = valid
            self.subject_alpha = alpha.detach()
            self.subject_coverage = rho.detach()
            self._summary_metrics = dict(
                localization_loss_unweighted=float(self.location_loss.detach()),
                localization_eligible_fraction=float(eligible.float().mean()),
                localization_eligible_samples=float(eligible.sum()),
                subject_valid_fraction=float(valid.float().mean()),
                subject_inside_mass=float(mass[eligible].detach().mean()) if bool(eligible.any()) else 0.,
                subject_attention_entropy=float(-(alpha.detach()*alpha.detach().clamp_min(1e-8).log()).sum(-1).mean()),
                query_anchor_cosine=float(F.cosine_similarity(q.detach(), anchor, dim=-1).mean()),
                query_norm_ratio=float((q.detach().norm(dim=-1)/anchor.norm(dim=-1).clamp_min(1e-12)).mean()),
                query_offset_norm=float(self.subject_offset.detach().norm()),
                conditional_offset_norm=float(shift.detach().square().mean().sqrt()),
                summarize_fp32=float(all(x.dtype == torch.float32 for x in result.values())),
                **{name+'_summary_rms': float(value.detach().square().mean().sqrt()) for name, value in result.items()})
        return result

    def forward(self, base, person, summaries, reliable, no_face=False, no_person=False,
                no_subject=False, no_global=False, query_person=None):
        with torch.autocast(device_type=base.device.type, enabled=False):
            base, person = base.detach().float(), person.detach().float()
            qp = person if query_person is None else query_person.detach().float()
            q = self.query(F.layer_norm(torch.cat([base, qp], -1), (2*base.shape[-1],)))
            use_face = reliable.detach().bool() & (not no_face)
            use_subject = self.subject_valid & (not no_subject)
            masks = {'global': torch.full_like(use_face, not no_global), 'subject': use_subject,
                     'person': torch.full_like(use_face, not no_person), 'face': use_face}
            memories, valid, pooled = [], [], []
            for i, name in enumerate(MEMORIES):
                memory = F.layer_norm(summaries[name].float(), (summaries[name].shape[-1],))+self.view_types[i]
                v = self.value(memory)
                pooled.append(torch.where(masks[name][:, None], v.mean(1), torch.zeros_like(v[:, 0])))
                memories.append(memory); valid.append(masks[name][:, None].expand(-1, memory.shape[1]))
            masses = {n: base.new_zeros(()) for n in MEMORIES}
            if hasattr(self, 'key'):
                memory = torch.cat(memories, 1); mask = F.pad(torch.cat(valid, 1), (0, 1), value=True)
                k, v = F.pad(self.key(memory), (0, 0, 0, 1)), F.pad(self.value(memory), (0, 0, 0, 1))
                b, n = mask.shape
                qh = q.reshape(b, self.heads, 1, -1)
                k = k.reshape(b, n, self.heads, -1).transpose(1, 2)
                v = v.reshape(b, n, self.heads, -1).transpose(1, 2)
                weights = ((qh @ k.transpose(-1, -2))/math.sqrt(self.dim/self.heads)).masked_fill(~mask[:, None, None], float('-inf')).softmax(-1)
                hidden = (weights @ v).reshape(b, self.dim)
                start = 0
                for name, memory in zip(MEMORIES, memories):
                    masses[name] = weights[..., start:start+memory.shape[1]].sum(-1).mean()
                    start += memory.shape[1]
            else:
                hidden = self.pool(torch.cat([q, *pooled, use_face[:, None].float(), use_subject[:, None].float()], -1))
            delta = bounded_delta(self.out(hidden), base, .01)
            ratio = delta.detach().square().mean(-1).sqrt()/base.square().mean(-1).sqrt().clamp_min(1e-12)
            if not bool(torch.isfinite(delta).all()) or bool((ratio > .010001).any()):
                raise FloatingPointError('Grounded summary final cap violation')
            self.last_metrics = dict(gate_mean=1., delta_to_base_rms=float(ratio.mean()),
                delta_to_base_rms_max=float(ratio.max()), delta_near_cap_fraction=float((ratio > .0095).float().mean()),
                **{name+'_attention_mass': float(value.detach()) for name, value in masses.items()},
                **self._summary_metrics)
            return delta

    def after_backward_metrics(self):
        values = [p.grad for p in (self.subject_offset, self.condition_slots,
            self.condition_input.weight, self.condition_output) if p.grad is not None]
        if any(not bool(torch.isfinite(g).all()) for g in values):
            raise FloatingPointError('Nonfinite subject query gradient')
        return dict(query_gradient_norm=math.sqrt(sum(float(g.float().square().sum()) for g in values)))


class PersonGroundedModel(PrivateSummaryModel):
    def initialize_interaction(self, arm, layer=5, dim=64, heads=4):
        if arm not in GROUNDED_ARMS or not 0 <= layer < len(self.visual_encoder.transformer.resblocks):
            raise ValueError(arm)
        self.grounded_summary_policy = arm
        self.interaction_arm, self.interaction_layer, self.interaction_enabled = 'C2', layer, True
        with torch.random.fork_rng(devices=[]):
            modules = []
            for task in range(self.num_tasks):
                torch.random.default_generator.manual_seed(930100+task)
                modules.append(GroundedReadout(self.width, self.head.in_features, self.num_selectors, arm, dim, heads))
            self.interaction_bank = nn.ModuleList(modules)
        self.interaction_bank.to(self.head.weight.device).requires_grad_(False)
        self._interaction_cache, self._private_sources = {}, {}

    def encode_lanes_with_views(self, images, all_seen_lanes):
        self._grounded_geometry = {k: images[k].detach() for k in ('spatial_bbox', 'spatial_visibility')}
        return super().encode_lanes_with_views(images, all_seen_lanes)

    def interaction_delta(self, no_face=False, no_person=False, no_subject=False, no_global=False,
                          zero_condition=False, early_donor=None, query_donor=None, memory_donors=None):
        c = self._interaction_cache; rows = []
        for index, task in enumerate(c['lane_ids']):
            module = self.interaction_bank[task]; person = c['features']['person'][:, index]
            summaries = module.summarize({v: self._private_sources[v][0] for v in VIEWS},
                {v: self._private_sources[v][1][index] for v in VIEWS},
                person if early_donor is None else person[early_donor],
                self._grounded_geometry['spatial_bbox'], self._grounded_geometry['spatial_visibility'], zero_condition)
            if memory_donors:
                summaries = {v: s[memory_donors[v]] if v in memory_donors else s for v, s in summaries.items()}
            rows.append(module(c['base'][:, index], person, summaries, c['reliable'], no_face, no_person,
                no_subject, no_global, None if query_donor is None else person[query_donor]))
        delta = torch.stack(rows, 1)
        if not any((no_face, no_person, no_subject, no_global, zero_condition,
                    early_donor is not None, query_donor is not None, bool(memory_donors))):
            self._native_delta = delta.detach()
        return delta

    def interaction_training_objective(self, classification_loss, epoch):
        module = self.interaction_bank[self.current_task_id]
        weight = .01*min(1., (epoch+1)/5) if module.grounded else 0.
        loc = module.location_loss
        return classification_loss+weight*loc, dict(classification_loss=float(classification_loss.detach()),
            localization_weight=weight, localization_weighted_loss=float(weight*loc.detach()))


def grounded_configuration(arm):
    if arm not in GROUNDED_ARMS: raise ValueError(arm)
    return dict(protocol='person_grounded_summary_3x2_v1', layer=5, dim=64, heads=4, selectors=10,
        full_global=8, full_subject=2, subject_reads_cls=False, global_reads_cls=True,
        query_parameters_per_task=22048, condition_rank=16, condition_source='task_constant' if '_CONST_' in arm else 'own_Person_CLS',
        readout='attention' if arm.endswith('ATTN') else 'pool_mlp_322_127_64',
        learning_rate=.0004, query_learning_rate=.0004, final_residual_cap=.01, fixed_gain=1.,
        localization_weight=.01 if '_GROUND_' in arm else 0., localization_warmup_epochs=5,
        localization_min_visibility=.9, localization_target='actual_summary_patch_overlap_mass',
        query_norm_matched=True, summarize_precision='fp32_autocast_disabled', global_full_summaries=True,
        target_region_constraint=False, new_private_summarize_queries=True,
        query_shared_across_views=False, source_only='three_views', learned_gate=False,
        common_projection_initialization_seed='930100+task_common_components_first', independent_optimizer_scaler_rng=True)
