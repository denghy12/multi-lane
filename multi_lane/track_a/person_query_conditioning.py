"""Two independent repairs of person-conditioned Summarize queries.

Projection arms only constrain the latent RMS immediately before tanh.
Gain arms only bound/amplify the person offset relative to its query anchor.
The original model and final one-percent feature correction stay unchanged.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .person_grounded_summary import GroundedReadout, PersonGroundedModel, patch_coverage
from .private_summarize_attention import norm_matched_query
from .protected_joint_attention import VIEWS, bounded_delta

QUERY_ARMS = tuple(prefix+'_'+operator
    for prefix in ('QPROJ_GROUND', 'QGAIN_CONTENT') for operator in ('POOL', 'ATTN'))


def base_arm(arm):
    if arm not in QUERY_ARMS:
        raise ValueError('Unknown single-factor query repair: '+str(arm))
    return 'PERSON_'+('GROUND' if arm.startswith('QPROJ_') else 'CONTENT')+'_'+arm.rsplit('_', 1)[1]


def cap_projected_rms(value, ceiling=1.):
    """Identity below ceiling, parameter-free radial shrinkage above it."""
    if not math.isfinite(ceiling) or ceiling <= 0:
        raise ValueError('Projection RMS ceiling must be finite and positive')
    x=value.float()
    rms=(x.square().mean(-1,keepdim=True)+1e-12).sqrt()
    return x/(rms/ceiling).clamp_min(1.)


def relative_person_offset(raw, anchor, ratio=.01, maximum_gain=32.):
    """Zero-preserving finite-gain map; per-slot RMS never exceeds ratio."""
    if not math.isfinite(maximum_gain) or maximum_gain < 1:
        raise ValueError('Person offset maximum gain must be finite and at least one')
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError('Person offset ratio must be finite and positive')
    return bounded_delta(raw.float()*maximum_gain, anchor, ratio)


class PersonQueryReadout(GroundedReadout):
    def __init__(self, width, output_dim, selectors, arm, dim=64, heads=4, rank=16):
        # The same constructor/RNG/parameters as the corresponding old control.
        super().__init__(width,output_dim,selectors,base_arm(arm),dim,heads,rank)
        self.query_repair_arm=arm
        self.projection_repair=arm.startswith('QPROJ_')
        self.projection_rms_ceiling=1.
        self.person_offset_ratio=.01
        self.person_offset_maximum_gain=32.

    def condition_components(self, anchors, person, zero_condition=False):
        """Expose the actual differentiable path for training and diagnostics."""
        with torch.autocast(device_type=person.device.type,enabled=False):
            hp=F.layer_norm(person.detach().float(),(person.shape[-1],))
            raw_z=self.condition_input(hp)
            used_z=cap_projected_rms(raw_z,self.projection_rms_ceiling) if self.projection_repair else raw_z
            condition=used_z.tanh()
            raw=(self.condition_slots[None]*condition[:,None]) @ self.condition_output
            anchor=anchors.detach().float()[:,-2:]
            effective=(raw if self.projection_repair else relative_person_offset(raw,anchor,
                self.person_offset_ratio,self.person_offset_maximum_gain))
            if zero_condition:effective=torch.zeros_like(effective)
            query=norm_matched_query(anchor,self.subject_offset+effective)
            if not all(bool(torch.isfinite(x).all()) for x in (raw_z,used_z,effective,query)):
                raise FloatingPointError('Nonfinite person query conditioning')
        return dict(person_ln=hp,raw_preactivation=raw_z,preactivation=used_z,condition=condition,
            raw_shift=raw,shift=effective,query=query,anchor=anchor)

    def summarize(self,tokens,anchors,person,boxes,visibility,zero_condition=False):
        with torch.autocast(device_type=person.device.type,enabled=False):
            g={v:tokens[v].detach().float() for v in VIEWS}
            s={v:anchors[v].detach().float() for v in VIEWS}
            c=self.condition_components(s['full'],person,zero_condition)
            q,anchor=c['query'],c['anchor']
            alpha=(q @ g['full'][:,1:].transpose(-1,-2)/math.sqrt(q.shape[-1])).softmax(-1)
            rho,valid,eligible=patch_coverage(boxes,visibility,alpha.shape[-1])
            result={'subject':alpha @ g['full'][:,1:]}
            for view,name in (('full','global'),('person','person'),('face','face')):
                query=s[view][:,:-2] if view=='full' else s[view]
                result[name]=(query @ g[view].transpose(-1,-2)/math.sqrt(query.shape[-1])).softmax(-1) @ g[view]
            mass=(alpha*rho[:,None]).sum(-1)
            self.location_loss=-(mass[eligible]+1e-8).log().mean() if bool(eligible.any()) else alpha.sum()*0.
            self.subject_valid=valid;self.subject_alpha=alpha.detach();self.subject_coverage=rho.detach()
            raw_rms=c['raw_shift'].detach().square().mean(-1).sqrt()
            effective_rms=c['shift'].detach().square().mean(-1).sqrt()
            anchor_rms=anchor.square().mean(-1).sqrt().clamp_min(1e-12)
            used_rms=c['preactivation'].detach().square().mean(-1).sqrt()
            raw_z_rms=c['raw_preactivation'].detach().square().mean(-1).sqrt()
            if self.projection_repair and bool((used_rms>self.projection_rms_ceiling+1e-6).any()):
                raise FloatingPointError('Projection RMS ceiling violated')
            if not self.projection_repair and bool((effective_rms/anchor_rms>self.person_offset_ratio+1e-6).any()):
                raise FloatingPointError('Person offset ratio violated')
            self._summary_metrics=dict(
                localization_loss_unweighted=float(self.location_loss.detach()),
                localization_eligible_fraction=float(eligible.float().mean()),
                localization_eligible_samples=float(eligible.sum()),
                subject_valid_fraction=float(valid.float().mean()),
                subject_inside_mass=float(mass[eligible].detach().mean()) if bool(eligible.any()) else 0.,
                subject_attention_entropy=float(-(alpha.detach()*alpha.detach().clamp_min(1e-8).log()).sum(-1).mean()),
                query_anchor_cosine=float(F.cosine_similarity(q.detach(),anchor,dim=-1).mean()),
                query_norm_ratio=float((q.detach().norm(dim=-1)/anchor.norm(dim=-1).clamp_min(1e-12)).mean()),
                query_offset_norm=float(self.subject_offset.detach().norm()),
                conditional_offset_norm=float(c['shift'].detach().square().mean().sqrt()),
                projection_raw_rms_mean=float(raw_z_rms.mean()),projection_raw_rms_max=float(raw_z_rms.max()),
                projection_used_rms_mean=float(used_rms.mean()),projection_used_rms_max=float(used_rms.max()),
                projection_shrink_fraction=float((raw_z_rms>self.projection_rms_ceiling).float().mean()) if self.projection_repair else 0.,
                condition_raw_saturated_gt3=float((c['raw_preactivation'].detach().abs()>3).float().mean()),
                condition_used_saturated_gt3=float((c['preactivation'].detach().abs()>3).float().mean()),
                condition_tanh_derivative=float((1-c['condition'].detach().square()).mean()),
                conditional_raw_rms=float(raw_rms.mean()),conditional_used_rms=float(effective_rms.mean()),
                conditional_to_anchor_rms=float((effective_rms/anchor_rms).mean()),
                conditional_to_anchor_rms_max=float((effective_rms/anchor_rms).max()),
                summarize_fp32=float(all(x.dtype==torch.float32 for x in result.values())),
                **{name+'_summary_rms':float(value.detach().square().mean().sqrt()) for name,value in result.items()})
        return result


class PersonQueryModel(PersonGroundedModel):
    def initialize_interaction(self,arm,layer=5,dim=64,heads=4):
        if arm not in QUERY_ARMS or not 0<=layer<len(self.visual_encoder.transformer.resblocks):
            raise ValueError(arm)
        self.grounded_summary_policy=arm
        self.interaction_arm,self.interaction_layer,self.interaction_enabled='C2',layer,True
        with torch.random.fork_rng(devices=[]):
            modules=[]
            for task in range(self.num_tasks):
                torch.random.default_generator.manual_seed(930100+task)
                modules.append(PersonQueryReadout(self.width,self.head.in_features,self.num_selectors,arm,dim,heads))
            self.interaction_bank=nn.ModuleList(modules)
        self.interaction_bank.to(self.head.weight.device).requires_grad_(False)
        self._interaction_cache,self._private_sources={},{}


def query_configuration(arm):
    from .person_grounded_summary import grounded_configuration
    cfg=grounded_configuration(base_arm(arm))
    projection=arm.startswith('QPROJ_')
    cfg.update(protocol='person_query_single_factor_v1',matched_control=base_arm(arm),
        conditioning_repair='post_projection_rms_ceiling' if projection else 'relative_person_offset',
        projection_rms_ceiling=1. if projection else None,
        conditional_query_ratio_cap=None if projection else .01,
        conditional_small_signal_maximum_gain=1. if projection else 32.,
        new_trainable_parameters=0,combined_repairs=False)
    return cfg


@torch.no_grad()
def trace_person_query(module,tokens,anchors,base,person,boxes,visibility,reliable,
                       condition_person=None,zero_condition=False):
    """Trace the modified path, retaining the existing own/wrong-person API."""
    from .person_grounded_breakpoints import trace_readout
    supplied=person if condition_person is None else condition_person
    summaries=module.summarize(tokens,anchors,supplied,boxes,visibility,zero_condition)
    components=module.condition_components(anchors['full'],supplied,zero_condition)
    with torch.autocast(device_type=person.device.type,enabled=False):
        normalized=F.layer_norm(summaries['subject'],(summaries['subject'].shape[-1],))+module.view_types[1]
        result=dict(components,condition_source_ln=components['person_ln'],alpha=module.subject_alpha.clone(),
            summary=summaries['subject'],normalized_summary=normalized,subject_value=module.value(normalized))
    result.update(trace_readout(module,base,person,summaries,reliable))
    return result,summaries
