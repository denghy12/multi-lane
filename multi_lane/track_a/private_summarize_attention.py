"""Protected 2x2: original/learnable summarization queries x pool/attention."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from .protected_joint_attention import ProtectedJointModel, VIEWS, bounded_delta

SUMMARY_ARMS=('SUMMARY_FIXED_POOL','SUMMARY_FIXED_ATTN','SUMMARY_LEARN_POOL','SUMMARY_LEARN_ATTN')


def norm_matched_query(anchor, offset):
    anchor=anchor.detach().float()
    proposal=anchor+offset.float()
    norm=proposal.norm(dim=-1,keepdim=True)
    value=proposal*(anchor.norm(dim=-1,keepdim=True)/norm.clamp_min(1e-12))
    return torch.where(norm>1e-12,value,anchor)


class PrivateSummaryReadout(nn.Module):
    def __init__(self,width,output_dim,selectors,arm,dim=64,heads=4):
        super().__init__()
        if arm not in SUMMARY_ARMS or dim%heads:raise ValueError(arm)
        self.arm,self.dim,self.heads=arm,dim,heads
        self.learnable='LEARN' in arm
        # Common components are constructed BEFORE operator-specific components.
        self.query=nn.Linear(2*output_dim,dim,bias=False)
        self.value=nn.Linear(width,dim,bias=False)
        self.out=nn.Linear(dim,output_dim,bias=False)
        self.view_types=nn.Parameter(torch.empty(3,width))
        nn.init.normal_(self.view_types,std=.01);nn.init.zeros_(self.out.weight)
        if arm.endswith('ATTN'):
            self.key=nn.Linear(width,dim,bias=False)
        else:
            self.pool=nn.Sequential(nn.Linear(4*dim+1,152),nn.GELU(),nn.Linear(152,dim))
        if self.learnable:self.query_offset=nn.Parameter(torch.zeros(selectors,width))
        else:self.register_buffer('query_offset',torch.zeros(selectors,width))
        self.last_metrics={}

    def summarize(self,tokens,anchors,zero_query=False):
        with torch.autocast(device_type=tokens['full'].device.type,enabled=False):
            return self._summarize_fp32(tokens,anchors,zero_query)

    def _summarize_fp32(self,tokens,anchors,zero_query=False):
        result={};cos=[];ratios=[];variation=[]
        for name in VIEWS:
            g=tokens[name].detach().float();s=anchors[name].detach().float()
            u=torch.zeros_like(self.query_offset) if zero_query else self.query_offset
            q=norm_matched_query(s,u)
            weights=(q@g.transpose(-1,-2)/math.sqrt(g.shape[-1])).softmax(-1)
            result[name]=weights@g
            cos.append(F.cosine_similarity(q.detach(),s,dim=-1).mean())
            ratios.append((q.detach().norm(dim=-1)/s.norm(dim=-1).clamp_min(1e-12)).mean())
            variation.append(result[name].detach().var(dim=1,unbiased=False).mean())
        self._query_statistics=dict(query_anchor_cosine=float(torch.stack(cos).mean()),
            query_norm_ratio=float(torch.stack(ratios).mean()),
            summary_slot_variance=float(torch.stack(variation).mean()),
            query_offset_norm=float(self.query_offset.detach().norm()))
        return result

    def forward(self,base,person,summaries,reliable,no_face=False,no_person=False,query_person=None):
        with torch.autocast(device_type=base.device.type,enabled=False):
            base=base.detach().float();person=person.detach().float()
            qp=person if query_person is None else query_person.detach().float()
            query=self.query(F.layer_norm(torch.cat([base,qp],-1),(2*base.shape[-1],)))
            use_face=reliable.detach().bool() & (not no_face)
            memories=[];valid=[];pooled=[]
            for i,name in enumerate(VIEWS):
                memory=F.layer_norm(summaries[name].float(),(summaries[name].shape[-1],))+self.view_types[i]
                allowed=use_face if name=='face' else torch.full_like(use_face,not no_person) if name=='person' else torch.ones_like(use_face)
                value=self.value(memory)
                pooled.append(torch.where(allowed[:,None],value.mean(1),torch.zeros_like(value[:,0])))
                memories.append(memory);valid.append(allowed[:,None].expand(-1,memory.shape[1]))
            if hasattr(self,'key'):
                memory=torch.cat(memories,1);mask=F.pad(torch.cat(valid,1),(0,1),value=True)
                k=F.pad(self.key(memory),(0,0,0,1));v=F.pad(self.value(memory),(0,0,0,1))
                b,n=mask.shape
                q=query.reshape(b,self.heads,1,-1)
                k=k.reshape(b,n,self.heads,-1).transpose(1,2);v=v.reshape(b,n,self.heads,-1).transpose(1,2)
                weights=((q@k.transpose(-1,-2))/math.sqrt(self.dim/self.heads)).masked_fill(~mask[:,None,None],float('-inf')).softmax(-1)
                hidden=(weights@v).reshape(b,self.dim)
                face_mass=weights[...,2*summaries['full'].shape[1]:-1].sum(-1).mean()
            else:
                hidden=self.pool(torch.cat([query,*pooled,use_face[:,None].float()],-1))
                face_mass=base.new_zeros(())
            delta=bounded_delta(self.out(hidden),base,.01)
            ratio=delta.detach().square().mean(-1).sqrt()/base.square().mean(-1).sqrt().clamp_min(1e-12)
            if not bool(torch.isfinite(delta).all()) or bool((ratio>.010001).any()):
                raise FloatingPointError('Private summary final cap violation')
            self.last_metrics=dict(gate_mean=1.,delta_to_base_rms=float(ratio.mean()),
                delta_to_base_rms_max=float(ratio.max()),delta_near_cap_fraction=float((ratio>.0095).float().mean()),
                face_attention_mass=float(face_mass.detach()),**self._query_statistics)
            return delta

    def after_backward_metrics(self):
        g=self.query_offset.grad if self.learnable else None
        if g is not None and not bool(torch.isfinite(g).all()):raise FloatingPointError('Non-finite query gradient')
        return dict(query_gradient_norm=float(g.float().norm()) if g is not None else 0.)


class PrivateSummaryModel(ProtectedJointModel):
    def initialize_interaction(self,arm,layer=5,dim=64,heads=4):
        if arm not in SUMMARY_ARMS or not 0<=layer<len(self.visual_encoder.transformer.resblocks):raise ValueError(arm)
        self.private_summary_policy=arm
        self.interaction_arm,self.interaction_layer,self.interaction_enabled='C2',layer,True
        with torch.random.fork_rng(devices=[]):
            modules=[]
            for task in range(self.num_tasks):
                torch.random.default_generator.manual_seed(929100+task)
                modules.append(PrivateSummaryReadout(self.width,self.head.in_features,self.num_selectors,arm,dim,heads))
            self.interaction_bank=nn.ModuleList(modules)
        self.interaction_bank.to(self.head.weight.device).requires_grad_(False)
        self._interaction_cache={};self._private_sources={}

    def encode_lanes_with_views(self,images,all_seen_lanes):
        self._private_sources={}
        return super().encode_lanes_with_views(images,all_seen_lanes)

    def _task_attention_ffn(self,block,lane_tokens,lane_ids,layer_id,task_cls,selectors,selected,image_view='full',summary_valid=None):
        if layer_id==self.interaction_layer:
            _,image_tokens=self._interaction_image_source
            with torch.no_grad():tokens=block.ln_1(image_tokens).detach()
            self._private_sources[image_view]=(tokens,selectors.detach())
        return super()._task_attention_ffn(block,lane_tokens,lane_ids,layer_id,task_cls,selectors,selected,image_view,summary_valid)

    def interaction_delta(self,no_face=False,no_person=False,zero_query=False,query_donor=None,memory_donors=None):
        c=self._interaction_cache;rows=[]
        for index,task in enumerate(c['lane_ids']):
            module=self.interaction_bank[task]
            summaries=module.summarize({v:self._private_sources[v][0] for v in VIEWS},
                {v:self._private_sources[v][1][index] for v in VIEWS},zero_query)
            if memory_donors:
                summaries={v:s[memory_donors[v]] if v in memory_donors else s for v,s in summaries.items()}
            person=c['features']['person'][:,index]
            qp=None if query_donor is None else person[query_donor]
            rows.append(module(c['base'][:,index],person,summaries,c['reliable'],no_face,no_person,qp))
        delta=torch.stack(rows,1)
        if not any((no_face,no_person,zero_query,query_donor is not None,bool(memory_donors))):self._native_delta=delta.detach()
        return delta


def summary_configuration(arm):
    if arm not in SUMMARY_ARMS:raise ValueError(arm)
    return dict(protocol='private_summary_2x2_v1',layer=5,dim=64,heads=4,selectors=10,
        learning_rate=.0004,query_learning_rate=.0004,final_residual_cap=.01,fixed_gain=1.,
        readout='attention' if arm.endswith('ATTN') else 'pool_mlp_257_152_64',
        new_private_summarize_queries='LEARN' in arm,query_offset_zero_initialized=True,
        query_norm_matched=True,query_shared_across_views=True,query_parameters_per_task=7680 if 'LEARN' in arm else 0,
        summarize_precision='fp32_common_recompute',global_full_summaries=True,target_region_constraint=False,
        common_projection_initialization_seed='929100+task_common_components_first',
        independent_optimizer_scaler_rng=True,source_only='three_views',learned_gate=False)
