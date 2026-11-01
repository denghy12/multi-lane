"""Independent frozen MULTI-LANE branches with task-local CLS-agent exchange."""
import copy
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

VIEWS = ('full', 'person', 'face')
ARMS = ('I_W', 'I_M', 'C_W', 'C_M', 'P_M')


class CLSReadout(nn.Module):
    """One receiver; single CLS queries two frozen patch memories and itself."""
    def __init__(self, width, dim=64, heads=4, pool=False):
        super().__init__()
        if dim % heads:
            raise ValueError('Attention dimension must divide into heads')
        self.dim, self.heads, self.pool = dim, heads, pool
        self.types = nn.Parameter(torch.empty(2, width))
        nn.init.normal_(self.types, std=.02)
        self.v = nn.Linear(width, dim, bias=False)
        if pool:
            # Value projection plus CLS/two pools MLP ~= the CA budget.
            hidden = max(1, round((3*width*dim)/(2*width+2*dim)))
            self.mlp = nn.Linear(width+2*dim, hidden, bias=False)
            self.out = nn.Linear(hidden, width, bias=False)
        else:
            self.q = nn.Linear(width, dim, bias=False)
            self.k = nn.Linear(width, dim, bias=False)
            self.out = nn.Linear(dim, width, bias=False)
        nn.init.zeros_(self.out.weight)
        self.last = {}

    def forward(self, cls, memories, masks):
        # cls Bx1xD; memories do not depend on any trainable source branch.
        sources = [F.layer_norm(x.detach().float()+self.types[i].float(), (x.shape[-1],))
                   for i, x in enumerate(memories)]
        query = F.layer_norm(cls.float(), (cls.shape[-1],))
        if self.pool:
            pooled = [self.v(x.to(cls.dtype)).mean(1)*m[:, None].to(cls.dtype)
                      for x, m in zip(sources, masks)]
            update = self.out(F.gelu(self.mlp(torch.cat([query[:, 0].to(cls.dtype), *pooled], -1))))[:, None]
        else:
            memory = torch.cat([query, *sources], 1).to(cls.dtype)
            valid = torch.cat([torch.ones(cls.shape[0], 1, device=cls.device, dtype=torch.bool),
                              *[m[:, None].expand(-1, x.shape[1]) for m,x in zip(masks, sources)]], 1)
            b = cls.shape[0]; d = self.dim//self.heads
            q = self.q(query.to(cls.dtype)).reshape(b, 1, self.heads, d).transpose(1, 2)
            k = self.k(memory).reshape(b, -1, self.heads, d).transpose(1, 2)
            v = self.v(memory).reshape(b, -1, self.heads, d).transpose(1, 2)
            with torch.autocast(device_type=cls.device.type, enabled=False):
                logits = (q.float() @ k.float().transpose(-2, -1)) * d**-.5
                weights = logits.masked_fill(~valid[:, None, None], float('-inf')).softmax(-1)
                value = (weights @ v.float()).transpose(1,2).reshape(b,1,self.dim)
            update = self.out(value.to(cls.dtype))
        with torch.no_grad():
            self.last = dict(delta_rms=float(update.float().square().mean().sqrt()),
                             cls_rms=float(cls.float().square().mean().sqrt()))
        return cls + update


class IndependentThreeView(nn.Module):
    """Three independent parameter objects; no trained external task weights."""
    adapter_mode = 'image_token'
    selector_conditioner = joint_summary_bank = selector_view_residuals = None
    detach_view_fusion_features = False
    view_fusion = 'fixed_three_view'
    view_classifier_mode = 'private_per_view'
    cross_view_face_attention = True  # permits unused Face-only CA on invalid batches
    historical_b32_modules = strong_full_asl = False

    def __init__(self, base, arm, layer=5, dim=64, heads=4):
        super().__init__()
        if arm not in ARMS or base.cross_view_bank is not None:
            raise ValueError('Expected declared arm and unmodified historical BASE')
        if not 0 <= layer < len(base.visual_encoder.transformer.resblocks):
            raise ValueError('Invalid exchange layer')
        self.arm, self.layer = arm, layer
        self.branches = nn.ModuleDict(dict(full=base, person=copy.deepcopy(base), face=copy.deepcopy(base)))
        self._task_sizes = base.task_sizes
        self.view_fusion_module = SimpleNamespace(logit_residual=False, view_names=VIEWS)
        self.exchange = nn.ModuleList()
        if arm.startswith(('C_', 'P_')):
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(20261101)
                self.exchange.extend(nn.ModuleDict({v: CLSReadout(base.width, dim, heads, arm=='P_M')
                                        for v in VIEWS}) for _ in self._task_sizes)
        self.exchange_enabled = True
        self._weights = None
        self._face_any = True
        self._frozen_cache = None
        self.assert_independent()

    @property
    def current_task_id(self): return self.branches['full'].current_task_id
    @property
    def seen_classes(self): return self.branches['full'].seen_classes
    @property
    def task_sizes(self): return self._task_sizes

    def assert_independent(self):
        stores = [set(p.data_ptr() for p in b.parameters()) for b in self.branches.values()]
        if any(stores[i] & stores[j] for i in range(3) for j in range(i)):
            raise AssertionError('Cross-view parameter alias')
        for b in self.branches.values(): b.assert_visual_frozen()

    def activate_task(self, task):
        for b in self.branches.values(): b.activate_task(task)
        self.configure_current(task)

    def restore_task(self, task):
        for b in self.branches.values(): b.restore_task(task)
        self.configure_current(task)

    def configure_current(self, task):
        self.requires_grad_(False)
        for p in self.parameters(): p.grad = None
        if task < 0: return
        for b in self.branches.values():
            b.selectors.requires_grad_(True)
            b.prompts.requires_grad_(True)
            b.head.requires_grad_(True)
            b.adapter_bank.task_adapters[task].requires_grad_(True)
        if self.exchange: self.exchange[task].requires_grad_(True)

    def representation_optimizer_parameters(self):
        for b in self.branches.values(): yield from b.representation_optimizer_parameters()
    def classifier_optimizer_parameters(self):
        for b in self.branches.values(): yield from b.head.parameters()
    def fusion_optimizer_parameters(self):
        if self.exchange and self.current_task_id >= 0:
            yield from self.exchange[self.current_task_id].parameters()
    def prediction_optimizer_parameters(self):
        yield from self.classifier_optimizer_parameters()
        yield from self.fusion_optimizer_parameters()
    def base_optimizer_parameters(self):
        yield from self.representation_optimizer_parameters()
        yield from self.prediction_optimizer_parameters()
    def adapter_optimizer_parameters(self):
        for b in self.branches.values(): yield from b.adapter_optimizer_parameters()
    def optimizer_parameter_names(self):
        ids={id(p) for p in (*self.base_optimizer_parameters(), *self.adapter_optimizer_parameters())}
        return tuple(n for n,p in self.named_parameters() if id(p) in ids)

    def _features(self, images, all_seen):
        lanes = self.branches['full']._lane_ids(all_seen)
        batch = images['full'].shape[0]
        reliable = images['face_reliable'].bool()
        self._face_any = bool(reliable.any())
        with torch.no_grad():
            tokens = {}
            for v,b in self.branches.items():
                if self._frozen_cache is not None and (v,-1) in self._frozen_cache:
                    tokens[v]=self._frozen_cache[v,-1]
                else:
                    tokens[v]=b._visual_tokens(images[v])
                    if self._frozen_cache is not None:self._frozen_cache[v,-1]=tokens[v]
        state = {v:b._initial_lane_tokens(batch, lanes) for v,b in self.branches.items()}
        for b in self.branches.values(): b.adapter_bank.reset_auxiliary_metrics()
        depth = len(self.branches['full'].visual_encoder.transformer.resblocks)
        for layer in range(depth):
            for v,b in self.branches.items():
                block=b.visual_encoder.transformer.resblocks[layer]
                state[v]=b._lane_block(block,tokens[v],state[v],lanes,layer)
                with torch.no_grad():
                    if self._frozen_cache is not None and (v,layer) in self._frozen_cache:
                        tokens[v]=self._frozen_cache[v,layer]
                    else:
                        tokens[v]=block(tokens[v].permute(1,0,2)).permute(1,0,2)
                        if self._frozen_cache is not None:self._frozen_cache[v,layer]=tokens[v]
            if layer == self.layer and self.exchange and self.exchange_enabled:
                # Read only the PRE-exchange states. Three writes are simultaneous.
                original=state
                updated={}
                for v in VIEWS:
                    if v=='face' and not self._face_any:
                        updated[v]=original[v]; continue
                    others=[u for u in VIEWS if u!=v]
                    masks=[reliable if u=='face' else torch.ones_like(reliable) for u in others]
                    patches=[tokens[u][:,1:].detach() for u in others]
                    cls=[]
                    for i,t in enumerate(lanes):
                        c=original[v][i,:,:1]
                        if v=='face':
                            new=c.clone()
                            new[reliable]=self.exchange[t][v](c[reliable],
                                [p[reliable] for p in patches],[m[reliable] for m in masks])
                        else:
                            new=self.exchange[t][v](c,patches,masks)
                        cls.append(torch.cat([new,original[v][i,:,1:]],1))
                    updated[v]=torch.stack(cls)
                state=updated
        features={}
        for v,b in self.branches.items():
            x=b.visual_encoder.ln_post(state[v])
            if b.visual_encoder.proj is not None: x=x @ b.visual_encoder.proj
            features[v]=F.normalize(x[:,:,0].permute(1,0,2).float(),dim=-1)
        return features,lanes

    def _forward(self, images, all_seen):
        features, lanes=self._features(images,all_seen)
        logits={v:self.branches[v].head(features[v]) for v in VIEWS}
        reliable=images['face_reliable'].bool()
        if self.arm.endswith('_W'):
            good=features['full'].new_tensor([.64,.16,.20])
            bad=features['full'].new_tensor([.8,.2,0.])
        else:
            good=features['full'].new_tensor([1/3,1/3,1/3])
            bad=features['full'].new_tensor([.5,.5,0.])
        weights=torch.where(reliable[:,None],good,bad)
        self._weights=weights[:,None,:].expand(-1,len(lanes),-1).detach()
        fused=sum(logits[v]*weights[:,i,None,None] for i,v in enumerate(VIEWS))
        return fused,logits,features,lanes

    def current_all_logits_with_view_features(self,images):
        z,vs,fs,_=self._forward(images,False)
        return z[:,0],{k:v[:,0] for k,v in vs.items()},fs
    def current_all_logits_with_views(self,images):
        z,vs,_=self.current_all_logits_with_view_features(images);return z,vs
    def seen_logits_with_views(self,images):
        z,vs,_,lanes=self._forward(images,True)
        mask=self.branches['full'].task_class_mask[lanes]
        combine=lambda value:(value*mask[None]).sum(1)[:,:self.seen_classes]
        return combine(z),{k:combine(v) for k,v in vs.items()}
    def seen_logits(self,images): return self.seen_logits_with_views(images)[0]
    def last_fusion_weights(self): return self._weights
    def fusion_weight_means(self):
        return tuple(self._weights.float().mean((0,1)).tolist()) if self._weights is not None else None

    def adapter_gradient_diagnostics(self,scale=1.):
        # A zero loss gradient must not decay an existing Face Adam momentum.
        if not self._face_any:
            for p in self.branches['face'].parameters(): p.grad=None
            if self.exchange:
                for p in self.exchange[self.current_task_id]['face'].parameters(): p.grad=None
        return {}
    def adapter_view_diagnostics(self): return {}
    def parax_gradient_diagnostics(self,*args): return {}
    def parax_gate_diagnostics(self): return {}
    def logit_residual_diagnostics(self): return {}
    def _zero(self): return self.branches['full'].selectors.new_zeros(())
    def parax_residual_penalty(self): return self._zero()
    def parax_alignment_penalty(self): return self._zero()
    def moe_balance_loss(self): return self._zero()
    def selector_view_residual_metric(self): return self._zero()


def optimizer_groups(model, **kwargs):
    rep=list(model.representation_optimizer_parameters())
    heads=list(model.classifier_optimizer_parameters())
    interaction=list(model.fusion_optimizer_parameters())
    adapter=list(model.adapter_optimizer_parameters())
    groups=[dict(params=rep,weight_decay=0.),dict(params=heads,weight_decay=0.)]
    if interaction: groups.append(dict(params=interaction,lr=.0004,weight_decay=0.))
    groups.append(dict(params=adapter,lr=.0004,weight_decay=0.))
    ids=[id(p) for g in groups for p in g['params']]
    if len(ids)!=len(set(ids)) or set(ids)!={id(p) for p in model.parameters() if p.requires_grad}:
        raise AssertionError('Incomplete or overlapping gradient routing')
    return rep+heads+interaction,adapter,groups


def compact_state(model):
    return {n:v.detach().cpu().clone() for n,v in model.state_dict().items() if '.visual_encoder.' not in n}


def old_state(model,task):
    result={};start=sum(model.task_sizes[:task])
    for v,b in model.branches.items():
        result[v+'.selectors']=b.selectors[:task]
        for i,p in enumerate(b.prompts):result[f'{v}.prompts.{i}']=p[:,:task]
        result[v+'.head.weight']=b.head.weight[:start]
        result[v+'.head.bias']=b.head.bias[:start]
        for i in range(task):
            for n,p in b.adapter_bank.task_adapters[i].state_dict().items():result[f'{v}.adapter.{i}.{n}']=p
    for i in range(min(task,len(model.exchange))):
        for n,p in model.exchange[i].state_dict().items():result[f'exchange.{i}.{n}']=p
    return result
