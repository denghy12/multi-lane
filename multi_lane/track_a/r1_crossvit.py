"""CrossViT-style receivers on the archived independently supervised R1 recipe."""
import copy
import json
import os
import random
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model import MultiLaneModel

VIEWS=('full','person','face')
KINDS=('native','attention','pool')


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def restore_rng(d):
    random.setstate(d['python']);np.random.set_state(d['numpy']);torch.set_rng_state(d['cpu'])
    if d['cuda']:torch.cuda.set_rng_state_all(d['cuda'])


def atomic_save(d,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name,delete=False) as f:
        tmp=Path(f.name);torch.save(d,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def compact(model):
    return {n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not n.startswith('visual_encoder.')}


class CLSAgent(nn.Module):
    def __init__(self,width,dim=64,heads=4,pool=False,query_mode='cls',pool_cap=None):
        super().__init__();self.dim,self.heads,self.pool=dim,heads,pool
        if dim%heads:raise ValueError('Invalid head width')
        if query_mode not in ('cls','selector_mean','selectors'):
            raise ValueError('Unknown query source')
        if pool and query_mode != 'cls':
            raise ValueError('Pool has no query-source ablation')
        if pool_cap is not None and (not pool or not 0 < pool_cap <= 1):
            raise ValueError('Pool residual cap must be in (0, 1]')
        self.pool_cap = pool_cap
        self.query_mode=query_mode
        self.diagnostic_enabled=False
        self.last_diagnostic=None
        self.types=nn.Parameter(torch.randn(2,width)*.02)
        self.v=nn.Linear(width,dim,bias=False)
        if pool:
            hidden=max(1,round(3*width*dim/(2*width+2*dim)))
            self.mlp=nn.Linear(width+2*dim,hidden,bias=False)
            self.out=nn.Linear(hidden,width,bias=False)
        else:
            self.q=nn.Linear(width,dim,bias=False);self.k=nn.Linear(width,dim,bias=False)
            self.out=nn.Linear(dim,width,bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self,cls,memories,masks,selectors=None):
        # Keep only the NEW readout in FP32; the original R1 task path is unchanged.
        with torch.autocast(device_type=cls.device.type,enabled=False):
            c=F.layer_norm(cls.float(),(cls.shape[-1],))
            sources=[F.layer_norm(x.detach().float()+self.types[i],(x.shape[-1],)) for i,x in enumerate(memories)]
            if self.pool:
                parts=[self.v(x).mean(1)*m[:,None].float() for x,m in zip(sources,masks)]
                delta=self.out(F.gelu(self.mlp(torch.cat([c[:,0],*parts],-1))))[:,None]
                if self.pool_cap is not None:
                    cap = self.pool_cap * cls.detach().float().norm(dim=-1, keepdim=True)
                    delta = delta * (cap / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1)
                if self.diagnostic_enabled:
                    with torch.no_grad():
                        self.last_diagnostic = {
                            'residual_to_cls_norm': float((delta.norm(dim=-1) /
                                cls.float().norm(dim=-1).clamp_min(1e-8)).mean())
                        }
            else:
                if self.query_mode == 'cls':
                    queries=c
                else:
                    if selectors is None or selectors.ndim != 3 or selectors.shape[1] == 0:
                        raise ValueError('Selector queries require nonempty selector states')
                    queries=selectors.float()
                    if self.query_mode == 'selector_mean':
                        queries=queries.mean(dim=1,keepdim=True)
                    queries=F.layer_norm(queries,(queries.shape[-1],))
                # The memory remains identical for all query sources, including own CLS.
                memory=torch.cat([c,*sources],1)
                b=len(c);d=self.dim//self.heads
                q=self.q(queries).reshape(b,-1,self.heads,d).transpose(1,2)
                k=self.k(memory).reshape(b,-1,self.heads,d).transpose(1,2)
                v=self.v(memory).reshape(b,-1,self.heads,d).transpose(1,2)
                valid=torch.cat([torch.ones(b,1,dtype=torch.bool,device=c.device),
                    *[m[:,None].expand(-1,x.shape[1]) for m,x in zip(masks,sources)]],1)
                a=((q@k.transpose(-2,-1))*d**-.5).masked_fill(~valid[:,None,None],float('-inf')).softmax(-1)
                contexts=(a@v).transpose(1,2).reshape(b,-1,self.dim)
                delta=self.out(contexts.mean(dim=1,keepdim=True))
                if self.diagnostic_enabled:
                    with torch.no_grad():
                        masses=[];offset=0
                        for size in [1,*[x.shape[1] for x in sources]]:
                            masses.append(float(a[...,offset:offset+size].sum(-1).mean()))
                            offset+=size
                        entropy=-(a*a.clamp_min(1e-30).log()).sum(-1)
                        self.last_diagnostic=dict(query_count=queries.shape[1],source_masses=masses,
                            query_batch_std=float(queries.std(dim=0,unbiased=False).mean()),
                            query_slot_std=float(queries.std(dim=1,unbiased=False).mean()),
                            projected_query_norm=float(q.norm(dim=-1).mean()),
                            projected_key_norm=float(k.norm(dim=-1).mean()),
                            attention_entropy=float(entropy.mean()),
                            residual_to_cls_norm=float((delta.norm(dim=-1)/cls.float().norm(dim=-1).clamp_min(1e-8)).mean()))
        return cls+delta


class R1Receiver(MultiLaneModel):
    def __init__(self,*args,r1_kind='native',r1_view='full',r1_root=None,r1_layer=5,r1_dim=64,r1_heads=4,r1_sender_root=None,r1_query_mode='cls',**kwargs):
        super().__init__(*args,**kwargs)
        if r1_kind not in KINDS or r1_view not in VIEWS:raise ValueError('Unknown R1 arm/view')
        if not 0<=r1_layer<len(self.visual_encoder.transformer.resblocks):raise ValueError('Invalid layer')
        self.r1_kind,self.r1_view,self.r1_layer=r1_kind,r1_view,r1_layer
        self.r1_query_mode=r1_query_mode
        if r1_query_mode != 'cls' and (r1_kind != 'attention' or r1_view != 'face' or r1_sender_root is not None):
            raise ValueError('Selector-query ablation requires frozen-patch Face Attention')
        self.r1_root=Path(r1_root) if r1_root else None
        if r1_sender_root is not None and (r1_view != 'face' or r1_kind == 'native'):
            raise ValueError('Task summaries require an interaction-enabled Face receiver')
        self.r1_sender_root = Path(r1_sender_root) if r1_sender_root else None
        self._senders = {}
        self._sender_stage = None
        self.r1_bank=nn.ModuleList();self.r1_enabled=True;self.r1_config=None;self.r1_resume=None
        if r1_kind!='native':
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(20261102)
                self.r1_bank.extend(CLSAgent(self.width,r1_dim,r1_heads,r1_kind=='pool',r1_query_mode) for _ in self.task_sizes)

    def activate_task(self,task):
        super().activate_task(task);self._configure_bank(task);self.r1_resume=None
        if self.r1_root:
            files=sorted((self.r1_root/'epoch_recovery'/f'task{task}').glob('epoch*.pth'))
            if files:
                saved=torch.load(files[-1],map_location='cpu',weights_only=False)
                if any(r['skipped_optimizer_steps'] for r in saved['training']['history']):
                    raise FloatingPointError('Recovery contains skipped steps; do not resume as a successful run')
                self.load_own(saved);self.r1_resume=saved['training'];restore_rng(saved['rng'])

    def restore_task(self,task):
        super().restore_task(task);self._configure_bank(task)

    def _configure_bank(self,task):
        for i,m in enumerate(self.r1_bank):m.requires_grad_(i==task)

    def r1_parameters(self):
        if self.r1_bank and self.current_task_id>=0:yield from self.r1_bank[self.current_task_id].parameters()

    def base_optimizer_parameters(self):
        yield from super().base_optimizer_parameters();yield from self.r1_parameters()

    def optimizer_parameter_names(self):
        extra=[f'r1_bank.{self.current_task_id}.{n}' for n,p in self.r1_bank[self.current_task_id].named_parameters()] if self.r1_bank and self.current_task_id>=0 else []
        return (*super().optimizer_parameter_names(),*extra)

    @torch.no_grad()
    def sender_summaries(self, images, lanes):
        stage = self.current_task_id
        if self._sender_stage != stage:
            for view in ('full', 'person'):
                if view not in self._senders:
                    # Keep sender parameters out of the receiver optimizer and checkpoints.
                    with torch.random.fork_rng(devices=[]):
                        sender = MultiLaneModel(
                            self.visual_encoder, self.task_sizes,
                            num_selectors=self.num_selectors, num_prompts=self.num_prompts,
                            num_prompt_layers=self.num_prompt_layers, normalize=self.normalize,
                            adapter_mode='image_token', adapter_bottleneck_dim=32,
                            adapter_layer_indices=(1,), adapter_residual_scale=.1,
                            adapter_activation='relu',
                        ).to(images['full'].device).eval().requires_grad_(False)
                    self._senders[view] = sender
                sender = self._senders[view]
                path = self.r1_sender_root / view / 'train' / 'compact_checkpoints' / f'task{stage}.pth'
                payload = torch.load(path, map_location='cpu', weights_only=False)
                if payload['task_id'] != stage:
                    raise ValueError(f'Sender checkpoint has the wrong stage: {path}')
                missing = sender.load_state_dict(payload['model'], strict=False)
                if missing.unexpected_keys or any(not k.startswith('visual_encoder.') for k in missing.missing_keys):
                    raise ValueError(f'Incomplete sender checkpoint: {path}')
                sender.restore_task(stage)
                sender.requires_grad_(False)
            self._sender_stage = stage

        summaries = []
        for view, sender in self._senders.items():
            image = images['evidence_' + view]
            tokens = sender._visual_tokens(image)
            state = sender._initial_lane_tokens(len(image), lanes)
            for layer, block in enumerate(self.visual_encoder.transformer.resblocks):
                if layer == self.r1_layer:
                    summaries.append(sender._lane_block(block, tokens, state, lanes, layer, summary_only=True))
                    break
                state = sender._lane_block(block, tokens, state, lanes, layer)
                tokens = block(tokens.permute(1, 0, 2)).permute(1, 0, 2)
        return summaries

    def encode_lanes(self,images,all_seen_lanes):
        own=images['full'] if isinstance(images,dict) else images
        if not self.r1_bank or not self.r1_enabled:return super().encode_lanes(own,all_seen_lanes)
        if not isinstance(images,dict):raise ValueError('Cross-view evidence missing')
        others=[v for v in VIEWS if v!=self.r1_view]
        lanes=self._lane_ids(all_seen_lanes)
        summaries = self.sender_summaries(images, lanes) if self.r1_sender_root else None
        with torch.no_grad():
            memories=[];masks=[]
            for v in others:
                if summaries is not None:
                    masks.append(torch.ones(len(own), device=own.device, dtype=torch.bool))
                    continue
                x=self._visual_tokens(images['evidence_'+v])
                for block in list(self.visual_encoder.transformer.resblocks)[:self.r1_layer+1]:
                    x=block(x.permute(1,0,2)).permute(1,0,2)
                memories.append(x[:,1:].detach())
                masks.append(images['evidence_face_reliable'].bool() if v=='face'
                    else torch.ones(len(own),device=own.device,dtype=torch.bool))
        self.adapter_bank.reset_auxiliary_metrics()
        tokens=self._visual_tokens(own);state=self._initial_lane_tokens(len(own),lanes)
        for layer,block in enumerate(self.visual_encoder.transformer.resblocks):
            state=self._lane_block(block,tokens,state,lanes,layer)
            with torch.no_grad():tokens=block(tokens.permute(1,0,2)).permute(1,0,2)
            if layer==self.r1_layer:
                state=torch.stack([torch.cat([self.r1_bank[t](state[i,:,:1],memories if summaries is None else [x[i] for x in summaries],masks,selectors=state[i,:,1:]),state[i,:,1:]],1)
                                   for i,t in enumerate(lanes)])
        state=self.visual_encoder.ln_post(state)
        if self.visual_encoder.proj is not None:state=state@self.visual_encoder.proj
        return F.normalize(state[:,:,0].permute(1,0,2).float(),dim=-1)

    def load_own(self,saved):
        if self.r1_config is not None and saved['config']!=self.r1_config:raise ValueError('Recovery config changed')
        missing=self.load_state_dict(saved['model'],strict=False)
        if missing.unexpected_keys or any(not n.startswith('visual_encoder.') for n in missing.missing_keys):raise ValueError('Missing task state')
        self.restore_task(saved['task'])

    def save_training_epoch(self,task,epoch,history,optimizer,scheduler,scaler,updates):
        if not self.r1_root:return
        if torch.cuda.is_available():
            history[-1].setdefault('cuda_peak_reserved_mib', torch.cuda.max_memory_reserved() / 1024**2)
        names={id(p):n for n,p in self.named_parameters()}
        training=dict(optimizer=copy.deepcopy(optimizer.state_dict()),scheduler=copy.deepcopy(scheduler.state_dict()),
            scaler=scaler.state_dict(),history=copy.deepcopy(history),next_epoch=epoch,completed_task_updates=updates,
            optimizer_parameter_names=[[names[id(p)] for p in g['params']] for g in optimizer.param_groups])
        saved=dict(task=task,config=self.r1_config,model=compact(self),training=training,rng=rng_state())
        atomic_save(saved,self.r1_root/'epoch_recovery'/f'task{task}'/f'epoch{epoch:02d}.pth')
        (self.r1_root/'progress.json').write_text(json.dumps(dict(task=task,epoch=epoch,updates=updates,
            skipped_steps=sum(r['skipped_optimizer_steps'] for r in history)))+'\n')
        if any(r['skipped_optimizer_steps'] for r in history):raise FloatingPointError('R1 run stopped at first skipped epoch; recovery retained')

    def restore_completed_task(self,task):
        if not self.r1_root:return None
        p=self.r1_root/'task_recovery'/f'task{task}.pth'
        if not p.exists():return None
        saved=torch.load(p,map_location='cpu',weights_only=False)
        self.load_own(saved);restore_rng(saved['rng']);return saved

    def save_completed_task(self,task,history,row):
        if self.r1_root:
            atomic_save(dict(task=task,config=self.r1_config,model=compact(self),rng=rng_state(),
                history=history,row=row),self.r1_root/'task_recovery'/f'task{task}.pth')


class EvidenceSource:
    """Native receiver augmentation first; deterministic TRAIN evidence afterward."""
    def __init__(self,source,auxiliary,receiver):
        self.source,self.auxiliary,self.receiver=source,auxiliary,receiver
        for a in auxiliary.values():
            if source.sample_ids!=a.sample_ids:raise ValueError('Evidence sample alignment differs')
    def __getattr__(self,name):
        if name=='source':raise AttributeError(name)
        return getattr(self.source,name)
    def __len__(self):return len(self.source)
    def __getitem__(self,index):
        own,y=self.source[index]
        evidence={'full':own}
        for view,source in self.auxiliary.items():
            evidence['evidence_'+view]=source[index][0]
        f=self.source if self.receiver=='face' else self.auxiliary['face']
        evidence['evidence_face_reliable']=torch.tensor(f.face_reliable[index],dtype=torch.bool)
        return evidence,y


def attach_evidence(source,receiver,dataset_parent,manifest_root):
    from multi_lane.continual_datasets.continual_datasets import EMOTIC
    from .runner import build_transforms
    auxiliary={}
    for v in VIEWS:
        if v==receiver:continue
        _,transform=build_transforms('clip',(0.05,1.),input_mode={'full':'full','person':'person_crop','face':'face_crop'}[v],
            person_transform_mode='letterbox' if v=='person' else 'legacy_crop',
            person_color_jitter_strength=0.,person_color_jitter_probability=0.)
        auxiliary[v]=EMOTIC(str(dataset_parent),train=tuple(source.splits)==('train',),
            eval_splits=source.splits,transform=transform,
            input_mode={'full':'full','person':'person_crop','face':'face_crop'}[v],
            person_crop_margin=.15 if v=='person' else 0.,
            face_manifest_root=manifest_root if v=='face' else None)
    return EvidenceSource(source,auxiliary,receiver)
