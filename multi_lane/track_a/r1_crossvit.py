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
    def __init__(self,width,dim=64,heads=4,pool=False):
        super().__init__();self.dim,self.heads,self.pool=dim,heads,pool
        if dim%heads:raise ValueError('Invalid head width')
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

    def forward(self,cls,memories,masks):
        # Keep only the NEW readout in FP32; the original R1 task path is unchanged.
        with torch.autocast(device_type=cls.device.type,enabled=False):
            c=F.layer_norm(cls.float(),(cls.shape[-1],))
            sources=[F.layer_norm(x.detach().float()+self.types[i],(x.shape[-1],)) for i,x in enumerate(memories)]
            if self.pool:
                parts=[self.v(x).mean(1)*m[:,None].float() for x,m in zip(sources,masks)]
                delta=self.out(F.gelu(self.mlp(torch.cat([c[:,0],*parts],-1))))[:,None]
            else:
                memory=torch.cat([c,*sources],1)
                b=len(c);d=self.dim//self.heads
                q=self.q(c).reshape(b,1,self.heads,d).transpose(1,2)
                k=self.k(memory).reshape(b,-1,self.heads,d).transpose(1,2)
                v=self.v(memory).reshape(b,-1,self.heads,d).transpose(1,2)
                valid=torch.cat([torch.ones(b,1,dtype=torch.bool,device=c.device),
                    *[m[:,None].expand(-1,x.shape[1]) for m,x in zip(masks,sources)]],1)
                a=((q@k.transpose(-2,-1))*d**-.5).masked_fill(~valid[:,None,None],float('-inf')).softmax(-1)
                delta=self.out((a@v).transpose(1,2).reshape(b,1,self.dim))
        return cls+delta


class R1Receiver(MultiLaneModel):
    def __init__(self,*args,r1_kind='native',r1_view='full',r1_root=None,r1_layer=5,r1_dim=64,r1_heads=4,**kwargs):
        super().__init__(*args,**kwargs)
        if r1_kind not in KINDS or r1_view not in VIEWS:raise ValueError('Unknown R1 arm/view')
        if not 0<=r1_layer<len(self.visual_encoder.transformer.resblocks):raise ValueError('Invalid layer')
        self.r1_kind,self.r1_view,self.r1_layer=r1_kind,r1_view,r1_layer
        self.r1_root=Path(r1_root) if r1_root else None
        self.r1_bank=nn.ModuleList();self.r1_enabled=True;self.r1_config=None;self.r1_resume=None
        if r1_kind!='native':
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(20261102)
                self.r1_bank.extend(CLSAgent(self.width,r1_dim,r1_heads,r1_kind=='pool') for _ in self.task_sizes)

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

    def encode_lanes(self,images,all_seen_lanes):
        own=images['full'] if isinstance(images,dict) else images
        if not self.r1_bank or not self.r1_enabled:return super().encode_lanes(own,all_seen_lanes)
        if not isinstance(images,dict):raise ValueError('Cross-view evidence missing')
        others=[v for v in VIEWS if v!=self.r1_view]
        with torch.no_grad():
            memories=[];masks=[]
            for v in others:
                x=self._visual_tokens(images['evidence_'+v])
                for block in list(self.visual_encoder.transformer.resblocks)[:self.r1_layer+1]:
                    x=block(x.permute(1,0,2)).permute(1,0,2)
                memories.append(x[:,1:].detach())
                masks.append(images['evidence_face_reliable'].bool() if v=='face'
                    else torch.ones(len(own),device=own.device,dtype=torch.bool))
        lanes=self._lane_ids(all_seen_lanes)
        self.adapter_bank.reset_auxiliary_metrics()
        tokens=self._visual_tokens(own);state=self._initial_lane_tokens(len(own),lanes)
        for layer,block in enumerate(self.visual_encoder.transformer.resblocks):
            state=self._lane_block(block,tokens,state,lanes,layer)
            with torch.no_grad():tokens=block(tokens.permute(1,0,2)).permute(1,0,2)
            if layer==self.r1_layer:
                state=torch.stack([torch.cat([self.r1_bank[t](state[i,:,:1],memories,masks),state[i,:,1:]],1)
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
