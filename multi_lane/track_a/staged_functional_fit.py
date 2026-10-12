"""Fit-only optimization and read-only held-out diagnostics for staged experts."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.nn import functional as F
from .post_task_calibration import base_hash, parameter_hash
from .runner import average_precision
from .staged_functional_experts import held_out


def check_partition(cache, expected):
    if cache.get('source_split') != 'train' or cache.get('module_split') != expected:
        raise ValueError('Wrong module partition')
    if any(held_out(s) != (expected == 'holdout') for s in cache['sample_ids']):
        raise ValueError('Cross-partition image detected')


def batch_forward(model, cache, indices, classes, device):
    feats = {v: x[indices].to(device) for v,x in cache['features'].items()}
    reliable = cache['face_reliable'][indices].to(device)
    labels = cache['targets'][indices].to(device)
    bank = model.parax_bank
    routed, ratios = {}, {}
    bank.reset_forward_diagnostics()
    for v,x in feats.items():
        y,_ = bank(0,x,v,task_id=model.current_task_id)
        routed[v] = model.normalize_routed_task_features(x,y)
        ratios[v] = bank.last_raw_residual_ratio
    head = lambda x: F.linear(x[:,0], model.head.weight[list(classes)], model.head.bias[list(classes)])
    fused,_ = model.view_fusion_module(routed,[model.current_task_id],reliable)
    original,_ = model.view_fusion_module(feats,[model.current_task_id],reliable)
    logits, reference = head(fused), head(original)
    auxiliary=[]
    for v in routed:
        mask = reliable if v=='face' else torch.ones_like(reliable)
        if bool(mask.any()): auxiliary.append(F.binary_cross_entropy_with_logits(head(routed[v])[mask],labels[mask]))
    objective=(F.binary_cross_entropy_with_logits(logits,labels)+.1*torch.stack(auxiliary).mean())*len(classes)/sum(model.task_sizes)
    return objective, logits, reference, labels, ratios


def fit_phase(model, cache, classes, device, phase, seed, epochs=5, batch_size=64):
    check_partition(cache,'fit')
    if cache['task_id']!=model.current_task_id:raise ValueError('Wrong task cache')
    bank=model.parax_bank
    model.eval().requires_grad_(False)
    for p in model.parameters():p.grad=None
    bank.set_phase(phase)
    parameters=list(bank.active_parameters())
    before=base_hash(model)
    fixed={k:v.detach().clone() for k,v in bank.state_dict().items()
           if k not in {n for n,p in bank.named_parameters() if p.requires_grad}}
    if not parameters:raise ValueError('Cannot optimize an empty phase')
    optimizer=torch.optim.Adam(parameters,lr=.0004,weight_decay=0.)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    generator=torch.Generator().manual_seed(seed)
    rows=[]
    for epoch in range(epochs):
        loss_sum=0.;steps=0;grad_sq={};ratios={v:0. for v in cache['features']}
        for index in torch.randperm(len(cache['targets']),generator=generator).split(batch_size):
            loss,_,_,_,rr=batch_forward(model,cache,index,classes,device)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite fit loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            for name,p in bank.named_parameters():
                if p.grad is not None:
                    if not bool(torch.isfinite(p.grad).all()):raise FloatingPointError('Nonfinite fit gradient')
                    grad_sq[name]=grad_sq.get(name,0.)+float(p.grad.float().square().sum())
            torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            loss_sum+=float(loss.detach())*len(index);steps+=1
            for v in ratios:ratios[v]+=rr[v]*len(index)
        scheduler.step()
        rows.append({'epoch':epoch,'loss':loss_sum/len(cache['targets']),'updates':steps,
                     'gradient_rms':{n:math.sqrt(s/steps) for n,s in grad_sq.items()},
                     'residual_ratios':{v:r/len(cache['targets']) for v,r in ratios.items()}})
    if base_hash(model)!=before:raise RuntimeError('Original model changed')
    if any(not torch.equal(v,bank.state_dict()[k]) for k,v in fixed.items()):raise RuntimeError('A frozen phase tensor changed')
    if any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('parax_bank.')):
        raise RuntimeError('Original model received gradients')
    bank.set_phase('off')
    return {'phase':phase,'epochs':epochs,'samples':len(cache['targets']),
            'trainable_parameters':sum(p.numel() for p in parameters),'base_unchanged':True,
            'phase_frozen_tensors_unchanged':True,'history':rows,'seed':seed}


@torch.no_grad()
def cache_metrics(model,cache,classes,device):
    check_partition(cache,cache['module_split'])
    before=parameter_hash(sorted(model.state_dict().items()))
    actual=[];reference=[];labels=[];ratios={v:0. for v in cache['features']};loss_sum=0.
    model.eval()
    for index in torch.arange(len(cache['targets'])).split(64):
        loss,z,b,y,rr=batch_forward(model,cache,index,classes,device)
        actual.append(z.cpu());reference.append(b.cpu());labels.append(y.cpu());loss_sum+=float(loss)*len(index)
        for v in ratios:ratios[v]+=rr[v]*len(index)
    y=torch.cat(labels).numpy()
    def metric(rows):
        z=torch.cat(rows);p=z.sigmoid().numpy()
        return {'mAP':float(np.mean([100*average_precision(p[:,i],y[:,i]) for i in range(y.shape[1])])),
                'bce':float(F.binary_cross_entropy_with_logits(z,torch.from_numpy(y)))}
    if before!=parameter_hash(sorted(model.state_dict().items())):raise RuntimeError('Read-only cache evaluation mutated model')
    return {'partition':cache['module_split'],'samples':len(y),'baseline':metric(reference),'adapted':metric(actual),
            'objective':loss_sum/len(y),'residual_ratios':{v:r/len(y) for v,r in ratios.items()},
            'optimization_updates':0,'used_for_selection':False}
