"""Read-only TRAIN forward/gradient audit of the interrupted CLS-agent arm."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from multi_lane.track_a.crossvit_experiment import build,restore
from multi_lane.track_a.crossvit_three_view import compact_state
from multi_lane.track_a.export_compact_test_scores import _sha256
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.protected_joint_experiment import training_source
from multi_lane.track_a.run_historical_b32_test_attribution import read,write
from multi_lane.track_a.runner import (dataset_view,task_indices,tensor_state_digest,
    compute_training_loss,compute_asymmetric_training_loss,compute_view_objective_losses,set_seed)
from multi_lane.track_a.task_recovery import restore_rng


def stats(x):
    dtype=str(x.dtype)
    x=x.detach().float();finite=torch.isfinite(x)
    f=x[finite]
    return dict(shape=list(x.shape),dtype=dtype,finite=bool(finite.all()),
        nonfinite=int((~finite).sum()),max_abs=float(f.abs().max()) if f.numel() else None,
        rms=float(f.square().mean().sqrt()) if f.numel() else None)


@contextmanager
def precision(model,mode):
    originals=[]
    if mode=='amp_fp32_exchange':
        for blocks in model.exchange:
            for module in blocks.values():
                forward=module.forward;originals.append((module,forward))
                def wrapped(cls,memories,masks,forward=forward):
                    with torch.autocast(device_type='cuda',enabled=False):
                        return forward(cls.float(),[m.float() for m in memories],masks)
                module.forward=wrapped
    if mode in ('amp_fp32_task_attention','amp_fp32_task_block'):
        attribute='_prompt_attention' if mode.endswith('_attention') else '_lane_block'
        for branch in model.branches.values():
            forward=getattr(branch,attribute);originals.append((branch,attribute,forward))
            def wrapped(*args,forward=forward,**kwargs):
                with torch.autocast(device_type='cuda',enabled=False):
                    args=tuple(x.float() if torch.is_tensor(x) and x.is_floating_point() else x for x in args)
                    return forward(*args,**kwargs)
            setattr(branch,attribute,wrapped)
    try:yield
    finally:
        for entry in originals:
            if len(entry)==2:entry[0].forward=entry[1]
            else:setattr(entry[0],entry[1],entry[2])


def main(args):
    set_seed(1,True)
    cfg=read(args.training_root/'config.json')
    model=build(cfg['model_template'],Path(cfg['clip_checkpoint']),cfg['arm'])
    torch.set_num_threads(2)
    last=args.training_root/'epoch_recovery/task3/epoch14.pth'
    payload=torch.load(last,map_location='cpu',weights_only=False)
    source=training_source(cfg['model_template'],SimpleNamespace(
        data_root=Path(cfg['data_root']),face_manifest_root=Path(cfg['face_manifest_root'])))
    # Same restored RNG and DataLoader recipe as the next training epoch.
    restore_rng(payload['rng'])
    loader=DataLoader(dataset_view(source,task_indices(3),True),batch_size=64,shuffle=True,
        num_workers=2,prefetch_factor=1,pin_memory=True,drop_last=False)
    x,y,ids=next(iter(loader));x=move_model_inputs(x,torch.device('cuda'));y=y.cuda().float()
    checkpoints=[args.training_root/'epoch_recovery/task3'/f'epoch{e:02d}.pth' for e in args.epochs]
    hashes={str(p):_sha256(p) for p in checkpoints}
    rows=[]
    write(args.output/'inputs.json',dict(split='train',task=3,sample_ids=list(ids),
        pixels_sha256=tensor_state_digest(x),face_reliable=int(x['face_reliable'].sum()),
        input_stats={k:stats(v) for k,v in x.items() if torch.is_tensor(v) and v.is_floating_point()},
        source_rng='epoch14_next_training_iterator'))
    for path in checkpoints:
        saved=restore(model,path);model.train()
        initial=tensor_state_digest(compact_state(model))
        weights={n:stats(p) for n,p in model.named_parameters()
            if n.startswith('exchange.3.') or n.endswith(('head.weight','head.bias'))}
        bad_state=[n for n,p in model.state_dict().items() if not torch.isfinite(p).all()]
        for mode in args.modes:
            trace=[];hooks=[]
            def hook(name):
                def capture(module,inputs,output):
                    if torch.is_tensor(output):trace.append(dict(name=name,**stats(output)))
                return capture
            for name,module in model.named_modules():
                if (name.startswith('exchange.3.') and name.endswith(('.q','.k','.v','.out'))
                    or name.endswith(('.mlp.c_fc','.mlp.c_proj','.ln_post','.head'))):
                    hooks.append(module.register_forward_hook(hook(name)))
            model.exchange_enabled=not mode.endswith('_off')
            amp=mode.startswith('amp')
            torch.backends.cuda.matmul.allow_tf32=amp
            torch.backends.cudnn.allow_tf32=amp
            row=dict(checkpoint=str(path),mode=mode,bad_parameters=bad_state)
            torch.autograd.set_detect_anomaly(mode=='amp_anomaly')
            try:
                with precision(model,mode),torch.cuda.amp.autocast(enabled=amp):
                    z,views=model.current_all_logits_with_views(x)
                    row['logits']={k:stats(v) for k,v in dict(fused=z,**views).items()}
                    if bool(torch.isfinite(z).all()):
                        bce=compute_training_loss(z,y,task_indices(3),1.,'legacy_full_zero')
                        asl=compute_asymmetric_training_loss(z,y,task_indices(3),1.,'legacy_full_zero')
                        losses=compute_view_objective_losses(views,x,y,task_indices(3),1.,'legacy_full_zero','bce')
                        av=compute_view_objective_losses(views,x,y,task_indices(3),1.,'legacy_full_zero','asl')
                        bce=bce+.1*torch.stack(list(losses.values())).mean()
                        asl=asl+.1*torch.stack(list(av.values())).mean()
                        row.update(bce=float(bce),asl=float(asl))
                if 'bce' in row and path.stem in ('epoch13','epoch14'):
                    gradients=[]
                    for label,loss,parameters in (
                        ('model',bce,tuple(model.base_optimizer_parameters())),
                        ('adapter',asl,tuple(model.adapter_optimizer_parameters()))):
                        names={id(p):n for n,p in model.named_parameters()}
                        for scale in (1.,65536.):
                            gs=torch.autograd.grad(loss*scale,parameters,retain_graph=True,allow_unused=True)
                            bad=[names[id(p)] for p,g in zip(parameters,gs) if g is not None and not bool(torch.isfinite(g).all())]
                            gradients.append(dict(group=label,scale=scale,nonfinite_parameters=bad))
                    row['gradients']=gradients
                row['first_nonfinite']=next((v for v in trace if not v['finite']),None)
                row['exchange_trace']=[v for v in trace if v['name'].startswith('exchange.')]
                row['largest_activations']=sorted(trace,key=lambda v:v.get('max_abs') or 0,reverse=True)[:8]
            except BaseException as e:row['error']=repr(e)
            finally:
                for h in hooks:h.remove()
                torch.autograd.set_detect_anomaly(False)
            rows.append(row)
            write(args.output/'progress.json',dict(status='running',records=rows))
            print(path.stem,mode,row.get('logits',{}).get('fused'),row.get('first_nonfinite'),row.get('error'),flush=True)
        if initial!=tensor_state_digest(compact_state(model)):raise AssertionError('Audit changed parameters')
        write(args.output/(path.stem+'_weights.json'),weights)
    for p,sha in hashes.items():
        if _sha256(Path(p))!=sha:raise AssertionError('Source checkpoint changed')
    write(args.output/'result.json',dict(status='complete',split='train',records=rows,
        source_checkpoints_sha256=hashes,read_only=True,parameters_unchanged=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',nargs='+',type=int,default=[1,10,13,14])
    p.add_argument('--modes',nargs='+',choices=('amp','amp_off','amp_fp32_exchange','fp32','fp32_off',
        'amp_fp32_task_attention','amp_fp32_task_block','amp_anomaly'),
        default=['amp','amp_off','amp_fp32_exchange','fp32','fp32_off'])
    main(p.parse_args())
