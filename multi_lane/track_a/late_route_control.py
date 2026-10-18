"""TRAIN-calibrated fixed Top-2 versus dynamic/off Late MoE, validation only."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .evaluate_joint26_checkpoint import make_source
from .expert_rollback import build
from .export_compact_test_scores import _sha256, _torch_load, restore_compact_model_state
from .gradient_attribution import grouped_batches
from .openai_clip_loader import load_openai_clip_visual
from .paired_transforms import move_model_inputs
from .runner import (LabelView, seen_indices, _intersects, evaluate, evaluate_view_diagnostics,
                     compact_model_state_dict, tensor_state_digest, git_metadata, set_seed)


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def choose_training_batches(source, task, count, seed):
    eligible=[i for i,y in enumerate(source.targets) if _intersects(y,seen_indices(task))]
    groups=grouped_batches([source.sample_ids[i] for i in eligible],64,count,seed)
    return [[eligible[j] for j in batch] for batch in groups]


class GateCalibration:
    def __init__(self, bank):
        self.bank=bank
        self.lookup={id(pool):(t,l) for t,layers in enumerate(bank.tasks) for l,pool in enumerate(layers)}
        self.mass=torch.zeros(bank._current_task_id+1,len(bank.layer_indices),3,bank.num_experts,dtype=torch.float64)
        self.samples=torch.zeros(self.mass.shape[:-1],dtype=torch.long)

    @contextmanager
    def capture(self):
        bank=self.bank;original=bank._route;existed='_route' in bank.__dict__
        def route(cls,pool,view):
            gates=original(cls,pool,view)
            task,layer=self.lookup[id(pool)]
            mask=torch.ones(len(cls),device=cls.device,dtype=torch.bool)
            if view==2:
                if bank._face_reliable is None or bank._face_reliable.shape!=mask.shape:
                    raise RuntimeError('Reliable Face mask required during calibration')
                mask=bank._face_reliable.to(cls.device)
            self.mass[task,layer,view]+=gates.detach()[mask].double().sum(0).cpu()
            self.samples[task,layer,view]+=int(mask.sum())
            return gates
        bank._route=route
        try:yield self
        finally:
            if existed:bank._route=original
            else:del bank._route

    def table(self):
        if bool((self.samples<=0).any()):raise ValueError('Each task/view requires valid TRAIN calibration samples')
        if not torch.allclose(self.mass.sum(-1),self.samples.double(),atol=1e-5,rtol=1e-6):
            raise AssertionError('Calibration gate mass does not conserve samples')
        fixed=torch.zeros_like(self.mass,dtype=torch.float32);rows={}
        for task in range(len(self.mass)):
            rows[str(task)]={}
            for layer,block in enumerate(self.bank.layer_indices):
                rows[str(task)][str(block)]={}
                for view,name in enumerate(self.bank.view_names):
                    mass=self.mass[task,layer,view]
                    selected=sorted(range(self.bank.num_experts),key=lambda i:(-float(mass[i]),i))[:self.bank.top_k]
                    if not bool((mass[selected]>0).all()):raise ValueError('Fixed Top-k needs positive mass for every selected expert')
                    weights=(mass[selected]/mass[selected].sum()).float()
                    fixed[task,layer,view,selected]=weights
                    rows[str(task)][str(block)][name]=dict(samples=int(self.samples[task,layer,view]),
                        expert_gate_mass=mass.tolist(),selected=selected,weights=weights.tolist())
        return fixed,rows


@contextmanager
def fixed_routing(bank, table):
    original=bank._route;existed='_route' in bank.__dict__
    lookup={id(pool):(t,l) for t,layers in enumerate(bank.tasks) for l,pool in enumerate(layers)}
    table=table.to(device=next(bank.parameters()).device,dtype=torch.float32)
    def route(cls,pool,view):
        task,layer=lookup[id(pool)]
        return table[task,layer,view].expand(len(cls),-1)
    bank._route=route
    try:yield
    finally:
        if existed:bank._route=original
        else:del bank._route


def ap(p,y):
    v=y[p.argsort()[::-1]]==1
    return float((np.cumsum(v)*v/np.arange(1,len(v)+1)).sum()/(v.sum()+1e-8)*100)


def paired_bootstrap(paths, replicates, seed):
    arrays={n:dict(np.load(p)) for n,p in paths.items()}
    a=arrays['dynamic'];groups=np.unique([str(s).rsplit('#person=',1)[0] for s in a['sample_ids']],return_inverse=True)[1]
    count=int(groups.max())+1;cache={}
    for name,z in arrays.items():
        for key in ('sample_ids','targets','class_indices'):
            if not np.array_equal(a[key],z[key]):raise AssertionError('Unpaired intervention scores')
        cache[name]=[]
        for c in range(z['targets'].shape[1]):
            order=z['probabilities'][:,c].argsort()[::-1]
            cache[name].append((groups[order],z['targets'][order,c]==1))
    rng=np.random.default_rng(seed);draws={n:[] for n in cache}
    for _ in range(replicates):
        w=np.bincount(rng.integers(0,count,count),minlength=count)
        for name,cols in cache.items():
            aps=[]
            for g,y in cols:
                labels=np.repeat(y,w[g]);aps.append(float((np.cumsum(labels)*labels/np.arange(1,len(labels)+1)).sum()/(labels.sum()+1e-8)*100))
            draws[name].append(float(np.mean(aps)))
    return {name:dict(CI95=np.quantile(np.array(draws['dynamic'])-np.array(draws[name]),[.025,.975]).tolist(),
                     point=float(np.mean([ap(a['probabilities'][:,c],a['targets'][:,c]) for c in range(a['targets'].shape[1])]))-
                           float(np.mean([ap(arrays[name]['probabilities'][:,c],arrays[name]['targets'][:,c]) for c in range(a['targets'].shape[1])])) )
            for name in ('fixed','off')}


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    config=json.loads((args.source/'train/config.json').read_text())
    summary=json.loads((args.source/'train/seed_summary.json').read_text())
    verification=json.loads((args.source/'verification.json').read_text())
    assert summary['status']=='complete' and summary['config']==config
    assert verification['status']=='verified' and len(verification['checks'])==8
    assert config['adapter_mode']=='moe_task_ffn_activate_freeze' and config['adapter_layer_indices']==[11]
    assert config['moe_top_k']==2 and config['moe_num_experts']==22 and config['moe_freeze_count']==2
    assert config['seed']==0 and not config['git']['dirty']
    metadata=git_metadata(Path(__file__).resolve().parents[2]);assert not metadata['dirty']
    assert sorted(set(args.tasks))==args.tasks and all(t in (0,2,7) for t in args.tasks)
    assert args.calibration_batches>0 and args.bootstrap>0
    if args.validation_limit is not None:assert args.validation_limit>0 and args.validation_limit%64==0
    set_seed(0,tf32=True);device=torch.device('cuda')
    model=build(config,load_openai_clip_visual(args.clip_checkpoint)).to(device)
    model.requires_grad_(False).eval();model.assert_visual_frozen()
    clip_before=tensor_state_digest(model.visual_encoder.state_dict())
    source_audit=json.loads((args.source/"train/paired_protocol_audit.json").read_text())
    assert source_audit["frozen_visual_unchanged"] and clip_before==source_audit["frozen_visual_sha256_after"]
    assert verification["source_git"]==config["git"]
    train=make_source(config,args.data_root,args.face_manifest_root,'train')
    validate=None;results={};source_hashes={};all_pixel_hashes={}
    args.output.mkdir(parents=True)
    save(args.output/'protocol.json',dict(source=str(args.source.resolve()),source_git=config['git'],evaluation_git=metadata,
        tasks=args.tasks,calibration_split='train',calibration_batches=args.calibration_batches,
        calibration_seed=20261018,selection='per_task_view_cpu_float64_gate_mass_top2_normalized',
        reliable_face_only=True,validation_limit=args.validation_limit,workers=0,batch_size=64,
        residual_scale=config['adapter_residual_scale'],bootstrap_replicates=args.bootstrap,test_accessed=False,optimizer_updates=0))
    # Lock every TRAIN-only route table before loading validation images.
    fixed_tables={}
    for task in args.tasks:
        path=args.source/'train/compact_checkpoints'/f'task{task}.pth'
        source_hashes[str(task)]=_sha256(path)
        assert source_hashes[str(task)]==verification['checks'][task]['checkpoint_sha256']
        restore_compact_model_state(model,_torch_load(path),task,config['git']);model.requires_grad_(False).eval()
        before=tensor_state_digest(compact_model_state_dict(model))
        batches=choose_training_batches(train,task,args.calibration_batches,20261018+task)
        capture=GateCalibration(model.adapter_bank);pixels=[]
        with torch.no_grad(),capture.capture():
            for indices in batches:
                loader=DataLoader(LabelView(train,indices,seen_indices(task),include_sample_id=True),batch_size=len(indices),num_workers=0,shuffle=False)
                images,_,ids=next(iter(loader));pixels.append(dict(ids=list(ids),pixel_sha256=tensor_state_digest(images)))
                with torch.cuda.amp.autocast(enabled=True):model.seen_logits(move_model_inputs(images,device))
        table,rows=capture.table();fixed_tables[task]=table
        assert tensor_state_digest(compact_model_state_dict(model))==before
        assert all(p.grad is None for p in model.parameters())
        all_pixel_hashes[str(task)]=pixels
        save(args.output/f'calibration/task{task}.json',dict(source_checkpoint_sha256=source_hashes[str(task)],
            batches=pixels,routes=rows,calibration_split='train',noise=False,dropout=False))
    save(args.output/'calibration/locked_tables.json',dict(source_checkpoint_sha256=source_hashes,locked_before_validation=True,train_batches=all_pixel_hashes))
    validate=make_source(config,args.data_root,args.face_manifest_root,'val')
    for task in args.tasks:
        restore_compact_model_state(model,_torch_load(args.source/'train/compact_checkpoints'/f'task{task}.pth'),task,config['git'])
        model.requires_grad_(False).eval();before=tensor_state_digest(compact_model_state_dict(model));task_results={}
        limit=args.validation_limit or len(validate);assert limit<=len(validate)
        loader=DataLoader(LabelView(validate,list(range(limit)),seen_indices(task),include_sample_id=True),batch_size=64,num_workers=0,shuffle=False)
        for mode in ('dynamic','fixed','off'):
            folder=args.output/f'task{task}'/mode;folder.mkdir(parents=True)
            model.set_adapter_runtime_enabled(mode!='off');model.adapter_bank.reset_route_statistics()
            with fixed_routing(model.adapter_bank,fixed_tables[task]) if mode=='fixed' else __import__('contextlib').nullcontext():
                row=evaluate(model,loader,device,task,config['threshold'],True,score_output_path=folder/'scores.npz')
                routes=model.adapter_bank.route_statistics()
                views=evaluate_view_diagnostics(model,loader,device,task,config['threshold'],True,score_output_path=folder/'view_scores.npz')
            assert tensor_state_digest(compact_model_state_dict(model))==before
            assert all(p.grad is None for p in model.parameters())
            if mode=='dynamic':
                for kind,new_name in (('scores','scores.npz'),('view_scores','view_scores.npz')):
                    with np.load(args.source/f'full_val/{kind}/task{task}.npz') as original,np.load(folder/new_name) as replay:
                        for key in ('sample_ids','targets','logits','probabilities') if kind=='scores' else ('sample_ids','targets','face_reliable','fused_logits','full_logits','person_logits','face_logits'):
                            assert np.array_equal(original[key][:limit],replay[key]),(task,kind,key)
            task_results[mode]=dict(metric=asdict(row),view_diagnostics=views,route_statistics=routes,model_state_unchanged=True)
            save(folder/'result.json',task_results[mode]);print(f'task={task} mode={mode} val_mAP={row.mAP:.6f}',flush=True)
        model.set_adapter_runtime_enabled(True)
        intervals=paired_bootstrap({n:args.output/f'task{task}'/n/'scores.npz' for n in task_results},args.bootstrap,20261018+task)
        results[str(task)]=dict(conditions=task_results,dynamic_minus=intervals)
        save(args.output/'progress.json',results)
        assert _sha256(args.source/'train/compact_checkpoints'/f'task{task}.pth')==source_hashes[str(task)]
    assert tensor_state_digest(model.visual_encoder.state_dict())==clip_before
    save(args.output/'result.json',dict(status='complete',source_git=config['git'],evaluation_git=metadata,
        source_checkpoint_sha256=source_hashes,clip_unchanged=True,source_checkpoint_unchanged=True,
        optimizer_updates=0,test_accessed=False,validation_samples=args.validation_limit or len(validate),tasks=results))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','output','data-root','clip-checkpoint','face-manifest-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tasks',type=int,nargs='+',default=[0,2,7]);p.add_argument('--calibration-batches',type=int,default=4)
    p.add_argument('--validation-limit',type=int,default=None);p.add_argument('--bootstrap',type=int,default=500)
    run(p.parse_args())

if __name__=='__main__':main()
