"""Six predeclared receiver configurations from four independently supervised paths."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from multi_lane.continual_datasets.continual_datasets import EMOTIC
from multi_lane.track_a.r1_crossvit import CLSAgent, attach_evidence
from multi_lane.track_a.r1_crossvit_experiment import (
    flags, read, write, validation, model_from_config, single_thread_worker)
from multi_lane.track_a.r1_frozen_cross_view import MODES, FrozenCrossAgent
from multi_lane.track_a.evaluation_scores import load_evaluation_scores, align_evaluation_scores
from multi_lane.track_a.fuse_face_endpoint_validation import load_face_reliability
from multi_lane.track_a.fuse_fixed_three_view_test import LOCKED_RELIABLE_WEIGHTS, LOCKED_FALLBACK_WEIGHTS
from multi_lane.track_a.runner import (
    build_transforms, compute_metrics, dataset_view, seen_indices, summarize_tasks,
    git_metadata, set_seed, evaluate)

VIEWS = ('person', 'face')
DIRECTIONS = ('face_only', 'person_only', 'person_face')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder(root, mode, view, split='train'):
    return root/mode/'attention'/view/split


def input_paths():
    return SimpleNamespace(
        reference_root=Path('../emotic_benchmark_runs'),
        data_root=Path('../multi-lane-main/datasets/EMOTIC'),
        clip_checkpoint=Path('../CODE_DDP/pretrained/clip/ViT-B-16.pt'),
        face_manifest_root=Path('../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909'))


def score_suffix(split, task):
    return Path('val/scores' if split == 'val' else 'train/test_scores')/f'task{task}.npz'


def queue(jobs, logs, root, phase, concurrency=2):
    pending, running, finished, failures = list(jobs), [], [], []
    while pending or running:
        for job, process, handle in running[:]:
            code = process.poll()
            if code is None:
                continue
            handle.close();running.remove((job, process, handle))
            record = dict(name=job['name'], exit_code=code)
            finished.append(record)
            if code:
                failures.append(record)
        if pending and len(running) < concurrency:
            free = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            output = subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid,used_memory',
                '--format=csv,noheader,nounits'], text=True)
            usage = {}
            for line in output.splitlines():
                values = [x.strip() for x in line.split(',')]
                if len(values) == 2 and all(x.isdigit() for x in values):
                    usage[int(values[0])] = int(values[1])
            reserved = sum(max(0, j.get('memory_mib',9000)-usage.get(p.pid,0)) for j,p,_ in running)
            job = pending[0]
            if free-reserved >= job.get('memory_mib',9000)+1024:
                pending.pop(0)
                handle = (logs/(job['name']+'.log')).open('a')
                process = subprocess.Popen(job['command'], stdout=handle, stderr=subprocess.STDOUT)
                running.append((job, process, handle))
                print('START', phase, job['name'], process.pid, flush=True)
        write(root/('queue_'+phase+'.json'), dict(
            status='running' if pending or running else 'finished',
            pending=[j['name'] for j in pending],
            running=[dict(name=j['name'],pid=p.pid) for j,p,_ in running],finished=finished))
        if pending or running:
            time.sleep(5)
    if failures:
        write(root/('failures_'+phase+'.json'), dict(failures=failures, other_jobs_completed=True))
        raise RuntimeError(f'{phase} failed: {failures}')


def load_source(train, inputs, split):
    c = read(train/'config.json')
    _, transform = build_transforms(c['input_normalization'],c['train_crop_scale'],
        input_mode=c['input_mode'],person_transform_mode=c['person_transform_mode'],
        person_color_jitter_strength=c['person_color_jitter_strength'],
        person_color_jitter_probability=c['person_color_jitter_probability'])
    parent = inputs.data_root.parent
    source = EMOTIC(str(parent),train=split=='train',eval_splits=(split,),transform=transform,
        input_mode=c['input_mode'],person_crop_margin=c['person_crop_margin'],
        face_manifest_root=inputs.face_manifest_root if c['input_mode']=='face_crop' else None)
    return attach_evidence(source,c['r1_receiver'],parent,inputs.face_manifest_root)


def restore(model, train, task):
    saved = torch.load(train/'compact_checkpoints'/f'task{task}.pth',map_location='cpu',weights_only=False)
    missing = model.load_state_dict(saved['model'],strict=False)
    if missing.unexpected_keys or any(not n.startswith('visual_encoder.') for n in missing.missing_keys):
        raise ValueError('Incomplete checkpoint')
    model.restore_task(task)


def preflight(train, inputs, output):
    """Real-CLIP zero identity, old-path replay, masks, gradients and eight-lane shape."""
    set_seed(1,True)
    model = model_from_config(read(train/'config.json'),inputs.clip_checkpoint)
    torch.set_num_threads(2)
    source = load_source(train,inputs,'val')
    data = dataset_view(source,seen_indices(1),include_sample_id=True)
    images, _, ids = next(iter(DataLoader(Subset(data,range(min(64,len(data)))),batch_size=64,num_workers=0)))
    images = {k:v.cuda() for k,v in images.items()}
    restore(model,train,0)
    old_state = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()
                 if n.startswith(('r1_bank.0.', 'adapter_bank.task_adapters.0.'))}
    old_selectors=model.selectors[0].detach().cpu().clone()
    old_prompts=[p[:,0].detach().cpu().clone() for p in model.prompts]
    old_head=(model.head.weight[:5].detach().cpu().clone(),model.head.bias[:5].detach().cpu().clone())
    with torch.no_grad(),torch.autocast('cuda'):
        old = model.seen_logits(images).float()
    # Hold lane cardinality fixed: changing T also changes AMP GEMM shapes.
    model.restore_task(1)
    with torch.no_grad(),torch.autocast('cuda'):
        old_same_shape=model.seen_logits(images)[:,:5].float()
    restore(model,train,1)
    with torch.no_grad(),torch.autocast('cuda'):
        later = model.seen_logits(images)[:,:5].float()
    drift = float((old_same_shape-later).abs().max())
    lane_count_rounding=float((old-later).abs().max())
    immutable=(all(torch.equal(v,model.state_dict()[n].cpu()) for n,v in old_state.items())
        and torch.equal(old_selectors,model.selectors[0].cpu())
        and all(torch.equal(v,p[:,0].cpu()) for v,p in zip(old_prompts,model.prompts))
        and torch.equal(old_head[0],model.head.weight[:5].cpu())
        and torch.equal(old_head[1],model.head.bias[:5].cpu()))
    if not immutable or drift > 2e-5:
        raise AssertionError(f'Old task changed: immutable={immutable}, same-shape drift={drift}')
    agents = [m for m in model.r1_bank.modules() if isinstance(m, (FrozenCrossAgent, CLSAgent))]
    saved_out = [m.out.weight.detach().clone() for m in agents]
    with torch.no_grad():
        for module in agents:
            module.out.weight.zero_()
    with torch.no_grad(),torch.autocast('cuda'):
        zero = model.seen_logits(images).float()
        model.r1_enabled=False
        native = model.seen_logits(images).float()
        model.r1_enabled=True
    identity_error = float((zero-native).abs().max())
    tf32=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            exact_zero=model.seen_logits(images).float()
            model.r1_enabled=False
            exact_native=model.seen_logits(images).float()
            model.r1_enabled=True
        fp32_identity_error=float((exact_zero-exact_native).abs().max())
    finally:
        torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=tf32
    if fp32_identity_error > 2e-5:
        raise AssertionError(f'FP32 zero identity failed: {fp32_identity_error}')
    with torch.no_grad():
        for module,value in zip(agents,saved_out):
            module.out.weight.copy_(value)
    if model.r1_view == 'person':
        images['evidence_face_reliable'].zero_()
        with torch.no_grad(),torch.autocast('cuda'):
            a = model.seen_logits(images)
            bad = dict(images,evidence_face=torch.randn_like(images['evidence_face'])*10)
            b = model.seen_logits(bad)
        if not torch.equal(a,b):
            raise AssertionError('Invalid Face evidence influenced Person')
    for module in agents:
        module.diagnostic_enabled=True
    with torch.no_grad(),torch.autocast('cuda'):
        probe_logits = model.seen_logits(images)
    diagnostics = {str(t): model.cross_agent(t, model.frozen_cross_layers[-1]).last_diagnostic
                   for t in range(2)}
    if model.task_pool_layer is not None:
        for t in range(2):
            diagnostics[str(t)]['pool'] = model.r1_bank[t][1].last_diagnostic
    # Gradient check uses trained smoke state, so Q/K/V must all participate.
    model.zero_grad(set_to_none=True)
    for k in images:
        if k.startswith('evidence_') and k!='evidence_face_reliable':
            images[k].requires_grad_(True)
    with torch.autocast('cuda'):
        loss = model.current_all_logits(images).float().square().mean()
    loss.backward()
    gradients = {n:None if p.grad is None else float(p.grad.float().norm())
                 for n,p in model.r1_bank[1].named_parameters()}
    if any(v is None or not np.isfinite(v) or v <= 0 for v in gradients.values()):
        raise AssertionError(f'Broken new-module gradients: {gradients}')
    if any(p.grad is not None for p in model.visual_encoder.parameters()):
        raise AssertionError('Visual weights received gradients')
    if any(v.grad is not None for k,v in images.items() if k.startswith('evidence_')):
        raise AssertionError('Sender input received gradients')
    model.zero_grad(set_to_none=True);model.restore_task(7);model.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(),torch.autocast('cuda'):
        eight = model.seen_logits(images)
    if eight.shape!=(len(ids),26) or not torch.isfinite(eight).all():
        raise AssertionError('Eight-lane inference failed')
    history = read(train/'training_history.json')
    peak = max(row.get('cuda_peak_reserved_mib',0) for rows in history.values() for row in rows)
    peak = max(peak,torch.cuda.max_memory_reserved()/1024**2)
    write(output,dict(status='passed',old_task_same_shape_max_drift=drift,old_parameters_bitwise_unchanged=immutable,
        changed_lane_count_amp_rounding=lane_count_rounding,
        zero_output_amp_max_difference=identity_error,zero_output_fp32_max_difference=fp32_identity_error,
        sender_and_visual_gradients_absent=True,gradient_norms=gradients,diagnostics=diagnostics,
        eight_lane_shape=list(eight.shape),smoke_train_peak_reserved_mib=peak,
        admission_memory_mib=int(peak*1.2+1024)))


def check_training(train, reference, view, tasks, smoke):
    summary = read(train/'seed_summary.json')
    if summary['status']!='complete':
        raise ValueError('Training did not complete')
    history = read(train/'training_history.json')
    archived = read(reference/'native'/view/'train/training_history.json')
    counts = []
    for task in range(tasks):
        rows = history[str(task)]
        if any(r['skipped_optimizer_steps'] for r in rows):
            raise ValueError('Skipped optimizer step')
        expected = 2 if smoke else sum(r['optimizer_steps'] for r in archived[str(task)])
        actual = sum(r['optimizer_steps'] for r in rows)
        if actual!=expected or (not smoke and len(rows)!=30):
            raise ValueError(f'Budget mismatch task{task}: {actual} != {expected}')
        if not (train/'compact_checkpoints'/f'task{task}.pth').exists():
            raise ValueError('Missing checkpoint')
        counts.append(actual)
    return dict(updates=counts,total=sum(counts),skips=0)


def off_evaluation(train, inputs, output):
    set_seed(1,True)
    model = model_from_config(read(train/'config.json'),inputs.clip_checkpoint)
    torch.set_num_threads(2)
    rows = {}
    for split in ('val','test'):
        source = load_source(train,inputs,split)
        rows[split] = []
        for task in (0,2,7):
            restore(model,train,task);model.r1_enabled=False;model.requires_grad_(False)
            data = dataset_view(source,seen_indices(task),include_sample_id=True)
            loader=DataLoader(data,batch_size=64,num_workers=2,prefetch_factor=1,
                worker_init_fn=single_thread_worker,generator=torch.Generator().manual_seed(77191+task))
            row=evaluate(model,loader,torch.device('cuda'),task,.5,True,
                score_output_path=output/split/f'task{task}.npz')
            rows[split].append(asdict(row))
            print('OFF',train,split,task,row.mAP,flush=True)
    write(output/'completion.json',dict(status='complete',metrics=rows,checkpoints_unmodified=True))


def train_probe(train, inputs, output):
    set_seed(1,True)
    model=model_from_config(read(train/'config.json'),inputs.clip_checkpoint)
    torch.set_num_threads(2)
    source=load_source(train,inputs,'train')
    rows=[]
    for task in (0,2,7):
        restore(model,train,task);model.requires_grad_(False).eval()
        for module in model.r1_bank:module.diagnostic_enabled=True
        data=dataset_view(source,seen_indices(task),include_sample_id=True)
        images,_,ids=next(iter(DataLoader(Subset(data,range(min(64,len(data)))),batch_size=64,num_workers=0)))
        images={k:v.cuda() for k,v in images.items()}
        with torch.no_grad(),torch.autocast('cuda'):
            logits=model.seen_logits(images)
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite TRAIN probe')
        rows.append(dict(task=task,sample_ids=list(ids),split='train',
            modules={str(t):model.r1_bank[t].last_diagnostic for t in range(task+1)}))
    write(output,dict(status='complete',diagnostic_only=True,rows=rows))


def aggregate(root,reference,manifest,off=False):
    results={}
    tasks=(0,2,7) if off else range(8)
    groups=['native','face_pool']+[f'{mode}_{d}' for mode in MODES for d in DIRECTIONS]
    for split in ('val','test'):
        reliability=load_face_reliability(manifest,split)
        rows={name:{pop:[] for pop in ('all','reliable','unreliable')} for name in groups}
        branch_rows={f'{mode}_{view}':[] for mode in MODES for view in VIEWS}
        for task in tasks:
            full=load_evaluation_scores(reference/'native/full'/score_suffix(split,task),64)
            person=load_evaluation_scores(reference/'native/person'/score_suffix(split,task),64)
            face=load_evaluation_scores(reference/'native/face'/score_suffix(split,task),64)
            ids,f,p,y,_,_=align_evaluation_scores(full,person)
            _,_,r,_,_,_=align_evaluation_scores(full,face)
            reliable=np.array([reliability[i] for i in ids],dtype=bool)
            weights=np.where(reliable[:,None],LOCKED_RELIABLE_WEIGHTS,LOCKED_FALLBACK_WEIGHTS)
            arrays={}
            for mode in MODES:
                for view in VIEWS:
                    path=(root/'off'/mode/view/split/f'task{task}.npz' if off else
                          root/mode/'attention'/view/score_suffix(split,task))
                    score=load_evaluation_scores(path,64)
                    _,_,z,_,_,_=align_evaluation_scores(full,score)
                    arrays[(mode,view)]=z
                    mask=reliable if view=='face' else np.ones(len(ids),dtype=bool)
                    branch_rows[f'{mode}_{view}'].append(compute_metrics(task,
                        torch.from_numpy(z[mask]).sigmoid(),torch.from_numpy(y[mask]),.5))
            baseline=weights[:,0,None]*f+weights[:,1,None]*p+weights[:,2,None]*r
            for name in groups:
                pp,rr=p,r
                if name=='face_pool':
                    pooled=load_evaluation_scores(reference/'pool/face'/score_suffix(split,task),64)
                    _,_,rr,_,_,_=align_evaluation_scores(full,pooled)
                elif name!='native':
                    mode,direction=next((m,name[len(m)+1:]) for m in MODES if name.startswith(m+'_'))
                    if direction in ('person_only','person_face'):pp=arrays[(mode,'person')]
                    if direction in ('face_only','person_face'):rr=arrays[(mode,'face')]
                logits=weights[:,0,None]*f+weights[:,1,None]*pp+weights[:,2,None]*rr
                if name.endswith('face_only') and not np.array_equal(logits[~reliable],baseline[~reliable]):
                    raise AssertionError('Face-only changed unreliable-person fusion')
                for pop,mask in [('all',np.ones(len(ids),dtype=bool)),('reliable',reliable),('unreliable',~reliable)]:
                    rows[name][pop].append(compute_metrics(task,torch.from_numpy(logits[mask]).sigmoid(),torch.from_numpy(y[mask]),.5))
            print('FUSION',split,task,{n:rows[n]['all'][-1].mAP for n in groups},flush=True)
        results[split]={name:{pop:dict(summary=summarize_tasks(values) if not off else None,
            metrics=[asdict(x) for x in values]) for pop,values in populations.items()} for name,populations in rows.items()}
        results[split]['branch_metrics']={name:[asdict(x) for x in values] for name,values in branch_rows.items()}
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','preflight','off','probe','verify-smoke'),nargs='?',default='run')
    parser.add_argument('--root',type=Path,default=Path('output/r1_frozen_cross_view/seed1_20261105'))
    parser.add_argument('--reference',type=Path,default=Path('../multi-lane-main-r1-crossvit-20261101/output/r1_crossvit/seed1_20261101_02'))
    parser.add_argument('--training-root',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();inputs=input_paths()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('GPU0 only')
    if args.action=='preflight':return preflight(args.training_root,inputs,args.output)
    if args.action=='off':return off_evaluation(args.training_root,inputs,args.output)
    if args.action=='probe':return train_probe(args.training_root,inputs,args.output)
    if args.action=='verify-smoke':
        # Reuse completed tiny training only when every training module is unchanged.
        old=read(args.root/'protocol.json')['git']['commit']
        changed=subprocess.check_output(['git','diff','--name-only',old,'HEAD','--','multi_lane'],text=True).strip()
        if changed:raise ValueError('Training code changed; a fresh smoke run is required')
        for m in MODES:
            for v in VIEWS:check_training(folder(args.root,m,v),args.reference,v,2,True)
        logs=Path('logs/r1_frozen_cross_view')/args.root.name
        jobs=[dict(name=f'{m}_{v}_preflight',command=[sys.executable,'-u',__file__,'preflight',
            '--training-root',str(folder(args.root,m,v)),'--output',str(args.root/'preflight'/f'{m}_{v}.json')],memory_mib=13000)
            for m in MODES for v in VIEWS]
        queue(jobs,logs,args.root,'preflight',1)
        original=read(args.root/'protocol.json')
        if any(sha(p)!=h for p,h in original['source_hashes'].items()):raise AssertionError('Reference changed')
        write(args.root/'completion.json',dict(status='complete',smoke=True,
            training_source=old,verification_source=git_metadata(Path(__file__).resolve().parents[2]),
            training_code_unchanged=True,sources_unchanged=True))
        return
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    logs=Path('logs/r1_frozen_cross_view')/root.name;logs.mkdir(parents=True,exist_ok=True)
    git=git_metadata(Path(__file__).resolve().parents[2])
    if git['dirty']:raise ValueError('Clean committed source required')
    sources=[args.reference/family/view/score_suffix(split,t) for family,view in
        [('native','full'),('native','person'),('native','face'),('pool','face')]
        for split in ('val','test') for t in range(8)]
    hashes={str(p):sha(p) for p in sources}
    protocol=dict(git=git,seed=1,tasks=2 if args.smoke else 8,epochs=30,batch_size=64,
        modes=list(MODES),directions=list(DIRECTIONS),trained_receivers=list(VIEWS),
        unique_trained_paths=4,predeclared_fusion_configurations=6,
        exchange='after complete frozen block6',task_private=True,dim=64,heads=4,
        qk_normalization='no-affine LN after projection per 16-d head',
        cap=.05,cap_reference='own pre-injection image CLS; total per-image Frobenius norm',
        own_view_supervision='BCE; Adapter ASL only',main_lr=.0125,adapter_lr=.0004,interaction_lr=.0004,
        source_memory='own image CLS + two unmodified frozen patch sets; invalid Face masked before softmax',
        fusion='fixed R1 logits then sigmoid',baselines_retrained=False,
        reference=str(args.reference),source_hashes=hashes,smoke=args.smoke)
    if (root/'protocol.json').exists() and read(root/'protocol.json')!=protocol:
        raise ValueError('Protocol changed; refuse overwrite')
    write(root/'protocol.json',protocol)
    jobs=[]
    for mode in MODES:
        for view in VIEWS:
            train=folder(root,mode,view)
            if (train/'seed_summary.json').exists() and read(train/'seed_summary.json')['status']=='complete':continue
            memory=10000
            if not args.smoke:
                memory=read(root.with_name(root.name+'_smoke')/'preflight'/f'{mode}_{view}.json')['admission_memory_mib']
            command=[sys.executable,'-u','-m','multi_lane.track_a.runner',
                *flags(SimpleNamespace(output=root/mode,**vars(inputs)),'attention',view,args.smoke),
                '--r1-frozen-mode',mode,'--r1-fusion-readout','logit']
            if args.smoke:command+=['--r1-smoke-eval-samples','64']
            jobs.append(dict(name=f'{mode}_{view}_train',command=command,memory_mib=memory))
    queue(jobs,logs,root,'train',1 if args.smoke else 2)
    counts={f'{m}_{v}':check_training(folder(root,m,v),args.reference,v,protocol['tasks'],args.smoke)
            for m in MODES for v in VIEWS}
    write(root/'training_checks.json',counts)
    if args.smoke:
        jobs=[dict(name=f'{m}_{v}_preflight',command=[sys.executable,'-u',__file__,'preflight',
            '--training-root',str(folder(root,m,v)),'--output',str(root/'preflight'/f'{m}_{v}.json')],memory_mib=13000)
              for m in MODES for v in VIEWS]
        queue(jobs,logs,root,'preflight',1)
    else:
        jobs=[]
        for m in MODES:
            for v in VIEWS:
                val=folder(root,m,v,'val')
                if (val/'result.json').exists() and read(val/'result.json')['status']=='complete':continue
                command=[sys.executable,'-u','-m','multi_lane.track_a.r1_crossvit_experiment','val',
                    '--training-root',str(folder(root,m,v)),'--output',str(val),
                    '--data-root',str(inputs.data_root),'--clip-checkpoint',str(inputs.clip_checkpoint),
                    '--face-manifest-root',str(inputs.face_manifest_root)]
                jobs.append(dict(name=f'{m}_{v}_val',command=command,memory_mib=8000))
        queue(jobs,logs,root,'val',2)
        write(root/'six_group_results.json',dict(status='complete',results=aggregate(root,args.reference,inputs.face_manifest_root)))
        jobs=[dict(name=f'{m}_{v}_probe',command=[sys.executable,'-u',__file__,'probe',
            '--training-root',str(folder(root,m,v)),'--output',str(root/'train_probes'/f'{m}_{v}.json')],memory_mib=8000)
            for m in MODES for v in VIEWS if not (root/'train_probes'/f'{m}_{v}.json').exists()]
        queue(jobs,logs,root,'probe',2)
        jobs=[dict(name=f'{m}_{v}_off',command=[sys.executable,'-u',__file__,'off',
            '--training-root',str(folder(root,m,v)),'--output',str(root/'off'/m/v)],memory_mib=8000)
            for m in MODES for v in VIEWS if not (root/'off'/m/v/'completion.json').exists()]
        queue(jobs,logs,root,'off',2)
        write(root/'off_results.json',dict(status='complete',tasks=[0,2,7],
            not_a_retrained_baseline=True,results=aggregate(root,args.reference,inputs.face_manifest_root,off=True)))
    if any(sha(p)!=h for p,h in hashes.items()):raise AssertionError('Reference predictions changed')
    write(root/'completion.json',dict(status='complete',smoke=args.smoke,training_checks=counts,
        sources_unchanged=True,protocol=protocol))


if __name__=='__main__':main()
