"""Three Face query ablations; reuse the locked R1/Pool references without training them."""
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
from multi_lane.track_a.r1_crossvit import attach_evidence
from multi_lane.track_a.r1_crossvit_experiment import flags, read, write, validation, model_from_config
from multi_lane.track_a.evaluation_scores import load_evaluation_scores, align_evaluation_scores
from multi_lane.track_a.fuse_face_endpoint_validation import load_face_reliability
from multi_lane.track_a.fuse_fixed_three_view_test import LOCKED_RELIABLE_WEIGHTS, LOCKED_FALLBACK_WEIGHTS
from multi_lane.track_a.runner import build_transforms, compute_metrics, dataset_view, seen_indices, summarize_tasks, git_metadata, set_seed

MODES = ('cls', 'selector_mean', 'selectors')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_path(root, mode):
    return root / mode / 'attention' / 'face' / 'train'


def val_path(root, mode):
    return root / mode / 'attention' / 'face' / 'val'


def queue(jobs, logs, root):
    pending, running, failures, states = list(jobs), [], [], []
    while pending or running:
        for job, process, handle in running[:]:
            code = process.poll()
            if code is None:
                continue
            handle.close();running.remove((job, process, handle))
            state = dict(name=job['name'], action=job['action'], exit_code=code)
            states.append(state)
            if code:
                failures.append(state)
        while pending and len(running) < 3:
            job = pending[0]
            if job['action'] == 'val' and sum(x[0]['action'] == 'val' for x in running) >= 2:
                break
            free = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True).strip())
            if free < 5000:
                break
            pending.pop(0)
            handle = (logs / (job['name'] + '.log')).open('a')
            process = subprocess.Popen(job['command'], stdout=handle, stderr=subprocess.STDOUT)
            running.append((job, process, handle))
            print('START', job['name'], process.pid, flush=True)
        write(root / 'queue.json', dict(status='running' if pending or running else 'finished',
              pending=[j['name'] for j in pending], running=[dict(name=j['name'], pid=p.pid) for j,p,_ in running], finished=states))
        if pending or running:
            time.sleep(5)
    if failures:
        write(root / 'failure.json', dict(failures=failures, other_jobs_completed=True))
        raise RuntimeError(f'Failed jobs: {failures}')


def check_training(folder, tasks, smoke):
    summary = read(folder / 'seed_summary.json')
    if summary['status'] != 'complete':
        raise ValueError('Incomplete Face training')
    counts = []
    histories = read(folder / 'training_history.json')
    for t in range(tasks):
        if not (folder / 'compact_checkpoints' / f'task{t}.pth').exists():
            raise FileNotFoundError(f'Missing checkpoint for task{t}')
        history = histories[str(t)]
        if any(row['skipped_optimizer_steps'] for row in history):
            raise FloatingPointError('Training contained skipped updates')
        if not smoke and len(history) != 30:
            raise ValueError('Incorrect training budget')
        counts.append(dict(task=t, epochs=len(history), updates=sum(row['optimizer_steps'] for row in history)))
    if smoke:
        saved = torch.load(sorted((folder / 'epoch_recovery/task1').glob('epoch*.pth'))[-1], map_location='cpu', weights_only=False)
        state = saved['training']['optimizer']
        names = saved['training']['optimizer_parameter_names']
        moments = {}
        for group, group_names in zip(state['param_groups'], names):
            for key, name in zip(group['params'], group_names):
                if name.startswith('r1_bank.1.'):
                    moments[name] = float(state['state'][key]['exp_avg'].norm())
        for name in ('q.weight','k.weight','v.weight','out.weight'):
            if not moments.get('r1_bank.1.' + name, 0) > 0:
                raise AssertionError(f'Real TRAIN optimizer has no nonzero moment for {name}')
    return counts


def probe(folder, inputs, stages):
    config = read(folder / 'config.json')
    set_seed(1, True)
    model = model_from_config(config, inputs.clip_checkpoint).eval().requires_grad_(False)
    torch.set_num_threads(2)
    _, transform = build_transforms(config['input_normalization'], config['train_crop_scale'], input_mode=config['input_mode'],
        person_transform_mode=config['person_transform_mode'], person_color_jitter_strength=config['person_color_jitter_strength'],
        person_color_jitter_probability=config['person_color_jitter_probability'])
    source = EMOTIC(str(inputs.data_root.parent), train=False, eval_splits=('val',), transform=transform,
        input_mode='face_crop', person_crop_margin=config['person_crop_margin'], face_manifest_root=inputs.face_manifest_root)
    source = attach_evidence(source, 'face', inputs.data_root.parent, inputs.face_manifest_root)
    records = []
    for stage in stages:
        saved = torch.load(folder / 'compact_checkpoints' / f'task{stage}.pth', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=False);model.restore_task(stage);model.requires_grad_(False)
        for module in model.r1_bank:
            module.diagnostic_enabled = True
        data = dataset_view(source, seen_indices(stage), include_sample_id=True)
        images, _, ids = next(iter(DataLoader(Subset(data, range(min(64, len(data)))), batch_size=64, num_workers=0)))
        images = {key:value.cuda() for key,value in images.items()}
        with torch.no_grad(), torch.autocast('cuda'):
            logits = model.seen_logits(images)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite probe logits')
        records.append(dict(stage=stage, sample_ids=list(ids), modules={str(t):model.r1_bank[t].last_diagnostic for t in range(stage+1)}))
    return records


def aggregate(root, reference, manifest):
    results = {}
    exact_cls = True
    for split in ('val', 'test'):
        reliability = load_face_reliability(manifest, split)
        rows = {mode:[[],[],[]] for mode in ('native','patch_pool',*MODES)}
        faces = {mode:[] for mode in MODES}
        for stage in range(8):
            suffix = Path('val/scores' if split == 'val' else 'train/test_scores') / f'task{stage}.npz'
            full = load_evaluation_scores(reference/'native/full'/suffix,64)
            person = load_evaluation_scores(reference/'native/person'/suffix,64)
            ids, f, p, targets, _, _ = align_evaluation_scores(full, person)
            reliable = np.array([reliability[i] for i in ids])
            weights = np.where(reliable[:,None], LOCKED_RELIABLE_WEIGHTS, LOCKED_FALLBACK_WEIGHTS)
            native_fallback = None
            for mode in rows:
                path = (reference/('pool' if mode=='patch_pool' else 'native')/'face'/suffix if mode in ('native','patch_pool')
                        else root/mode/'attention/face'/suffix)
                face = load_evaluation_scores(path,64)
                _, _, z, _, _, _ = align_evaluation_scores(full, face)
                logits = weights[:,0,None]*f + weights[:,1,None]*p + weights[:,2,None]*z
                if mode == 'native':
                    native_fallback = logits[~reliable].copy()
                if not np.array_equal(native_fallback, logits[~reliable]):
                    raise AssertionError('Unreliable Face fusion changed')
                if mode == 'cls':
                    old = load_evaluation_scores(reference/'attention/face'/suffix,64)
                    _, a, b, _, ap, bp = align_evaluation_scores(face, old)
                    exact_cls = exact_cls and np.array_equal(a,b) and np.array_equal(ap,bp)
                scores = torch.from_numpy(logits).sigmoid();y = torch.from_numpy(targets)
                for group, mask in enumerate((np.ones(len(ids),dtype=bool),reliable,~reliable)):
                    rows[mode][group].append(compute_metrics(stage,scores[mask],y[mask],.5))
                if mode in faces:
                    faces[mode].append(compute_metrics(stage,torch.from_numpy(z[reliable]).sigmoid(),y[reliable],.5))
            print('FUSION',split,stage,{mode:groups[0][-1].mAP for mode,groups in rows.items()},flush=True)
        results[split] = {mode:{pop:dict(summary=summarize_tasks(values),stages=[x.mAP for x in values],
                    per_class_ap=[x.per_class_ap for x in values]) for pop,values in zip(('all','reliable','unreliable'),groups)}
                    for mode,groups in rows.items()}
        for mode in faces:
            results[split][mode]['face_reliable'] = dict(summary=summarize_tasks(faces[mode]),stages=[x.mAP for x in faces[mode]])
    if not exact_cls:
        raise AssertionError('CLS control did not exactly replay archived Face Attention')
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('output/r1_selector_query/seed1_20261104'))
    parser.add_argument('--reference',type=Path,default=Path('../multi-lane-main-r1-crossvit-20261101/output/r1_crossvit/seed1_20261101_02'))
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();root=args.root
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('This experiment is restricted to GPU0')
    root.mkdir(parents=True,exist_ok=True)
    logs=Path('logs/r1_selector_query')/root.name;logs.mkdir(parents=True,exist_ok=True)
    git=git_metadata(Path(__file__).resolve().parents[2])
    if git['dirty']:
        raise ValueError('Clean committed source required')
    inputs=SimpleNamespace(reference_root=Path('../emotic_benchmark_runs'),data_root=Path('../multi-lane-main/datasets/EMOTIC'),
        clip_checkpoint=Path('../CODE_DDP/pretrained/clip/ViT-B-16.pt'),
        face_manifest_root=Path('../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909'))
    paths=[args.reference/family/view/suffix/f'task{t}.npz' for family,view in
           [('native','full'),('native','person'),('native','face'),('pool','face'),('attention','face')]
           for suffix in ('val/scores','train/test_scores') for t in range(8)]
    hashes={str(p):sha(p) for p in paths}
    protocol=dict(git=git,seed=1,tasks=2 if args.smoke else 8,epochs=30,batch_size=64,
          main_lr=.0125,adapter_lr=.0004,interaction_lr=.0004,query_modes=list(MODES),query_counts=[1,1,10],
          cross_attention_dim=64,heads=4,query_normalization='LayerNorm before Q projection, common archived rule',
          memory='own Face CLS plus two frozen spatial patch sets',location='after complete task block6; before block7',
          receiver='face_only',reduction='mean contexts, shared output, add to own CLS',residual_cap=None,
          query_gradient='normal Face BCE; no extra detach',supervision='own Face BCE / Adapter ASL',
          fusion='fixed R1 logits then sigmoid',reference=str(args.reference),source_hashes=hashes,
          baselines_retrained=False,full_person_predictions_reused=True,smoke=args.smoke)
    if (root/'protocol.json').exists() and read(root/'protocol.json')!=protocol:
        raise ValueError('Existing experiment protocol differs')
    write(root/'protocol.json',protocol)
    jobs=[]
    for mode in MODES:
        folder=train_path(root,mode)
        if (folder/'seed_summary.json').exists() and read(folder/'seed_summary.json')['status']=='complete':
            continue
        command=[sys.executable,'-u','-m','multi_lane.track_a.runner',*flags(SimpleNamespace(output=root/mode,**vars(inputs)),'attention','face',args.smoke),
                 '--r1-query-mode',mode,'--r1-fusion-readout','logit']
        if args.smoke:command+=['--r1-smoke-eval-samples','64']
        jobs.append(dict(name=mode+'_train',action='train',command=command))
    queue(jobs,logs,root)
    counts={mode:check_training(train_path(root,mode),protocol['tasks'],args.smoke) for mode in MODES}
    write(root/'training_checks.json',counts)
    jobs=[]
    for mode in MODES:
        if (val_path(root,mode)/'result.json').exists() and read(val_path(root,mode)/'result.json')['status']=='complete':
            continue
        command=[sys.executable,'-u','-m','multi_lane.track_a.r1_crossvit_experiment','val',
             '--training-root',str(train_path(root,mode)),'--output',str(val_path(root,mode)),
             '--data-root',str(inputs.data_root),'--clip-checkpoint',str(inputs.clip_checkpoint),
             '--face-manifest-root',str(inputs.face_manifest_root)]
        if args.smoke:command+=['--smoke']
        jobs.append(dict(name=mode+'_val',action='val',command=command))
    queue(jobs,logs,root)
    probes={mode:probe(train_path(root,mode),inputs,[0,1] if args.smoke else [0,2,7]) for mode in MODES}
    write(root/'query_probes.json',probes)
    if any(sha(path)!=value for path,value in hashes.items()):
        raise AssertionError('Reference predictions changed')
    results={} if args.smoke else aggregate(root,args.reference,inputs.face_manifest_root)
    write(root/'completion.json',dict(status='complete',smoke=args.smoke,protocol=protocol,
           source_hashes_unchanged=True,full_person_predictions_reused=True,
           cls_control_replay_exact=None if args.smoke else True,results=results))


if __name__=='__main__':
    main()
