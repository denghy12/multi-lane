"""Three shared-B0 arms; immutable recipe, GPU0 admission and complete evaluation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from multi_lane.track_a.historical_b32_search import training_arguments, replace_flag, validate_config
from multi_lane.track_a.run_historical_b32_search import verify_train, compare_archive

ARMS={'B0':'disabled','FACE_POOL':'pool','FACE_ATTENTION':'attention'}

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def verify_training(path, smoke):
    import torch
    from multi_lane.track_a.export_compact_test_scores import _torch_load
    result=verify_train(path,smoke)
    previous=None;cuda_rng=[]
    for t in range(2 if smoke else 8):
        state=_torch_load(path/'compact_checkpoints'/f'task{t}.pth')['model']
        if previous:
            for k,v in state.items():
                if k.startswith('face_cls_bank.tasks.') and int(k.split('.')[2])<t:
                    assert torch.equal(v,previous[k]),k
        previous=state
        from multi_lane.track_a.verify_spatial_summary import load_boundary
        boundary=load_boundary(path/'recovery_checkpoints'/f'task{t}_before_reporting.pth')
        cuda_rng.append([hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in boundary['rng']['torch_cuda']])
    result['cuda_rng_at_task_boundaries']=cuda_rng
    result['old_face_readouts_unchanged']=True
    return result


def compare_smoke(root, archive):
    import numpy as np
    import torch
    from multi_lane.track_a.export_compact_test_scores import _torch_load
    new=root/'B0/train'
    a=json.loads((archive/'training_history.json').read_text())
    b=json.loads((new/'training_history.json').read_text())
    comparisons=[]
    for t in range(2):
        assert len(a[str(t)])==len(b[str(t)])
        for x,y in zip(a[str(t)],b[str(t)]):
            assert all(x[k]==y[k] for k in ('current_loss','adapter_objective_loss','learning_rate','optimizer_steps','skipped_optimizer_steps'))
        x=_torch_load(archive/'compact_checkpoints'/f'task{t}.pth')['model']
        y=_torch_load(new/'compact_checkpoints'/f'task{t}.pth')['model']
        assert x.keys()==y.keys() and all(torch.equal(v,y[k]) for k,v in x.items())
        with np.load(archive/'val_scores'/f'task{t}.npz') as x,np.load(new/'val_scores'/f'task{t}.npz') as y:
            assert all(np.array_equal(x[k],y[k]) for k in ('sample_ids','targets','class_indices','logits','probabilities'))
        comparisons.append(t)
    return dict(status='exact',reference=str(archive),tasks=comparisons,all_losses_method_parameters_predictions_equal=True)


def probe(train, args):
    """Fixed deterministic TRAIN images, no updates or Test selection."""
    import torch
    from torch.utils.data import DataLoader
    from multi_lane.continual_datasets.continual_datasets import EMOTIC
    from multi_lane.track_a.runner import build_transforms, resolve_dataset_parent, dataset_view, move_model_inputs
    from multi_lane.track_a.paired_transforms import ThreeViewTransform
    from multi_lane.track_a.export_compact_test_scores import build_model,restore_compact_model_state,_torch_load
    from multi_lane.track_a.openai_clip_loader import load_openai_clip_visual
    c=json.loads((train/'config.json').read_text())
    model=build_model(c,load_openai_clip_visual(args.clip_checkpoint)).float().cuda()
    restore_compact_model_state(model,_torch_load(train/'compact_checkpoints/task0.pth'),0,c['git'])
    model.eval()
    _, transform=build_transforms(c['input_normalization'],c['train_crop_scale'])
    multi=ThreeViewTransform(train=False,normalization=c['input_normalization'],crop_scale=c['train_crop_scale'],
        margin=c['person_crop_margin'],jitter_strength=0.,jitter_probability=0.,full_crop_mode=c['full_crop_mode'])
    source=EMOTIC(str(resolve_dataset_parent(args.data_root)),train=True,transform=transform,input_mode='full',
        person_crop_margin=c['person_crop_margin'],multi_view_transform=multi,face_manifest_root=args.face_manifest_root)
    view=dataset_view(source,list(range(5)),include_sample_id=True)
    images,targets,ids=next(iter(DataLoader(view,batch_size=64,num_workers=0,shuffle=False)))
    images=move_model_inputs(images,torch.device('cuda'))
    model.zero_grad(set_to_none=True)
    z,views,_=model.current_all_logits_with_view_features(images)
    z[:,:5].square().mean().backward()
    gradients={n:float(p.grad.float().norm()) if p.grad is not None else None
               for n,p in model.face_cls_bank.tasks[0].named_parameters()}
    assert all(p.grad is None for p in model.visual_encoder.parameters())
    for n in ('v.weight','out.weight', *(['q.weight','k.weight'] if c['shared_face_readout']=='attention' else ['mlp.weight'])):
        assert gradients[n] is not None and gradients[n]>0 and torch.isfinite(torch.tensor(gradients[n])),(n,gradients)
    with torch.no_grad():
        on=z.detach();on_views={k:v.detach() for k,v in views.items()}
        model.face_cls_bank.runtime_enabled=False
        off,off_views,_=model.current_all_logits_with_view_features(images)
        assert torch.equal(on_views['full'],off_views['full']) and torch.equal(on_views['person'],off_views['person'])
        bad=~images['face_reliable'].bool()
        assert torch.equal(on[bad],off[bad])
    with torch.no_grad():
        model.restore_task(7)
        late=model.seen_logits(images)
        assert late.shape==(len(ids),26) and torch.isfinite(late).all()
    write(train.parent/'train_probe.json',dict(eight_lane_shape=list(late.shape),gradients=gradients,stats=model.face_cls_bank.statistics(),
        source_split='train',sample_ids=list(ids),full_person_inference_equal=True,
        unreliable_inference_equal=True,frozen_visual_no_gradient=True))


def eval_job(args,root,arm,split,off=False):
    folder=('off_' if off else 'historical_')+split
    flags=['--source-run',str(root/arm/'train'),'--output',str(root/arm/folder),'--split',split,
        '--data-root',str(args.data_root),'--clip-checkpoint',str(args.clip_checkpoint),
        '--face-manifest-root',str(args.face_manifest_root),'--batch-size','64','--workers','2',
        '--expected-tasks','2' if args.smoke else '8']
    if off: flags+=['--disable-shared-face-runtime','--checkpoint-tasks',*(['0','1'] if args.smoke else ['0','2','7'])]
    return dict(name=arm+'/'+folder,kind='eval',arm=arm,module='multi_lane.track_a.evaluate_moe_task_ffn',arguments=flags)


def run(args):
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Clean source required')
    root=args.root;logs=Path('logs/shared_b0_face_cross_view')/root.name
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);logs.mkdir(parents=True,exist_ok=True)
    if not args.smoke:
        smoke=json.loads((args.smoke_root/'completion.json').read_text())
        assert smoke['status']=='complete' and smoke['smoke'] and smoke['baseline_replay']['status']=='exact'
        assert smoke['head']==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        reservations=smoke['reservations_mib']
    else:reservations={a:7000 for a in ARMS}
    common=SimpleNamespace(data_root=args.data_root,clip_checkpoint=args.clip_checkpoint,face_manifest_root=args.face_manifest_root,smoke=args.smoke)
    jobs=[]
    for arm,mode in ARMS.items():
        flags=training_arguments(common,root/arm,1,'THREE_BASE')
        if mode!='disabled':
            flags+=['--shared-face-readout',mode,'--shared-face-layer','5','--shared-face-dim','64',
                    '--shared-face-heads','4','--shared-face-cap','.05']
            replace_flag(flags,'view_fusion_learning_rate',.0004)
        jobs.append(dict(name=arm+'/train',arm=arm,kind='train',module='multi_lane.track_a.runner',arguments=flags))
    protocol=dict(status='running',smoke=args.smoke,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        gpu=0,seed=1,tasks=2 if args.smoke else 8,epochs=30,batch=64,cap=.05,
        reference=dict(final_mAP=32.6800,average_mAP=39.5292),jobs=jobs.copy(),
        protected_full_in_training=False,train_threads=72,eval_threads=4,workers=2)
    write(root/'protocol.json',protocol)
    records=[];active=[];failures=[];audits={};baseline_replay=None;reserve={}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72',
        OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',MULTILANE_EVAL_THREADS='4')
    limit=1 if args.smoke else 3
    while jobs or active:
        for process,record in list(active):
            code=process.poll()
            if code is None:continue
            active.remove((process,record));record.update(exit=code,finished=time.time())
            print('FINISH',record['name'],code,flush=True)
            if code:failures.append(record['name']);continue
            try:
                arm=record['arm']
                if record['kind']=='train':
                    audits[arm]=verify_training(root/arm/'train',args.smoke)
                    write(root/arm/'training_verification.json',audits[arm])
                    audit=json.loads((root/arm/'train/paired_protocol_audit.json').read_text())
                    reserve[arm]=int(audit['peak_reserved_mib']*1.25+1024)
                    if arm=='B0':
                        baseline_replay=compare_smoke(root,args.archive_smoke) if args.smoke else compare_archive(root/arm/'train',args.archive_full)
                        write(root/'baseline_replay.json',baseline_replay)
                        if baseline_replay['status'] not in {'exact'}:raise RuntimeError('B0 historical replay mismatch')
                    if args.smoke:
                        jobs.append(eval_job(args,root,arm,'val'))
                        if arm!='B0':jobs.append(dict(name=arm+'/probe',kind='probe',arm=arm,module=None,
                            arguments=['--probe',str(root/arm/'train'),'--data-root',str(args.data_root),
                                '--clip-checkpoint',str(args.clip_checkpoint),'--face-manifest-root',str(args.face_manifest_root)]))
                    else:
                        jobs += [eval_job(args,root,arm,s) for s in ('val','test')]
                        if arm!='B0':jobs += [eval_job(args,root,arm,s,True) for s in ('val','test')]
            except Exception as error:
                record['verification_error']=repr(error);failures.append(record['name']);print('CHECK_FAILED',repr(error),flush=True)
        if failures:
            # Let existing work finish its own saved boundaries; never launch unverified new work.
            if not active:break
        elif jobs and len(active)<limit:
            job=jobs[0];need=reservations[job['arm']] if job['kind']=='train' else max(3500,reservations[job['arm']])
            free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free>=need+1024:
                jobs.pop(0);log=logs/(job['name']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                command=([sys.executable,'-u','-m',job['module']] if job['module'] else [sys.executable,'-u',__file__])+job['arguments']
                child_env=env.copy()
                if job['kind']!='train':child_env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                with log.open('x') as stream:
                    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=child_env)
                record=dict(job,pid=process.pid,started=time.time(),log=str(log),command=command,reservation_mib=need)
                active.append((process,record));records.append(record);print('START',job['name'],process.pid,'need',need,'free',free,flush=True)
        write(logs/'queue.json',dict(records=records,pending=jobs,failures=failures,active=[r['name'] for _,r in active]))
        if jobs or active:time.sleep(3)
    if failures:
        write(root/'failure.json',dict(failures=failures,pending=jobs));raise RuntimeError(str(failures))
    # New modules must not alter sampler/augmentation RNG consumption at boundaries.
    assert len({a['initial_base_sha256'] for a in audits.values()})==1
    streams=[a['cpu_rng_at_task_boundaries'] for a in audits.values()]
    assert all(s==streams[0] for s in streams)
    streams=[a['cuda_rng_at_task_boundaries'] for a in audits.values()]
    assert all(s==streams[0] for s in streams)
    rows=[]
    for arm in ARMS:
        row={'arm':arm}
        for split in (('val',) if args.smoke else ('val','test')):
            result=json.loads((root/arm/('historical_'+split)/'result.json').read_text())
            assert len(result['task_metrics'])==(2 if args.smoke else 8) and not result['full_cohort']
            for key in ('final_mAP','average_mAP','final_cF1','final_oF1'):row[split+'_'+key]=result['summary'][key]
        rows.append(row)
    with (root/'summary.csv').open('x') as handle:
        writer=csv.DictWriter(handle,rows[0]);writer.writeheader();writer.writerows(rows)
    write(root/'completion.json',dict(protocol,status='complete',baseline_replay=baseline_replay,
        reservations_mib=reserve,same_base_initialization=True,same_sampler_rng=True))
    print('SHARED_B0_FACE_COMPLETE',root,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('output/shared_b0_face_cross_view/seed1_20261105'))
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-root',type=Path)
    p.add_argument('--probe',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('../multi-lane-main/datasets/EMOTIC'))
    p.add_argument('--clip-checkpoint',type=Path,default=Path('../CODE_DDP/pretrained/clip/ViT-B-16.pt'))
    p.add_argument('--face-manifest-root',type=Path,default=Path('../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909'))
    old=Path('../multi-lane-main-historical-b32-modules-seed1-20261025/output/historical_b32_modules')
    p.add_argument('--archive-smoke',type=Path,default=old/'smoke_seed1_20261025_04/BASE/train')
    p.add_argument('--archive-full',type=Path,default=old/'formal_seed1_20261025_01/BASE/train')
    args=p.parse_args()
    if args.probe:probe(args.probe,args)
    else:run(args)

if __name__=='__main__':main()
