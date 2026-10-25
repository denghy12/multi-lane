"""GPU0-only five-arm seed1 b32 + summary-module experiment."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .historical_b32_modules import ARMS, training_arguments, validate_config
from .historical_b32_search import training_arguments as historical_arguments
from .run_historical_b32_search import verify_train, compare_archive
from .run_module_stacking import write


def verify_modules(train, smoke):
    import torch
    from .export_compact_test_scores import _torch_load
    result = verify_train(train, smoke, config_validator=validate_config)
    c = json.loads((train/'config.json').read_text())
    previous = None
    for t in range(2 if smoke else 8):
        state = _torch_load(train/'compact_checkpoints'/f'task{t}.pth')['model']
        if previous is not None:
            for k, value in state.items():
                if k.startswith(('cross_view_bank.tasks.', 'cross_view_bank.face_tasks.', 'spatial_summary_bank.tasks.')) and int(k.split('.')[2]) < t:
                    assert torch.equal(value, previous[k]), k
        previous = state
    result['old_summary_modules_unchanged'] = True
    return result


def verify_native(root):
    import numpy as np
    import torch
    from .export_compact_test_scores import _torch_load
    old=root/'NATIVE_BASE/train'; new=root/'BASE/train'
    a=json.loads((old/'training_history.json').read_text()); b=json.loads((new/'training_history.json').read_text())
    for t in range(2):
        for x,y in zip(a[str(t)],b[str(t)]):
            assert all(x[k]==y[k] for k in ('current_loss','adapter_objective_loss','learning_rate','optimizer_steps','skipped_optimizer_steps'))
        x=_torch_load(old/'compact_checkpoints'/f'task{t}.pth')['model']
        y=_torch_load(new/'compact_checkpoints'/f'task{t}.pth')['model']
        assert x.keys()==y.keys() and all(torch.equal(v,y[k]) for k,v in x.items())
        with np.load(old/'val_scores'/f'task{t}.npz') as x,np.load(new/'val_scores'/f'task{t}.npz') as y:
            assert all(np.array_equal(x[k],y[k]) for k in ('sample_ids','targets','class_indices','logits','probabilities'))
    return dict(all_parameters_losses_predictions_equal=True,tasks=2,updates=4)


def evaluation_job(args, root, arm, split, runtime=False, replay=False):
    name = 'runtime_off' if runtime else 'replay_val' if replay else 'historical_'+split
    flags = ['--source-run',root/arm/'train','--output',root/arm/name,'--split',split,
        '--data-root',args.data_root,'--clip-checkpoint',args.clip_checkpoint,'--face-manifest-root',args.face_manifest_root,
        '--expected-tasks',2 if args.smoke else 8,'--batch-size',64,'--workers',2]
    if runtime:
        flags += ['--checkpoint-tasks',*(['0','1'] if args.smoke else ['0','2','7']),
                  '--disable-stacking-modules','B' if arm.startswith('ATTENTION') else 'C']
    return dict(name=arm+'/'+name,kind='eval',arm=arm,module='multi_lane.track_a.evaluate_moe_task_ffn',arguments=list(map(str,flags)))


def summarize(root, smoke):
    import numpy as np
    rows=[]
    for arm in ARMS:
        row=dict(arm=arm)
        for split in ('val','test'):
            folder='replay_val' if smoke and split=='test' else 'historical_'+split
            result=json.loads((root/arm/folder/'result.json').read_text())
            assert not result['full_cohort'] and len(result['task_metrics'])==(2 if smoke else 8)
            for k in ('final_mAP','average_mAP','final_cF1','final_oF1'):
                row[split+'_'+k]=result['summary'][k]
            if smoke or split=='test':
                for t in range(2 if smoke else 8):
                    with np.load(root/arm/'train'/('val_scores' if smoke else 'test_scores')/f'task{t}.npz') as a, \
                         np.load(root/arm/folder/'scores'/f'task{t}.npz') as b:
                        assert all(np.array_equal(a[k],b[k]) for k in ('sample_ids','targets','class_indices','logits','probabilities'))
        rows.append(row)
    with (root/'summary.csv').open('x') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    changes={}
    index={r['arm']:r for r in rows}
    for arm in ARMS[1:]:
        changes[arm]={k:index[arm][k]-index['BASE'][k] for k in rows[0] if k!='arm'}
    write(root/'contrasts.json',changes)


def run(args):
    if not args.run_id or '/' in args.run_id or args.run_id in {'.','..'}:raise ValueError('Invalid run ID')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Clean checkout required')
    if shutil.disk_usage('.').free<25*1024**3:raise RuntimeError('Need 25GiB free output space')
    root=Path('output/historical_b32_modules')/args.run_id;logs=Path('logs/historical_b32_modules')/args.run_id
    if root.exists() or logs.exists():raise FileExistsError(args.run_id)
    if not args.smoke:
        gate=json.loads((args.smoke_root/'completion.json').read_text())
        assert gate['smoke'] and gate['status']=='complete' and gate['native_equivalence']['all_parameters_losses_predictions_equal']
        assert json.loads((args.smoke_root/'initialization_audit.json').read_text())['status']=='verified'
    root.mkdir(parents=True);logs.mkdir(parents=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72',PYTHONUNBUFFERED='1')
    protocol=dict(status='running',smoke=args.smoke,seed=1,tasks=2 if args.smoke else 8,gpu=0,
        train_parallelism=5,eval_parallelism=4,batch_size=64,cpu_threads=72,
        test_used_for_parameter_search=True,evaluation_scope='samples_intersect_seen_classes',
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        training_arguments={a:training_arguments(args,root/a,a) for a in ARMS})
    write(root/'protocol.json',protocol)
    if args.smoke:
        command=[sys.executable,'-u','-m','multi_lane.track_a.audit_historical_b32_modules',
            '--clip-checkpoint',str(args.clip_checkpoint),'--output',str(root/'initialization_audit.json')]
        with (logs/'initialization.log').open('x') as f:
            try:
                subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            except BaseException as error:
                write(root/'failure.json',dict(stage='initialization_audit',error=repr(error)))
                raise
    pending=[dict(name=a+'/train',kind='train',arm=a,module='multi_lane.track_a.runner',arguments=protocol['training_arguments'][a]) for a in ARMS]
    if args.smoke:
        path=root/'native_arguments.json';write(path,historical_arguments(args,root/'NATIVE_BASE',1,'THREE_BASE',native=True))
        pending.append(dict(name='NATIVE_BASE/train',kind='native',arm='NATIVE_BASE',module='multi_lane.track_a.native_b32_reference',
            arguments=['--legacy-root',str(args.legacy_root),'--arguments-json',str(path)]))
    active=[];records=[];audits={};failed=False;last_launch=0
    try:
        while pending or active:
            for process,r in list(active):
                code=process.poll()
                if code is None:continue
                active.remove((process,r));r.update(exit=code,finished_unix=time.time());failed|=code!=0
                print('FINISH',r['name'],code,flush=True)
                if code==0:
                    try:
                        arm=r['arm']
                        if r['kind']=='train':
                            audit=verify_modules(root/arm/'train',args.smoke);audits[arm]=audit
                            write(root/arm/'training_verification.json',audit)
                            if arm=='BASE' and not args.smoke:
                                anchor=compare_archive(root/arm/'train',args.historical_run)
                                write(root/'baseline_reproduction.json',anchor)
                                if anchor['status']!='exact':raise RuntimeError('32.6800 baseline did not exactly reproduce')
                            pending += [evaluation_job(args,root,arm,'val'),evaluation_job(args,root,arm,'val' if args.smoke else 'test',replay=args.smoke)]
                        elif r['kind']=='eval' and r['name'].endswith('/historical_val') and arm!='BASE':
                            pending.append(evaluation_job(args,root,arm,'val',runtime=True))
                    except Exception as error:
                        failed=True;r['verification_error']=repr(error);print('VERIFICATION_FAILED',r['name'],repr(error),flush=True)
            if failed and not active:raise RuntimeError('A job failed; commands preserved in queue.json')
            if pending and not failed and time.time()-last_launch>=12:
                free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                stat=os.statvfs('/dev/shm');shm=stat.f_bavail*stat.f_frsize/1024**2
                index=next((i for i,j in enumerate(pending) if len(active)<5 and
                    (j['kind']!='eval' or sum(r['kind']=='eval' for _,r in active)<4)),None)
                if index is not None and free>=6000 and shm>=1024:
                    job=pending.pop(index);log=logs/(job['name']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                    command=[sys.executable,'-u','-m',job['module'],*job['arguments']]
                    with log.open('x') as f:process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    r=dict(job,pid=process.pid,gpu=0,command=command,log=str(log),started_unix=time.time())
                    active.append((process,r));records.append(r);last_launch=time.time();print('START',r['name'],r['pid'],flush=True)
            write(logs/'queue.json',dict(records=records,pending=pending,failed=failed))
            if pending or active:time.sleep(3)
        assert len({r['initial_base_sha256'] for r in audits.values()})==1
        # Module initialization and extra metadata must not change train sampling.
        rng=[r['cpu_rng_at_task_boundaries'] for r in audits.values()]
        assert all(x==rng[0] for x in rng)
        protocol['native_equivalence']=verify_native(root) if args.smoke else None
        summarize(root,args.smoke)
        protocol.update(status='complete',same_initialization_and_sampler_rng=True)
        write(root/'completion.json',protocol);print('HISTORICAL_B32_MODULES_COMPLETE',root,flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    for key in ('data-root','clip-checkpoint','face-manifest-root','legacy-root','historical-run'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-root',type=Path)
    run(p.parse_args())
