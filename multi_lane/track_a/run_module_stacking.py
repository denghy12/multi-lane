"""GPU0 memory-aware queue: eight training arms, paired val/test, runtime controls."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .module_stacking import ARMS, STAGES, modes


def write(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def training_arguments(args, root, arm):
    result=['--module-stacking','--training-protocol','incremental','--moe-paired-audit','--seed','0',
        '--data-root',args.data_root,'--clip-checkpoint',args.clip_checkpoint,'--face-manifest-root',args.face_manifest_root,
        '--output-root',root/arm/'train','--epochs','30','--train-batch-size','64','--eval-batch-size','64',
        '--workers','0','--eval-workers','0','--save-pre-evaluation-recovery',
        '--amp-initial-scale','1024','--amp-growth-interval','1000000000',
        '--source-learning-rate','.05','--source-reference-batch-size','256','--weight-decay','0',
        '--scheduler-mode','cosine','--scheduler-min-lr-ratio','0','--scheduler-warmup-ratio','0',
        '--temperature','1','--threshold','.5','--training-loss-mode','legacy_full_zero',
        '--loss-routing','joint_bce','--supervised-loss-scale','1','--input-mode','full',
        '--input-normalization','clip','--train-crop-scale','.05','1','--full-crop-mode','legacy',
        '--person-transform-mode','letterbox','--person-crop-margin','.15',
        '--person-color-jitter-strength','0','--person-color-jitter-probability','0',
        '--selector-mode','shared','--prompt-mode','shared','--num-selectors','10',
        '--view-fusion','fixed_three_view','--view-classifier-mode','shared_post_fusion',
        '--view-auxiliary-loss-weight','.1','--view-gradient-routing','joint','--view-fusion-learning-rate','.0004',
        '--adapter-bottleneck-dim','32','--adapter-layer-indices','1','--adapter-residual-scale','.1',
        '--adapter-residual-gate-mode','fixed','--adapter-activation','relu','--adapter-task-init','independent',
        '--adapter-view-mode','shared','--adapter-view-bottleneck-dim','0','--adapter-learning-rate','.0004',
        '--adapter-weight-decay','0','--adapter-regularization','none',
        '--cross-view-layer','5','--cross-view-dim','64','--cross-view-heads','4','--cross-view-scale','.1',
        '--parax-mode','disabled','--joint-summary-mode','disabled','--selector-conditioning','disabled',
        '--calibration-fraction','.1','--save-calibration-scores','--save-evaluation-scores',
        '--evaluation-score-purpose','validation_search','--skip-validation-eval','--reporting-split','val',
        '--save-compact-checkpoints','--no-save-checkpoints','--max-tasks','2' if args.smoke else '8']
    for key,value in modes(arm).items(): result += ['--'+key.replace('_','-'),value]
    if args.smoke:result+=['--optimizer-updates-per-task','2']
    return list(map(str,result))


def free_memory():
    return int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())


def run_queue(jobs, logs, status, parallelism, fraction, minimum_free):
    pending=list(jobs);active=[];records=[];failed=False;last_wait=None
    while pending or active:
        for process, record in list(active):
            code=process.poll()
            if code is not None:
                record.update(exit=code,finished_unix=time.time())
                active.remove((process,record));failed |= code != 0
                print(f'FINISH job={record["name"]} pid={process.pid} exit={code}',flush=True)
        if failed:
            if not active:
                write(status,dict(records=records,pending=pending,failed=True))
                raise RuntimeError(f'Job failed; pending jobs preserved in {status}')
        elif pending and len(active)<parallelism:
            free=free_memory()
            if free>=minimum_free:
                job=pending.pop(0)
                log=logs/(job['name']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
                code=(f'import runpy,torch;torch.cuda.set_per_process_memory_fraction({fraction});'
                      f'runpy.run_module({job["module"]!r},run_name="__main__")')
                command=[sys.executable,'-u','-c',code,*job['arguments']]
                with log.open('x') as stream:process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
                record=dict(job,pid=process.pid,gpu=0,command=command,log=str(log),
                            initial_free_mib=free,allocator_fraction=fraction,started_unix=time.time())
                records.append(record);active.append((process,record));last_wait=None
                print(f'START job={job["name"]} pid={process.pid} gpu=0 free={free} log={log}',flush=True)
            elif last_wait is None or time.time()-last_wait>60:
                print(f'WAIT_GPU free={free} required={minimum_free} pending={len(pending)}',flush=True);last_wait=time.time()
        write(status,dict(records=records,pending=pending,failed=failed))
        if pending or active:time.sleep(5)
    return records


def evaluation_job(args,root,arm,split,folder,extra=()):
    return dict(name=arm+'/'+folder,module='multi_lane.track_a.evaluate_moe_task_ffn',arguments=list(map(str,[
        '--source-run',root/arm/'train','--output',root/arm/folder,'--split',split,
        '--data-root',args.data_root,'--clip-checkpoint',args.clip_checkpoint,'--face-manifest-root',args.face_manifest_root,
        '--expected-tasks',2 if args.smoke else 8,'--batch-size',64,'--workers',0,*extra])))


def run(args):
    if not args.run_id or '/' in args.run_id or args.run_id in {'.','..'}:raise ValueError('Invalid run ID')
    if args.gpu!=0:raise ValueError('User requested GPU0 only')
    if not 1<=args.train_parallelism<=4 or not 1<=args.eval_parallelism<=4:raise ValueError('Concurrency must be 1..4')
    if not .16<=args.memory_fraction<=.2 or args.minimum_free_mib<4000:
        raise ValueError('Invalid memory admission limits')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Clean Git checkout required')
    if shutil.disk_usage('.').free<20*1024**3:raise RuntimeError('Need at least 20 GiB of free output space')
    root=Path('output/three_module_stacking')/args.run_id;logs=Path('logs/three_module_stacking')/args.run_id
    if root.exists() or logs.exists():raise FileExistsError('Run ID already exists')
    root.mkdir(parents=True);logs.mkdir(parents=True)
    protocol=dict(status='running',smoke=args.smoke,seed=0,tasks=2 if args.smoke else 8,gpu=0,
        train_parallelism=args.train_parallelism,eval_parallelism=args.eval_parallelism,
        memory_fraction=args.memory_fraction,minimum_free_mib=args.minimum_free_mib,
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        inputs={k:str(getattr(args,k)) for k in ('data_root','clip_checkpoint','face_manifest_root')},
        stages=STAGES,test_used_for_selection=False,
        training_arguments={arm:training_arguments(args,root,arm) for arm in ARMS})
    write(root/'protocol.json',protocol)
    try:
        for index,stage in enumerate(STAGES):
            jobs=[dict(name=arm+'/train',module='multi_lane.track_a.runner',arguments=protocol['training_arguments'][arm]) for arm in stage]
            run_queue(jobs,logs,logs/f'training_stage{index}.json',args.train_parallelism,args.memory_fraction,args.minimum_free_mib)
        verify=[sys.executable,'-m','multi_lane.track_a.verify_module_stacking',str(root),'--tasks',str(protocol['tasks'])]
        if args.smoke:verify+=['--smoke']
        with (logs/'training_verification.log').open('x') as stream:
            subprocess.run(verify,stdout=stream,stderr=subprocess.STDOUT,check=True)
        jobs=[]
        for arm in ARMS:
            jobs.append(evaluation_job(args,root,arm,'val','full_val',['--full-cohort']))
            jobs.append(evaluation_job(args,root,arm,'val' if args.smoke else 'test',
                                       'replay_val' if args.smoke else 'full_test',[] if args.smoke else ['--full-cohort']))
        run_queue(jobs,logs,logs/'evaluation_queue.json',args.eval_parallelism,args.memory_fraction,args.minimum_free_mib)
        tasks=['0','1'] if args.smoke else ['0','2','7']
        jobs=[evaluation_job(args,root,'F111','val','runtime_'+name,
            ['--full-cohort','--checkpoint-tasks',*tasks,'--disable-stacking-modules',*disabled])
            for name,disabled in [('off_A',['A']),('off_B',['B']),('off_C',['C']),('off_ABC',['A','B','C'])]]
        run_queue(jobs,logs,logs/'runtime_queue.json',args.eval_parallelism,args.memory_fraction,args.minimum_free_mib)
        from .summarize_module_stacking import summarize
        summarize(root,args.smoke)
        protocol['status']='complete';write(root/'completion.json',protocol)
        (logs/'batch_exit_status.txt').write_text('0\n')
        print('MODULE_STACKING_COMPLETE '+str(root),flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error),pending_status='see logs/*queue*.json and training_stage*.json'))
        (logs/'batch_exit_status.txt').write_text('1\n');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    for key in ('data-root','clip-checkpoint','face-manifest-root'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--train-parallelism',type=int,default=4);parser.add_argument('--eval-parallelism',type=int,default=4)
    parser.add_argument('--memory-fraction',type=float,default=.18);parser.add_argument('--minimum-free-mib',type=int,default=5600)
    parser.add_argument('--smoke',action='store_true');run(parser.parse_args())
