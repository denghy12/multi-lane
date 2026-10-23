"""Full-TRAIN seven-arm transfer, historical reference, multi-GPU admission queue."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .strong_full_asl import ARMS, modes
from .run_module_stacking import write


def training_arguments(args, root, arm):
    # Reuse declared numeric options, then replace only the new protocol factors.
    from .run_module_stacking import training_arguments as old_arguments
    result=old_arguments(args,root,'F100')
    for flag in ('--module-stacking','--moe-paired-audit','--save-calibration-scores'):
        result.remove(flag)
    result[result.index('--output-root')+1]=str(root/arm/'train')
    for flag,value in {'--calibration-fraction':'0','--view-auxiliary-loss-weight':'0',
            '--adapter-weight-decay':'0'}.items():result[result.index(flag)+1]=value
    if arm.startswith('S_'):
        i=result.index('--face-manifest-root');del result[i:i+2]
    result+=['--strong-full-asl']
    for key,value in modes(arm).items():result[result.index('--'+key.replace('_','-'))+1]=value
    return result


def memory_state(gpus):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(a):dict(free=int(b),utilization=int(c)) for a,b,c in (r.split(',') for r in rows.splitlines()) if int(a) in gpus}


def run_queue(jobs, logs, status, gpus, per_gpu=1, fraction=.3, minimum_free=6500, on_success=None):
    pending=list(jobs);active=[];records=[];failed=False;last_wait=0
    while pending or active:
        for process,record in list(active):
            code=process.poll()
            if code is not None:
                record.update(exit=code,finished_unix=time.time());active.remove((process,record));failed|=code!=0
                print(f'FINISH {record["name"]} gpu={record["gpu"]} exit={code}',flush=True)
                if code == 0 and on_success is not None:
                    pending.extend(on_success(record))
        if failed and not active:
            write(status,dict(records=records,pending=pending,failed=True));raise RuntimeError(f'Queue failed: {status}')
        if pending and not failed:
            state=memory_state(gpus);counts={g:sum(r['gpu']==g for _,r in active) for g in gpus}
            training_gpus={r['gpu'] for _,r in active if r['name'].endswith('/train')}
            eligible=[g for g in gpus if counts[g]<per_gpu and state[g]['free']>=minimum_free
                      and g not in training_gpus]
            # One training job per card; evaluations only fill released cards.
            if eligible:
                gpu=min(eligible,key=lambda g:(counts[g],state[g]['utilization']//20,-state[g]['free'],g))
                job=pending.pop(0);log=logs/(job['name']+'.log');log.parent.mkdir(parents=True,exist_ok=True)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
                code=(f'import runpy,torch;torch.cuda.set_per_process_memory_fraction({fraction});'
                      f'runpy.run_module({job["module"]!r},run_name="__main__")')
                command=[sys.executable,'-u','-c',code,*job['arguments']]
                with log.open('x') as stream:process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
                record=dict(job,pid=process.pid,gpu=gpu,log=str(log),command=command,initial_free_mib=state[gpu]['free'],
                    allocator_fraction=fraction,started_unix=time.time())
                records.append(record);active.append((process,record))
                print(f'START {job["name"]} pid={process.pid} gpu={gpu} free={state[gpu]["free"]}',flush=True)
            elif time.time()-last_wait>60:
                print(f'WAIT_GPU pending={len(pending)} state={state}',flush=True);last_wait=time.time()
        write(status,dict(records=records,pending=pending,failed=failed))
        if pending or active:time.sleep(3)
    return records


def evaluation_job(args,root,arm,split,folder,extra=()):
    source=root/arm/('export' if arm=='R_LEGACY' else 'train')
    return dict(name=arm+'/'+folder,module='multi_lane.track_a.evaluate_moe_task_ffn',arguments=list(map(str,[
        '--source-run',source,'--output',root/arm/folder,'--split',split,
        '--data-root',args.data_root,'--clip-checkpoint',args.clip_checkpoint,'--face-manifest-root',args.face_manifest_root,
        '--expected-tasks',2 if args.smoke else 8,'--batch-size',64,'--workers',0,
        *(['--full-cohort'] if getattr(args,'evaluation_cohort','historical')=='full' else []),*extra])))


def summarize(root, arms, smoke):
    fields=['arm','val_final','val_average','test_final','test_average','added_parameters_per_task']
    rows=[]
    for arm in arms:
        val=json.loads((root/arm/'full_val/result.json').read_text())
        test=json.loads((root/arm/('replay_val' if smoke else 'full_test')/'result.json').read_text())
        source=root/arm/('export' if arm=='R_LEGACY' else 'train');c=json.loads((source/'config.json').read_text())
        rows.append(dict(arm=arm,val_final=val['summary']['final_mAP'],val_average=val['summary']['average_mAP'],
            test_final=test['summary']['final_mAP'],test_average=test['summary']['average_mAP'],
            added_parameters_per_task=49952+(196608 if c.get('cross_view_mode')=='task_full' else 0)+(60 if c.get('spatial_summary_mode')=='target' else 0)))
    with (root/'summary.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
    contrasts={}
    for split in ['val','test']:
        for metric in ['final','average']:
            m={r['arm']:r[split+'_'+metric] for r in rows}
            contrasts[split+'_'+metric]=dict(single_asl=m['S_ASL']-m['S_BCE'],three_asl=m['T_ASL']-m['T_BCE'],
                transfer_bce=m['T_BCE']-m['S_BCE'],transfer_asl=m['T_ASL']-m['S_ASL'],
                attention=m['T_ASL_ATTN']-m['T_ASL'],position=m['T_ASL_POS']-m['T_ASL'],
                joint_interaction=m['T_ASL_BOTH']-m['T_ASL_ATTN']-m['T_ASL_POS']+m['T_ASL'])
    write(root/'contrasts.json',contrasts)
    if not smoke:
        # The existing eight-cell bootstrap is not a seven-arm estimator.
        # Store a protocol-specific paired original-image bootstrap below.
        from .strong_full_asl_bootstrap import summarize_bootstrap
        summarize_bootstrap(root)


def run(args):
    if not args.run_id or '/' in args.run_id or args.run_id in {'.','..'}:raise ValueError('Invalid run ID')
    if args.train_per_gpu != 1 or not 1 <= args.eval_per_gpu <= 3:
        raise ValueError('Use one training process and 1..3 evaluation processes per card')
    if not .2 <= args.memory_fraction <= .4 or args.minimum_free_mib < 4000:
        raise ValueError('Invalid memory admission limits')
    if len(set(args.gpus))!=len(args.gpus) or not set(args.gpus)<=set(range(8)):raise ValueError('Invalid GPUs')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Clean checkout required')
    if shutil.disk_usage('.').free<30*1024**3:raise RuntimeError('Need 30 GiB of output space')
    root=Path('output/strong_full_asl_three_view')/args.run_id;logs=Path('logs/strong_full_asl_three_view')/args.run_id
    if root.exists() or logs.exists():raise FileExistsError('Run ID exists')
    root.mkdir(parents=True);logs.mkdir(parents=True)
    arms=list(ARMS)+(['R_LEGACY'] if args.legacy_root else [])
    protocol=dict(status='running',seed=0,tasks=2 if args.smoke else 8,smoke=args.smoke,gpus=args.gpus,
        train_per_gpu=args.train_per_gpu,eval_per_gpu=args.eval_per_gpu,memory_fraction=args.memory_fraction,
        minimum_free_mib=args.minimum_free_mib,arms=arms,test_used_for_selection=False,
        evaluation_cohort=args.evaluation_cohort,
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        inputs={k:str(getattr(args,k)) for k in ['data_root','clip_checkpoint','face_manifest_root']},
        training_arguments={arm:training_arguments(args,root,arm) for arm in ARMS})
    write(root/'protocol.json',protocol)
    try:
        jobs=[dict(name=arm+'/train',module='multi_lane.track_a.runner',arguments=protocol['training_arguments'][arm]) for arm in ARMS]
        if args.legacy_root:
            jobs.append(dict(name='R_LEGACY/train',module='multi_lane.track_a.legacy_full_reference',arguments=list(map(str,[
                '--legacy-root',args.legacy_root,'--output',root/'R_LEGACY',
                '--data-root',args.data_root,'--clip-checkpoint',args.clip_checkpoint,*(['--smoke'] if args.smoke else [])]))))
        completed=set();evaluation_submitted=set()
        tasks=['0','1'] if args.smoke else ['0','2','7']
        controls={'T_ASL':[('off_A',['A'])], 'T_ASL_ATTN':[('off_B',['B'])],
                  'T_ASL_POS':[('off_C',['C'])], 'T_ASL_BOTH':[('off_B',['B']),('off_C',['C']),('off_BC',['B','C'])]}
        def ready(record):
            arm,kind=record['name'].split('/');extra=[]
            if kind=='train':
                completed.add(arm)
                if 'S_BCE' in completed:
                    for candidate in arms:
                        if candidate not in completed or candidate in evaluation_submitted:continue
                        if candidate!='R_LEGACY':
                            from .verify_strong_full_asl import verify
                            selected=('S_BCE',) if candidate=='S_BCE' else ('S_BCE',candidate)
                            with (logs/(candidate+'_verification.log')).open('x') as stream:
                                from contextlib import redirect_stdout
                                with redirect_stdout(stream):verify(root,protocol['tasks'],args.smoke,
                                    arms=selected,output_name='source_verification_'+candidate+'.json')
                        else:
                            audit=json.loads((root/candidate/'reference_verification.json').read_text())
                            if audit['status']!='verified' or audit['skipped']!=0:raise ValueError('Legacy source incomplete')
                        evaluation_submitted.add(candidate)
                        extra += [evaluation_job(args,root,candidate,'val','full_val'),evaluation_job(args,root,candidate,
                            'val' if args.smoke else 'test','replay_val' if args.smoke else 'full_test')]
            elif kind=='full_val':
                for name,disabled in controls.get(arm,[]):extra.append(evaluation_job(args,root,arm,'val','runtime_'+name,
                    ['--checkpoint-tasks',*tasks,'--disable-stacking-modules',*disabled]))
            return extra
        run_queue(jobs,logs,logs/'pipeline_queue.json',args.gpus,args.eval_per_gpu,args.memory_fraction,args.minimum_free_mib,on_success=ready)
        from .verify_strong_full_asl import verify
        with (logs/'training_verification.log').open('x') as stream:
            from contextlib import redirect_stdout
            with redirect_stdout(stream):verify(root,protocol['tasks'],args.smoke)
        summarize(root,arms,args.smoke)
        protocol['status']='complete';write(root/'completion.json',protocol)
        (logs/'batch_exit_status.txt').write_text('0\n');print('STRONG_FULL_ASL_COMPLETE '+str(root),flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error)));(logs/'batch_exit_status.txt').write_text('1\n');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True)
    for key in ['data-root','clip-checkpoint','face-manifest-root','legacy-root']:p.add_argument('--'+key,type=Path,required=key!='legacy-root')
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--train-per-gpu',type=int,default=1);p.add_argument('--eval-per-gpu',type=int,default=2)
    p.add_argument('--memory-fraction',type=float,default=.3);p.add_argument('--minimum-free-mib',type=int,default=6500)
    p.add_argument('--evaluation-cohort',choices=['historical','full'],default='historical',
                   help='Historical seen-positive people by default; full cohort is a separate sensitivity analysis')
    p.add_argument('--smoke',action='store_true');run(p.parse_args())
