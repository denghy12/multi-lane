"""GPU0-only matched training, then automatic complete validation/test and OFF."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

from .run_historical_b32_test_attribution import read,write
from .run_protected_view_forward import common,sources
from .self_trained_view_transfer import ARMS


def gpu_free_mib():
    return int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())


def verify_training(root,smoke):
    tasks=range(2 if smoke else 8)
    values={arm:read(root/arm/'train/audits.json') for arm in ARMS}
    for task in tasks:
        rows=[v[str(task)] for v in values.values()]
        assert all(r['warmup_batches']==rows[0]['warmup_batches'] for r in rows)
        adaptation=[values[a][str(task)]['adaptation_batches'] for a in ARMS[1:]]
        assert all(v==adaptation[0] for v in adaptation)
        assert all(r['old_task_parameters_exact'] and r['old_TRAIN_probe_predictions_exact']
                   and r['student_external_parameter_loads']==0 for r in rows)
    checkpoints={arm:read(root/arm/'train/training_complete.json')['checkpoints'] for arm in ARMS}
    write(root/'training_verification.json',dict(status='verified',
        matched_consumed_warmup_and_adaptation_pixels_targets_ids=True,
        external_student_parameter_loads=0,old_tasks_exact=True,checkpoints=checkpoints))


def run(args):
    assert args.run_id and '/' not in args.run_id and args.run_id not in ('.','..')
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    root=Path('output/self_trained_view_transfer')/args.run_id
    logs=Path('logs/self_trained_view_transfer')/args.run_id
    root.mkdir(parents=True,exist_ok=True);logs.mkdir(parents=True,exist_ok=True)
    source=sources(args)
    protocol=dict(status='running',head=head,seed=1,gpu=0,smoke=args.smoke,
        tasks=list(range(2 if args.smoke else 8)),arms=list(ARMS),batch_size=64,
        workers=2,prefetch_factor=1,evaluation_workers=0,threads=72,amp=True,tf32=True,
        maximum_concurrent_jobs=4,minimum_free_gpu_mib=6000,minimum_free_shared_memory_mib=1024,
        source_checkpoint_sha256=source,student_external_parameter_loads=0,
        evaluation_scope='samples_intersect_seen_classes',test_used_for_parameter_search=True,
        students_trained_from_fresh_initialization=True,teacher_only_used_during_training=True)
    if (root/'protocol.json').exists():assert read(root/'protocol.json')==protocol
    else:write(root/'protocol.json',protocol)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72',PYTHONUNBUFFERED='1')
    records=[];active=[];done=set();last_start=0.

    def start(name,command):
        nonlocal last_start
        path=logs/(name+'.log');path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:
            process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        row=dict(name=name,pid=process.pid,arguments=command,log=str(path),started_unix=time.time())
        records.append(row);active.append((process,row));last_start=time.time()
        print('START',name,process.pid,flush=True)

    module=[sys.executable,'-u','-m','multi_lane.track_a.self_trained_view_transfer']
    try:
        audit=root/'initialization_audit.json'
        if not audit.exists():
            start('initialization_audit',module+['audit',*common(args),'--output',str(audit)])
            process,row=active.pop();code=process.wait();row.update(exit=code,finished_unix=time.time())
            assert code==0,'Initialization audit failed'
        assert read(audit)['status']=='verified'
        pending=[(arm+'/train',module+['train',*common(args),'--arm',arm,'--output',str(root/arm/'train')]
                  +(['--smoke'] if args.smoke else [])) for arm in ARMS]
        def complete_path(name):
            return root/name/('training_complete.json' if name.endswith('/train') else 'result.json')
        while pending or active:
            for process,row in list(active):
                code=process.poll()
                if code is None:continue
                row.update(exit=code,finished_unix=time.time());active.remove((process,row))
                if code:raise RuntimeError(f"{row['name']} failed with {code}; {row['log']}")
                assert read(complete_path(row['name']))['status']=='complete'
                done.add(row['name']);print('FINISH',row['name'],flush=True)
            for name,command in list(pending):
                if complete_path(name).exists():
                    assert read(complete_path(name))['status']=='complete'
                    done.add(name);pending.remove((name,command));print('REUSE_COMPLETED',name,flush=True)
            # Only after all training is checked can formal Test be opened.
            if not any(n.endswith('/train') for n,_ in pending) and not any(r['name'].endswith('/train') for _,r in active):
                if not (root/'training_verification.json').exists():verify_training(root,args.smoke)
                for arm in ARMS:
                    for split in ('val',) if args.smoke else ('val','test'):
                        name=arm+'/'+split
                        if name in done or any(n==name for n,_ in pending) or any(r['name']==name for _,r in active):continue
                        pending.append((name,module+['eval','--training-root',str(root/arm/'train'),
                            '--data-root',str(args.data_root),'--clip-checkpoint',str(args.clip_checkpoint),
                            '--face-manifest-root',str(args.face_manifest_root),'--split',split,
                            '--output',str(root/name)]))
            shm=os.statvfs('/dev/shm');free_shm=shm.f_bavail*shm.f_frsize/1024**2
            if (pending and len(active)<4 and time.time()-last_start>=12
                    and gpu_free_mib()>=6000 and free_shm>=1024):
                name,command=pending.pop(0);start(name,command)
            write(logs/'queue.json',dict(status='running',records=records,pending=[n for n,_ in pending],
                active=[r['name'] for _,r in active],completed=sorted(done)))
            if pending or active:time.sleep(3)
        assert sources(args)==source
        rows=[]
        if not args.smoke:
            for arm in ARMS:
                row=dict(arm=arm)
                for split in ('val','test'):
                    result=read(root/arm/split/'result.json')
                    assert result['tasks']==list(range(8)) and not result['teacher_loaded_for_inference']
                    row.update({split+'_'+k:v for k,v in result['summary'].items()
                                if k in ('final_mAP','average_mAP','final_cF1','final_oF1')})
                rows.append(row)
            with (root/'summary.csv').open('w') as f:
                writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
        write(logs/'queue.json',dict(status='complete',records=records,completed=sorted(done)))
        write(root/'completion.json',dict(protocol,status='complete',source_checkpoints_unchanged=True,
            student_external_parameter_loads=0,all_data_streams_matched=True))
        print('SELF_TRAINED_VIEW_TRANSFER_COMPLETE',root,flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-id',required=True)
    for name in ('source-run','person-run','dual-run','reference-root','data-root','clip-checkpoint','face-manifest-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');run(parser.parse_args())
