"""Bounded GPU0 queue: six fresh eight-task runs, complete Val/Test and OFF."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .protected_joint_attention import ARMS
from .run_historical_b32_test_attribution import read, write


def common(args):
    return sum((['--'+name.replace('_','-'),str(getattr(args,name))] for name in
                ('source_run','data_root','clip_checkpoint','face_manifest_root')), [])


def memory_available():
    free = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True).strip())
    shm = os.statvfs('/dev/shm')
    return free, shm.f_bavail*shm.f_frsize/1024**2


def run(args):
    from .protected_joint_experiment import runtime
    git = runtime()
    args.output.mkdir(parents=True, exist_ok=True); args.logs.mkdir(parents=True, exist_ok=True)
    lock = (args.output/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = [0,1] if args.smoke else list(range(8))
    protocol = dict(git=git,seed=1,gpu=0,arms=list(ARMS),tasks=tasks,smoke=args.smoke,
        epochs_per_task=30,batch_size=64,threads=72,train_workers=2,eval_workers=2,
        prefetch_factor=1,maximum_jobs=args.maximum_jobs,maximum_eval_jobs=2,
        minimum_free_gpu_mib=6000,minimum_free_shm_mib=1024,
        source_task_weights_loaded=False,teacher_predictions=False,
        test_used_for_development=True,evaluation_scope='samples_intersect_seen_classes',
        source_run=str(args.source_run),data_root=str(args.data_root),clip_checkpoint=str(args.clip_checkpoint),
        face_manifest_root=str(args.face_manifest_root),smoke_updates=args.smoke_updates if args.smoke else None)
    if (args.output/'protocol.json').exists():
        if read(args.output/'protocol.json') != protocol:
            raise ValueError('Queue protocol changed on restart')
    else:
        write(args.output/'protocol.json',protocol)
    module = [sys.executable,'-u','-m','multi_lane.track_a.protected_joint_experiment']
    pending, active, records, failures = [], [], [], []
    for arm in ARMS:
        root = args.output/arm/'train'
        command = module+['train',*common(args),'--arm',arm,'--tasks','8','--output',str(root)]
        if args.smoke:
            command += ['--smoke','--smoke-updates',str(args.smoke_updates)]
        else:
            command += ['--historical-reference-root',str(args.source_run)]
        pending.append(dict(name=arm+'_train',kind='train',arm=arm,command=command,output=str(root)))
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72')
    completed, last = set(), 0.

    def completion(job):
        return Path(job['output'])/('training_complete.json' if job['kind']=='train' else 'result.json')

    def ready(job):
        path = completion(job)
        return path.exists() and read(path).get('status') == 'complete' and read(path).get('tasks') == tasks

    def add_evaluation(arm):
        for split in ('val',) if args.smoke else ('val','test'):
            root = args.output/arm/split
            pending.append(dict(name=arm+'_'+split,kind='eval',arm=arm,output=str(root),
                command=module+['eval','--training-root',str(args.output/arm/'train'),
                    '--split',split,'--workers','2','--output',str(root)]))
    try:
        while pending or active:
            for process,job in list(active):
                code = process.poll()
                if code is None:
                    continue
                active.remove((process,job)); job.update(exit=code,finished_unix=time.time())
                if code or not ready(job):
                    failures.append(job)
                else:
                    completed.add(job['name'])
                    if job['kind']=='train':
                        add_evaluation(job['arm'])
                print('FINISH',job['name'],code,flush=True)
            if failures:
                pending = []
            if pending and len(active)<args.maximum_jobs and time.time()-last>=12:
                # Eval uses at most two processes; train gets the remaining slots.
                jobs = [j for j in pending if j['kind']=='train' or
                        sum(r['kind']=='eval' for _,r in active)<2]
                if jobs:
                    job = jobs[0]
                    if ready(job):
                        pending.remove(job); completed.add(job['name'])
                        if job['kind']=='train':
                            add_evaluation(job['arm'])
                        continue
                    free, shm = memory_available()
                    if free>=6000 and shm>=1024:
                        pending.remove(job)
                        log = args.logs/(job['name']+'.log')
                        with log.open('a') as handle:
                            process = subprocess.Popen(job['command'],env=env,stdout=handle,
                                stderr=subprocess.STDOUT,start_new_session=True)
                        job.update(pid=process.pid,log=str(log),started_unix=time.time())
                        active.append((process,job)); records.append(job); last=time.time()
                        print('START',job['name'],process.pid,flush=True)
            write(args.logs/'queue.json',dict(status='running',records=records,
                pending=[j['name'] for j in pending],active=[j['name'] for _,j in active],
                completed=sorted(completed),failures=failures))
            if pending or active:
                time.sleep(3)
        if failures:
            raise RuntimeError('Job failure: '+repr(failures))
        rows = []
        for arm in ARMS:
            row = dict(arm=arm)
            for split in ('val',) if args.smoke else ('val','test'):
                result = read(args.output/arm/split/'result.json')
                for key in ('final_mAP','average_mAP'):
                    row[split+'_'+key] = result['summary'][key]
                    row[split+'_off_'+key] = result['off_summary'][key]
                base = read(args.output/'C0'/split/'result.json')
                for key in ('final_mAP','average_mAP'):
                    if result['off_summary'][key] != base['summary'][key]:
                        raise AssertionError(('OFF did not replay paired B0',arm,split,key))
            rows.append(row)
        with (args.output/'summary.csv').open('w') as handle:
            writer = csv.DictWriter(handle, list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write(args.output/'completion.json',dict(status='complete',protocol=protocol,
            all_off_summaries_exactly_match_paired_B0=True,records=records))
        write(args.logs/'queue.json',dict(status='complete',records=records))
        print('PROTECTED_JOINT_ATTENTION_COMPLETE',args.output,flush=True)
    except BaseException as error:
        write(args.output/'failure.json',dict(error=repr(error),records=records)); raise


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','data-root','clip-checkpoint','face-manifest-root','output','logs'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--maximum-jobs',type=int,choices=(1,2,3,4),default=4)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-updates',type=int,default=4)
    run(p.parse_args())
