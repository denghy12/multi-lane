"""GPU0 queue: wait for the population experiment, then continue FP32 recovery.

The population worktree is read-only. This launcher requires an already passed
Person epoch14 probe and pins this execution worktree's clean source HEAD.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BOUNDARY_SHA = '15d570f6d928854312d473ccdf0a41de3110ce111fc964581feeadcbcceedfd8'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_source(expected_head=None):
    head = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    if expected_head is not None and head != expected_head:
        raise RuntimeError('Source HEAD changed while waiting')
    if subprocess.check_output(['git','status','--porcelain'], text=True).strip():
        raise RuntimeError('Execution source worktree must remain clean')
    return head


def validate_probe(root):
    root = Path(root)
    if (root/'abort.json').exists():
        raise ValueError('Recovery output already contains an abort; do not resume failed state')
    if (root/'completion.json').exists():
        raise ValueError('Recovery is already complete; do not launch it again')
    manifest = read(root/'recovery_provenance.json')
    if (manifest.get('person_precision') != 'fp32' or manifest.get('smoke') is not False
            or (manifest.get('task'),manifest.get('completed_epoch')) != (3,13)
            or manifest.get('source_sha256') != BOUNDARY_SHA):
        raise ValueError('Only the clean task3 epoch13 FP32 continuation is accepted')
    if sha(Path(manifest['source_whole_checkpoint'])) != BOUNDARY_SHA:
        raise ValueError('Safe source checkpoint changed')
    for name in manifest['prefix_files']:
        path=Path(name)
        if 'task3' in path.parts and path.name.startswith('epoch'):
            digits=path.name[5:7]
            if digits.isdigit() and int(digits)>13:
                raise ValueError('Unsafe failed epoch imported into recovery prefix')
    probe = read(root/'person_epoch_probe.json')
    if (probe.get('status') != 'passed' or (probe.get('task'),probe.get('epoch')) != (3,14)
            or probe.get('effective_updates') != 156 or probe.get('skipped_updates') != 0
            or probe.get('amp') is not False or probe.get('scaler_disabled') is not True
            or probe.get('source_prefix_unchanged') is not True):
        raise ValueError('Person FP32 epoch14 must pass all update/finite-state checks')
    expected=root/'branches/attention/person/train/epoch_recovery/task3/epoch14.pth'
    if Path(probe['checkpoint']).resolve()!=expected.resolve() or sha(expected)!=probe['sha256']:
        raise ValueError('Person successful probe checkpoint differs')
    return {'source_epoch13_sha256': BOUNDARY_SHA,
            'person_epoch14_sha256': probe['sha256'], 'probe': str(root/'person_epoch_probe.json')}


def dependency_state(population_root, batch_status_path):
    population_root, batch_status_path = Path(population_root), Path(batch_status_path)
    if (population_root/'abort.json').exists():
        raise RuntimeError('Population experiment failed; automatic dependency gate stopped')
    if not batch_status_path.exists():
        return 'waiting', {'reason':'population_batch_status_missing'}
    batch=read(batch_status_path)
    jobs=[j for j in batch.get('jobs',[]) if j.get('name')=='fusion_population']
    if len(jobs)!=1:
        raise ValueError('Population batch must contain exactly one fusion_population job')
    job=jobs[0]
    if job.get('status')=='failed' or job.get('exit_code') not in (None,0):
        raise RuntimeError('Population job exited unsuccessfully')
    if job.get('status')!='complete' or job.get('exit_code')!=0:
        return 'waiting', {'reason':'population_process_not_complete','job_status':job.get('status')}
    # The batch may have an earlier AMP32 recovery failure; population itself
    # must still have exited successfully and the owning queue must be terminal.
    if batch.get('status') not in ('complete','finished_with_failure'):
        return 'waiting', {'reason':'population_queue_not_terminal','batch_status':batch.get('status')}
    completion=population_root/'completion.json';result=population_root/'result.json'
    if not completion.exists() or not result.exists():
        raise RuntimeError('Successful population exit lacks complete evaluation artifacts')
    if read(completion).get('status')!='complete':
        raise RuntimeError('Population completion marker is not complete')
    values=read(result)
    for split in ('val','test'):
        if split not in values or not values[split]:
            raise ValueError('Population result lacks both evaluation splits')
        for method, populations in values[split].items():
            if len(populations.get('all',{}).get('metrics',[]))!=8:
                raise ValueError(f'Population result is not eight stages: {split}/{method}')
    return 'ready', {'population_result_sha256':sha(result), 'batch_status':batch['status'],
                     'population_exit_code':job['exit_code']}


def memory_ready(free_mib, total_mib, min_free_mib=20480):
    if min_free_mib<=0 or total_mib<=0 or not 0<=free_mib<=total_mib:
        raise ValueError('Invalid GPU memory measurement/budget')
    return free_mib>=min_free_mib


def gpu0_memory():
    row=subprocess.check_output(['nvidia-smi','--id=0',
        '--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    free,total=(int(x.strip()) for x in row.split(','))
    return {'free_mib':free,'total_mib':total}


def launch(args):
    if args.poll_seconds<=0 or args.poll_seconds>60 or args.max_wait_hours<=0:
        raise ValueError('Poll must be 1–60 seconds and timeout positive')
    if args.queue_root.exists():
        raise FileExistsError('A fresh queue record is required')
    head=check_source();probe=validate_probe(args.recovery_root)
    args.queue_root.mkdir(parents=True)
    lock=(args.recovery_root/'fp32_queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'waiting_population','git_head':head,'cwd':str(Path.cwd()),
        'launcher_pid':os.getpid(),'created_utc':now(),'gpu':0,'probe':probe,
        'population_root':str(args.population_root),'population_batch_status':str(args.population_batch_status),
        'recovery_root':str(args.recovery_root),'min_free_mib':args.min_free_mib,
        'gpu_ready_consecutive_required':3,'automatic_retry':False}
    write(args.queue_root/'status.json',state)
    child=None
    try:
        deadline=time.monotonic()+3600*args.max_wait_hours;stable=0
        while True:
            check_source(head)
            if time.monotonic()>=deadline:
                raise TimeoutError('Population/GPU0 dependency wait timed out')
            status,detail=dependency_state(args.population_root,args.population_batch_status)
            state['dependency']=detail;state['updated_utc']=now()
            if status=='ready':
                memory=gpu0_memory();state['gpu0_memory']=memory
                stable=stable+1 if memory_ready(**memory,min_free_mib=args.min_free_mib) else 0
                state.update(status='waiting_gpu0',consecutive_ready=stable)
                if stable>=3:
                    break
            else:
                stable=0;state['status']='waiting_population'
            write(args.queue_root/'status.json',state)
            time.sleep(args.poll_seconds)
        check_source(head);validate_probe(args.recovery_root)
        command=[sys.executable,'-u','-m','scripts.emotic.run_r1_full_extend_recovery','run',
                 '--root',str(args.recovery_root),'--logs',str(args.logs)]
        args.logs.mkdir(parents=True,exist_ok=True)
        controller_log=args.logs/'controller.log'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72')
        with controller_log.open('x') as handle:
            child=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running',controller_pid=child.pid,command=command,
                         controller_log=str(controller_log),started_utc=now())
            write(args.queue_root/'status.json',state)
            print('FP32_RECOVERY_STARTED',child.pid,flush=True)
            while child.poll() is None:
                check_source(head)
                progress=args.recovery_root/'progress.json'
                if progress.exists():state['progress']=read(progress)
                state['updated_utc']=now();write(args.queue_root/'status.json',state)
                time.sleep(args.poll_seconds)
            code=child.wait()
        if code:
            raise RuntimeError(f'FP32 recovery failed with exit code {code}; no automatic retry')
        if (read(args.recovery_root/'completion.json').get('status')!='complete'
                or read(args.recovery_root/'numerical_recovery_complete.json').get('status')!='complete'):
            raise RuntimeError('Recovery exit lacks completed whole-model evaluations')
        result=read(args.recovery_root/'result.json')
        for split in ('val','test'):
            if len(result[split]['fixed']['all']['metrics'])!=8:
                raise RuntimeError('Recovery did not finish eight-stage Val/Test')
        state.update(status='complete',exit_code=0,finished_utc=now(),
                     result=str(args.recovery_root/'result.json'))
        write(args.queue_root/'status.json',state)
        (args.queue_root/'exit.status').write_text('0\n')
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(status='failed',error=repr(error),exit_code=1,finished_utc=now())
        write(args.queue_root/'status.json',state)
        (args.queue_root/'exit.status').write_text('1\n')
        raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery-root',type=Path,required=True)
    p.add_argument('--population-root',type=Path,required=True)
    p.add_argument('--population-batch-status',type=Path,required=True)
    p.add_argument('--queue-root',type=Path,default=Path('output/r1_fp32_recovery_queue/seed1_20261113'))
    p.add_argument('--logs',type=Path,default=Path('logs/r1_full_extend_fp32_recovery/seed1_20261113'))
    p.add_argument('--min-free-mib',type=int,default=20480)
    p.add_argument('--poll-seconds',type=float,default=15.)
    p.add_argument('--max-wait-hours',type=float,default=24.)
    launch(p.parse_args())


if __name__=='__main__':main()
