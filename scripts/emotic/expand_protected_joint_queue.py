"""Adopt the four live trainers; dispatch the two waiting arms on GPU0.

Run this dispatcher from a separate Git checkout. All training/evaluation children
keep the immutable original checkout, configuration and checkpoint progress.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from expand_historical_b32_queue import proc_info, detached_trainers, eligible

POLICY = dict(max_parallel=6, max_eval_parallel=2, minimum_free_gpu_mib=6000,
              minimum_free_shm_mib=1024, launch_interval_seconds=12)
ARMS = ('C0','C1','C2','C3','C4','FULL_ONLY')


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def exit_status(record, info, complete, log_text):
    same = info is not None and info['start']==record['adopted_start_ticks']
    if same and info['state']!='Z':
        return None, None
    status = info.get('wait_status') if same else None
    if status is not None:
        code = os.waitstatus_to_exitcode(status)
        if code:
            return code, 'linux_proc_wait_status'
    ok = complete and 'Traceback (most recent call last)' not in log_text
    evidence = 'linux_proc_wait_status' if status is not None else 'pid_exited_and_complete_artifact'
    return (0 if ok else 1), evidence


def pending_train_jobs(protocol, snapshot, root):
    known = {r['name'] for r in snapshot['records']}
    module = [sys.executable,'-u','-m','multi_lane.track_a.protected_joint_experiment']
    common = sum((['--'+name.replace('_','-'),protocol[name]] for name in
                  ('source_run','data_root','clip_checkpoint','face_manifest_root')), [])
    jobs = []
    for arm in ARMS:
        name = arm+'_train'
        if name not in known:
            output = root/arm/'train'
            jobs.append(dict(name=name, kind='train', arm=arm, output=str(output),
                command=module+['train',*common,'--arm',arm,'--tasks','8','--output',str(output),
                                '--historical-reference-root',protocol['source_run']]))
    return jobs


def main(args):
    source = args.source_worktree.resolve()
    root = source/'output/protected_joint_attention'/args.run_id
    logs = source/'logs/protected_joint_attention'/args.run_id
    status_path = logs/'queue.json'
    protocol = json.loads((root/'protocol.json').read_text())
    assert protocol['tasks']==list(range(8)) and protocol['gpu']==0 and not protocol['smoke']
    assert not (root/'completion.json').exists() and not (root/'failure.json').exists()
    assert not (root/'concurrency_handoff.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert head==protocol['git']['commit']
    old = proc_info(args.old_controller)
    assert old and 'multi_lane.track_a.run_protected_joint_attention' in old['command']
    assert args.run_id in old['command']
    assert Path(os.readlink(f'/proc/{args.old_controller}/cwd'))==source
    snapshot = json.loads(status_path.read_text())
    assert not snapshot['failures']
    active = [r for r in snapshot['records'] if 'exit' not in r]
    assert detached_trainers(active, {r['pid']:proc_info(r['pid']) for r in active})
    for row in active:
        assert Path(os.readlink(f"/proc/{row['pid']}/cwd"))==source
    pending = pending_train_jobs(protocol,snapshot,root)
    print('PREFLIGHT',json.dumps(dict(active=[(r['name'],r['pid']) for r in active],
        pending=[j['name'] for j in pending],policy=POLICY)),flush=True)
    if args.dry_run:
        return
    write(root/'concurrency_handoff_preflight.json',dict(old_controller=old,
        verified_detached_trainers=active,policy=POLICY,unix=time.time()))
    # Kill the dispatcher PID only. Each training child is a session leader
    # without a controlling terminal, and is neither signalled nor restarted.
    os.kill(args.old_controller,signal.SIGTERM)
    for _ in range(100):
        info = proc_info(args.old_controller)
        if info is None or info['state']=='Z':
            break
        time.sleep(.05)
    else:
        raise RuntimeError('Original dispatcher did not exit')
    lock = (root/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    snapshot = json.loads(status_path.read_text())
    write(root/'queue_before_concurrency_handoff.json',snapshot)
    records = snapshot['records']
    assert not snapshot['failures'] and all(r.get('exit',0)==0 for r in records)
    active = [r for r in records if 'exit' not in r]
    for row in active:
        info = proc_info(row['pid'])
        assert info and info['state']!='Z' and args.run_id in info['command']
        assert 'multi_lane.track_a.protected_joint_experiment train' in info['command']
        assert Path(os.readlink(f"/proc/{row['pid']}/cwd"))==source
        row['adopted_start_ticks'] = info['start']
    pending = pending_train_jobs(protocol,snapshot,root)
    owned, known = {}, {r['name'] for r in records}|{j['name'] for j in pending}
    completed = set(snapshot['completed']); failures = []
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72')
    handoff = dict(status='adopted',model_head=head,old_controller=args.old_controller,
        controller_pid=os.getpid(),controller_head=subprocess.check_output(['git','rev-parse','HEAD'],
            cwd=Path(__file__).resolve().parents[2],text=True).strip(),policy=POLICY,
        training_restarted=False,batch_size_changed=False,gpu=0,adopted_unix=time.time(),
        preserved_processes=[dict(name=r['name'],pid=r['pid'],start=r['adopted_start_ticks']) for r in active])
    write(root/'concurrency_handoff.json',handoff)
    write(root/'runtime_scheduling_policy.json',POLICY)
    module = [sys.executable,'-u','-m','multi_lane.track_a.protected_joint_experiment']

    def complete(job):
        path = source/Path(job['output'])/('training_complete.json' if job['kind']=='train' else 'result.json')
        if not path.exists():
            return False
        value = json.loads(path.read_text())
        return value.get('status')=='complete' and value.get('tasks')==list(range(8))

    def add_evaluation(arm):
        for split in ('val','test'):
            name = arm+'_'+split
            if name in known:
                continue
            known.add(name); output = root/arm/split
            pending.append(dict(name=name,kind='eval',arm=arm,output=str(output),
                command=module+['eval','--training-root',str(root/arm/'train'),
                                '--split',split,'--workers','2','--output',str(output)]))

    def queue_state(status='running'):
        return dict(status=status,records=records,pending=[j['name'] for j in pending],
            active=[j['name'] for j in active],completed=sorted(completed),failures=failures,
            controller_pid=os.getpid(),controller='six_way_handoff',policy=POLICY)

    try:
        for job in records:
            if job.get('exit')==0 and job['kind']=='train':
                add_evaluation(job['arm'])
        last_launch = 0.
        while active or pending:
            for job in list(active):
                if job['pid'] in owned:
                    code = owned[job['pid']].poll(); evidence='subprocess_returncode'
                else:
                    code,evidence = exit_status(job,proc_info(job['pid']),complete(job),
                        (source/Path(job['log'])).read_text())
                if code is None:
                    continue
                if code==0 and not complete(job):
                    code=1; evidence='missing_complete_artifact'
                job.update(exit=code,exit_evidence=evidence,finished_unix=time.time())
                active.remove(job)
                if code:
                    failures.append(job)
                else:
                    completed.add(job['name'])
                    if job['kind']=='train':
                        add_evaluation(job['arm'])
                print('FINISH',job['name'],code,flush=True)
            if failures and not active:
                raise RuntimeError('A job failed; preserved pending work and stopped dispatch')
            if pending and not failures and time.time()-last_launch>=POLICY['launch_interval_seconds']:
                free = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'],text=True).strip())
                shm=os.statvfs('/dev/shm'); available_shm=shm.f_bavail*shm.f_frsize/1024**2
                job = next((j for j in pending if eligible(j,active,free,available_shm,POLICY)),None)
                if job is not None:
                    pending.remove(job)
                    if complete(job):
                        completed.add(job['name'])
                        if job['kind']=='train': add_evaluation(job['arm'])
                        continue
                    log=logs/(job['name']+'.log')
                    with log.open('x') as stream:
                        child=subprocess.Popen(job['command'],cwd=source,env=env,stdout=stream,
                            stderr=subprocess.STDOUT,start_new_session=True)
                    job.update(pid=child.pid,log=str(log),started_unix=time.time(),model_head=head)
                    owned[child.pid]=child; active.append(job); records.append(job); last_launch=time.time()
                    print('START',job['name'],child.pid,'GPU0',flush=True)
            write(status_path,queue_state())
            if active or pending: time.sleep(3)
        sys.path.insert(0,str(source))
        from multi_lane.track_a.protected_joint_audit import verify_outputs
        verify_outputs(root,list(range(8)),False)
        rows=[]
        for arm in ARMS:
            row=dict(arm=arm)
            for split in ('val','test'):
                value=json.loads((root/arm/split/'result.json').read_text())
                baseline=json.loads((root/'C0'/split/'result.json').read_text())
                for key in ('final_mAP','average_mAP'):
                    row[split+'_'+key]=value['summary'][key]
                    row[split+'_off_'+key]=value['off_summary'][key]
                    assert value['off_summary'][key]==baseline['summary'][key]
            rows.append(row)
        with (root/'summary.csv').open('w') as stream:
            writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)
        handoff.update(status='complete',finished_unix=time.time())
        write(root/'concurrency_handoff.json',handoff)
        write(root/'completion.json',dict(status='complete',protocol=protocol,records=records,
            all_off_summaries_exactly_match_paired_B0=True,runtime_scheduling_handoff=handoff))
        write(status_path,queue_state('complete'))
        print('PROTECTED_JOINT_ATTENTION_COMPLETE',root,flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error),records=records,pending=pending,
            controller='six_way_handoff',training_processes_not_signalled=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-worktree',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--old-controller',type=int,required=True)
    parser.add_argument('--dry-run',action='store_true')
    main(parser.parse_args())
