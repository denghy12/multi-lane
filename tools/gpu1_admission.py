"""GPU1 reservations plus exact observation of inherited, non-child trainers."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from gpu0_shared_admission import write


def process_state(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return dict(state=fields[0],start=fields[19],exit_status=int(fields[49]))
    except FileNotFoundError:
        return None


def inherited_exit(pid,start):
    value=process_state(pid)
    if value is None:
        raise RuntimeError('Inherited trainer vanished before exact exit was recorded')
    if value['start']!=start:raise RuntimeError('Inherited trainer PID was reused')
    return os.waitstatus_to_exitcode(value['exit_status']) if value['state']=='Z' else None


def can_start(total,free,leases,kind):
    budget=4500 if kind=='train' else 2500
    return (len(leases)<8 and (kind!='train' or sum(x['kind']=='train' for x in leases)<4)
            and sum(x['budget'] for x in leases)+budget+2000<=total and free>=budget+2000)


class Admission:
    def __init__(self,root,family):self.root=Path(root);self.family=family

    def acquire(self,key,kind,command,log):
        with (self.root/'gpu1.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=self.root/'gpu1_jobs.json'
            state=json.loads(path.read_text()) if path.exists() else dict(leases={},history=[])
            total,free=map(int,subprocess.check_output(['nvidia-smi','-i','1',
                '--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(','))
            state['last_check']=dict(gpu=1,total=total,free=free,headroom=2000,checked_unix=time.time())
            process=None
            if can_start(total,free,list(state['leases'].values()),kind):
                log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('x') as stream:
                    process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1'))
                state['leases'][key]=dict(pid=process.pid,start=process_state(process.pid)['start'],gpu=1,
                    kind=kind,budget=4500 if kind=='train' else 2500,family=self.family,
                    command=command,log=str(log.resolve()),started_unix=time.time())
            write(path,state)
            return process

    def release(self,key,code):
        with (self.root/'gpu1.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);path=self.root/'gpu1_jobs.json';state=json.loads(path.read_text())
            record=state['leases'].pop(key);state['history'].append(dict(record,exit=code,finished_unix=time.time()))
            write(path,state)
