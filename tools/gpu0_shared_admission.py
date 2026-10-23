"""Cross-experiment GPU0 admission; reserve before starting a child process."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def identity(pid):
    try:
        parts=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return None if parts[0]=='Z' else parts[19]
    except (FileNotFoundError,ProcessLookupError):
        return None


def gpu_memory():
    values=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.total,memory.free',
                                    '--format=csv,noheader,nounits'],text=True)
    return tuple(int(x.strip()) for x in values.strip().split(','))


def fits(total,free,new_count,legacy_active,budget=4500,headroom=2000):
    # The existing stacking controller may advance from one trainer to four
    # evaluators without taking this lock. Keep its future evaluation allowance.
    reserve=8000 if legacy_active else 0
    return (new_count<5 and (new_count+1)*budget+reserve+headroom<=total
            and free>=budget+headroom)


class SharedAdmission:
    def __init__(self,root,family,legacy_pid,legacy_start):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.family=family;self.legacy_pid=legacy_pid;self.legacy_start=legacy_start
        self.phase='preflight'

    def state(self):
        path=self.root/'admission.json'
        return json.loads(path.read_text()) if path.exists() else dict(leases={},requests={},history=[])

    def acquire(self,key,command,log,environment):
        with (self.root/'admission.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=self.state()
            for job,lease in list(state['leases'].items()):
                if identity(lease['pid'])!=lease['start_ticks']:
                    state['history'].append(dict(lease,released_unix=time.time(),reason='process_finished'))
                    del state['leases'][job]
            for job,request in list(state['requests'].items()):
                if identity(request['driver_pid'])!=request['driver_start']:
                    del state['requests'][job]
            state['requests'].setdefault(key,dict(family=self.family,driver_pid=os.getpid(),
                driver_start=identity(os.getpid()),phase=self.phase,submitted_unix=time.time()))
            counts={family:sum(x['family']==family for x in state['leases'].values())
                    for family in ('regional','context')}
            rank=lambda item:({'preflight':0,'smoke':1,'formal':2}[item[1]['phase']],
                               counts.get(item[1]['family'],0),item[1]['submitted_unix'])
            first=min(state['requests'].items(),key=rank)[0]
            total,free=gpu_memory()
            legacy=identity(self.legacy_pid)==self.legacy_start
            state['last_admission_check']=dict(total_mib=total,free_mib=free,
                active_new_jobs=len(state['leases']),legacy_active=legacy,
                legacy_reservation_mib=8000 if legacy else 0,headroom_mib=2000,job_budget_mib=4500,
                checked_unix=time.time())
            process=None
            if first==key and fits(total,free,len(state['leases']),legacy):
                log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('x') as stream:
                    process=subprocess.Popen(command,env=environment,stdout=stream,stderr=subprocess.STDOUT)
                lease=dict(key=key,family=self.family,phase=self.phase,pid=process.pid,
                    start_ticks=identity(process.pid),command=command,log=str(log.resolve()),
                    started_unix=time.time(),reserved_budget_mib=4500,allocator_fraction=.16,gpu=0)
                state['leases'][key]=lease;del state['requests'][key]
            write(self.root/'admission.json',state)
            return process

    def release(self,key,exit_code):
        with (self.root/'admission.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);state=self.state()
            lease=state['leases'].pop(key,None)
            if lease:state['history'].append(dict(lease,exit=exit_code,released_unix=time.time()))
            write(self.root/'admission.json',state)
