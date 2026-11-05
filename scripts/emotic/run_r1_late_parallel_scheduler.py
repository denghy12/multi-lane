"""Take over a live legacy queue without restarting any training child.

Only the old dispatcher is stopped. Model jobs keep their PID and RNG state.
New jobs continue to execute from the original, unmodified model worktree.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def info(pid):
    try:
        value=Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:return None
    fields=value[value.rfind(')')+2:].split()
    return dict(pid=pid,state=fields[0],ppid=int(fields[1]),start=int(fields[19]),
        exit_code=os.waitstatus_to_exitcode(int(fields[49])) if len(fields)>49 and fields[0]=='Z' else None)


def priority(job):
    patch=job['name'].startswith('layer9_patch_') and not job['name'].endswith('_validation')
    return (0 if patch else 1,0 if patch and job['name'].endswith('_person') else 1,job['name'])


def measured_memory(job, smoke):
    name=job['name']
    if name.startswith('layer9_'):
        key=name[len('layer9_'):].removesuffix('_validation')
        p=smoke/'preflight'/f'{key}.json'
        d=json.loads(p.read_text())
        # Measured allocator peak + measured ~528 MiB CUDA context + slack.
        return math.ceil(d['smoke_train_peak_reserved_mib']+768)
    return job.get('memory_mib',9000)


def startup_reserve(record):
    if record.get('steady'):return 0
    command=record['job']['command']
    if '--output-root' in command:
        progress=Path(command[command.index('--output-root')+1])/'progress.json'
        if progress.exists() and progress.stat().st_mtime>=record['started']:
            record['steady']=True;return 0
    return record['memory_mib']


class LiveDispatcher:
    def __init__(self, snapshot, smoke, source, old_pid, max_jobs=4):
        self.snapshot,self.smoke,self.source,self.old_pid=snapshot,smoke,source,old_pid
        self.max_jobs=max_jobs
        self.started_jobs=0

    def __call__(self,jobs,root,logs,concurrency=2,available=None,poll=5,on_finish=None):
        root=Path(root);control=root/'parallel_control.json'
        names={j['name']:j for j in jobs}
        expected=set(self.snapshot['pending'])|set(self.snapshot['finished'])|{j['name'] for j in self.snapshot['running']}
        if set(names)!=expected:raise ValueError('Legacy job graph differs from live snapshot')
        done=dict(self.snapshot['finished']);running=[]
        for item in self.snapshot['running']:
            process=info(item['pid'])
            if process is None:raise ValueError('Live training child disappeared during takeover')
            if process['ppid']!=self.old_pid:raise ValueError('Training child ownership changed')
            running.append(dict(job=names[item['name']],pid=item['pid'],start=process['start'],
                started=0,steady=True,memory_mib=measured_memory(names[item['name']],self.smoke),owned=False))
        pending=[names[n] for n in self.snapshot['pending']]
        while pending or running:
            for record in running[:]:
                child=info(record['pid'])
                if child is not None and child['start']!=record['start']:
                    raise RuntimeError('PID reused; refusing to manage unrelated process')
                if child is not None and child['state']!='Z':continue
                if record['owned']:
                    code=record['process'].poll();record['handle'].close()
                else:code=None if child is None else child['exit_code']
                if code is None:raise RuntimeError('Cannot establish adopted job exit status')
                name=record['job']['name'];done[name]='complete' if code==0 else 'failed'
                running.remove(record)
                if on_finish:on_finish(record['job'],done[name])
            for job in pending[:]:
                if any(done.get(d) in ('failed','blocked') for d in job.get('depends',())):
                    done[job['name']]='blocked';pending.remove(job)
            ready=sorted([j for j in pending if all(done.get(d)=='complete' for d in j.get('depends',()))],key=priority)
            free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                '--format=csv,noheader,nounits'],text=True).strip())
            reserved=sum(startup_reserve(r) for r in running)
            if ready and len(running)<self.max_jobs:
                job=ready[0];need=measured_memory(job,self.smoke)
                if free-reserved>=need+512:
                    handle=(Path(logs)/(job['name']+'.log')).open('a')
                    process=subprocess.Popen(job['command'],cwd=self.source,stdout=handle,stderr=subprocess.STDOUT)
                    state=info(process.pid)
                    running.append(dict(job=job,pid=process.pid,start=state['start'],started=time.time(),steady=False,
                        memory_mib=need,owned=True,process=process,handle=handle))
                    self.started_jobs+=1
                    pending.remove(job);print('START_PARALLEL',job['name'],process.pid,'budget_mib',need,flush=True)
            serial=[dict(name=r['job']['name'],pid=r['pid'],start=r['start'],adopted=not r['owned'],
                         steady=r['steady'],memory_budget_mib=r['memory_mib']) for r in running]
            state=dict(status='running' if pending or running else 'finished',owner_pid=os.getpid(),
                paused_old_dispatcher=self.old_pid,max_late_jobs=self.max_jobs,
                free_gpu_mib=free,startup_reserve_mib=reserved,
                pending=[j['name'] for j in pending],running=serial,finished=done)
            write(root/'queue.json',state);write(control,state)
            if pending and not running and not ready:raise RuntimeError('Unresolved dependency graph')
            if pending or running:time.sleep(poll)
        return done


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-worktree',type=Path,required=True)
    p.add_argument('--old-controller',type=int,required=True)
    p.add_argument('--root',type=Path,default=Path('output/r1_frozen_late_stable/seed1_20261105'))
    a=p.parse_args();scheduler_root=Path(__file__).resolve().parents[2]
    source=a.source_worktree.resolve();root=(source/a.root).resolve()
    old=info(a.old_controller)
    command=Path(f'/proc/{a.old_controller}/cmdline').read_bytes().replace(b'\0',b' ')
    if old is None or b'run_r1_frozen_late_stable.py' not in command:
        raise ValueError('Target is not the expected dispatcher')
    if Path(f'/proc/{a.old_controller}/cwd').resolve()!=source:
        raise ValueError('Dispatcher worktree differs')
    if (root/'parallel_control.json').exists():raise ValueError('A takeover already exists; inspect before retry')
    os.kill(a.old_controller,signal.SIGSTOP)
    for _ in range(100):
        if info(a.old_controller)['state']=='T':break
        time.sleep(.02)
    snapshot=json.loads((root/'queue.json').read_text())
    children=Path(f'/proc/{a.old_controller}/task/{a.old_controller}/children').read_text().split()
    live_model_children=set()
    for pid in children:
        try:cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError:continue
        if b'multi_lane.track_a.' in cmd:live_model_children.add(int(pid))
    if live_model_children!={j['pid'] for j in snapshot['running']}:
        os.kill(a.old_controller,signal.SIGCONT)
        raise RuntimeError('Queue was between updates; old dispatcher resumed without takeover')
    write(root/'parallel_handoff_before.json',dict(old_dispatcher=old,queue=snapshot,
        note='Only dispatcher stopped; all model children continue with original PID/RNG'))
    dispatcher=None
    try:
        os.chdir(source);sys.path.insert(0,str(source))
        spec=importlib.util.spec_from_file_location('legacy_late_entry',source/'scripts/emotic/run_r1_frozen_late_stable.py')
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        dispatcher=LiveDispatcher(snapshot,root.with_name(root.name+'_smoke'),source,a.old_controller)
        legacy.run_graph=dispatcher
        meta=dict(scheduler_head=subprocess.check_output(['git','-C',str(scheduler_root),'rev-parse','HEAD'],text=True).strip(),
            model_head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
            scheduler_pid=os.getpid(),old_dispatcher=a.old_controller,model_code_unchanged=True,
            memory_rule='measured reserved +768MiB per process; 512MiB global headroom; full start-up reservation',
            cuda_driver_pid_namespace_not_used=True)
        write(root/'parallel_scheduler_provenance.json',meta)
        sys.argv=[str(source/'scripts/emotic/run_r1_frozen_late_stable.py'),'--root',str(a.root)]
        legacy.main()
    except BaseException:
        resumed=dispatcher is None or dispatcher.started_jobs==0
        if resumed:os.kill(a.old_controller,signal.SIGCONT)
        write(root/'parallel_scheduler_error.json',dict(status='error',old_dispatcher_paused=not resumed,
            note='Do not resume old dispatcher while externally launched jobs are running'))
        raise
    # The adopted children have exited; no live model belongs to the old dispatcher now.
    current=info(a.old_controller)
    if current is not None and current['start']==old['start']:
        os.kill(a.old_controller,signal.SIGKILL)
    write(root/'parallel_handoff_complete.json',dict(status='complete',old_dispatcher_retired=True))


if __name__=='__main__':main()
