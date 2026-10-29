"""Run six existing-checkpoint diagnostics on GPU0, with bounded concurrency."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ARMS=tuple('PERSON_'+c+'_'+op for c in ('CONST','CONTENT','GROUND') for op in ('POOL','ATTN'))


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);args.logs.mkdir(parents=True,exist_ok=True)
    pending=list(ARMS);active=[];records=[];failed=[];last=0.
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72',OPENBLAS_NUM_THREADS='1')
    while pending or active:
        for process,record,stream in active[:]:
            code=process.poll()
            if code is None:continue
            active.remove((process,record,stream));stream.close()
            record.update(exit_code=code,finished_unix=time.time())
            result=args.output/record['arm']/'result.json'
            if code or not result.exists() or json.loads(result.read_text())['status']!='complete':failed.append(record)
            records.append(record)
        free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        shared=os.statvfs('/dev/shm');shm=shared.f_bavail*shared.f_frsize/1024**2
        if pending and len(active)<args.maximum_jobs and free>=3500 and shm>=512 and time.time()-last>=12:
            arm=pending.pop(0);command=[sys.executable,'-u','-m','multi_lane.track_a.person_grounded_breakpoints',
                '--training-root',str(args.source/arm/'train'),'--output',str(args.output/arm)]
            stream=(args.logs/(arm+'.log')).open('a',buffering=1)
            process=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            record=dict(arm=arm,pid=process.pid,command=command,started_unix=time.time())
            active.append((process,record,stream));last=time.time();print('START',arm,process.pid,flush=True)
        write(args.output/'queue.json',dict(status='running',pending=pending,active=[r for _,r,_ in active],completed=records,failures=failed))
        if pending or active:time.sleep(5)
    write(args.output/('failure.json' if failed else 'completion.json'),dict(status='failed' if failed else 'complete',
        arms=list(ARMS),tasks=[0,2,6,7],splits=['val','test'],records=records,failures=failed,training_updates=0))
    if failed:raise SystemExit(1)
    command=[sys.executable,'-u','scripts/emotic/analyze_person_grounded_breakpoints.py',
        '--root',str(args.output),'--output',str(args.output/'analysis'),'--bootstrap','500']
    with (args.logs/'analysis.log').open('w') as stream:
        code=subprocess.call(command,stdout=stream,stderr=subprocess.STDOUT,
            env=dict(env,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
    write(args.output/('analysis_failure.json' if code else 'analysis_complete.json'),
        dict(status='failed' if code else 'complete',command=command,exit_code=code,training_updates=0))
    if code:raise SystemExit(code)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--logs',type=Path,required=True);p.add_argument('--maximum-jobs',type=int,choices=range(1,7),default=6)
    main(p.parse_args())
