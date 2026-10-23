"""Execute unchanged source experiments using one shared memory admission policy.

This external driver replaces only queued orchestrators, never running trainers.
Source worktrees stay clean and at their original scientific code commits.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from gpu0_shared_admission import SharedAdmission,identity,write


def checked_source(args):
    root=args.source.resolve()
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('Source checkout must remain clean')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if head!=args.source_head:raise RuntimeError('Source code version changed')
    if args.family=='context':
        record=json.loads((root/'logs/summary_context_attention/implementation_20261023/cpu_tests.json').read_text())
        if record['status']!='passed' or record['head']!=head:raise RuntimeError('CPU tests mismatch')
    return root


def run(args):
    root=checked_source(args)
    os.chdir(root)
    sys.path.insert(0,str(root))
    admission=SharedAdmission(args.control,args.family,args.legacy_pid,args.legacy_start)
    module_name='run_regional_summary' if args.family=='regional' else 'run_summary_context'
    module=importlib.import_module('multi_lane.track_a.'+module_name)
    family_kind='target_region_summary' if args.family=='regional' else 'summary_context_attention'
    queue=(root/'output'/family_kind/(('queue_'+args.run_id) if args.family=='regional' else args.run_id+'_queue'))
    old_state=queue/'status.json'
    original=json.loads(old_state.read_text())
    assert original['status'] in ('handoff_to_shared_scheduler','superseded_by_shared_scheduler')
    control_head=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()
    status=dict(status='preflight',family=args.family,source_head=args.source_head,control_head=control_head,
                gpu=0,pid=os.getpid(),scheduler='shared_memory_admission',
                previous_controller_pid=original.get('pid',args.previous_pid),run_id=args.run_id,
                full_training_budget_unchanged=True,changed_only_resource_admission=True,
                effective_allocator_fraction=.16,effective_local_concurrency=3 if args.family=='regional' else 4)
    def save_status():
        write(old_state,status);write(args.control/(args.family+'_status.json'),status)
    save_status()

    def run_jobs(jobs,logs,path,parallelism=None):
        limit=(3 if args.family=='regional' else 4) if parallelism is None else parallelism
        # Preserve original scientific launch declarations and separately record
        # effective resource overrides, rather than silently rewriting provenance.
        for job in jobs:
            arguments=list(map(str,job['arguments']))
            for option in ('--output-root','--source-run'):
                if option in arguments:
                    batch_root=Path(arguments[arguments.index(option)+1]).parent.parent
                    write(batch_root/'resource_override.json',dict(source_head=args.source_head,
                        control_head=control_head,allocator_fraction=.16,local_concurrency=limit,
                        global_admission=str(args.control),changed_only_resources=True,
                        original_protocol_preserved=True))
        pending,active,records=list(jobs),[],[]
        failed=False
        while pending or active:
            for process,record,key in list(active):
                code=process.poll()
                if code is not None:
                    record.update(exit=code,finished_unix=time.time())
                    active.remove((process,record,key));admission.release(key,code);failed|=code!=0
                    print('FINISH',record['name'],code,flush=True)
            if failed and not active:
                write(path,dict(status='failed',records=records,pending=pending,failed=True))
                raise RuntimeError('Job failed; remaining jobs and artifacts preserved')
            if not failed and pending and len(active)<limit:
                job=pending[0];key=args.family+'/'+str(path)+'/'+job['name']
                log=logs/(job['name']+'.log')
                code=("import runpy,torch;torch.cuda.set_per_process_memory_fraction(.16);"
                      f"runpy.run_module({job['module']!r},run_name='__main__')")
                command=[sys.executable,'-u','-c',code,*map(str,job['arguments'])]
                environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
                process=admission.acquire(key,command,log,environment)
                if process is not None:
                    pending.pop(0)
                    record=dict(job,pid=process.pid,command=command,gpu=0,allocator_fraction=.16,
                                log=str(log),started_unix=time.time(),shared_reservation_mib=4500)
                    records.append(record);active.append((process,record,key))
                    print('START',record['name'],process.pid,'GPU0 shared admission',flush=True)
            write(path,dict(status='running' if pending or active else 'complete',
                            records=records,pending=pending,failed=failed,shared_scheduler=True))
            if pending or active:time.sleep(5)
        return records

    try:
        if args.family=='regional':
            module.run_queue=run_jobs
            logs=Path('logs/target_region_summary')/('preflight_'+args.run_id)
            arguments=['--reference-run',args.reference_run,'--output',queue/'initialization.json',*module.resources(args)]
            run_jobs([module.job('real_initialization','audit_spatial_initialization',arguments)],logs,logs/'initialization_queue.json',1)
            assert json.loads((queue/'initialization.json').read_text())['status']=='verified'
            admission.phase='smoke';status['status']='running_gpu_smoke';save_status()
            module.batch(args,'smoke_'+args.run_id,True)
            admission.phase='formal';status['status']='formal_training_and_evaluation';save_status()
            module.batch(args,args.run_id,False)
        else:
            module.run_jobs=run_jobs
            admission.phase='smoke';status['status']='running_gpu_smoke';save_status()
            module.run_batch(args,'smoke_'+args.run_id,True)
            admission.phase='formal';status['status']='formal_training_and_evaluation';save_status()
            module.run_batch(args,args.run_id,False)
        status.update(status='complete',finished_unix=time.time())
    except BaseException as error:
        status.update(status='failed',error=repr(error),finished_unix=time.time())
        raise
    finally:
        save_status()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=('regional','context'),required=True)
    for key in ('source','control','data-root','clip-checkpoint','face-manifest-root'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('run-id','source-head','legacy-start'):p.add_argument('--'+key,required=True)
    p.add_argument('--reference-run',type=Path)
    p.add_argument('--legacy-pid',type=int,required=True);p.add_argument('--previous-pid',type=int,required=True)
    p.add_argument('--gpu',type=int,choices=[0],default=0)
    args=p.parse_args();args.control=args.control.resolve()
    if args.family=='regional' and args.reference_run is None:p.error('regional reference required')
    run(args)
