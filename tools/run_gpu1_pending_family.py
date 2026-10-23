"""Keep inherited trainers on GPU0; route only not-yet-started jobs to GPU1."""
import argparse
import csv
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gpu0_shared_admission import write
from gpu1_admission import Admission,inherited_exit,process_state


def read(path):return json.loads(path.read_text())


def main(args):
    manifest=read(args.manifest);source=Path(manifest['source']);os.chdir(source);sys.path.insert(0,str(source))
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==manifest['source_head']
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    assert read(Path(manifest['smoke'])/'completion.json')['status']=='complete'
    family=manifest['family'];kind='target_region_summary' if family=='regional' else 'summary_context_attention'
    module=importlib.import_module('multi_lane.track_a.'+('run_regional_summary' if family=='regional' else 'run_summary_context'))
    root=Path('output')/kind/args.run_id;logs=Path('logs')/kind/args.run_id
    admission=Admission(args.control,family)
    protocol=read(root/('launch.json' if family=='regional' else 'protocol.json'))
    status=dict(status='training',family=family,pid=os.getpid(),source_head=manifest['source_head'],
                control_head=args.control_head,adopted_gpu0=[x['pid'] for x in manifest['active']],pending_gpu=1,
                original_parent=manifest['parent_pid'],optimizer_updates_restarted=0)
    status_path=args.control/(family+'_status.json')
    def save():
        write(status_path,status);write(Path(manifest['queue_status']),status)
    save()
    parent_released=False
    def release_parent():
        nonlocal parent_released
        if parent_released:return
        state=process_state(manifest['parent_pid'])
        assert state and state['start']==manifest['parent_start'] and state['state']=='T'
        os.kill(manifest['parent_pid'],signal.SIGTERM)
        os.kill(manifest['parent_pid'],signal.SIGCONT)
        parent_released=True
        status['old_controller_released']=True;save()

    def run_jobs(jobs,path,adopted=()):
        active=[];records=[];pending=list(jobs);failed=False
        standard_path=logs/({'training_gpu01.json':'training_queue.json',
                             'evaluation_gpu1.json':'evaluation_queue.json',
                             'attribution_gpu1.json':'attribution_queue.json'}[path.name])
        def save_jobs():
            value=dict(records=records,pending=pending,failed=failed,actual_gpu_routing=True)
            write(path,value);write(standard_path,value)
        for item in adopted:
            match=[j for j in pending if j['name']==item['name']]
            assert len(match)==1 and match[0]['arguments']==item['arguments']
            pending.remove(match[0]);record=dict(item,gpu=0,adopted=True)
            records.append(record);active.append((None,record))
        inherited_remaining=len(adopted)
        if adopted:
            write(root/'gpu01_training_handoff.json',dict(manifest=manifest,adopted=records,pending_gpu1=pending))
        while pending or active:
            for process,record in list(active):
                code=inherited_exit(record['pid'],record['start']) if process is None else process.poll()
                if code is not None:
                    record.update(exit=code,finished_unix=time.time());active.remove((process,record));failed|=code!=0
                    if process is None:inherited_remaining-=1
                    else:admission.release(record['lease_key'],code)
                    print('FINISH',record['name'],'gpu',record['gpu'],'exit',code,flush=True)
            if adopted and inherited_remaining==0 and not parent_released:release_parent()
            if failed and not active:
                save_jobs();raise RuntimeError('Job failed; outputs preserved')
            if not failed and pending:
                job=pending[0];key=family+'/'+str(path)+'/'+job['name']
                job_kind='train' if job['module'].endswith('.runner') else 'eval'
                log=logs/(job['name']+'.log')
                code=("import runpy,torch;torch.cuda.set_per_process_memory_fraction(.16);"
                      f"runpy.run_module({job['module']!r},run_name='__main__')")
                command=[sys.executable,'-u','-c',code,*map(str,job['arguments'])]
                process=admission.acquire(key,job_kind,command,log)
                if process is not None:
                    pending.pop(0);record=dict(job,pid=process.pid,gpu=1,lease_key=key,command=command,
                                              log=str(log),started_unix=time.time(),adopted=False)
                    records.append(record);active.append((process,record));print('START',job['name'],'gpu1',process.pid,flush=True)
            save_jobs()
            if pending or active:time.sleep(5)
        return records

    try:
        arms=[x[0] for x in module.ARMS] if family=='regional' else list(module.ARMS)
        jobs=[dict(name=arm+'/train',module='multi_lane.track_a.runner',arguments=module.training_arguments(args,root,arm,False)) for arm in arms]
        for job in jobs:assert job['arguments']==protocol['training_arguments'][job['name'].split('/')[0]]
        records=run_jobs(jobs,logs/'training_gpu01.json',manifest['active'])
        write(root/'resource_override_gpu01.json',dict(source_head=manifest['source_head'],control_head=args.control_head,
            training_devices={x['name']:x['gpu'] for x in records},future_evaluation_gpu=1,allocator_fraction=.16,
            no_training_restarted=True,original_protocol_preserved=True))
        status['status']='verification';save()
        if family=='regional':
            importlib.import_module('multi_lane.track_a.verify_spatial_summary').verify(root,8,False,module.ARMS)
        else:
            for arm in arms:
                with (logs/arm/'verification_gpu01.log').open('x') as stream:
                    subprocess.run([sys.executable,'-m','multi_lane.track_a.verify_joint_summary',str(root/arm),
                        '--expected-tasks','8','--baseline',str(root/'baseline/train')],stdout=stream,stderr=subprocess.STDOUT,check=True)
        status['status']='validation_and_test';save();jobs=[]
        for arm in arms:
            for split in ('val','test'):
                folder='full_val' if split=='val' else 'full_test'
                if family=='context':
                    jobs.append(module.evaluation_job(args,root,arm,split,folder,False,['--full-cohort']))
                else:
                    jobs.append(module.job(arm+'/'+folder,'evaluate_moe_task_ffn',[
                        '--source-run',root/arm/'train','--output',root/arm/folder,'--split',split,
                        '--expected-tasks',8,'--batch-size',64,'--workers',0,*module.resources(args),'--full-cohort']))
        run_jobs(jobs,logs/'evaluation_gpu1.json')
        summaries={arm:{folder:read(root/arm/folder/'result.json')['summary'] for folder in ('full_val','full_test')} for arm in arms}
        if family=='context':
            write(root/'summary.json',dict(results=summaries,test_used_for_selection=False))
            jobs=[dict(name=arm+'/attribution_val',module='multi_lane.track_a.evaluate_cross_view_attribution',arguments=list(map(str,[
                '--source-run',root/arm/'train','--output',root/arm/'attribution_val','--split','val',
                '--data-root',args.data_root,'--clip-checkpoint',args.clip_checkpoint,'--face-manifest-root',args.face_manifest_root,
                '--batch-size',64,'--checkpoint-tasks',0,2,7]))) for arm in arms[1:]]
        else:
            rows=[]
            for arm in arms:
                for split,folder in [('val','full_val'),('test','full_test')]:
                    value=read(root/arm/folder/'result.json')
                    assert len(value['task_metrics'])==8 and value['full_cohort']
                    assert value['cohort_samples']==(2397 if split=='val' else 5368)
                    rows.append(dict(arm=arm,split=split,folder=folder,**value['summary']))
            write(root/'evaluation_verification.json',dict(status='verified',summaries=rows))
            with (root/'summary.csv').open('x',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            jobs=[module.job(arm+'/attribution','evaluate_spatial_attribution',[
                '--source-run',root/arm/'train','--output',root/arm/'spatial_attribution_val',*module.resources(args),
                '--checkpoint-tasks','0','2','7']) for arm in ('old_bias','regional')]
        status['status']='validation_attribution';save();run_jobs(jobs,logs/'attribution_gpu1.json')
        protocol.update(status='complete',validation_and_test_complete=True,
                        actual_training_gpus=sorted({x['gpu'] for x in records}),actual_pending_and_evaluation_gpu=1)
        write(root/'completion.json',protocol);(logs/'batch_exit_status.txt').write_text('0\n')
        status.update(status='complete',finished_unix=time.time());save()
    except BaseException as error:
        status.update(status='failed',error=repr(error));save();write(root/'failure_gpu01.json',status)
        write(root/'failure.json',status);(logs/'batch_exit_status.txt').write_text('1\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','control','data-root','clip-checkpoint','face-manifest-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-run',type=Path)
    p.add_argument('--run-id',required=True);p.add_argument('--control-head',required=True)
    args=p.parse_args();args.manifest=args.manifest.resolve();args.control=args.control.resolve();main(args)
