"""Continue all three spatial arms and their locked evaluations on GPU0 only."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from .pause_spatial_boundary import identity
from .run_spatial_summary import launch, wait_all
from .spatial_resume import sha256


def command_arguments(record, output, boundary):
    old = record['command']
    arguments = old[old.index('--training-protocol'):]
    position = arguments.index('--output-root')
    arguments[position + 1] = str(output)
    return arguments + ['--resume-spatial-boundary', str(boundary)]


def retire_paused(pause, source_root, destination):
    for arm, row in pause['processes'].items():
        old = torch.load(row['checkpoint'], map_location='cpu', weights_only=False)
        compact = torch.load(destination/arm/'train/compact_checkpoints'/f"task{pause['task']}.pth",
                             map_location='cpu', weights_only=False)
        assert old['model'].keys() == compact['model'].keys()
        assert all(torch.equal(v, compact['model'][n]) for n, v in old['model'].items()), arm
        assert sha256(row['checkpoint']) == row['sha256']
    pids = [(r['pid'], 'multi_lane.track_a.runner') for r in pause['processes'].values()]
    pids.append((pause['controller'], 'multi_lane.track_a.run_spatial_summary'))
    for pid, expected in pids:
        identity(pid, expected, source_root)
        state = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[0]
        if state not in ('T', 't'):
            raise RuntimeError('Refusing to terminate a process that is not paused')
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
    for _ in range(50):
        alive = []
        for pid, _ in pids:
            path = Path('/proc')/str(pid)/'stat'
            if path.exists() and path.read_text().rsplit(')', 1)[1].split()[0] != 'Z':
                alive.append(pid)
        if not alive:
            break
        time.sleep(.1)
    if alive:
        raise RuntimeError(f'Old processes did not exit: {alive}')
    return dict(status='old_processes_retired_after_equal_restoration', pids=[p for p, _ in pids],
                recovered_models_bitwise_equal=True, source_checkpoint_sha_unchanged=True)


def run(args):
    root = Path('output/target_anchored_summarization') / args.run_id
    logs = Path('logs/target_anchored_summarization') / args.run_id
    if root.exists() or logs.exists():
        raise FileExistsError('Run ID exists')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Clean worktree required')
    source = args.source_root / 'output/target_anchored_summarization' / args.source_run_id
    original = json.loads((source/'launch.json').read_text())
    assert bool(original['smoke']) == args.smoke
    tasks = 2 if args.smoke else 8
    assert original['tasks'] == tasks
    root.mkdir(parents=True); logs.mkdir(parents=True)
    try:
        pause = None
        if not args.smoke:
            if args.pause_record is None:
                raise ValueError('A validated boundary pause record is required')
            print('WAITING_FOR_TASK3_BOUNDARIES', flush=True)
            while True:
                if args.pause_record.exists():
                    pause = json.loads(args.pause_record.read_text())
                    if pause['status'] == 'paused_at_saved_boundary':
                        break
                    if pause['status'].startswith('pause_failed'):
                        raise RuntimeError('Boundary pause failed')
                time.sleep(15)
            assert pause['task'] == args.boundary_task == 3
            for arm, row in pause['processes'].items():
                path = source/arm/'train/recovery_checkpoints/task3_before_reporting.pth'
                assert path.resolve() == Path(row['checkpoint']).resolve() and sha256(path) == row['sha256']
        free = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                                            '--format=csv,noheader,nounits'],text=True).strip())
        if free < 14000:
            raise RuntimeError('Need at least 14000 MiB free on GPU0 for three capped workers')
        jobs, records = [], []
        for record in original['train']:
            arm = record['phase'].removeprefix('train_')
            boundary = source/arm/'train/recovery_checkpoints'/f'task{args.boundary_task}_before_reporting.pth'
            process, new = launch('multi_lane.track_a.runner', command_arguments(record,root/arm/'train',boundary),
                                  0,logs/arm/'train.log','train_'+arm)
            jobs.append(process); records.append(new)
        (root/'launch.json').write_text(json.dumps(dict(seed=0,tasks=tasks,smoke=args.smoke,
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_launch=original,source_run=str(source),boundary_task=args.boundary_task,
            train=records,all_workers_gpu0=True,no_training_updates_repeated=True),indent=2)+'\n')
        if pause is not None:
            while True:
                ready = all((root/r['phase'].removeprefix('train_')/'train/compact_checkpoints/task3.pth').exists()
                            and 'task=3 val_mAP=' in Path(r['log']).read_text()
                            for r in records)
                if ready:
                    retired = retire_paused(pause,args.source_root,root)
                    (root/'migration_verification.json').write_text(json.dumps(retired,indent=2)+'\n')
                    old_run = args.source_root/'output/target_anchored_summarization'/args.source_run_id
                    (old_run/'migration_to_gpu0.json').write_text(json.dumps({**retired,
                        'destination':str(root.resolve()),'status':'continued_on_gpu0','boundary_task':3},indent=2)+'\n')
                    print('GPU0_MIGRATION_VERIFIED_OLD_GPUS_RELEASED',flush=True)
                    break
                if any(p.poll() is not None and p.returncode != 0 for p in jobs):
                    raise RuntimeError('Resume failed; old paused processes and snapshots retained')
                time.sleep(5)
        wait_all(jobs,records,logs/'training_status.json')
        verify = [sys.executable,'-m','multi_lane.track_a.verify_spatial_summary',str(root),'--tasks',str(tasks)]
        if args.smoke:
            verify.append('--smoke')
        with (logs/'verification.log').open('x') as stream:
            subprocess.run(verify,stdout=stream,stderr=subprocess.STDOUT,check=True)
        if args.smoke:
            comparison=[]
            for arm in ('baseline','absolute','target'):
                for task in range(tasks):
                    old=torch.load(source/arm/'train/compact_checkpoints'/f'task{task}.pth',map_location='cpu',weights_only=False)['model']
                    new=torch.load(root/arm/'train/compact_checkpoints'/f'task{task}.pth',map_location='cpu',weights_only=False)['model']
                    assert old.keys()==new.keys() and all(torch.equal(v,new[n]) for n,v in old.items()),(arm,task)
                    comparison.append(dict(arm=arm,task=task,method_state_bitwise_equal=True))
            (root/'uninterrupted_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
        jobs,records=[],[]
        splits=[('val','full_val'),('val','replay_val')] if args.smoke else [('val','full_val'),('test','full_test')]
        for arm in ('baseline','absolute','target'):
            config=json.loads((root/arm/'train/config.json').read_text())
            for split,folder in splits:
                arguments=['--source-run',root/arm/'train','--output',root/arm/folder,
                    '--split',split,'--expected-tasks',str(tasks),'--data-root',config['data_root'],
                    '--clip-checkpoint',config['clip_checkpoint'],'--face-manifest-root',config['face_manifest_root'],
                    '--batch-size','64','--workers','0']
                if folder!='replay_val':arguments+=['--full-cohort']
                process,record=launch('multi_lane.track_a.evaluate_moe_task_ffn',arguments,0,logs/arm/(folder+'.log'),arm+'_'+folder)
                jobs.append(process);records.append(record)
        wait_all(jobs,records,logs/'evaluation_status.json')
        checks=[]
        for arm in ('baseline','absolute','target'):
            for split,folder in splits:
                result=json.loads((root/arm/folder/'result.json').read_text())
                assert len(result['task_metrics'])==tasks
                if folder!='replay_val':assert result['full_cohort'] and result['cohort_samples']==(2397 if split=='val' else 5368)
                else:
                    import numpy as np
                    for task in range(tasks):
                        with np.load(root/arm/'train/val_scores'/f'task{task}.npz') as old,np.load(root/arm/folder/'scores'/f'task{task}.npz') as new:
                            for key in ('sample_ids','targets','class_indices','logits','probabilities'):assert np.array_equal(old[key],new[key]),(arm,task,key)
                checks.append(dict(arm=arm,split=split,folder=folder,summary=result['summary']))
        (root/'evaluation_verification.json').write_text(json.dumps(dict(status='verified',checks=checks),indent=2)+'\n')
        config=json.loads((root/'target/train/config.json').read_text())
        arguments=['--source-run',root/'target/train','--output',root/'target/spatial_attribution_val',
            '--data-root',config['data_root'],'--clip-checkpoint',config['clip_checkpoint'],
            '--face-manifest-root',config['face_manifest_root'],'--checkpoint-tasks']
        arguments+=['0','1','--smoke-batches','2'] if args.smoke else ['0','2','7']
        process,record=launch('multi_lane.track_a.evaluate_spatial_attribution',arguments,0,logs/'attribution.log','attribution')
        wait_all([process],[record],logs/'attribution_status.json')
        (root/'completion.json').write_text(json.dumps(dict(status='complete',tasks=tasks,smoke=args.smoke,
            all_workers_gpu0=True,validation_and_test_complete=not args.smoke,test_used_for_selection=False),indent=2)+'\n')
        (logs/'batch_exit_status.txt').write_text('0\n')
        print('GPU0_CONTINUATION_COMPLETE',flush=True)
    except BaseException:
        (logs/'batch_exit_status.txt').write_text('1\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--source-run-id',required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--boundary-task',type=int,required=True)
    p.add_argument('--pause-record',type=Path)
    p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
