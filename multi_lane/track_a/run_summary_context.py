"""GPU0-only FIFO experiment, following a declared existing experiment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ARMS = ('baseline', 'legacy_task_residual', 'task_context', 'person_context')
MODES = dict(zip(ARMS, ('disabled', 'task_full', 'task_full_context', 'person_full_context')))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def dependency_ready(root):
    if (root / 'failure.json').exists():
        raise RuntimeError('Preceding experiment failed; inspect its preserved artifacts')
    preceding_queue = root.parent / ('queue_' + root.name) / 'status.json'
    if preceding_queue.exists() and read(preceding_queue).get('status') == 'failed':
        raise RuntimeError('Preceding experiment queue failed; inspect its preserved status')
    completion = root / 'completion.json'
    if not completion.exists():
        return False
    value = read(completion)
    if (value.get('status') != 'complete' or value.get('smoke') is not False
            or value.get('tasks') != 8 or value.get('validation_and_test_complete') is not True):
        raise ValueError('Dependency must be a completed formal eight-task experiment')
    return True


def gpu_state():
    values = subprocess.check_output(['nvidia-smi', '-i', '0',
        '--query-gpu=uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    uuid, free, util = (x.strip() for x in values.strip().split(','))
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                                    '--format=csv,noheader'], text=True).splitlines()
    return dict(free_mib=int(free), utilization=int(util), compute_busy=uuid in [x.strip() for x in apps])


def training_arguments(args, root, arm, smoke):
    values = ['--training-protocol', 'incremental', '--moe-paired-audit', '--seed', '0',
        '--data-root', args.data_root, '--clip-checkpoint', args.clip_checkpoint,
        '--face-manifest-root', args.face_manifest_root, '--output-root', root / arm / 'train',
        '--epochs', '30', '--train-batch-size', '64', '--eval-batch-size', '64',
        '--workers', '0', '--eval-workers', '0', '--save-pre-evaluation-recovery',
        '--amp-initial-scale', '1024', '--amp-growth-interval', '1000000000',
        '--source-learning-rate', '.05', '--source-reference-batch-size', '256', '--weight-decay', '0',
        '--scheduler-mode', 'cosine', '--scheduler-min-lr-ratio', '0', '--scheduler-warmup-ratio', '0',
        '--temperature', '1', '--threshold', '.5', '--training-loss-mode', 'legacy_full_zero',
        '--loss-routing', 'joint_bce', '--supervised-loss-scale', '1', '--input-mode', 'full',
        '--input-normalization', 'clip', '--train-crop-scale', '.05', '1', '--full-crop-mode', 'legacy',
        '--person-transform-mode', 'letterbox', '--person-crop-margin', '.15',
        '--person-color-jitter-strength', '0', '--person-color-jitter-probability', '0',
        '--selector-mode', 'shared', '--prompt-mode', 'shared', '--num-selectors', '10',
        '--view-fusion', 'fixed_three_view', '--view-classifier-mode', 'shared_post_fusion',
        '--view-auxiliary-loss-weight', '.1', '--view-gradient-routing', 'joint',
        '--adapter-mode', 'disabled', '--parax-mode', 'disabled', '--joint-summary-mode', 'disabled',
        '--selector-conditioning', 'disabled', '--cross-view-mode', MODES[arm],
        '--cross-view-layer', '5', '--cross-view-dim', '64', '--cross-view-heads', '4',
        '--cross-view-scale', '.1', '--cross-view-context-max-mass', '.1',
        '--view-fusion-learning-rate', '.0004', '--skip-validation-eval', '--reporting-split', 'val',
        '--save-compact-checkpoints', '--no-save-checkpoints', '--max-tasks', '2' if smoke else '8']
    if smoke:
        values += ['--optimizer-updates-per-task', '2']
    return list(map(str, values))


def evaluation_job(args, root, arm, split, folder, smoke, extra=()):
    return dict(name=arm + '/' + folder, module='multi_lane.track_a.evaluate_moe_task_ffn', arguments=list(map(str, [
        '--source-run', root/arm/'train', '--output', root/arm/folder, '--split', split,
        '--data-root', args.data_root, '--clip-checkpoint', args.clip_checkpoint,
        '--face-manifest-root', args.face_manifest_root, '--batch-size', '64', '--workers', '0',
        '--expected-tasks', '2' if smoke else '8', *extra])))


def run_jobs(jobs, logs, status):
    pending, active, records = list(jobs), [], []
    failed = False
    while pending or active:
        for process, record in list(active):
            code = process.poll()
            if code is not None:
                record.update(exit=code, finished_unix=time.time())
                active.remove((process, record)); failed |= code != 0
                print('FINISH', record['name'], code, flush=True)
        if failed and not active:
            write(status, dict(records=records, pending=pending, failed=True))
            raise RuntimeError('Job failed; remaining jobs and output preserved')
        if not failed and pending and len(active) < 2 and gpu_state()['free_mib'] >= 8500:
            job = pending.pop(0)
            log = logs/(job['name']+'.log'); log.parent.mkdir(parents=True, exist_ok=True)
            code = ("import runpy,torch;torch.cuda.set_per_process_memory_fraction(.3);"
                    f"runpy.run_module({job['module']!r},run_name='__main__')")
            command = [sys.executable, '-u', '-c', code, *job['arguments']]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            with log.open('x') as stream:
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            record = dict(job, pid=process.pid, command=command, gpu=0, log=str(log), started_unix=time.time())
            records.append(record); active.append((process, record))
            print('START', job['name'], process.pid, flush=True)
        write(status, dict(records=records, pending=pending, failed=failed))
        if pending or active:
            time.sleep(5)


def run_batch(args, run_id, smoke):
    root = Path('output/summary_context_attention')/run_id
    logs = Path('logs/summary_context_attention')/run_id
    if root.exists() or logs.exists():
        raise FileExistsError('Never overwrite an existing run')
    if shutil.disk_usage('.').free < 20 * 1024**3:
        raise RuntimeError('Need 20 GiB free disk')
    root.mkdir(parents=True); logs.mkdir(parents=True)
    protocol = dict(status='running', seed=0, tasks=2 if smoke else 8, smoke=smoke, gpu=0,
                    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
                    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    training_arguments={arm:training_arguments(args, root, arm, smoke) for arm in ARMS},
                    test_used_for_selection=False, auxiliary_weight=.1, calibration_fraction=0,
                    train_parallelism=2, eval_parallelism=2, allocator_fraction=.3)
    write(root/'protocol.json', protocol)
    try:
        jobs = [dict(name=arm+'/train', module='multi_lane.track_a.runner',
                     arguments=protocol['training_arguments'][arm]) for arm in ARMS]
        run_jobs(jobs, logs, logs/'training_queue.json')
        for arm in ARMS:
            command=[sys.executable,'-m','multi_lane.track_a.verify_joint_summary',str(root/arm),
                     '--expected-tasks',str(protocol['tasks']),'--baseline',str(root/'baseline/train')]
            with (logs/arm/'verification.log').open('x') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        jobs=[]
        for arm in ARMS:
            jobs.append(evaluation_job(args,root,arm,'val','full_val',smoke,['--full-cohort']))
            jobs.append(evaluation_job(args,root,arm,'val' if smoke else 'test',
                                       'replay_val' if smoke else 'full_test',smoke,
                                       [] if smoke else ['--full-cohort']))
        run_jobs(jobs,logs,logs/'evaluation_queue.json')
        if smoke:
            for arm in ARMS:
                a=read(root/arm/'train/seed_summary.json')['task_metrics']
                b=read(root/arm/'replay_val/result.json')['task_metrics']
                assert len(a)==len(b)==2
                assert all(abs(x['mAP']-y['mAP'])<1e-7 for x,y in zip(a,b)), arm
            jobs=[evaluation_job(args,root,arm,'val','context_off',True,
                  ['--full-cohort','--disable-cross-view-runtime']) for arm in ARMS[1:]]
        else:
            jobs=[dict(name=arm+'/attribution_val',module='multi_lane.track_a.evaluate_cross_view_attribution',
                       arguments=list(map(str,['--source-run',root/arm/'train','--output',root/arm/'attribution_val',
                         '--split','val','--data-root',args.data_root,'--clip-checkpoint',args.clip_checkpoint,
                         '--face-manifest-root',args.face_manifest_root,'--batch-size',64,
                         '--checkpoint-tasks',0,2,7]))) for arm in ARMS[1:]]
        run_jobs(jobs,logs,logs/'attribution_queue.json')
        summaries={arm:{split:read(root/arm/split/'result.json')['summary']
                       for split in ('full_val','replay_val' if smoke else 'full_test')} for arm in ARMS}
        write(root/'summary.json',dict(results=summaries,test_used_for_selection=False))
        protocol.update(status='complete',validation_and_test_complete=not smoke,
                        validation_replay_verified=smoke)
        write(root/'completion.json',protocol)
        (logs/'batch_exit_status.txt').write_text('0\n')
        print('SUMMARY_CONTEXT_COMPLETE',run_id,flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error)))
        (logs/'batch_exit_status.txt').write_text('1\n')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--after-run',type=Path,required=True)
    for field in ('data-root','clip-checkpoint','face-manifest-root'):
        parser.add_argument('--'+field,type=Path,required=True)
    args=parser.parse_args()
    if not args.run_id or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in args.run_id):
        parser.error('Invalid run ID')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Clean versioned code required')
    for item in (args.data_root/'CVPR17_Annotations.mat',args.clip_checkpoint,
                 *(args.face_manifest_root/'manifests'/f'{split}.jsonl' for split in ('train','val','test'))):
        if not item.is_file(): raise FileNotFoundError(item)
    control=Path('logs/summary_context_attention')/(args.run_id+'_queue')
    control.mkdir(parents=True,exist_ok=True)
    lock=(control/'lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    record=Path('output/summary_context_attention')/(args.run_id+'_queue')
    record.mkdir(parents=True,exist_ok=True)
    # A recorded successful CPU suite is required before any queued GPU work.
    tests=Path('logs/summary_context_attention/implementation_20261023/cpu_tests.json')
    test_record=read(tests)
    expected_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if test_record.get('status')!='passed' or test_record.get('head')!=expected_head:
        raise RuntimeError('CPU test record does not match queued HEAD')
    stable=0; deadline=time.monotonic()+72*3600
    while True:
        if time.monotonic()>deadline: raise TimeoutError('GPU0 queue exceeded 72 hours')
        ready=dependency_ready(args.after_run)
        state=gpu_state()
        idle=state['free_mib']>=22000 and state['utilization']<=5 and not state['compute_busy']
        stable=stable+1 if ready and idle else 0
        value=dict(status='waiting_for_preceding_experiment' if not ready else 'waiting_for_gpu0',
                   dependency=str(args.after_run.resolve()),dependency_complete=ready,gpu=0,
                   gpu_state=state,stable_idle_checks=stable,head=expected_head,
                   checked_unix=time.time(),pid=os.getpid(),cpu_tests=test_record)
        write(record/'status.json',value)
        if stable>=3: break
        print('QUEUED',value['status'],state,flush=True);time.sleep(20)
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Code changed while queued')
    if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=expected_head:
        raise RuntimeError('HEAD changed while queued')
    value['status']='running_gpu_smoke';write(record/'status.json',value)
    run_batch(args,'smoke_'+args.run_id,True)
    value['status']='running_formal';write(record/'status.json',value)
    run_batch(args,args.run_id,False)
    value['status']='complete';write(record/'status.json',value)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        # Preserve failure status even when the preceding experiment failed
        # before this queue was allowed to create a GPU batch.
        if '--run-id' in sys.argv:
            index=sys.argv.index('--run-id')
            if index+1<len(sys.argv):
                name=sys.argv[index+1]
                if name and all(x in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in name):
                    root=Path('output/summary_context_attention')/(name+'_queue')
                    write(root/'failure.json',dict(status='failed',error=repr(error),finished_unix=time.time()))
                    path=root/'status.json'
                    state=read(path) if path.exists() else {}
                    state.update(status='failed',error=repr(error));write(path,state)
        raise
