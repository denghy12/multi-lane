"""GPU0-only, dependency-ordered regional summarization preflight and experiment."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ARMS = [('baseline', 'disabled'), ('old_bias', 'target'), ('regional', 'regional')]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def dependency_ready(path):
    """Do not mistake the other branch's smoke or training-only exit for completion."""
    if (path / 'failure.json').exists():
        raise RuntimeError('Preceding batch failed; preserve it and inspect ' + str(path))
    complete = path / 'completion.json'
    if not complete.exists():
        return False
    result = json.loads(complete.read_text())
    if result.get('status') != 'complete' or result.get('smoke') is not False or result.get('tasks') != 8:
        raise ValueError('Dependency is not a complete formal eight-task batch')
    return True


def free_memory():
    return int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True).strip())


def run_queue(jobs, logs, status, parallelism=3):
    pending, active, records = list(jobs), [], []
    failed, last_wait = False, 0
    while pending or active:
        for process, record in list(active):
            code = process.poll()
            if code is not None:
                record.update(exit=code, finished_unix=time.time())
                active.remove((process, record)); failed |= code != 0
                print(f'FINISH {record["name"]} exit={code}', flush=True)
        if failed and not active:
            write(status, dict(status='failed', records=records, pending=pending))
            raise RuntimeError('Job failed; pending jobs preserved in ' + str(status))
        if not failed and pending and len(active) < parallelism:
            free = free_memory()
            if free >= 5600:
                job = pending.pop(0)
                log = logs / (job['name'] + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1')
                code = ("import runpy,torch;torch.cuda.set_per_process_memory_fraction(.16);"
                        f"runpy.run_module({job['module']!r},run_name='__main__')")
                command = [sys.executable, '-u', '-c', code, *map(str, job['arguments'])]
                with log.open('x') as stream:
                    process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                record = dict(job, gpu=0, pid=process.pid, command=command, log=str(log),
                              allocator_fraction=.16, initial_free_mib=free, started_unix=time.time())
                records.append(record); active.append((process, record))
                print(f'START {job["name"]} gpu=0 pid={process.pid}', flush=True)
            elif time.time() - last_wait > 60:
                print(f'WAIT_GPU0 free_mib={free} required=5600', flush=True); last_wait = time.time()
        write(status, dict(status='running' if pending or active else 'complete', records=records, pending=pending))
        if pending or active:
            time.sleep(5)
    return records


def training_arguments(args, root, arm, smoke):
    mode = dict(ARMS)[arm]
    result = ['--training-protocol', 'incremental', '--moe-paired-audit', '--seed', '0',
        '--data-root', args.data_root, '--clip-checkpoint', args.clip_checkpoint,
        '--face-manifest-root', args.face_manifest_root, '--output-root', root/arm/'train',
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
        '--view-fusion-learning-rate', '.0004', '--adapter-mode', 'disabled', '--parax-mode', 'disabled',
        '--joint-summary-mode', 'disabled', '--cross-view-mode', 'disabled',
        '--calibration-fraction', '.1', '--save-calibration-scores', '--save-evaluation-scores',
        '--evaluation-score-purpose', 'validation_search', '--skip-validation-eval',
        '--reporting-split', 'val', '--save-compact-checkpoints', '--no-save-checkpoints',
        '--max-tasks', '2' if smoke else '8', '--spatial-summary-mode', mode]
    if smoke:
        result += ['--optimizer-updates-per-task', '2']
    return list(map(str, result))


def resources(args):
    return ['--data-root', str(args.data_root), '--clip-checkpoint', str(args.clip_checkpoint),
            '--face-manifest-root', str(args.face_manifest_root)]


def job(name, module, arguments):
    return dict(name=name, module='multi_lane.track_a.' + module, arguments=list(map(str, arguments)))


def verify_recovery(args, root, logs):
    import numpy as np
    import torch
    from .export_compact_test_scores import _sha256
    from .verify_spatial_summary import load_boundary
    source = root/'regional'/'train'
    paths = [source/'recovery_checkpoints'/f'task{t}_before_reporting.pth' for t in (0, 1)]
    hashes = {str(p): _sha256(p) for p in paths}
    arguments = training_arguments(args, root, 'regional', True)
    arguments[arguments.index('--output-root') + 1] = str(root/'recovery_probe'/'train')
    arguments += ['--resume-spatial-boundary', str(paths[0])]
    run_queue([job('recovery_probe', 'runner', arguments)], logs, logs/'recovery_queue.json', 1)
    def equal(a, b):
        if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, np.ndarray): return isinstance(b, np.ndarray) and np.array_equal(a, b)
        if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a == b
    checks = {}
    for task, path in enumerate(paths):
        original = load_boundary(path)
        replay = load_boundary(root/'recovery_probe'/'train'/'recovery_checkpoints'/path.name)
        for key in ('model', 'requires_grad', 'training_state', 'rng', 'sampler_generator'):
            checks[f'task{task}_{key}'] = equal(original[key], replay[key])
            assert checks[f'task{task}_{key}'], (task, key, 'Recovery changed trajectory')
        with np.load(source/'val_scores'/f'task{task}.npz') as old, np.load(root/'recovery_probe'/'train'/'val_scores'/f'task{task}.npz') as new:
            for key in ('sample_ids','targets','class_indices','logits','probabilities'):
                assert np.array_equal(old[key], new[key]), (task, key, 'Recovery reporting changed')
    assert all(_sha256(Path(p)) == sha for p,sha in hashes.items())
    write(root/'recovery_verification.json', dict(status='verified', checks=checks, source_sha256=hashes))


def batch(args, run_id, smoke):
    root = Path('output/target_region_summary')/run_id
    logs = Path('logs/target_region_summary')/run_id
    if root.exists() or logs.exists(): raise FileExistsError('Run ID exists: ' + run_id)
    root.mkdir(parents=True); logs.mkdir(parents=True)
    tasks = 2 if smoke else 8
    protocol = dict(status='running', seed=0, smoke=smoke, tasks=tasks, gpu=0, arms=ARMS,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        training_arguments={arm:training_arguments(args, root, arm, smoke) for arm,_ in ARMS},
        test_used_for_selection=False)
    write(root/'launch.json', protocol)
    try:
        run_queue([job(arm+'/train', 'runner', protocol['training_arguments'][arm]) for arm,_ in ARMS],
                  logs, logs/'training_queue.json')
        from .verify_spatial_summary import verify
        verify(root, tasks, smoke, ARMS)
        if smoke: verify_recovery(args, root, logs)
        jobs = []
        for arm,_ in ARMS:
            for split,folder in ([('val','full_val'),('val','replay_val')] if smoke else [('val','full_val'),('test','full_test')]):
                arguments = ['--source-run', root/arm/'train', '--output', root/arm/folder,
                    '--split', split, '--expected-tasks', tasks, '--batch-size', 64, '--workers', 0, *resources(args)]
                if folder != 'replay_val': arguments += ['--full-cohort']
                jobs.append(job(arm+'/'+folder, 'evaluate_moe_task_ffn', arguments))
        run_queue(jobs, logs, logs/'evaluation_queue.json')
        import numpy as np
        summaries = []
        for arm,_ in ARMS:
            for split,folder in ([('val','full_val'),('val','replay_val')] if smoke else [('val','full_val'),('test','full_test')]):
                result = json.loads((root/arm/folder/'result.json').read_text())
                assert len(result['task_metrics']) == tasks
                if folder == 'replay_val':
                    for task in range(tasks):
                        with np.load(root/arm/'train'/'val_scores'/f'task{task}.npz') as old, np.load(root/arm/folder/'scores'/f'task{task}.npz') as new:
                            for key in ('sample_ids','targets','class_indices','logits','probabilities'):
                                assert np.array_equal(old[key],new[key]), (arm,task,key,'Replay failed')
                else:
                    assert result['full_cohort'] and result['cohort_samples'] == (2397 if split == 'val' else 5368)
                summaries.append(dict(arm=arm,split=split,folder=folder,**result['summary']))
        write(root/'evaluation_verification.json',dict(status='verified',summaries=summaries))
        with (root/'summary.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
        controls = []
        for arm in ('old_bias','regional'):
            arguments = ['--source-run',root/arm/'train','--output',root/arm/'spatial_attribution_val',
                         *resources(args),'--checkpoint-tasks',*(['0','1'] if smoke else ['0','2','7'])]
            if smoke: arguments += ['--smoke-batches','2']
            controls.append(job(arm+'/attribution','evaluate_spatial_attribution',arguments))
        run_queue(controls, logs, logs/'attribution_queue.json',2)
        protocol.update(status='complete',validation_and_test_complete=not smoke)
        write(root/'completion.json',protocol); (logs/'batch_exit_status.txt').write_text('0\n')
        return root
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error),pending='see logs/*queue.json'))
        (logs/'batch_exit_status.txt').write_text('1\n'); raise


def main(args):
    if not args.run_id or '/' in args.run_id or args.run_id in ('.','..'):
        raise ValueError('Invalid run ID')
    if args.gpu != 0: raise ValueError('GPU0 only')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Clean Git checkout required')
    if shutil.disk_usage('.').free < 12*1024**3: raise RuntimeError('Need 12 GiB free storage')
    queue = Path('output/target_region_summary')/('queue_'+args.run_id)
    if queue.exists(): raise FileExistsError(queue)
    queue.mkdir(parents=True)
    protocol = dict(status='waiting_for_preceding_batch',gpu=0,run_id=args.run_id,
        dependency=str(args.after_batch),seed=0,tasks=8,
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        stages=['real_initialization','two_task_smoke','exact_recovery','full_smoke_validation',
                'formal_eight_task_training','full_validation_and_test','validation_attribution'])
    write(queue/'status.json',protocol)
    try:
        last = 0
        while not dependency_ready(args.after_batch):
            if time.time()-last > 60:
                print('WAIT_PRECEDING_BATCH '+str(args.after_batch),flush=True);last=time.time()
            protocol['checked_unix']=time.time();write(queue/'status.json',protocol);time.sleep(30)
        if subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()!=protocol['head']:
            raise RuntimeError('Queued code version changed')
        if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
            raise RuntimeError('Queued worktree was modified')
        protocol['status']='preflight';write(queue/'status.json',protocol)
        logs=Path('logs/target_region_summary')/('preflight_'+args.run_id)
        arguments=['--reference-run',args.reference_run,'--output',queue/'initialization.json',*resources(args)]
        run_queue([job('real_initialization','audit_spatial_initialization',arguments)],logs,logs/'initialization_queue.json',1)
        assert json.loads((queue/'initialization.json').read_text())['status']=='verified'
        smoke=batch(args,'smoke_'+args.run_id,True)
        assert json.loads((smoke/'completion.json').read_text())['status']=='complete'
        protocol['status']='formal_training_and_evaluation';write(queue/'status.json',protocol)
        formal=batch(args,args.run_id,False)
        protocol.update(status='complete',formal_output=str(formal));write(queue/'status.json',protocol)
        print('REGIONAL_SUMMARY_COMPLETE '+str(formal),flush=True)
    except BaseException as error:
        protocol.update(status='failed',error=repr(error));write(queue/'status.json',protocol);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--gpu',type=int,choices=[0],default=0)
    for key in ('after-batch','reference-run','data-root','clip-checkpoint','face-manifest-root'):
        parser.add_argument('--'+key,type=Path,required=True)
    main(parser.parse_args())
