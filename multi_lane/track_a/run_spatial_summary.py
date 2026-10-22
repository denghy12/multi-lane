"""Locked three-arm seed0 run; use one memory-capped process per chosen GPU."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def launch(module, arguments, gpu, log, phase):
    # Leave room for CUDA context overhead and external workloads. Never stop them.
    last = None
    while True:
        free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                                            '--format=csv,noheader,nounits'], text=True).strip())
        if free >= 5600:
            break
        if last != free:
            print(f'WAIT_GPU phase={phase} gpu={gpu} free_mib={free} required=5600', flush=True)
            last = free
        time.sleep(30)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1')
    code = ("import runpy,torch;torch.cuda.set_per_process_memory_fraction(.16);"
            f"runpy.run_module('{module}',run_name='__main__')")
    command = [sys.executable, '-u', '-c', code, *map(str, arguments)]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x') as stream:
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
    print(f'START phase={phase} gpu={gpu} pid={process.pid} log={log}', flush=True)
    return process, dict(phase=phase, gpu=gpu, pid=process.pid, command=command, log=str(log),
                         cuda_allocator_fraction=.16, initial_free_mib=free)


def wait_all(jobs, records, path):
    failed = False
    for process, record in zip(jobs, records):
        record['exit'] = process.wait()
        failed |= record['exit'] != 0
        path.write_text(json.dumps(records, indent=2) + '\n')
        print(f'FINISH phase={record["phase"]} pid={record["pid"]} exit={record["exit"]}', flush=True)
    if failed:
        raise RuntimeError(f'At least one process failed; see {path}')


def run(args):
    assert args.run_id and '/' not in args.run_id
    root = Path('output/target_anchored_summarization') / args.run_id
    logs = Path('logs/target_anchored_summarization') / args.run_id
    if root.exists() or logs.exists():
        raise FileExistsError('Run ID already exists')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Clean Git checkout required')
    if shutil.disk_usage('.').free < 8 * 1024**3:
        raise RuntimeError('Need at least 8 GiB of free output storage')
    assert len(args.train_gpus) == len(set(args.train_gpus)) == 3
    assert len(args.eval_gpus) == len(set(args.eval_gpus)) == 6
    root.mkdir(parents=True); logs.mkdir(parents=True)
    status = logs / 'batch_exit_status.txt'
    try:
        tasks = 2 if args.smoke else 8
        common = ['--training-protocol', 'incremental', '--moe-paired-audit', '--seed', '0',
                  '--data-root', args.data_root, '--clip-checkpoint', args.clip_checkpoint,
                  '--face-manifest-root', args.face_manifest_root,
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
                  '--max-tasks', str(tasks)]
        if args.smoke:
            common += ['--optimizer-updates-per-task', '2']
        jobs, records = [], []
        for (arm, mode), gpu in zip([('baseline', 'disabled'), ('absolute', 'absolute'), ('target', 'target')], args.train_gpus):
            process, record = launch('multi_lane.track_a.runner',
                                      common + ['--spatial-summary-mode', mode, '--output-root', root/arm/'train'],
                                      gpu, logs/arm/'train.log', 'train_'+arm)
            jobs.append(process); records.append(record)
        (root/'launch.json').write_text(json.dumps(dict(seed=0, tasks=tasks, smoke=args.smoke,
             git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
             train=records, no_binding_loss=True), indent=2)+'\n')
        wait_all(jobs, records, logs/'training_status.json')
        command = [sys.executable, '-m', 'multi_lane.track_a.verify_spatial_summary', str(root), '--tasks', str(tasks)]
        if args.smoke:
            command += ['--smoke']
        with (logs/'verification.log').open('x') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        jobs, records = [], []
        splits = [('val', 'full_val'), ('val', 'replay_val')] if args.smoke else [('val', 'full_val'), ('test', 'full_test')]
        for arm in ('baseline', 'absolute', 'target'):
            for split, folder in splits:
                arguments = ['--source-run', root/arm/'train', '--output', root/arm/folder,
                             '--split', split, '--expected-tasks', str(tasks), '--data-root', args.data_root,
                             '--clip-checkpoint', args.clip_checkpoint, '--face-manifest-root', args.face_manifest_root,
                             '--batch-size', '64', '--workers', '0']
                if folder != 'replay_val':
                    arguments += ['--full-cohort']
                process, record = launch('multi_lane.track_a.evaluate_moe_task_ffn', arguments,
                                          args.eval_gpus[len(jobs)], logs/arm/(folder+'.log'), arm+'_'+folder)
                jobs.append(process); records.append(record)
        wait_all(jobs, records, logs/'evaluation_status.json')
        evaluation = []
        for arm in ('baseline', 'absolute', 'target'):
            for split, folder in splits:
                result = json.loads((root/arm/folder/'result.json').read_text())
                assert len(result['task_metrics']) == tasks
                if folder != 'replay_val':
                    assert result['full_cohort'] and result['cohort_samples'] == (2397 if split == 'val' else 5368)
                else:
                    import numpy as np
                    for task in range(tasks):
                        with np.load(root/arm/'train'/'val_scores'/f'task{task}.npz') as old, np.load(root/arm/folder/'scores'/f'task{task}.npz') as new:
                            for key in ('sample_ids','targets','class_indices','logits','probabilities'):
                                assert np.array_equal(old[key], new[key]), (arm, task, key, 'replay')
                evaluation.append(dict(arm=arm, split=split, folder=folder, summary=result['summary']))
        (root/'evaluation_verification.json').write_text(json.dumps(dict(status='verified', checks=evaluation), indent=2)+'\n')
        attribution_args = ['--source-run', root/'target'/'train', '--output', root/'target'/'spatial_attribution_val',
                            '--data-root', args.data_root, '--clip-checkpoint', args.clip_checkpoint,
                            '--face-manifest-root', args.face_manifest_root, '--checkpoint-tasks']
        attribution_args += ['0', '1', '--smoke-batches', '2'] if args.smoke else ['0', '2', '7']
        process, record = launch('multi_lane.track_a.evaluate_spatial_attribution', attribution_args,
                                  args.eval_gpus[-1], logs/'attribution.log', 'spatial_attribution_val')
        wait_all([process], [record], logs/'attribution_status.json')
        (root/'completion.json').write_text(json.dumps(dict(status='complete', smoke=args.smoke, tasks=tasks,
                  validation_and_test_complete=not args.smoke, test_used_for_selection=False),indent=2)+'\n')
        status.write_text('0\n')
        print(f'SPATIAL_SUMMARY_BATCH_COMPLETE {args.run_id}', flush=True)
    except BaseException:
        status.write_text('1\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--train-gpus', type=int, nargs=3, default=[4,5,6])
    parser.add_argument('--eval-gpus', type=int, nargs=6, default=[1,2,3,4,5,6])
    parser.add_argument('--data-root', default='../multi-lane-main/datasets/EMOTIC')
    parser.add_argument('--clip-checkpoint', default='../CODE_DDP/pretrained/clip/ViT-B-16.pt')
    parser.add_argument('--face-manifest-root', default='../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909')
    run(parser.parse_args())
