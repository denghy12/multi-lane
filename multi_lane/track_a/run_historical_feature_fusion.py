"""GPU0 queue for the four locked feature-fusion/Adapter-ASL controls."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .historical_feature_fusion import ARMS, training_arguments, validate_config
from .run_module_stacking import write


def verify_train(root, arm, tasks, smoke):
    import torch
    from .export_compact_test_scores import _torch_load, _sha256
    from .verify_spatial_summary import load_boundary
    train = root / arm / 'train'
    c = json.loads((train / 'config.json').read_text())
    s = json.loads((train / 'seed_summary.json').read_text())
    h = json.loads((train / 'training_history.json').read_text())
    assert validate_config(c) == arm and s['config'] == c and s['status'] == 'complete'
    assert c['max_tasks'] == tasks and not c['git']['dirty']
    audit = json.loads((train / 'paired_protocol_audit.json').read_text())
    assert audit['frozen_visual_unchanged'] and not audit['tasks']
    previous = None
    hashes = {}
    rng = []
    for t in range(tasks):
        path = train / 'compact_checkpoints' / f'task{t}.pth'
        payload = _torch_load(path)
        state = payload['model']
        assert payload['source_git'] == c['git'] and payload['task_id'] == t
        assert all(torch.isfinite(v).all() for v in state.values())
        if previous is not None:
            assert torch.equal(previous['selectors'][:t], state['selectors'][:t])
            for name in state:
                if name.startswith('prompts.'):
                    assert torch.equal(previous[name][:, :t], state[name][:, :t])
                if name.startswith('adapter_bank.task_adapters.') and int(name.split('.')[2]) < t:
                    assert torch.equal(previous[name], state[name])
            classes = sum(c['task_sizes'][:t])
            for name in ('head.weight', 'head.bias'):
                assert torch.equal(previous[name][:classes], state[name][:classes])
        boundary = load_boundary(train / 'recovery_checkpoints' / f'task{t}_before_reporting.pth')
        assert state.keys() == boundary['model'].keys()
        assert all(torch.equal(value, boundary['model'][name]) for name, value in state.items())
        assert boundary['sampler_generator'] is None and boundary['rng']['torch_cuda']
        assert all(key in boundary['training_state'] for key in ('optimizer', 'scheduler', 'scaler', 'optimizer_parameter_names'))
        rng.append(boundary['rng']['torch_cpu'].tolist())
        assert all(row['skipped_optimizer_steps'] == 0 for row in h[str(t)])
        hashes[str(t)] = _sha256(path)
        previous = state
    count = sum(row['optimizer_steps'] for rows in h.values() for row in rows)
    assert count == (tasks * 2 if smoke else 13950) == s['completed_optimizer_updates']
    result = dict(status='verified', arm=arm, updates=count, skipped=0,
        old_parameters_unchanged=True, recovery_equals_compact=True, global_sampler_rng=True,
        initial_base_sha256=c['initial_selector_prompt_classifier_sha256'],
        frozen_visual_unchanged=True, checkpoint_sha256=hashes,
        cpu_rng_at_task_boundaries=rng, peak_allocated_mib=audit['peak_allocated_mib'])
    write(root / arm / 'training_verification.json', result)
    return result


def verify_smoke_full(root):
    import torch
    from .export_compact_test_scores import _torch_load
    old = root / 'NATIVE_FULL' / 'train'
    new = root / 'FULL_ASL' / 'train'
    a = json.loads((old / 'training_history.json').read_text())
    b = json.loads((new / 'training_history.json').read_text())
    keys = ('current_loss', 'adapter_objective_loss', 'learning_rate', 'optimizer_steps', 'skipped_optimizer_steps')
    for t in range(2):
        assert len(a[str(t)]) == len(b[str(t)])
        for x, y in zip(a[str(t)], b[str(t)]):
            assert all(x[key] == y[key] for key in keys), (t, x, y)
        x = _torch_load(old / 'compact_checkpoints' / f'task{t}.pth')['model']
        y = _torch_load(new / 'compact_checkpoints' / f'task{t}.pth')['model']
        assert x.keys() == y.keys(), (x.keys(), y.keys())
        assert all(torch.equal(value, y[key]) for key, value in x.items()), t
    return dict(native_old_full_equals_current=True, tasks=2, updates_per_task=2,
                every_method_tensor_equal=True, compared_loss_lr_steps_equal=True)


def evaluation_job(args, root, arm, split, folder):
    return dict(name=arm + '/' + folder, module='multi_lane.track_a.evaluate_moe_task_ffn',
        arguments=list(map(str, ['--source-run', root / arm / 'train', '--output', root / arm / folder,
            '--split', split, '--data-root', args.data_root, '--clip-checkpoint', args.clip_checkpoint,
            '--face-manifest-root', args.face_manifest_root, '--expected-tasks', 2 if args.smoke else 8,
            '--batch-size', 64, '--workers', 0])))


def run(args):
    if not args.run_id or '/' in args.run_id or args.run_id in {'.', '..'}:
        raise ValueError('Invalid run ID')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Clean checkout required')
    if shutil.disk_usage('.').free < 20 * 1024 ** 3:
        raise RuntimeError('Need at least 20 GiB output space')
    root = Path('output/feature_fusion_adapter_asl') / args.run_id
    logs = Path('logs/feature_fusion_adapter_asl') / args.run_id
    if root.exists() or logs.exists():
        raise FileExistsError(args.run_id)
    if not args.smoke:
        gate = json.loads((args.smoke_root / 'completion.json').read_text())
        assert gate['smoke'] and gate['native_full_equivalence']['native_old_full_equals_current']
    root.mkdir(parents=True); logs.mkdir(parents=True)
    protocol = dict(status='running', seed=0, tasks=2 if args.smoke else 8, smoke=args.smoke,
        gpu=0, parallelism=4, cpu_threads=72, evaluation_scope='samples_intersect_seen_classes',
        test_used_for_selection=False,
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        training_arguments={arm: training_arguments(args, root, arm) for arm in ARMS})
    write(root / 'protocol.json', protocol)
    pending = [dict(name=arm + '/train', module='multi_lane.track_a.runner',
                    arguments=protocol['training_arguments'][arm]) for arm in ARMS]
    if args.smoke:
        pending.append(dict(name='NATIVE_FULL/train', module='multi_lane.track_a.legacy_full_reference',
            arguments=list(map(str, ['--legacy-root', args.legacy_root, '--output', root / 'NATIVE_FULL',
                '--data-root', args.data_root, '--clip-checkpoint', args.clip_checkpoint, '--smoke']))))
    active = []; records = []; failed = False; audits = {}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='72', MKL_NUM_THREADS='72', PYTHONUNBUFFERED='1')
    try:
        while pending or active:
            for process, record in list(active):
                code = process.poll()
                if code is None:
                    continue
                active.remove((process, record));record.update(exit=code, finished_unix=time.time())
                print('FINISH', record['name'], code, flush=True)
                failed |= code != 0
                if code == 0:
                    arm, kind = record['name'].split('/')
                    if kind == 'train' and arm != 'NATIVE_FULL':
                        audits[arm] = verify_train(root, arm, protocol['tasks'], args.smoke)
                        pending += [evaluation_job(args, root, arm, 'val', 'historical_val'),
                                    evaluation_job(args, root, arm, 'val' if args.smoke else 'test',
                                                   'replay_val' if args.smoke else 'historical_test')]
            if failed and not active:
                raise RuntimeError('A job failed; see queue.json; unstarted jobs preserved')
            if pending and not failed and len(active) < 4:
                free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip())
                if free >= 6000:
                    job = pending.pop(0); log = logs / (job['name'] + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, '-u', '-m', job['module'], *job['arguments']]
                    with log.open('x') as stream:
                        p = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    record = dict(job, pid=p.pid, gpu=0, command=command, log=str(log), started_unix=time.time())
                    active.append((p, record));records.append(record)
                    print('START', job['name'], p.pid, 'gpu=0', flush=True)
            write(logs / 'queue.json', dict(records=records, pending=pending, failed=failed))
            if pending or active:
                time.sleep(3)
        assert len({a['initial_base_sha256'] for a in audits.values()}) == 1
        three = [audits[arm]['cpu_rng_at_task_boundaries'] for arm in ARMS[1:]]
        assert all(value == three[0] for value in three), 'Three-view sampler RNG diverged'
        if args.smoke:
            protocol['native_full_equivalence'] = verify_smoke_full(root)
        summaries = []
        for arm in ARMS:
            row = dict(arm=arm)
            for split in ('val', 'test'):
                folder = 'replay_val' if args.smoke and split == 'test' else 'historical_' + split
                result = json.loads((root / arm / folder / 'result.json').read_text())
                assert not result['full_cohort'] and len(result['task_metrics']) == protocol['tasks']
                for name in ('final_mAP', 'average_mAP', 'final_cF1', 'final_oF1'):
                    row[split + '_' + name] = result['summary'][name]
            summaries.append(row)
        with (root / 'summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
        protocol['status'] = 'complete'; protocol['three_view_sampler_rng_equal'] = True
        write(root / 'completion.json', protocol)
        print('HISTORICAL_FEATURE_FUSION_COMPLETE', root, flush=True)
    except BaseException as error:
        write(root / 'failure.json', dict(error=repr(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    for key in ('data-root', 'clip-checkpoint', 'face-manifest-root', 'legacy-root'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-root', type=Path)
    run(parser.parse_args())
