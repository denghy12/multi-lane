"""GPU0-only read-only Test on/off replay of existing b32 checkpoints."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .historical_b32_test_attribution import ARMS, TASKS, validate_request

MODEL_HEAD = 'ab1557142c95d0d8574ea172a968c55d78466a25'
VIEWS = ('full', 'person', 'face', 'fused')
FIELDS = ('sample_ids', 'targets', 'class_indices', 'logits', 'probabilities')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_hashes(source):
    records = {}
    for arm in ('BASE', *ARMS):
        audit = read(source / arm / 'training_verification.json')
        for task in range(8):
            value = sha(source / arm / 'train/compact_checkpoints' / f'task{task}.pth')
            assert value == audit['checkpoint_sha256'][str(task)]
            records[f'{arm}/task{task}'] = value
    return records


def command_arguments(args, root, arm, mode):
    flags = ['--source-run', args.source_root / arm / 'train', '--output', root / arm / mode,
        '--split', 'test', '--data-root', args.data_root, '--clip-checkpoint', args.clip_checkpoint,
        '--face-manifest-root', args.face_manifest_root, '--expected-tasks', 8,
        '--batch-size', 64, '--workers', 2, '--checkpoint-tasks', *TASKS,
        '--historical-b32-test-attribution']
    if mode == 'off':
        flags += ['--disable-stacking-modules', 'B' if arm.startswith('ATTENTION') else 'C']
    return list(map(str, flags))


def verify_job(args, root, arm, mode, head):
    import numpy as np
    folder = root / arm / mode; result = read(folder / 'result.json')
    source = args.source_root / arm
    assert result['split'] == 'test' and result['checkpoint_tasks'] == list(TASKS)
    assert result['historical_b32_test_attribution'] and result['summary_is_selected_boundaries_only']
    assert not result['full_cohort'] and result['moe_runtime_enabled']
    assert result['source_git']['commit'] == MODEL_HEAD and not result['source_git']['dirty']
    assert result['evaluation_git']['commit'] == head and not result['evaluation_git']['dirty']
    expected = [] if mode == 'on' else ['B' if arm.startswith('ATTENTION') else 'C']
    assert result['disabled_stacking_modules'] == expected
    reference = read(source / 'historical_test/result.json')
    rows = []
    for task in TASKS:
        recorded = read(source / 'training_verification.json')['checkpoint_sha256'][str(task)]
        assert result['checkpoint_sha256'][str(task)] == recorded
        original = source / 'historical_test'
        with np.load(folder / 'scores' / f'task{task}.npz') as actual, \
             np.load(original / 'scores' / f'task{task}.npz') as archived:
            assert all(np.array_equal(actual[k], archived[k]) for k in FIELDS[:3])
            if mode == 'on':
                assert all(np.array_equal(actual[k], archived[k]) for k in FIELDS)
        with np.load(folder / 'view_scores' / f'task{task}.npz') as actual, \
             np.load((original if mode == 'on' else root / arm / 'on') / 'view_scores' / f'task{task}.npz') as other:
            keys = [name + suffix for name in VIEWS for suffix in ('_logits', '_probabilities')]
            if mode == 'on':
                assert all(np.array_equal(actual[k], other[k]) for k in keys)
            else:
                unaffected = ('full', 'face') if arm.startswith('ATTENTION') else ('person', 'face')
                assert all(np.array_equal(actual[name + '_logits'], other[name + '_logits']) for name in unaffected)
            assert np.array_equal(actual['face_reliable'], other['face_reliable'])
        metric = next(r for r in result['task_metrics'] if r['task_id'] == task)
        if mode == 'on':
            assert all(metric[k] == reference['task_metrics'][task][k] for k in ('mAP', 'cF1', 'oF1'))
        rows.append(dict(task=task, mAP=metric['mAP'], checkpoint_sha256=recorded))
    return dict(status='verified', arm=arm, mode=mode, tasks=rows,
        original_on_predictions_exact=mode == 'on', unaffected_views_exact=mode == 'off')


def summarize(args, root):
    rows, classes = [], []
    baseline = read(args.source_root / 'BASE/historical_test/result.json')
    for arm in ARMS:
        on = read(root / arm / 'on/result.json'); off = read(root / arm / 'off/result.json')
        names = read(args.source_root / arm / 'train/config.json')['class_order']
        for t in TASKS:
            a = next(r for r in on['task_metrics'] if r['task_id'] == t)
            b = next(r for r in off['task_metrics'] if r['task_id'] == t)
            base = baseline['task_metrics'][t]
            rows.append(dict(arm=arm, task=t, samples=a['samples'], base_mAP=base['mAP'],
                on_mAP=a['mAP'], off_mAP=b['mAP'], on_minus_off=a['mAP']-b['mAP'],
                off_minus_base=b['mAP']-base['mAP'], on_minus_base=a['mAP']-base['mAP'],
                on_cF1=a['cF1'], off_cF1=b['cF1'], on_oF1=a['oF1'], off_oF1=b['oF1'],
                **{name+'_mAP_change':on['view_diagnostics'][str(t)]['metrics'][name]['mAP']-
                    off['view_diagnostics'][str(t)]['metrics'][name]['mAP'] for name in VIEWS}))
            for i, (left, right) in enumerate(zip(a['per_class_ap'], b['per_class_ap'])):
                classes.append(dict(arm=arm, task=t, class_name=names[i], on_AP=left, off_AP=right, delta=left-right))
    for name, values in (('summary.csv', rows), ('final_task7.csv', [r for r in rows if r['task'] == 7]), ('class_effects.csv', classes)):
        with (root / name).open('x') as f:
            writer = csv.DictWriter(f, list(values[0])); writer.writeheader(); writer.writerows(values)


def run(args):
    if not args.run_id or '/' in args.run_id or args.run_id in {'.', '..'}:
        raise ValueError('Invalid run ID')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Clean evaluator checkout required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    assert read(args.source_root / 'completion.json')['status'] == 'complete'
    assert read(args.source_root / 'baseline_reproduction.json')['status'] == 'exact'
    for arm in ARMS:
        config = read(args.source_root / arm / 'train/config.json')
        assert config['git']['commit'] == MODEL_HEAD and not config['git']['dirty']
        assert read(args.source_root / arm / 'train/seed_summary.json')['status'] == 'complete'
        validate_request(config, 'test', TASKS, (), 8)
    before = checkpoint_hashes(args.source_root)
    root = Path('output/historical_b32_test_attribution') / args.run_id
    logs = Path('logs/historical_b32_test_attribution') / args.run_id
    if root.exists() or logs.exists():
        raise FileExistsError(args.run_id)
    root.mkdir(parents=True); logs.mkdir(parents=True)
    protocol = dict(status='running', source_root=str(args.source_root), model_head=MODEL_HEAD,
        evaluator_head=head, seed=1, split='test', tasks=list(TASKS), batch_size=64, workers=2,
        cpu_threads=72, gpu=0, max_parallel=4, training=False, test_used_for_selection=True,
        evaluation_scope='samples_intersect_seen_classes', full_eight_stage_average=False,
        checkpoint_sha256_before=before, started_unix=time.time())
    write(root / 'protocol.json', protocol)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='72', MKL_NUM_THREADS='72', PYTHONUNBUFFERED='1')
    pending = [(arm, 'on') for arm in ARMS]; active = []; records = []; audits = []; failed = False; last_launch = 0
    try:
        while pending or active:
            for process, record in list(active):
                code = process.poll()
                if code is None:continue
                active.remove((process, record)); record.update(exit=code, finished_unix=time.time()); failed |= code != 0
                print('FINISH', record['arm'], record['mode'], code, flush=True)
                if code == 0:
                    try:
                        audit = verify_job(args, root, record['arm'], record['mode'], head); audits.append(audit)
                        write(root / record['arm'] / (record['mode']+'_verification.json'), audit)
                        if record['mode'] == 'on':pending.append((record['arm'], 'off'))
                    except Exception as error:
                        record['verification_error'] = repr(error); failed = True
                        print('VERIFICATION_FAILED', record['arm'], record['mode'], repr(error), flush=True)
            if failed and not active:raise RuntimeError('A job or verification failed; queue preserved')
            if pending and not failed and len(active) < 4 and time.time()-last_launch >= 12:
                free = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                stat = os.statvfs('/dev/shm'); shm = stat.f_bavail*stat.f_frsize/1024**2
                if free >= 6000 and shm >= 1024:
                    arm, mode = pending.pop(0)
                    command = [sys.executable, '-u', '-m', 'multi_lane.track_a.evaluate_moe_task_ffn', *command_arguments(args, root, arm, mode)]
                    log = logs / arm / (mode+'.log'); log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open('x') as f:
                        process = subprocess.Popen(command, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                    record = dict(arm=arm, mode=mode, command=command, log=str(log), pid=process.pid, gpu=0, started_unix=time.time())
                    records.append(record); active.append((process,record)); last_launch=time.time()
                    print('START', arm, mode, process.pid, flush=True)
            write(logs / 'queue.json', dict(failed=failed, records=records, pending=pending))
            if pending or active:time.sleep(3)
        after = checkpoint_hashes(args.source_root); assert after == before
        summarize(args, root)
        write(root / 'completion.json', dict(protocol, status='complete', jobs=len(records),
            checkpoint_sha256_after=after, source_checkpoints_unchanged=True, audits=audits, finished_unix=time.time()))
        print('HISTORICAL_B32_TEST_ATTRIBUTION_COMPLETE', root, flush=True)
    except BaseException as error:
        write(root / 'failure.json', dict(error=repr(error))); raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run-id', required=True)
    for name in ('source-root', 'data-root', 'clip-checkpoint', 'face-manifest-root'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
