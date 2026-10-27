"""Resume unchanged student evaluation with isolated, single-thread data workers."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _sha256, _torch_load
from .head_anchored_eight import runtime
from .historical_b32_backbone_attribution import build, method_state
from .paired_transforms import move_model_inputs
from .run_historical_b32_test_attribution import read, write
from .runner import (TaskMetrics, compute_metrics, dataset_view, seen_indices,
                     set_seed, summarize_tasks, tensor_state_digest)
from .self_trained_view_transfer import ARMS
from .strong_full_evaluation import evaluate_fused_and_views


def single_thread_worker(_):
    # PyTorch already does this in its worker loop; make the contract explicit.
    torch.set_num_threads(1)


def loader_for(view, task, workers):
    options = dict(batch_size=64, shuffle=False, num_workers=workers,
                   generator=torch.Generator().manual_seed(87001 + task))
    if workers:
        options.update(prefetch_factor=1, persistent_workers=True, pin_memory=True,
                       multiprocessing_context='spawn', worker_init_fn=single_thread_worker)
    return DataLoader(view, **options)


def code_audit(source_git):
    root = Path(__file__).resolve().parents[2]
    changed = subprocess.check_output(['git', 'diff', '--name-only', source_git,
                                      'HEAD', '--', 'multi_lane/'], cwd=root, text=True).splitlines()
    if set(changed) - {'multi_lane/track_a/self_trained_fast_evaluation.py'}:
        raise ValueError('Evaluation optimization changed existing model/evaluation code: ' + str(changed))
    return dict(existing_package_files_byte_identical=True, changed_package_files=changed,
                training_commit=source_git)


def load_stage(model, training, complete, cfg, task):
    path = training / 'compact_checkpoints' / f'task{task}.pth'
    assert _sha256(path) == complete['checkpoints'][str(task)]['sha256']
    payload = _torch_load(path)
    assert payload['source_git'] == cfg['git'] and payload['task_id'] == task
    missing = model.load_state_dict(payload['model'], strict=False)
    assert not missing.unexpected_keys and all(k.startswith('visual_encoder.') for k in missing.missing_keys)
    model.restore_task(task)
    model.requires_grad_(False).eval()
    return path


def input_digest(batch):
    images, targets, ids = batch
    h = hashlib.sha256()
    for name, tensor in sorted(dict(images, targets=targets).items()):
        h.update(name.encode())
        h.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        h.update(tensor.contiguous().numpy().tobytes())
    h.update(repr(tuple(ids)).encode())
    return h.hexdigest()


def benchmark(args):
    git = runtime(); set_seed(1, True)
    training = args.source_root / 'SELF_NO_CONSTRAINT/train'
    cfg = read(training / 'config.json'); complete = read(training / 'training_complete.json')
    audit = code_audit(cfg['git']['commit'])
    model = build(cfg['model_template'], Path(cfg['clip_checkpoint']))
    data = make_source(cfg['model_template'], Path(cfg['data_root']), Path(cfg['face_manifest_root']), 'val')
    records = []
    for task in (0, 7):
        load_stage(model, training, complete, cfg, task)
        model.cross_view_runtime_enabled = True
        view = Subset(dataset_view(data, seen_indices(task), True), range(128))
        references = None
        for workers in (0, 2, 4):
            loader = loader_for(view, task, workers)
            start = time.monotonic(); batches = list(loader)
            loading = time.monotonic() - start
            if workers:
                # Also measure steady state without process startup.
                start = time.monotonic(); again = list(loader)
                steady = time.monotonic() - start
                assert [input_digest(b) for b in again] == [input_digest(b) for b in batches]
            else:
                steady = loading
            outputs = []; repeated = []
            for batch in batches:
                images = move_model_inputs(batch[0], torch.device('cuda'))
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                    z, vs = model.seen_logits_with_views(images)
                    rz, rvs = model.seen_logits_with_views(images)
                outputs.append({k: v.float().cpu().numpy() for k, v in dict(fused=z, **vs).items()})
                repeated.append({k: v.float().cpu().numpy() for k, v in dict(fused=rz, **rvs).items()})
            digests = [input_digest(b) for b in batches]
            if references is None:
                references = (digests, outputs)
            assert digests == references[0], 'DataLoader changed images, labels, masks or sample ordering'
            differences = {k: max(float(np.max(np.abs(a[k] - b[k])))
                for a, b in zip(references[1], outputs)) for k in outputs[0]}
            repeat = {k: max(float(np.max(np.abs(a[k] - b[k])))
                for a, b in zip(outputs, repeated)) for k in outputs[0]}
            # Same batches, same model/kernel settings: require exact output replay.
            assert all(v == 0. for v in differences.values()), differences
            row = dict(task=task, workers=workers, samples=128, loading_seconds=loading,
                       steady_loading_seconds=steady, input_sha256=digests,
                       max_logit_difference=differences, unchanged_repeat_difference=repeat,
                       inputs_and_predictions_exact=True)
            records.append(row)
            print('LOADER_BENCHMARK', row, flush=True)
            del loader, batches
    write(args.output / 'benchmark.json', dict(status='verified', git=git, code_audit=audit,
          records=records, training_updates=0, batch_size=64, model_threads=72,
          worker_threads=1, amp=True, tf32=True))


def validate_saved(folder, task, view):
    """Only a complete JSON+two NPZ triplet is eligible for reuse."""
    paths = [folder / f'task{task}.json', folder / 'scores' / f'task{task}.npz',
             folder / 'view_scores' / f'task{task}.npz']
    if not all(p.exists() for p in paths):
        return None
    row = read(paths[0])
    source = view.source
    ids = np.asarray([source.sample_ids[i] for i in view.indices])
    with np.load(paths[1]) as fused, np.load(paths[2]) as vs:
        assert int(fused['task_id']) == task and int(vs['task_id']) == task
        assert np.array_equal(vs['sample_ids'], ids)
        assert np.array_equal(vs['class_indices'], np.asarray(view.class_indices))
        assert np.array_equal(vs['face_reliable'], np.asarray([source.face_reliable[i] for i in view.indices]))
        expected = np.zeros((len(ids), len(view.class_indices)), dtype=np.float32)
        for n, index in enumerate(view.indices):
            expected[n] = [float(c in source.targets[index]) for c in view.class_indices]
        assert np.array_equal(vs['targets'], expected)
        lengths = np.asarray([min(64, len(ids) - i) for i in range(0, len(ids), 64)])
        assert np.array_equal(vs['batch_lengths'], lengths)
        for k in ('sample_ids', 'targets', 'class_indices', 'batch_lengths'):
            assert np.array_equal(fused[k], vs[k]), k
        for suffix in ('logits', 'probabilities'):
            assert np.array_equal(fused[suffix], vs['fused_' + suffix])
        for name in ('fused', 'full', 'person', 'face'):
            scores = vs[name + '_probabilities'].copy()
            assert np.isfinite(scores).all() and np.isfinite(vs[name + '_logits']).all()
            actual = asdict(compute_metrics(task, torch.from_numpy(scores), torch.from_numpy(expected), .5))
            assert actual == row['views']['metrics'][name], name
        assert row['metrics'] == row['views']['metrics']['fused']
    return dict(row=row, files=[dict(path=str(p), sha256=_sha256(p)) for p in paths])


def evaluate(args):
    start = time.time(); git = runtime(); set_seed(1, True)
    training = args.source_root / args.arm / 'train'
    cfg = read(training / 'config.json'); complete = read(training / 'training_complete.json')
    assert cfg == complete['config'] and complete['status'] == 'complete'
    assert not cfg['smoke'] and cfg['tasks'] == list(range(8))
    code = code_audit(cfg['git']['commit'])
    model = build(cfg['model_template'], Path(cfg['clip_checkpoint']))
    data = make_source(cfg['model_template'], Path(cfg['data_root']), Path(cfg['face_manifest_root']), args.split)
    results = []; off = []; audits = []
    for task in cfg['tasks']:
        path = load_stage(model, training, complete, cfg, task)
        before = tensor_state_digest(method_state(model))
        view = dataset_view(data, seen_indices(task), True)
        loader = None
        for mode in ('on', 'off') if model.cross_view_bank is not None else ('on',):
            target = args.output / mode
            reused = validate_saved(target, task, view)
            origin = 'this_fast_run'
            if reused is None:
                prior = args.source_root / args.arm / args.split / mode
                reused = validate_saved(prior, task, view)
                origin = 'original_evaluation'
                if reused is not None:
                    for item in reused['files']:
                        src = Path(item['path']); dst = target / src.relative_to(prior)
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                        assert _sha256(dst) == item['sha256']
            tick = time.time()
            if reused is not None:
                row = reused['row']
                reuse_audit = dict(origin=origin, source_files=reused['files'])
            else:
                if loader is None:
                    loader = loader_for(view, task, args.workers)
                model.cross_view_runtime_enabled = mode == 'on' and model.cross_view_bank is not None
                with torch.no_grad():
                    value, views = evaluate_fused_and_views(model, loader, torch.device('cuda'), task, .5, True,
                        target / 'scores' / f'task{task}.npz', target / 'view_scores' / f'task{task}.npz')
                row = dict(metrics=asdict(value), views=views)
                write(target / f'task{task}.json', row)
                assert validate_saved(target, task, view) is not None
                reuse_audit = None
            (results if mode == 'on' else off).append(row['metrics'])
            audits.append(dict(task=task, mode=mode, own_checkpoint_sha256=_sha256(path),
                reused=reuse_audit, elapsed_seconds=time.time() - tick, samples=len(view)))
            write(args.output / 'progress.json', dict(status='running', arm=args.arm, split=args.split, passes=audits))
            print('FAST_EVALUATE', args.arm, args.split, task, mode,
                  'reuse=' + str(reused is not None), row['metrics']['mAP'], 'seconds=', time.time() - tick, flush=True)
        assert before == tensor_state_digest(method_state(model))
        del loader
    write(args.output / 'result.json', dict(status='complete', git=git, training_git=cfg['git'],
        code_audit=code, split=args.split, smoke=False, seed=1, arm=args.arm, tasks=cfg['tasks'],
        metrics=results, off_metrics=off, summary=summarize_tasks([TaskMetrics(**v) for v in results]),
        off_summary=summarize_tasks([TaskMetrics(**v) for v in off]) if off else {},
        teacher_loaded_for_inference=False, external_parameter_assembly=False, audits=audits,
        parameters_unchanged=True, batch_size=64, model_threads=72, workers=args.workers,
        worker_threads=1, prefetch_factor=1, evaluation_scope='samples_intersect_seen_classes',
        elapsed_seconds=time.time() - start, peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2))


def run(args):
    git = runtime()
    assert read(args.output / 'benchmark.json')['status'] == 'verified'
    assert read(args.source_root / 'training_verification.json')['status'] == 'verified'
    logs = Path('logs/self_trained_fast_evaluation') / args.output.name
    logs.mkdir(parents=True, exist_ok=True)
    hashes = {str(p): _sha256(p) for p in args.source_root.glob('*/train/compact_checkpoints/task*.pth')}
    assert len(hashes) == 32
    pending = [(arm, split) for arm in ARMS for split in ('val', 'test')]
    active = []; records = []; last = 0.
    write(args.output / 'protocol.json', dict(git=git, source_root=str(args.source_root), checkpoint_sha256=hashes,
        seed=1, gpu=0, workers=args.workers, batch_size=64, model_threads=72, worker_threads=1,
        prefetch_factor=1, concurrency=4, amp=True, tf32=True, training_updates=0))
    try:
        while pending or active:
            for process, row in list(active):
                code = process.poll()
                if code is None: continue
                row.update(exit=code, finished_unix=time.time()); active.remove((process, row))
                assert code == 0, row
                assert read(Path(row['output']) / 'result.json')['status'] == 'complete'
            if pending and len(active) < 4 and time.time() - last >= 6:
                arm, split = pending.pop(0); output = args.output / arm / split
                if (output / 'result.json').exists():
                    assert read(output / 'result.json')['status'] == 'complete'
                    continue
                free = int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                          '--format=csv,noheader,nounits'], text=True).strip())
                shm = os.statvfs('/dev/shm')
                if free < 4000 or shm.f_bavail * shm.f_frsize < 1024**3:
                    pending.insert(0, (arm, split)); time.sleep(3); continue
                cmd = [sys.executable, '-u', '-m', __spec__.name, 'eval', '--source-root', str(args.source_root),
                       '--output', str(output), '--arm', arm, '--split', split, '--workers', str(args.workers)]
                log = logs / (arm + '_' + split + '.log')
                with log.open('a') as f:
                    process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                row = dict(arm=arm, split=split, pid=process.pid, output=str(output), log=str(log), started_unix=time.time())
                records.append(row); active.append((process, row)); last=time.time()
            write(logs / 'queue.json', dict(status='running', records=records, pending=pending,
                  active=[r['pid'] for _, r in active]))
            if active or pending: time.sleep(2)
        assert hashes == {p: _sha256(Path(p)) for p in hashes}
        rows=[]
        for arm in ARMS:
            row=dict(arm=arm)
            for split in ('val','test'):
                result=read(args.output / arm / split / 'result.json')
                row.update({split+'_'+k:v for k,v in result['summary'].items()
                            if k in ('final_mAP','average_mAP','final_cF1','final_oF1')})
            rows.append(row)
        with (args.output / 'summary.csv').open('w') as f:
            writer=csv.DictWriter(f, list(rows[0]));writer.writeheader();writer.writerows(rows)
        write(args.output / 'completion.json', dict(status='complete', git=git, source_checkpoints_unchanged=True,
              training_updates=0, source_root=str(args.source_root)))
        write(logs / 'queue.json', dict(status='complete', records=records))
    except BaseException as error:
        write(args.output / 'failure.json', dict(error=repr(error), records=records))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('benchmark','eval','run'))
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(2,4), default=2)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--split', choices=('val','test'))
    args=parser.parse_args()
    {'benchmark':benchmark,'eval':evaluate,'run':run}[args.action](args)
