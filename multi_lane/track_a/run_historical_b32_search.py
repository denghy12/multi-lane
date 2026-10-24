"""GPU0-only queue for historical b32 reproduction and an 18-model search."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .historical_b32_search import experiment_cells, recipe, training_arguments, validate_config, LEGACY_COMMIT
from .run_module_stacking import write


def verify_train(train, smoke):
    import torch
    from .export_compact_test_scores import _torch_load, _sha256
    from .verify_spatial_summary import load_boundary
    c = json.loads((train / 'config.json').read_text())
    s = json.loads((train / 'seed_summary.json').read_text())
    h = json.loads((train / 'training_history.json').read_text())
    profile = validate_config(c)
    assert s['config'] == c and s['status'] == 'complete' and not c['git']['dirty']
    tasks = 2 if smoke else 8
    assert c['max_tasks'] == tasks and len(h) == tasks
    visual = json.loads((train / 'paired_protocol_audit.json').read_text())
    assert visual['frozen_visual_unchanged']
    previous = None; hashes = {}; rng = []
    for t in range(tasks):
        path = train / 'compact_checkpoints' / f'task{t}.pth'
        payload = _torch_load(path); state = payload['model']
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
        assert all(torch.equal(v, boundary['model'][k]) for k, v in state.items())
        assert boundary['sampler_generator'] is None and boundary['rng']['torch_cuda']
        assert all(k in boundary['training_state'] for k in ('optimizer', 'scheduler', 'scaler', 'optimizer_parameter_names'))
        assert all(row['skipped_optimizer_steps'] == 0 for row in h[str(t)])
        rng.append(hashlib.sha256(boundary['rng']['torch_cpu'].numpy().tobytes()).hexdigest())
        hashes[str(t)] = _sha256(path); previous = state
    updates = sum(row['optimizer_steps'] for rows in h.values() for row in rows)
    assert updates == (4 if smoke else 13950) == s['completed_optimizer_updates']
    return dict(status='verified', seed=c['seed'], profile=profile, updates=updates, skipped=0,
        old_parameters_unchanged=True, frozen_visual_unchanged=True, recovery_equals_compact=True,
        global_sampler_rng=True, checkpoint_sha256=hashes, cpu_rng_at_task_boundaries=rng,
        initial_base_sha256=c['initial_selector_prompt_classifier_sha256'],
        peak_allocated_mib=visual['peak_allocated_mib'])


def verify_native(root):
    import numpy as np
    import torch
    from .export_compact_test_scores import _torch_load
    results = {}
    for profile in ('THREE_BASE', 'FULL_BASE'):
        native = root / ('NATIVE_' + profile) / 'train'
        current = root / 'seed1' / profile / 'train'
        a = json.loads((native / 'training_history.json').read_text())
        b = json.loads((current / 'training_history.json').read_text())
        for t in range(2):
            assert len(a[str(t)]) == len(b[str(t)])
            for x, y in zip(a[str(t)], b[str(t)]):
                for key in ('current_loss', 'adapter_objective_loss', 'learning_rate', 'optimizer_steps', 'skipped_optimizer_steps'):
                    assert x[key] == y[key], (profile, t, key, x[key], y[key])
            x = _torch_load(native / 'compact_checkpoints' / f'task{t}.pth')['model']
            y = _torch_load(current / 'compact_checkpoints' / f'task{t}.pth')['model']
            assert x.keys() == y.keys() and all(torch.equal(v, y[k]) for k, v in x.items()), (profile, t)
            with np.load(native / 'val_scores' / f'task{t}.npz') as n, np.load(current / 'val_scores' / f'task{t}.npz') as m:
                assert all(np.array_equal(n[k], m[k]) for k in ('sample_ids', 'class_indices', 'targets', 'logits', 'probabilities'))
        results[profile] = dict(all_method_tensors_losses_and_predictions_equal=True, tasks=2, updates=4)
    return results


def compare_archive(train, archive):
    """Record exact reproduction, never replace a mismatched score with the anchor."""
    import numpy as np
    source = json.loads((archive / 'seed_summary.json').read_text())
    current = json.loads((train / 'seed_summary.json').read_text())
    metrics = {k: dict(expected=source['metrics'][k], actual=current['metrics'][k],
        equal=source['metrics'][k] == current['metrics'][k])
        for k in ('final_mAP', 'average_mAP', 'final_cF1', 'final_oF1')}
    phases = []
    for t in range(8):
        with np.load(archive / 'test_scores' / f'task{t}.npz') as a, np.load(train / 'test_scores' / f'task{t}.npz') as b:
            phases.append(dict(task=t, fields_equal={k: bool(np.array_equal(a[k], b[k]))
                for k in ('sample_ids', 'targets', 'class_indices', 'logits', 'probabilities')}))
    exact = all(v['equal'] for v in metrics.values()) and all(all(v['fields_equal'].values()) for v in phases)
    return dict(status='exact' if exact else 'mismatch', archive=str(archive), metrics=metrics, phases=phases)


def evaluation_job(args, root, name, split, replay=False):
    folder = 'replay_val' if replay else 'historical_' + split
    return dict(name=name + '/' + folder, kind='eval', cell=name, module='multi_lane.track_a.evaluate_moe_task_ffn',
        arguments=list(map(str, ['--source-run', root / name / 'train', '--output', root / name / folder,
            '--split', split, '--data-root', args.data_root, '--clip-checkpoint', args.clip_checkpoint,
            '--face-manifest-root', args.face_manifest_root, '--expected-tasks', 2 if args.smoke else 8,
            '--batch-size', 64, '--workers', 2])))


def summarize(root, cells, smoke):
    import numpy as np
    rows = []
    for seed, profile in cells:
        name = f'seed{seed}/{profile}'; row = dict(seed=seed, profile=profile, **recipe(profile),
            test_used_for_parameter_search=not smoke)
        for split in ('val', 'test'):
            folder = 'replay_val' if smoke and split == 'test' else 'historical_' + split
            d = json.loads((root / name / folder / 'result.json').read_text())
            assert not d['full_cohort'] and len(d['task_metrics']) == (2 if smoke else 8)
            for k in ('final_mAP', 'average_mAP', 'final_cF1', 'final_oF1'):
                row[split + '_' + k] = d['summary'][k]
            # Evaluating outside training must reproduce the native reporting score.
            for t in range(2 if smoke else 8):
                with np.load(root / name / 'train' / ('val_scores' if smoke else 'test_scores') / f'task{t}.npz') as a:
                    if smoke or split == 'test':
                        with np.load(root / name / folder / 'scores' / f'task{t}.npz') as b:
                            assert all(np.array_equal(a[k], b[k]) for k in ('sample_ids', 'targets', 'class_indices', 'logits', 'probabilities'))
        rows.append(row)
    with (root / 'summary.csv').open('x') as stream:
        writer = csv.DictWriter(stream, list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if not smoke:
        choices = {}
        for seed in (0, 3):
            three = [r for r in rows if r['seed'] == seed and r['profile'] != 'FULL_BASE']
            # Both endpoints were explicitly authorized for this exploratory search.
            choices[str(seed)] = {metric: max(three, key=lambda r: r[metric])['profile']
                for metric in ('val_final_mAP', 'val_average_mAP', 'test_final_mAP', 'test_average_mAP')}
        write(root / 'exploratory_selection.json', dict(test_used_for_selection=True,
            interpretation='Exploratory parameter search; all configurations reported, not an untouched test estimate',
            choices_by_endpoint=choices))
    return rows


def run(args):
    if not args.run_id or '/' in args.run_id or args.run_id in {'.', '..'}:
        raise ValueError('Invalid run ID')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Clean checkout required')
    if shutil.disk_usage('.').free < 30 * 1024 ** 3:
        raise RuntimeError('Need 30 GiB free output space')
    if args.max_parallel != 4:
        raise ValueError('This queue declares four slots on GPU0')
    if not args.smoke:
        gate = json.loads((args.smoke_root / 'completion.json').read_text())
        assert gate['smoke'] and gate['status'] == 'complete' and gate['native_equivalence']
        assert all(v['all_method_tensors_losses_and_predictions_equal'] for v in gate['native_equivalence'].values())
    root = Path('output/historical_b32_search') / args.run_id
    logs = Path('logs/historical_b32_search') / args.run_id
    if root.exists() or logs.exists():
        raise FileExistsError(args.run_id)
    root.mkdir(parents=True); logs.mkdir(parents=True)
    cells = experiment_cells(args.smoke)
    protocol = dict(status='running', smoke=args.smoke, gpu=0, parallelism=4, cpu_threads=72,
        seeds=[0, 1, 3], tasks=2 if args.smoke else 8, test_used_for_parameter_search=not args.smoke,
        evaluation_scope='samples_intersect_seen_classes', source_commit=LEGACY_COMMIT,
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        jobs=[dict(seed=s, profile=p, **recipe(p)) for s, p in cells])
    pending = []
    for seed, profile in cells:
        name = f'seed{seed}/{profile}'
        pending.append(dict(name=name + '/train', cell=name, kind='train', module='multi_lane.track_a.runner',
            arguments=training_arguments(args, root / name, seed, profile)))
    if args.smoke:
        for profile in ('THREE_BASE', 'FULL_BASE'):
            name = 'NATIVE_' + profile
            flags = training_arguments(args, root / name, 1, profile, native=True)
            path = root / (name + '_arguments.json'); write(path, flags)
            pending.append(dict(name=name + '/train', cell=name, kind='native', module='multi_lane.track_a.native_b32_reference',
                arguments=['--legacy-root', str(args.legacy_root), '--arguments-json', str(path)]))
    protocol['training_arguments'] = {r['cell']: r['arguments'] for r in pending}
    write(root / 'protocol.json', protocol)
    active = []; records = []; failed = False; audits = {}; native_equivalence = None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='72', MKL_NUM_THREADS='72', PYTHONUNBUFFERED='1')
    try:
        while pending or active:
            for process, record in list(active):
                code = process.poll()
                if code is None:
                    continue
                active.remove((process, record)); record.update(exit=code, finished_unix=time.time())
                print('FINISH', record['name'], code, flush=True); failed |= code != 0
                if code == 0 and record['kind'] == 'train':
                    try:
                        name = record['cell']; audit = verify_train(root / name / 'train', args.smoke)
                        audits[name] = audit; write(root / name / 'training_verification.json', audit)
                        if not args.smoke and name == 'seed1/THREE_BASE':
                            anchor = compare_archive(root / name / 'train', args.historical_run)
                            write(root / 'historical_seed1_reproduction.json', anchor)
                            if anchor['status'] != 'exact':
                                raise RuntimeError('Historical seed1 b32 did not reproduce; fix before further search')
                        pending += [evaluation_job(args, root, name, 'val'),
                                    evaluation_job(args, root, name, 'val' if args.smoke else 'test', replay=args.smoke)]
                    except Exception as error:
                        # Let already running jobs reach their saved boundaries;
                        # keep every unstarted command for a reviewable recovery.
                        failed = True; record['verification_error'] = repr(error)
                        print('VERIFICATION_FAILED', record['name'], repr(error), flush=True)
            if failed and not active:
                raise RuntimeError('A child failed; unstarted work remains in queue.json')
            if pending and not failed and len(active) < 4:
                free = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip())
                if free >= 6000:
                    job = pending.pop(0); log = logs / (job['name'] + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, '-u', '-m', job['module'], *job['arguments']]
                    with log.open('x') as stream:
                        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    record = dict(job, pid=process.pid, gpu=0, command=command, log=str(log), started_unix=time.time())
                    active.append((process, record)); records.append(record)
                    print('START', job['name'], process.pid, 'gpu=0', flush=True)
            write(logs / 'queue.json', dict(records=records, pending=pending, failed=failed))
            if pending or active:
                time.sleep(3)
        if args.smoke:
            native_equivalence = verify_native(root)
        for seed in {s for s, _ in cells}:
            same = [a for a in audits.values() if a['seed'] == seed]
            assert len({a['initial_base_sha256'] for a in same}) == 1
            three = [a['cpu_rng_at_task_boundaries'] for a in same if a['profile'] != 'FULL_BASE']
            assert all(v == three[0] for v in three)
        summarize(root, cells, args.smoke)
        protocol.update(status='complete', native_equivalence=native_equivalence,
            checkpoint_and_frozen_dependency_audits_passed=True,
            same_seed_initialization_equal=True, same_seed_three_view_sampler_rng_equal=True)
        write(root / 'completion.json', protocol)
        print('HISTORICAL_B32_SEARCH_COMPLETE', root, flush=True)
    except BaseException as error:
        write(root / 'failure.json', dict(error=repr(error), active=records, pending=pending))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    for key in ('data-root', 'clip-checkpoint', 'face-manifest-root', 'legacy-root', 'historical-run'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-root', type=Path)
    parser.add_argument('--max-parallel', type=int, default=4)
    run(parser.parse_args())
