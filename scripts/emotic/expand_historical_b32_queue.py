"""Adopt the live b32 dispatcher and raise GPU0 concurrency without restarting jobs.

The model checkout stays immutable. The old dispatcher's tmux pane remains alive
and stopped until all work finishes, so its training children retain their session.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace


def proc_info(pid):
    try:
        root = Path('/proc') / str(pid)
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(state=fields[0], start=fields[19],
            command=(root / 'cmdline').read_bytes().replace(b'\0', b' ').decode(),
            wait_status=int(fields[49]) if len(fields) > 49 else None)
    except FileNotFoundError:
        return None


def adopted_status(record, info, text, complete):
    same = info is not None and info['start'] == record['adopted_start_ticks']
    if same and info['state'] != 'Z':
        return None, None
    status = info.get('wait_status') if same else None
    if status is not None:
        code = os.waitstatus_to_exitcode(status)
        if code:
            return code, 'linux_proc_wait_status'
    evidence = 'linux_proc_wait_status' if status is not None else 'artifacts_and_log_after_pid_exit'
    ok = complete and 'MULTI_LANE_TRACK_A_COMPLETE' in text and 'Traceback (most recent call last)' not in text
    return (0 if ok else 1), evidence


DEFAULT_POLICY = dict(max_parallel=6, max_eval_parallel=4, minimum_free_gpu_mib=6000,
                      minimum_free_shm_mib=1024, launch_interval_seconds=12)


def validate_policy(policy):
    if set(policy) != set(DEFAULT_POLICY):
        raise ValueError('Runtime policy must contain exactly the declared resource controls')
    if (type(policy['max_parallel']) is not int or not 1 <= policy['max_parallel'] <= 6
            or type(policy['max_eval_parallel']) is not int or not 1 <= policy['max_eval_parallel'] <= policy['max_parallel']
            or policy['minimum_free_gpu_mib'] < 6000 or policy['minimum_free_shm_mib'] < 1024
            or policy['launch_interval_seconds'] < 5):
        raise ValueError('Runtime policy exceeds the audited concurrency/resource limits')
    return policy


def eligible(job, active, free_gpu, free_shm, policy):
    validate_policy(policy)
    return (len(active) < policy['max_parallel']
        and (job['kind'] != 'eval' or sum(r['kind'] == 'eval' for r in active) < policy['max_eval_parallel'])
        and free_gpu >= policy['minimum_free_gpu_mib'] and free_shm >= policy['minimum_free_shm_mib'])


def get_flag(flags, flag):
    return flags[flags.index(flag) + 1]


def main(args):
    source = args.source_worktree.resolve()
    sys.path.insert(0, str(source))
    from multi_lane.track_a.run_module_stacking import write
    from multi_lane.track_a.run_historical_b32_search import verify_train, compare_archive, evaluation_job, summarize
    from multi_lane.track_a.historical_b32_search import experiment_cells
    root = source / 'output/historical_b32_search' / args.run_id
    logs = source / 'logs/historical_b32_search' / args.run_id
    status_path = logs / 'queue.json'
    handoff_path = root / 'concurrency_handoff.json'
    policy_path = root / 'runtime_scheduling_policy.json'
    protocol = json.loads((root / 'protocol.json').read_text())
    assert not protocol['smoke'] and protocol['tasks'] == 8 and protocol['gpu'] == 0
    assert not (root / 'completion.json').exists() and not (root / 'failure.json').exists()
    assert not handoff_path.exists()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() == protocol['head']
    old = proc_info(args.old_controller)
    assert old and args.run_id in old['command'] and 'multi_lane.track_a.run_historical_b32_search' in old['command']
    assert Path(os.readlink(f'/proc/{args.old_controller}/cwd')) == source
    snapshot = json.loads(status_path.read_text())
    assert not snapshot['failed']
    print('PREFLIGHT', json.dumps(dict(active=[r['name'] for r in snapshot['records'] if 'exit' not in r],
        pending=len(snapshot['pending']), batch_size_unchanged=64, policy=DEFAULT_POLICY)), flush=True)
    if args.dry_run:
        return
    # Stop only dispatch; existing training processes continue without a signal.
    os.kill(args.old_controller, signal.SIGSTOP)
    for _ in range(100):
        current = proc_info(args.old_controller)
        if current and current['state'] == 'T':
            break
        time.sleep(.05)
    else:
        raise RuntimeError('Original dispatcher did not stop')
    try:
        snapshot = json.loads(status_path.read_text())
        assert not snapshot['failed']
        records = snapshot['records']; pending = snapshot['pending']
        assert all(r.get('exit', 0) == 0 for r in records)
        active = [r for r in records if 'exit' not in r]
        for r in active:
            assert r['kind'] == 'train', 'This handoff adopts live training, not in-progress evaluations'
            info = proc_info(r['pid'])
            if info and info['state'] != 'Z':
                assert r['module'] in info['command'] and args.run_id in info['command']
                assert Path(os.readlink(f"/proc/{r['pid']}/cwd")) == source
            r['adopted_start_ticks'] = info['start'] if info else 'already_exited'
        write(root / 'queue_before_concurrency_handoff.json', snapshot)
        write(policy_path, DEFAULT_POLICY)
        handoff = dict(status='adopted', old_controller=args.old_controller, old_controller_start=old['start'],
            controller_pid=os.getpid(), model_head=protocol['head'],
            controller_head=subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]),
                'rev-parse', 'HEAD'], text=True).strip(), training_restarted=False, batch_size_changed=False,
            gpu=0, initial_policy=DEFAULT_POLICY, adopted_unix=time.time(),
            preserved_processes=[dict(name=r['name'], pid=r['pid'], start=r['adopted_start_ticks']) for r in active])
        write(handoff_path, handoff)
    except BaseException:
        # Before adoption succeeds the old queue remains the only dispatcher.
        os.kill(args.old_controller, signal.SIGCONT)
        raise
    first = protocol['training_arguments']['seed1/THREE_BASE']
    params = SimpleNamespace(smoke=False, data_root=Path(get_flag(first, '--data-root')),
        clip_checkpoint=Path(get_flag(first, '--clip-checkpoint')),
        face_manifest_root=Path(get_flag(first, '--face-manifest-root')))
    archive = (source / args.historical_run).resolve()
    owned = {}; audits = {}; known = {r['name'] for r in records} | {j['name'] for j in pending}
    failed = False; last_launch = 0.; last_policy = None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='72', MKL_NUM_THREADS='72', PYTHONUNBUFFERED='1')

    def enqueue(job):
        if job['name'] not in known:
            known.add(job['name']); pending.append(job)

    def ready(record):
        if record['kind'] != 'train':
            return
        cell = record['cell']; audit = verify_train(root / cell / 'train', False)
        audits[cell] = audit; write(root / cell / 'training_verification.json', audit)
        if cell == 'seed1/THREE_BASE':
            anchor = compare_archive(root / cell / 'train', archive)
            write(root / 'historical_seed1_reproduction.json', anchor)
            if anchor['status'] != 'exact':
                raise RuntimeError('Historical b32 seed1 mismatch; stop further dispatch')
        for split in ('val', 'test'):
            # Keep relative data paths relative to the original model checkout.
            enqueue(evaluation_job(params, root, cell, split))

    try:
        for r in records:
            if r.get('exit') == 0:
                ready(r)
        while pending or active:
            policy = validate_policy(json.loads(policy_path.read_text()))
            if policy != last_policy:
                with (root / 'runtime_policy_history.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(unix=time.time(), policy=policy)) + '\n')
                print('POLICY', json.dumps(policy), flush=True); last_policy = policy
            for r in list(active):
                if r['pid'] in owned:
                    code = owned[r['pid']].poll(); evidence = 'subprocess_returncode'
                else:
                    summary = root / r['cell'] / 'train/seed_summary.json'
                    complete = summary.exists() and json.loads(summary.read_text()).get('status') == 'complete'
                    code, evidence = adopted_status(r, proc_info(r['pid']), (source / r['log']).read_text(), complete)
                if code is None:
                    continue
                r.update(exit=code, exit_evidence=evidence, finished_unix=time.time()); active.remove(r)
                print('FINISH', r['name'], code, evidence, flush=True); failed |= code != 0
                if code == 0:
                    try:
                        ready(r)
                    except Exception as error:
                        failed = True; r['verification_error'] = repr(error)
                        print('VERIFICATION_FAILED', r['name'], repr(error), flush=True)
            if failed and not active:
                raise RuntimeError('A job failed verification; remaining commands preserved')
            if pending and not failed and time.time() - last_launch >= policy['launch_interval_seconds']:
                free_gpu = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.free',
                    '--format=csv,noheader,nounits'], text=True).strip())
                shm = os.statvfs('/dev/shm'); free_shm = shm.f_bavail * shm.f_frsize / 1024 ** 2
                index = next((i for i, j in enumerate(pending) if eligible(j, active, free_gpu, free_shm, policy)), None)
                if index is not None:
                    job = pending.pop(index); log = logs / (job['name'] + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, '-u', '-m', job['module'], *job['arguments']]
                    with log.open('x') as stream:
                        process = subprocess.Popen(command, cwd=source, env=env, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True)
                    r = dict(job, pid=process.pid, gpu=0, command=command, log=str(log),
                        started_unix=time.time(), model_source_head=protocol['head'])
                    active.append(r); records.append(r); owned[process.pid] = process; last_launch = time.time()
                    print('START', r['name'], process.pid, 'gpu=0', flush=True)
            write(status_path, dict(records=records, pending=pending, failed=failed,
                controller='six_slot_handoff', controller_pid=os.getpid(), policy=policy,
                handoff=str(handoff_path), model_source_head=protocol['head']))
            if pending or active:
                time.sleep(3)
        for seed in (0, 1, 3):
            same = [a for a in audits.values() if a['seed'] == seed]
            assert len({a['initial_base_sha256'] for a in same}) == 1
            three = [a['cpu_rng_at_task_boundaries'] for a in same if a['profile'] != 'FULL_BASE']
            assert all(x == three[0] for x in three)
        summarize(root, experiment_cells(False), False)
        protocol.update(status='complete', native_equivalence=None,
            checkpoint_and_frozen_dependency_audits_passed=True, same_seed_initialization_equal=True,
            same_seed_three_view_sampler_rng_equal=True, runtime_scheduler_handoff=handoff,
            initial_parallelism=protocol['parallelism'], parallelism=last_policy['max_parallel'])
        write(root / 'completion.json', protocol)
        # Only now release the old tmux pane; all of its training children ended.
        old_now = proc_info(args.old_controller)
        if old_now and old_now['start'] == old['start']:
            os.kill(args.old_controller, signal.SIGTERM); os.kill(args.old_controller, signal.SIGCONT)
        handoff.update(status='complete', finished_unix=time.time()); write(handoff_path, handoff)
        print('HISTORICAL_B32_SEARCH_COMPLETE', root, flush=True)
    except BaseException as error:
        write(root / 'failure.json', dict(error=repr(error), controller='six_slot_handoff',
            queue=str(status_path), old_controller_retained_stopped=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-worktree', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--old-controller', type=int, required=True)
    parser.add_argument('--historical-run', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
