"""Adopt an existing strong-ASL queue without restarting training; GPU0 evaluations only.

Run this Git-tracked script from the original experiment checkout, whose model
code and protocol remain unchanged. Stop the original dispatcher before use.
"""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def proc_info(pid):
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(state=fields[0], start=fields[19],
                    command=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except FileNotFoundError:
        return None


def adopted_exit(record, info, log_text):
    if info and info['start'] == record['adopted_start_ticks'] and info['state'] != 'Z':
        return None
    marker = ('HISTORICAL_FULL_REFERENCE_COMPLETE' if record['name'] == 'R_LEGACY/train'
              else 'MULTI_LANE_TRACK_A_COMPLETE')
    return 0 if marker in log_text and 'Traceback (most recent call last)' not in log_text else 1


def gpu0_eligible(active_records, available_mib, limit=2):
    return (not any(r['gpu'] == 0 and r['name'].endswith('/train') for r in active_records)
            and sum(r['gpu'] == 0 for r in active_records) < limit and available_mib >= 6500)


def main(args):
    # Import the original experiment's modules, not the dispatcher checkout's model.
    sys.path.insert(0, str(Path.cwd()))
    from multi_lane.track_a.run_module_stacking import write
    from multi_lane.track_a.run_strong_full_asl import evaluation_job, summarize, memory_state
    from multi_lane.track_a.verify_strong_full_asl import verify

    root = Path('output/strong_full_asl_three_view') / args.run_id
    logs = Path('logs/strong_full_asl_three_view') / args.run_id
    status = logs / 'pipeline_queue.json'
    protocol = json.loads((root / 'protocol.json').read_text())
    assert protocol['smoke'] is False and protocol['tasks'] == 8
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == protocol['head']
    assert not (root / 'completion.json').exists()
    old = proc_info(args.old_controller)
    assert old and old['state'] == 'T' and 'multi_lane.track_a.run_strong_full_asl' in old['command']
    snapshot = json.loads(status.read_text())
    assert not snapshot['failed']
    records = snapshot['records']
    assert all(r.get('exit', 0) == 0 for r in records)
    active = [r for r in records if 'exit' not in r]
    for r in active:
        # No active evaluation is interrupted or silently assigned a fake exit code.
        assert r['name'].endswith('/train'), r['name']
        info = proc_info(r['pid'])
        if info and info['state'] != 'Z':
            assert r['module'] in info['command'] and str(root) in info['command']
        r['adopted_start_ticks'] = info['start'] if info else 'already_exited'
    pending = list(snapshot['pending'])
    assert all(not j['name'].endswith('/train') for j in pending)
    print(json.dumps(dict(adopting=[(r['name'], r['pid'], r['gpu']) for r in active],
                          evaluation_gpus=[0], pending=len(pending)), ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    policy = root / 'gpu0_evaluation_handoff.json'
    if policy.exists():
        raise FileExistsError(policy)
    write(policy, dict(status='adopted', changed_unix=time.time(), old_controller=args.old_controller,
                       controller_pid=os.getpid(), original_model_head=protocol['head'],
                       script_head=subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[2]),
                                                           'rev-parse', 'HEAD'], text=True).strip(),
                       evaluation_gpus=[0], eval_per_gpu=2, training_restarted=False,
                       preserved_training=[dict(name=r['name'], pid=r['pid'], gpu=r['gpu'],
                                                start_ticks=r['adopted_start_ticks']) for r in active]))
    # SIGTERM is delivered on continuing the stopped dispatcher; children are untouched.
    import signal
    os.kill(args.old_controller, signal.SIGTERM)
    os.kill(args.old_controller, signal.SIGCONT)
    completed = {r['name'].split('/')[0] for r in records
                 if r['name'].endswith('/train') and r.get('exit') == 0}
    known = {r['name'] for r in records} | {j['name'] for j in pending}
    params = SimpleNamespace(smoke=False, **protocol['inputs'])
    controls = {'T_ASL': [('off_A', ['A'])], 'T_ASL_ATTN': [('off_B', ['B'])],
                'T_ASL_POS': [('off_C', ['C'])],
                'T_ASL_BOTH': [('off_B', ['B']), ('off_C', ['C']), ('off_BC', ['B', 'C'])]}
    owned = {}

    def enqueue(job):
        if job['name'] not in known:
            pending.append(job)
            known.add(job['name'])

    def ready(r):
        arm, kind = r['name'].split('/')
        if kind == 'train':
            completed.add(arm)
            if arm != 'R_LEGACY':
                with (logs / (arm + '_verification.log')).open('x') as stream:
                    with redirect_stdout(stream):
                        verify(root, 8, False, arms=('S_BCE', arm),
                               output_name='source_verification_' + arm + '.json')
            else:
                assert json.loads((root / arm / 'reference_verification.json').read_text())['status'] == 'verified'
            for split in ('val', 'test'):
                enqueue(evaluation_job(params, root, arm, split, 'full_' + split))
        elif kind == 'full_val':
            for name, disabled in controls.get(arm, []):
                enqueue(evaluation_job(params, root, arm, 'val', 'runtime_' + name,
                                       ['--checkpoint-tasks', '0', '2', '7',
                                        '--disable-stacking-modules', *disabled]))

    failed = False
    try:
        while pending or active:
            for r in list(active):
                if r['pid'] in owned:
                    code = owned[r['pid']].poll()
                else:
                    code = adopted_exit(r, proc_info(r['pid']), Path(r['log']).read_text(errors='replace'))
                if code is None:
                    continue
                r.update(exit=code, finished_unix=time.time())
                active.remove(r)
                print('FINISH %s gpu=%s exit=%s' % (r['name'], r['gpu'], code), flush=True)
                failed |= code != 0
                write(status, dict(records=records, pending=pending, failed=failed, evaluation_gpus=[0]))
                if code == 0:
                    ready(r)
            if failed:
                raise RuntimeError('An adopted training/evaluation failed; no further jobs dispatched')
            if pending and gpu0_eligible(active, memory_state([0])[0]['free']):
                job = pending.pop(0)
                log = logs / (job['name'] + '.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
                code = ('import runpy,torch;torch.cuda.set_per_process_memory_fraction(0.3);'
                        'runpy.run_module(%r,run_name="__main__")' % job['module'])
                command = [sys.executable, '-u', '-c', code, *job['arguments']]
                with log.open('x') as stream:
                    process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
                r = dict(job, pid=process.pid, gpu=0, log=str(log), command=command,
                         allocator_fraction=.3, started_unix=time.time())
                records.append(r)
                active.append(r)
                owned[process.pid] = process
                print('START %s pid=%s gpu=0' % (job['name'], process.pid), flush=True)
            write(status, dict(records=records, pending=pending, failed=failed, evaluation_gpus=[0]))
            if pending or active:
                time.sleep(3)
        with (logs / 'training_verification.log').open('x') as stream:
            with redirect_stdout(stream):
                verify(root, 8, False)
        summarize(root, protocol['arms'], False)
        protocol['status'] = 'complete'
        protocol['evaluation_handoff'] = str(policy)
        write(root / 'completion.json', protocol)
        (logs / 'batch_exit_status.txt').write_text('0\n')
        print('STRONG_FULL_ASL_COMPLETE ' + str(root), flush=True)
    except BaseException as error:
        write(root / 'failure.json', dict(error=repr(error), controller='gpu0_handoff'))
        (logs / 'batch_exit_status.txt').write_text('1\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--old-controller', type=int, required=True)
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args())
