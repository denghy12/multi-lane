"""Pause only identified experiment processes after an atomic task snapshot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def identity(pid, expected, root):
    proc = Path('/proc') / str(pid)
    command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    if expected not in command or (proc / 'cwd').resolve() != root.resolve():
        raise RuntimeError(f'Unexpected process identity: {pid}')
    # /proc/stat field 22; comm may contain spaces and parentheses.
    birth = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
    return birth


def pause(args):
    root = args.source_root.resolve()
    run = root / 'output/target_anchored_summarization' / args.run_id
    source = json.loads((run / 'launch.json').read_text())
    records = source['train']
    if len(records) != 3 or source['smoke'] or source['tasks'] != 8:
        raise ValueError('Expected the running full three-arm experiment')
    identities = {r['pid']: identity(r['pid'], 'multi_lane.track_a.runner', root) for r in records}
    identities[args.controller] = identity(args.controller, 'multi_lane.track_a.run_spatial_summary', root)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stopped = []
    report = dict(status='waiting_for_boundary', task=args.task, source=str(run),
                  controller=args.controller, processes={}, no_process_terminated=True)

    def stop(pid, expected):
        if identity(pid, expected, root) != identities[pid]:
            raise RuntimeError('Process ID was reused')
        os.kill(pid, signal.SIGSTOP)
        stopped.append(pid)

    try:
        stop(args.controller, 'multi_lane.track_a.run_spatial_summary')
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        pending = dict((r['phase'].removeprefix('train_'), r['pid']) for r in records)
        while pending:
            for arm, pid in list(pending.items()):
                path = run / arm / 'train/recovery_checkpoints' / f'task{args.task}_before_reporting.pth'
                identity(pid, 'multi_lane.track_a.runner', root)
                if not path.exists():
                    continue
                # atomic_save publishes only a complete file. Stop before extra
                # reporting or activation of the next task can advance anything.
                stop(pid, 'multi_lane.track_a.runner')
                report['processes'][arm] = dict(pid=pid, checkpoint=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), paused_at=time.time())
                del pending[arm]
                args.output.write_text(json.dumps(report, indent=2) + '\n')
                print(f'PAUSED arm={arm} pid={pid} task={args.task}', flush=True)
            if pending:
                time.sleep(.5)
        report['status'] = 'paused_at_saved_boundary'
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('ALL_THREE_PAUSED_WITH_RECOVERY', flush=True)
    except BaseException:
        for pid in stopped:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        report['status'] = 'pause_failed_processes_continued'
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--controller', type=int, required=True)
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    pause(p.parse_args())
