"""Resume interrupted task-boundary snapshots; all remaining work uses GPU0.

Run from the original clean model checkout. Completed arms are read-only links;
interrupted arms resume into a fresh output tree without overwriting originals.
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


def recovery_arguments(original, destination, checkpoint):
    args = list(original)
    assert '--resume-spatial-boundary' not in args
    args[args.index('--output-root') + 1] = str(destination)
    return args + ['--resume-spatial-boundary', str(checkpoint)]


def eligible(job, active, free_mib):
    if free_mib < 6500 or len(active) >= 5:
        return False
    return job['name'].endswith('/train') or sum(
        not r['name'].endswith('/train') for _, r in active) < 2


def main(args):
    sys.path.insert(0, str(Path.cwd()))
    from multi_lane.track_a.run_module_stacking import write
    from multi_lane.track_a.run_strong_full_asl import evaluation_job, summarize, memory_state
    from multi_lane.track_a.verify_strong_full_asl import verify
    from multi_lane.track_a.spatial_resume import sha256
    source = Path('output/strong_full_asl_three_view') / args.source_run_id
    root = Path('output/strong_full_asl_three_view') / args.run_id
    logs = Path('logs/strong_full_asl_three_view') / args.run_id
    assert source != root and not root.exists() and not logs.exists()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    protocol = json.loads((source / 'protocol.json').read_text())
    assert protocol['head'] == subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    assert protocol['tasks'] == 8 and not protocol['smoke']
    arms = ['T_BCE', 'T_ASL', 'T_ASL_ATTN', 'T_ASL_POS', 'T_ASL_BOTH']
    checkpoints = {}
    for arm in arms:
        candidates = list((source / arm / 'train/recovery_checkpoints').glob('task*_before_reporting.pth'))
        checkpoints[arm] = max(candidates, key=lambda p: int(p.name.split('_')[0][4:]))
    if args.dry_run:
        print(json.dumps(dict(source=str(source), destination=str(root), gpu=0,
                              checkpoints={a:str(p) for a,p in checkpoints.items()}), indent=2), flush=True)
        return
    root.mkdir(parents=True)
    logs.mkdir(parents=True)
    # Reuse completed Full runs and their validated evaluations without copying weights.
    for arm in ['S_BCE', 'S_ASL', 'R_LEGACY']:
        assert (source / arm / 'full_val/result.json').is_file()
        assert (source / arm / 'full_test/result.json').is_file()
        (root / arm).symlink_to((source / arm).resolve(), target_is_directory=True)
    jobs = []
    recovery = {}
    for arm in arms:
        cp = checkpoints[arm]
        recovery[arm] = dict(checkpoint=str(cp), sha256=sha256(cp),
                             boundary_task=int(cp.name.split('_')[0][4:]))
        jobs.append(dict(name=arm+'/train', module='multi_lane.track_a.runner',
                         arguments=recovery_arguments(protocol['training_arguments'][arm], root/arm/'train', cp)))
    protocol.update(status='running', original_run=str(source), gpus=[0], train_per_gpu=5,
                    eval_per_gpu=2, recovery=recovery,
                    recovery_reason='tmux session closure interrupted training during dispatcher handoff',
                    original_training_restarted_from_task0=False,
                    controller_head=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[2]),
                                                             'rev-parse','HEAD'],text=True).strip())
    write(root/'protocol.json', protocol)
    params = SimpleNamespace(smoke=False, **protocol['inputs'])
    controls = {'T_ASL':[('off_A',['A'])], 'T_ASL_ATTN':[('off_B',['B'])],
                'T_ASL_POS':[('off_C',['C'])],
                'T_ASL_BOTH':[('off_B',['B']),('off_C',['C']),('off_BC',['B','C'])]}
    pending = list(jobs)
    active = []
    records = []
    known = {j['name'] for j in pending}

    def enqueue(job):
        assert job['name'] not in known
        known.add(job['name'])
        pending.append(job)

    def ready(record):
        arm, kind = record['name'].split('/')
        if kind == 'train':
            with (logs/(arm+'_verification.log')).open('x') as stream:
                with redirect_stdout(stream):
                    verify(root,8,False,arms=('S_BCE',arm),output_name='source_verification_'+arm+'.json')
            for split in ['val','test']:
                enqueue(evaluation_job(params,root,arm,split,'full_'+split))
        elif kind == 'full_val':
            for name, disabled in controls.get(arm,[]):
                enqueue(evaluation_job(params,root,arm,'val','runtime_'+name,
                                       ['--checkpoint-tasks','0','2','7','--disable-stacking-modules',*disabled]))

    try:
        failed = False
        while active or pending:
            for process, record in list(active):
                code = process.poll()
                if code is None:
                    continue
                record.update(exit=code,finished_unix=time.time())
                active.remove((process,record))
                failed |= code != 0
                print('FINISH %s gpu=0 exit=%s'%(record['name'],code),flush=True)
                write(logs/'pipeline_queue.json',dict(records=records,pending=pending,failed=failed,gpus=[0]))
                if code == 0:
                    ready(record)
            if failed:
                # Keep this tmux/controller alive until any other owned children exit.
                # Do not dispatch more work or close their session while they run.
                if not active:
                    raise RuntimeError('Recovery job failed; existing child jobs allowed to finish')
            elif pending and eligible(pending[0],active,memory_state([0])[0]['free']):
                job = pending.pop(0)
                log = logs/(job['name']+'.log')
                log.parent.mkdir(parents=True,exist_ok=True)
                env = dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
                code = ('import runpy,torch;torch.cuda.set_per_process_memory_fraction(0.3);'
                        'runpy.run_module(%r,run_name="__main__")'%job['module'])
                command = [sys.executable,'-u','-c',code,*job['arguments']]
                with log.open('x') as stream:
                    process = subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,
                                               start_new_session=True)
                record = dict(job,pid=process.pid,gpu=0,command=command,log=str(log),started_unix=time.time())
                active.append((process,record))
                records.append(record)
                print('START %s pid=%s gpu=0'%(job['name'],process.pid),flush=True)
            write(logs/'pipeline_queue.json',dict(records=records,pending=pending,failed=failed,gpus=[0]))
            if active or pending:
                time.sleep(3)
        # Confirm immutable original checkpoints are still identical.
        assert all(sha256(Path(r['checkpoint']))==r['sha256'] for r in recovery.values())
        with (logs/'training_verification.log').open('x') as stream:
            with redirect_stdout(stream):
                verify(root,8,False)
        summarize(root,protocol['arms'],False)
        protocol['status']='complete'
        write(root/'completion.json',protocol)
        (logs/'batch_exit_status.txt').write_text('0\n')
        print('STRONG_FULL_ASL_RECOVERY_COMPLETE '+str(root),flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error)))
        (logs/'batch_exit_status.txt').write_text('1\n')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run-id',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--dry-run',action='store_true')
    main(parser.parse_args())
