"""GPU0 exact archived Full protocol, native historical test, then historical val."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(args):
    sys.path.insert(0,str(Path.cwd()))
    if args.native_source is not None and (args.native_pid is None or args.native_log is None):
        raise ValueError('Native adoption needs its PID and immutable log path')
    root=Path('output/strong_full_asl_three_view')/args.run_id
    logs=Path('logs/strong_full_asl_three_view')/args.run_id
    if root.exists() or logs.exists():raise FileExistsError(args.run_id)
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Clean checkout required')
    root.mkdir(parents=True);logs.mkdir(parents=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72',PYTHONUNBUFFERED='1')
    protocol=dict(status='running',gpu=0,seed=args.seed,tasks=8,epochs=30,batch_size=64,workers=2,
                  cpu_initialization_threads=72,
                  evaluation_scope='samples_intersect_seen_classes',source_commit='f1ada2291ae40aaf17416ae96d6431505f2bc575',
                  wrapper_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  historical_config=str(args.historical_config),test_used_for_selection=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    jobs=[('train',[sys.executable,'-u','-m','multi_lane.track_a.legacy_full_reference',
                    '--legacy-root',str(args.legacy_root),'--output',str(root/'reference'),
                    '--data-root',str(args.data_root),'--clip-checkpoint',str(args.clip_checkpoint),
                    '--historical-config',str(args.historical_config),'--seed',str(args.seed)]),
          ('validation',[sys.executable,'-u','-m','multi_lane.track_a.evaluate_moe_task_ffn',
                         '--source-run',str(root/'reference/export'),'--output',str(root/'historical_val'),
                         '--split','val','--data-root',str(args.data_root),'--clip-checkpoint',str(args.clip_checkpoint),
                         '--face-manifest-root',str(args.face_manifest_root),'--expected-tasks','8',
                         '--batch-size','64','--workers','0'])]
    try:
        if args.native_source is not None:
            from multi_lane.track_a.legacy_full_reference import historical_configuration_differences, export_historical_reference
            from multi_lane.track_a.runner import git_metadata
            expected=json.loads(args.historical_config.read_text());expected['seed']=args.seed
            actual=json.loads((args.native_source/'train/config.json').read_text())
            assert not historical_configuration_differences(actual,expected)
            native_process=json.loads((args.native_source/'process.json').read_text())
            assert native_process['pid']==args.native_pid and native_process['gpu']==0
            protocol['adopted_native_source']=str(args.native_source)
            protocol['adopted_native_pid']=args.native_pid
            (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
            print('ADOPT_NATIVE',args.native_pid,'gpu=0',flush=True)
            while True:
                path=Path('/proc')/str(args.native_pid)/'stat'
                if not path.exists() or path.read_text().rsplit(')',1)[1].split()[0]=='Z':break
                time.sleep(3)
            summary=json.loads((args.native_source/'train/seed_summary.json').read_text())
            assert summary['status']=='complete' and summary['completed_optimizer_updates']==13950
            assert 'MULTI_LANE_TRACK_A_COMPLETE' in args.native_log.read_text()
            reference=root/'reference';reference.mkdir()
            (reference/'train').symlink_to((args.native_source/'train').resolve(),target_is_directory=True)
            export_historical_reference(reference/'train',reference/'export',git_metadata(Path.cwd()))
            anchor={'final_mAP':32.53651921448899,'average_mAP':39.144491444469374}
            record=dict(status='verified',strict_configuration_verified=True,original_metrics=summary['metrics'],
                        historical_anchor=anchor,historical_anchor_matched=(all(abs(summary['metrics'][k]-v)<1e-4 for k,v in anchor.items()) if args.seed==0 else None))
            (reference/'reference_verification.json').write_text(json.dumps(record,indent=2)+'\n')
            jobs=jobs[1:]
        for name,command in jobs:
            record=dict(name=name,command=command,gpu=0,started_unix=time.time())
            with (logs/(name+'.log')).open('x') as stream:
                p=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                record['pid']=p.pid
                (logs/(name+'_process.json')).write_text(json.dumps(record,indent=2)+'\n')
                print('START',name,'pid',p.pid,'gpu=0',flush=True)
                status=p.wait()
            record.update(exit=status,finished_unix=time.time())
            (logs/(name+'_process.json')).write_text(json.dumps(record,indent=2)+'\n')
            if status:raise RuntimeError('%s exited %d'%(name,status))
        reference=json.loads((root/'reference/reference_verification.json').read_text())
        assert reference['strict_configuration_verified']
        val=json.loads((root/'historical_val/result.json').read_text())
        result=dict(status='complete',historical_anchor_matched=reference['historical_anchor_matched'],
                    val=val['summary'],test=reference['original_metrics'],historical_anchor=reference['historical_anchor'])
        (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        protocol['status']='complete'
        (root/'completion.json').write_text(json.dumps(protocol,indent=2)+'\n')
        print('HISTORICAL_EXACT_REFERENCE_COMPLETE',json.dumps(result),flush=True)
    except BaseException as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e)),indent=2)+'\n')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    p.add_argument('--native-source',type=Path)
    p.add_argument('--native-pid',type=int)
    p.add_argument('--native-log',type=Path)
    for key in ['legacy-root','data-root','clip-checkpoint','face-manifest-root','historical-config']:
        p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
