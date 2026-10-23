"""GPU0 exact archived Full protocol, native historical test, then historical val."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main(args):
    root=Path('output/strong_full_asl_three_view')/args.run_id
    logs=Path('logs/strong_full_asl_three_view')/args.run_id
    if root.exists() or logs.exists():raise FileExistsError(args.run_id)
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise RuntimeError('Clean checkout required')
    root.mkdir(parents=True);logs.mkdir(parents=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    protocol=dict(status='running',gpu=0,seed=0,tasks=8,epochs=30,batch_size=64,workers=2,
                  evaluation_scope='samples_intersect_seen_classes',source_commit='f1ada2291ae40aaf17416ae96d6431505f2bc575',
                  wrapper_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  historical_config=str(args.historical_config),test_used_for_selection=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    jobs=[('train',[sys.executable,'-u','-m','multi_lane.track_a.legacy_full_reference',
                    '--legacy-root',str(args.legacy_root),'--output',str(root/'reference'),
                    '--data-root',str(args.data_root),'--clip-checkpoint',str(args.clip_checkpoint),
                    '--historical-config',str(args.historical_config)]),
          ('validation',[sys.executable,'-u','-m','multi_lane.track_a.evaluate_moe_task_ffn',
                         '--source-run',str(root/'reference/export'),'--output',str(root/'historical_val'),
                         '--split','val','--data-root',str(args.data_root),'--clip-checkpoint',str(args.clip_checkpoint),
                         '--face-manifest-root',str(args.face_manifest_root),'--expected-tasks','8',
                         '--batch-size','64','--workers','0'])]
    try:
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
    for key in ['legacy-root','data-root','clip-checkpoint','face-manifest-root','historical-config']:
        p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
