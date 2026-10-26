"""Verify complete sources, profile in isolation, evaluate eight stages on GPU0."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

from .fixed_model_mean_forward import CONDITIONS
from .run_historical_b32_test_attribution import read, write
from .run_protected_view_forward import common, sources


def run(args):
    assert args.run_id and '/' not in args.run_id and args.run_id not in ('.','..')
    assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    root=Path('output/fixed_model_mean_forward')/args.run_id
    logs=Path('logs/fixed_model_mean_forward')/args.run_id
    root.mkdir(parents=True,exist_ok=False); logs.mkdir(parents=True,exist_ok=True)
    source=sources(args)
    protocol=dict(status='running',head=head,seed=1,gpu=0,training_updates=0,tasks=list(range(8)),
        conditions=list(CONDITIONS),batch_size=64,workers=2,threads=72,amp=True,tf32=True,
        evaluation_scope='samples_intersect_seen_classes',test_used_for_parameter_search=True,
        source_checkpoint_sha256=source,visual_weights_shared=True,frozen_tokens_cached=False,
        classifier_head='corresponding_stage_B0_shared_post_feature_mean',source_full_summaries_preserved=True,
        fixed_mean_coefficient=.5,mean_before_head=True,new_normalization=False,
        queue_parallelism=2,resource_profiles_serial=True)
    write(root/'protocol.json',protocol)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='72',MKL_NUM_THREADS='72',PYTHONUNBUFFERED='1')
    records=[]

    def start(action,name,flags):
        path=logs/(name+'.log'); path.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u','-m','multi_lane.track_a.fixed_model_mean_forward',action,
                 *common(args),*flags,'--output',str(root/name)]
        with path.open('a') as f:
            process=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(name=name,pid=process.pid,arguments=command,started_unix=time.time(),log=str(path))
        records.append(record); write(logs/'queue.json',dict(records=records,status='running'))
        print('START',name,process.pid,flush=True)
        return process,record

    def finish(process,record):
        code=process.wait(); record.update(exit=code,finished_unix=time.time())
        write(logs/'queue.json',dict(records=records,status='running' if code==0 else 'failed'))
        if code: raise RuntimeError(f"{record['name']} failed: {code}; see {record['log']}")
        value=read(root/record['name']/'result.json')
        assert value['status']=='complete' and value['git']['commit']==head
        print('FINISH',record['name'],flush=True)

    try:
        finish(*start('preflight','preflight',['--split','val']))
        assert read(root/'preflight/result.json')['source_predictions_exact']
        # Each benchmark runs in its own process, before concurrent evaluation.
        active=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid',
            '--format=csv,noheader,nounits'],text=True).splitlines()
        assert not [x for x in active if x.strip().isdigit()], 'GPU0 busy: cannot isolate timing'
        for arm in ('A','R5'):
            finish(*start('benchmark','benchmark/'+arm,['--arm',arm]))
        pending=[start('eval','evaluation/'+split,['--split',split]) for split in ('val','test')]
        while pending:
            for process,record in list(pending):
                if process.poll() is not None:
                    finish(process,record); pending.remove((process,record))
            if pending: time.sleep(3)
        assert sources(args)==source
        results={split:read(root/'evaluation'/split/'result.json') for split in ('val','test')}
        rows=[]
        for arm in CONDITIONS:
            row=dict(condition=arm)
            for split,result in results.items():
                assert result['tasks']==list(range(8)) and result['source_predictions_exact']
                row.update({split+'_'+k:v for k,v in result['summaries'][arm].items()
                            if k in ('final_mAP','average_mAP','final_cF1','final_oF1')})
            rows.append(row)
        with (root/'summary.csv').open('w') as f:
            writer=csv.DictWriter(f,list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write(root/'resource_summary.json',{arm:read(root/'benchmark'/arm/'result.json') for arm in ('A','R5')})
        write(logs/'queue.json',dict(records=records,status='complete'))
        write(root/'completion.json',dict(protocol,status='complete',source_checkpoints_unchanged=True,
            all_source_predictions_exact=True))
        print('FIXED_MODEL_MEAN_COMPLETE',root,flush=True)
    except BaseException as error:
        write(root/'failure.json',dict(error=repr(error))); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-id',required=True)
    for name in ('source-run','person-run','dual-run','reference-root','data-root','clip-checkpoint','face-manifest-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
