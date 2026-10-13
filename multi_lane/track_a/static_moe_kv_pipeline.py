"""Run paired validation-only KV smoke/full diagnostics across available GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import BoundedSemaphore

from .runner import git_metadata


def assess_value_only_gate(summaries):
    """Predeclared descriptive gate; passing is not proof of retraining benefit."""
    rows={}
    for key in ('v1_MOE_OVERLAP','v2_MOE_OVERLAP'):
        curves=summaries[key]['curve_summary']
        rows[key]={}
        for metric in ('final_mAP','average_mAP'):
            rows[key][metric]={
                'value_minus_both':curves['value_only'][metric]-curves['both'][metric],
                'value_minus_none':curves['value_only'][metric]-curves['none'][metric],
                'key_minus_none':curves['key_only'][metric]-curves['none'][metric],
            }
    def passes(row):
        f,a=row['final_mAP'],row['average_mAP']
        return (f['value_minus_both']>=.1 and f['value_minus_none']>=.1
                and f['key_minus_none']<=0 and a['value_minus_both']>=0
                and a['value_minus_none']>=0 and a['key_minus_none']<=0)
    approved=all(passes(r) for r in rows.values())
    return dict(supports_implementing_value_only=approved,required_versions=['v1','v2'],
                rule='Both overlap versions: final value-both >=0.1, value-none >=0.1, key-none <=0; same average directions',
                details=rows,training_started=False,
                interpretation='Inference intervention on joint checkpoints; conditional support, not proof of training improvement')


def launch(batch_id,v1_batch,v2_batch,gpus,data_root,clip_checkpoint,face_manifest_root):
    root=Path(__file__).resolve().parents[2]
    metadata=git_metadata(root)
    if metadata['dirty'] or Path(batch_id).name!=batch_id or not 1<=len(gpus)<=5 or len(set(gpus))!=len(gpus):
        raise ValueError('Clean worktree, simple batch ID and 1--5 different GPUs required')
    output=root/'output/emotic_static_moe_kv'/batch_id
    logs=root/'logs/emotic_static_moe_kv'/batch_id
    output.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    jobs=[('v1','SHARED',v1_batch),('v1','MOE_UNIFORM',v1_batch),('v1','MOE_OVERLAP',v1_batch),
          ('v2','SHARED',v2_batch),('v2','MOE_OVERLAP',v2_batch)]
    jobs=[(*job,str(gpus[i%len(gpus)])) for i,job in enumerate(jobs)]
    slots={str(g):BoundedSemaphore(2) for g in gpus}
    state=dict(status='preflight',git=metadata,split='val',test_accessed=False,training_performed=False,
               jobs=[{'version':v,'method':m,'source':str(b),'gpu':g} for v,m,b,g in jobs],
               started_at=datetime.now(timezone.utc).isoformat())
    state_path=output/'pipeline_state.json'
    def write():state_path.write_text(json.dumps(state,indent=2)+'\n')
    write()
    try:
        for gpu in gpus:
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free',
                                             '--format=csv,noheader,nounits'],text=True).strip())
            required=8192*min(2,sum(job[-1]==str(gpu) for job in jobs))
            if free<required:raise RuntimeError(f'GPU {gpu} has insufficient free memory: {free} < {required}')
        def run_unlimited(job,smoke):
            version,method,batch,gpu=job;key=version+'_'+method
            phase='smoke' if smoke else 'full_val'
            command=[sys.executable,'-u','-m','multi_lane.track_a.evaluate_static_moe_kv',
                     '--source-batch',str(batch),'--output',str(output/phase/key),
                     '--version',version,'--method',method,'--data-root',str(data_root),
                     '--clip-checkpoint',str(clip_checkpoint),'--face-manifest-root',str(face_manifest_root)]
            if smoke:command+=['--smoke']
            destination=output/phase/key;destination.parent.mkdir(exist_ok=True)
            with (logs/(phase+'_'+key+'.log')).open('w') as h:
                p=subprocess.run(command,cwd=root,stdout=h,stderr=subprocess.STDOUT,
                                 env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1'))
            print(f'STATIC_MOE_KV_PIPELINE phase={phase} source={key} exit={p.returncode}',flush=True)
            return key,p.returncode
        def run(job,smoke):
            with slots[job[-1]]:
                return run_unlimited(job,smoke)
        for smoke in (True,False):
            state['status']='preflight' if smoke else 'validation_running';write()
            with ThreadPoolExecutor(max_workers=5) as pool:exits=dict(pool.map(lambda job:run(job,smoke),jobs))
            state['smoke_exits' if smoke else 'formal_exits']=exits;write()
            if any(exits.values()):raise RuntimeError(f'KV phase failed: {exits}; outputs retained')
        summaries={v+'_'+m:json.loads((output/'full_val'/(v+'_'+m)/'summary.json').read_text()) for v,m,b,g in jobs}
        gate=assess_value_only_gate(summaries)
        (output/'decision.json').write_text(json.dumps(gate,indent=2)+'\n')
        (output/'comparison.json').write_text(json.dumps(dict(status='complete',split='val',results=summaries,decision=gate),indent=2)+'\n')
        state.update(status='complete',decision=gate,finished_at=datetime.now(timezone.utc).isoformat());write()
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}');write();raise
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch-id',required=True)
    for key in ('v1-batch','v2-batch','data-root','clip-checkpoint','face-manifest-root'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gpus',nargs='+',required=True)
    launch(**vars(p.parse_args()))
