"""TRAIN-only timing and exact-state comparison for execution changes."""
import argparse, json, os, time
from pathlib import Path
import numpy as np
import torch
from scripts.emotic.d9_task_patch_run import CONFIG, FIT
from scripts.emotic.d9_task_patch_readout import fit_one
from scripts.emotic.d9_task_patch_extract import load_features
from scripts.emotic.d9_final_patch_readout import predict
from scripts.emotic.d9_a_eight_eval import read
from scripts.emotic.evaluate_d9_relation import write_json
from multi_lane.track_a.task_patch_fusion import candidates, restore_gate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=30)
    parser.add_argument('--case',choices=('small','large','both'),default='both')
    args=parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_num_threads(1)
    cfg=read(CONFIG);out=Path('output/d9_task_patch_speed/seed1_20261128/benchmark');out.mkdir(parents=True,exist_ok=True)
    pairs=[('equal90',6),('full',3)]
    if args.case!='both':pairs=[pairs[args.case=='large']]
    rows=[]
    for name,task in pairs:
        stage=load_features(cfg,name,out,'train',task)
        for key in ('cls__common','cls_regions__common','cls_patch__common'):
            spec=next(s for s in candidates() if s['key']==key)
            # Both implementations warm up their kernels before measured fits.
            for fast in (False,True):fit_one(stage,spec,dict(FIT,steps=2),'cuda',fast=fast)
            states={};timings={}
            for fast in (False,True):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                states[fast]=fit_one(stage,spec,dict(FIT,steps=args.steps),'cuda',fast=fast)
                torch.cuda.synchronize();timings[str(fast)]=dict(seconds=time.perf_counter()-start,
                    peak_mib=torch.cuda.max_memory_allocated()/2**20)
            if states[False]!=states[True]:
                for k,v in states[False]['gate']['state'].items():
                    np.testing.assert_array_equal(v,states[True]['gate']['state'][k],err_msg=key+'/'+k)
                raise AssertionError('Training metadata or histories differ')
            a=predict(stage,restore_gate(states[False]['gate']),'cuda')
            b=predict(stage,restore_gate(states[True]['gate']),'cuda')
            for x,y in zip(a[:2],b[:2]):torch.testing.assert_close(x,y,rtol=0,atol=0)
            row=dict(source=name,task=task,method=key,steps=args.steps,images=len(stage.base.ids),
                     people=len(stage.roster_ids),timings=timings,exact_state_and_predictions=True,
                     speedup=timings['False']['seconds']/timings['True']['seconds'])
            if name=='equal90':
                start=time.perf_counter();fit_one(stage,spec,dict(FIT,steps=args.steps),'cpu',fast=True)
                row['cpu_seconds']=time.perf_counter()-start
            rows.append(row);print(json.dumps(row),flush=True)
            write_json(out/f'{args.case}_{args.steps}.json',dict(rows=rows,torch=torch.__version__,
                gpu=torch.cuda.get_device_name(),deterministic=True,tf32=False))

if __name__=='__main__':main()
