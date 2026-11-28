"""Measure 1/2/4 independent exact-math fits on one otherwise idle GPU."""
import hashlib,json,multiprocessing as mp,os,time
from pathlib import Path
import torch
from scripts.emotic.d9_task_patch_run import CONFIG,FIT
from scripts.emotic.d9_task_patch_readout import fit_one
from scripts.emotic.d9_task_patch_extract import load_features
from scripts.emotic.d9_a_eight_eval import read
from scripts.emotic.evaluate_d9_relation import write_json
from multi_lane.track_a.task_patch_fusion import candidates


def child(queue,event):
    try:
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        cfg=read(CONFIG);stage=load_features(cfg,'full',Path('.'),'train',3)
        spec=next(s for s in candidates() if s['key']=='cls_patch__common')
        fit_one(stage,spec,dict(FIT,steps=2),'cuda');torch.cuda.synchronize()
        queue.put(('ready',os.getpid()));event.wait()
        start=time.perf_counter();result=fit_one(stage,spec,dict(FIT,steps=100),'cuda');torch.cuda.synchronize()
        queue.put(('done',dict(seconds=time.perf_counter()-start,
            digest=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest())))
    except BaseException as e:
        queue.put(('error',repr(e)));raise


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    ctx=mp.get_context('spawn');rows=[]
    for count in (1,2,4):
        queue=ctx.Queue();event=ctx.Event();workers=[ctx.Process(target=child,args=(queue,event)) for _ in range(count)]
        for w in workers:w.start()
        for _ in workers:
            kind,value=queue.get(timeout=180)
            if kind!='ready':raise RuntimeError((kind,value))
        start=time.perf_counter();event.set();results=[]
        for _ in workers:
            kind,value=queue.get(timeout=180)
            if kind!='done':raise RuntimeError((kind,value))
            results.append(value)
        seconds=time.perf_counter()-start
        for w in workers:
            w.join()
            if w.exitcode:raise RuntimeError('Worker failed')
        row=dict(workers=count,wall_seconds=seconds,fits_per_second=count/seconds,results=results)
        rows.append(row);print(json.dumps(row),flush=True)
    if len({r['digest'] for row in rows for r in row['results']})!=1:raise ValueError('Concurrent fitting altered states')
    out=Path('output/d9_task_patch_speed/seed1_20261128/benchmark/concurrency.json')
    write_json(out,dict(rows=rows,exact_concurrent_states=True))

if __name__=='__main__':main()
