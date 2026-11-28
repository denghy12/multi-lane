"""Schedule independent task fits on GPUs 0/1; retain the paired Val/Test barrier."""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from scripts.emotic.d9_task_patch_run import CONFIG, FIT, protocol
from scripts.emotic.d9_task_patch_readout import train, finalize_train, evaluate, METHODS
from scripts.emotic.d9_task_patch_extract import extract
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_final_patch_run import lock_protocol, verify_validation
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.evaluate_d9_relation import write_json
from scripts.emotic.d9_summary_eight import BENCHMARK, KEYS
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.task_patch_fusion import candidates

CONFIG_FAST=CONFIG.with_name('d9_task_patch_speed_20261128.json')
OLD_DIGEST='e3afb150ab6f5bdb59c3a8aa34a11a4d49c697a5930912c6c0df4c8de6db90b6'


def configuration(phase):
    cfg=read(CONFIG_FAST);fit=dict(FIT);smoke=phase=='preflight'
    if smoke:fit.update(steps=5,permutations=2,patch_swaps=2)
    return cfg,fit,smoke,[0,7] if smoke else cfg['tasks']


def import_fits(cfg,base,previous):
    """Only immutable completed candidate fits; never an unfinished optimizer state."""
    if (base/'resume_import.json').exists():
        report=read(base/'resume_import.json')
        for row in report['files']:require_digest(base/row['relative'],row['sha256'])
        return
    old=read(previous/'protocol.json')
    if old['digest']!=OLD_DIGEST:raise ValueError('Unexpected source experiment')
    expected={k:v for k,v in cfg.items() if k not in ('output','logs')}
    if {k:v for k,v in old['config'].items() if k not in ('output','logs')}!=expected:
        raise ValueError('Resume changes scientific configuration')
    manifest=[]
    for name in cfg['sources']:
        for task in range(8):
            for spec in candidates():
                rel=Path(name)/'fits'/f'task{task}'/(spec['key']+'.json');source=previous/rel
                if not source.exists():continue
                payload=read(source)
                if payload['config']!=FIT or payload['spec']!=spec or payload['smoke']:
                    raise ValueError('Imported fit differs from formal protocol')
                target=base/rel;target.parent.mkdir(parents=True,exist_ok=True)
                digest=sha256_file(source)
                if target.exists():require_digest(target,digest)
                else:shutil.copy2(source,target)
                require_digest(target,digest)
                manifest.append(dict(relative=str(rel),sha256=digest))
    write_verified(base/'resume_import.json',dict(previous=str(previous),protocol_sha256=sha256_file(previous/'protocol.json'),
        runtime_head=old['head'],files=manifest))
    print(f'IMPORTED {len(manifest)} completed candidate fits',flush=True)


def schedule(cfg,phase,jobs,gpus,label,workers_per_gpu=1):
    logdir=Path(cfg['logs'])/phase;logdir.mkdir(parents=True,exist_ok=True)
    queue=list(jobs);active={};finished=[]
    slots=[(gpu,slot) for slot in range(workers_per_gpu) for gpu in gpus]
    try:
        while queue or active:
            for gpu,slot in slots:
                if (gpu,slot) in active or not queue:continue
                args=queue.pop(0);tag='_'.join(args)
                logfile=logdir/(tag+'.log');stream=logfile.open('a')
                command=[sys.executable,'-u','-m','scripts.emotic.d9_task_patch_parallel','--phase',phase,'--worker',*args]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                         CUBLAS_WORKSPACE_CONFIG=':4096:8')
                proc=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
                active[gpu,slot]=(proc,stream,time.monotonic(),tag,logfile)
                print(f'START GPU{gpu} worker{slot} {tag} pid={proc.pid}',flush=True)
            for (gpu,slot),(proc,stream,start,tag,logfile) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                stream.close();del active[gpu,slot]
                elapsed=time.monotonic()-start
                if code:raise RuntimeError(f'{tag} exited {code}; see {logfile}')
                finished.append(dict(job=tag,gpu=gpu,worker=slot,seconds=elapsed,log=str(logfile)))
                write_json(Path(cfg['output'])/phase/(label+'_jobs.json'),dict(finished=finished))
                print(f'DONE GPU{gpu} {tag} {elapsed:.1f}s',flush=True)
            time.sleep(.2)
    finally:
        for proc,stream,_,_,_ in active.values():
            proc.terminate();proc.wait();stream.close()


def worker(phase,args):
    cfg,fit,smoke,tasks=configuration(phase);base=Path(cfg['output'])/phase
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    kind,name,*rest=args
    if name not in cfg['sources']:raise ValueError('Unknown source')
    if kind=='fit':train(cfg,name,base/name,fit,smoke,'cuda',tasks=[int(rest[0])],finalize=False)
    elif kind in ('val','test'):
        if kind=='test':
            for source in cfg['sources']:verify_validation(base/source)
        for task in tasks:extract(cfg,name,base/name,kind,task,smoke)
        evaluate(cfg,name,base/name,kind,tasks,fit,smoke,'cuda')
    else:raise ValueError('Unknown worker operation')


def run(phase,gpus,previous):
    cfg,fit,smoke,tasks=configuration(phase);base=Path(cfg['output'])/phase;base.mkdir(parents=True,exist_ok=True)
    record=protocol(cfg)
    record['execution']=dict(gpus=list(gpus),fit_jobs='four independent task workers per GPU, longest first',
        frozen_input_cache=True,roster_reduction='original CPU prefix sum, batched transfer',
        eval_jobs='one source per worker, both Val sealed before any Test')
    # GPU count is execution-only; preflight may use the idle GPU1.
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Committed code required')
    if not smoke:
        pre=read(Path(cfg['output'])/'preflight/complete.json')
        if pre['digest']!=record['digest']:raise ValueError('Matching completed preflight required')
    record=lock_protocol(base,record)
    if previous and not smoke:import_fits(cfg,base,previous)
    jobs=[['fit',n,str(t)] for n in cfg['sources'] for t in range(8)]
    def cost(job):
        _,n,t=job
        remain=sum(not (base/n/'fits'/f'task{t}'/(s['key']+'.json')).exists() for s in candidates())
        return remain*(Path(cfg['parent'])/n/'features/train'/f'task{t}.npz').stat().st_size
    schedule(cfg,phase,sorted(jobs,key=cost,reverse=True),gpus,'train',workers_per_gpu=4)
    for name in cfg['sources']:finalize_train(cfg,name,base/name,fit,smoke)
    for split in ('val','test'):
        if split=='test':
            for name in cfg['sources']:verify_validation(base/name)
        schedule(cfg,phase,[[split,n] for n in cfg['sources']],gpus,split)
        for task in tasks:
            with np.load(base/'equal90'/split/f'task{task}.npz') as a,np.load(base/'full'/split/f'task{task}.npz') as b:
                for k in ('image_ids','person_ids','targets','inverse','reliable','counts','reliable_counts'):
                    np.testing.assert_array_equal(a[k],b[k])
    rows=[]
    for name in cfg['sources']:
        reports={s:read(base/name/f'{s}_results.json') for s in ('val','test')}
        selected=read(base/name/'selection.json')['selected']
        for method in METHODS:
            row=dict(source=name,method=method,selected_by_val=method==selected)
            if not smoke:
                for split,r in reports.items():
                    row[split]={k:r['summary'][method]['all'][k] for k in KEYS}
                    row[split+'_delta_original']={k:row[split][k]-BENCHMARK[split][k] for k in KEYS}
                    row[split+'_both_targets']=all(row[split][k]>=BENCHMARK[split][k] for k in KEYS)
            rows.append(row)
    write_verified(base/'comparison.json',dict(rows=rows,tasks=tasks,smoke=smoke,target_comparable=not smoke,
        metric='final=task7 all26 mAP, average=mean of eight all-seen mAPs',original=BENCHMARK))
    write_verified(base/'complete.json',dict(digest=record['digest'],comparison_sha256=sha256_file(base/'comparison.json'),
                                           sources=record['sources']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('preflight','formal'),required=True)
    parser.add_argument('--gpus',default='0,1')
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--worker',nargs='+')
    args=parser.parse_args();os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if args.worker:return worker(args.phase,args.worker)
    gpus=[int(g) for g in args.gpus.split(',')]
    if len(set(gpus))!=len(gpus) or not set(gpus)<= {0,1} or not gpus:parser.error('Use only GPUs 0 and 1')
    cfg=read(CONFIG_FAST);statepath=Path(cfg['output'])/'run_state.json';statepath.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();state=dict(phase=args.phase,status='running',gpus=gpus,started=datetime.now(timezone.utc).isoformat())
    write_json(statepath,state)
    try:run(args.phase,gpus,args.resume_from)
    except BaseException as e:
        state.update(status='failed',error=repr(e),elapsed_seconds=time.monotonic()-start);write_json(statepath,state);raise
    state.update(status='complete',elapsed_seconds=time.monotonic()-start,finished=datetime.now(timezone.utc).isoformat())
    write_json(statepath,state);print(json.dumps(state),flush=True)

if __name__=='__main__':main()
