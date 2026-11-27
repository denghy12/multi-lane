"""Eight final CLS/patch controls, both TRAIN sources, full eight-stage Val then Test."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from multi_lane.track_a.final_patch_fusion import METHODS, LEARNED, WIDTH, HIDDEN, PROJECTION_SEED, FinalPatchFusion
from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_a_eight_eval import read, require_digest
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.d9_summary_run import ROOT, code_record, source_check
from scripts.emotic.d9_summary_eight import BENCHMARK, KEYS
from scripts.emotic.d9_final_patch_extract import extract, CAPTURE_FILES
from scripts.emotic.d9_final_patch_readout import train, evaluate
from scripts.emotic.evaluate_d9_relation import write_json

CONFIG=ROOT/'scripts/emotic/configs/d9_final_patch_20261127.json'
FIT=dict(seed=1,steps=300,lr=.001,l2=.001,amplitude=.25,folds=3,permutations=100,permutation_seed=20261128,
         patch_swaps=10,patch_swap_seed=20261129,spatial_seed=20261130)


def check(cfg):
    if (cfg['schema']!=1 or cfg['seed']!=1 or cfg['tasks']!=list(range(8)) or cfg['methods']!=list(METHODS)
            or cfg['fit']!=FIT or cfg['extraction']!=dict(batch_size=64,workers=2)
            or cfg['representation']!=dict(layer='final',task='current',patches=196,width=WIDTH,hidden=HIDDEN,
                                           projection_seed=PROJECTION_SEED,patch_dtype='float16')
            or cfg['initialization']!='Full1_aux0_no_prior'
            or cfg['selection']!='Val average_mAP descending, then final_mAP descending, then declared order'
            or cfg['evaluation']!='current_stage_gate_broadcast_all_seen'):
        raise ValueError('Final CLS/patch controls and shared budgets are locked')
    if set(cfg['sources'])!={'equal90','full'} or any(set(s)!={'path','mode','exposure'} for s in cfg['sources'].values()):
        raise ValueError('No source-specific hyperparameters')
    for key in ('output','logs'):
        p=Path(cfg[key])
        if p.is_absolute() or '..' in p.parts or p.parts[0]!=key:raise ValueError('Use project-relative artifact paths')


def protocol(cfg):
    check(cfg);prior=Path(cfg['cache']);old=read(prior/'protocol.json');complete=read(prior/'complete.json')
    require_digest(prior/'comparison.json',complete['comparison_sha256'])
    sources=source_check(cfg)
    if old['config']['tasks']!=list(range(8)) or old['digest']!=complete['digest'] or sources!=old['sources']:
        raise ValueError('Expected the same completed eight-stage branch/cache identities')
    # Exactly four existing model files gain optional capture arguments; restoration
    # is checked independently against native predictions before each extraction.
    allowed=set(CAPTURE_FILES[:4]);changed={}
    for path,digest in old['code_sha256'].items():
        current=sha256_file(ROOT/path)
        if current!=digest:
            if path not in allowed:raise ValueError(f'Unexpected change in historical cache dependency: {path}')
            changed[path]=dict(old=digest,new=current)
    code=code_record();digest=hashlib.sha256(json.dumps(dict(config=cfg,code=code),sort_keys=True).encode()).hexdigest()
    return dict(digest=digest,config=cfg,sources=sources,code_sha256=code,capture_dependency_changes=changed,
                cache_protocol_sha256=sha256_file(prior/'protocol.json'),cache_complete_sha256=sha256_file(prior/'complete.json'),
                head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
                parameters={m:sum(p.numel() for p in FinalPatchFusion(m).parameters()) for m in METHODS[1:]})


def verify_validation(out):
    seal=read(out/'validation_seal.json')
    for file,key in [('fit.json','fit_sha256'),('val_results.json','val_sha256'),('selection.json','selection_sha256')]:
        require_digest(out/file,seal[key])


def run(cfg,phase):
    record=protocol(cfg)
    if phase=='check':
        print(json.dumps({k:record[k] for k in ('digest','head','branch','parameters')},indent=2));return
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Run committed code only')
    if not torch.cuda.is_available():raise RuntimeError('Native final-token extraction requires Linux CUDA')
    smoke=phase=='preflight';base=Path(cfg['output'])/phase;base.mkdir(parents=True,exist_ok=True)
    if not smoke:
        preflight=read(Path(cfg['output'])/'preflight/complete.json')
        if preflight['digest']!=record['digest'] or preflight['sources']!=record['sources']:
            raise ValueError('Complete matching real preflight before formal evaluation')
    write_verified(base/'protocol.json',record)
    tasks=[0,7] if smoke else cfg['tasks'];fitting=dict(cfg['fit'])
    if smoke:fitting.update(steps=5,permutations=2,patch_swaps=2)
    for name in cfg['sources']:
        out=base/name
        for task in tasks:extract(cfg,name,out/'features','train',task,smoke)
        train(cfg,name,out,tasks,fitting,smoke,'cuda')
    for split in ('val','test'):
        if split=='test':
            for name in cfg['sources']:verify_validation(base/name)
        for name in cfg['sources']:
            out=base/name
            for task in tasks:extract(cfg,name,out/'features',split,task,smoke)
            evaluate(cfg,name,out,split,tasks,fitting,smoke,'cuda')
        for task in tasks:
            with np.load(base/'equal90'/split/f'task{task}.npz',allow_pickle=False) as a,np.load(base/'full'/split/f'task{task}.npz',allow_pickle=False) as b:
                for key in ('image_ids','person_ids','inverse','targets','reliable','counts','reliable_counts'):
                    np.testing.assert_array_equal(a[key],b[key])
    rows=[]
    for name in cfg['sources']:
        reports={s:read(base/name/f'{s}_results.json') for s in ('val','test')}
        selection=read(base/name/'selection.json')
        for method in METHODS:
            row=dict(coverage=name,method=method,selected_by_val=selection['selected']==method)
            if not smoke:
                row.update({s:{k:reports[s]['summary'][method]['all'][k] for k in KEYS} for s in reports})
                row['comparisons']={s:reports[s]['comparisons'][method] for s in reports}
            rows.append(row)
    write_verified(base/'comparison.json',dict(tasks=tasks,smoke=smoke,target_comparable=not smoke,rows=rows,original_D=BENCHMARK,
                   metric='final=stage7 all26 mAP; average=mean eight all-seen stage mAPs'))
    write_verified(base/'complete.json',dict(digest=record['digest'],sources=record['sources'],comparison_sha256=sha256_file(base/'comparison.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True,choices=('check','preflight','formal'))
    parser.add_argument('--config',type=Path,default=CONFIG);args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:parser.error('Run from the Linux worktree root')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.use_deterministic_algorithms(True)
    cfg=read(args.config);check(cfg);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    state=dict(phase=args.phase,status='running',started=datetime.now(timezone.utc).isoformat());start=time.monotonic()
    write_json(out/'run_state.json',state)
    try:run(cfg,args.phase)
    except BaseException as error:
        state.update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-start);write_json(out/'run_state.json',state);raise
    state.update(status='complete',elapsed_seconds=time.monotonic()-start,finished=datetime.now(timezone.utc).isoformat())
    write_json(out/'run_state.json',state);print(json.dumps(state),flush=True)

if __name__=='__main__':main()
