"""Declared paired task-head experiment, with both sources sealed before Test."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from multi_lane.track_a.task_patch_fusion import candidates
from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_a_eight_eval import read,require_digest
from scripts.emotic.d9_summary_run import source_check,code_record,ROOT
from scripts.emotic.d9_final_patch_run import lock_protocol,verify_validation
from scripts.emotic.d9_final_patch_extract import CAPTURE_FILES
from scripts.emotic.d9_task_patch_extract import extract
from scripts.emotic.d9_task_patch_readout import train,evaluate,METHODS
from scripts.emotic.d9_paired_a_evaluate import write_verified
from scripts.emotic.evaluate_d9_relation import write_json
from scripts.emotic.d9_summary_eight import BENCHMARK,KEYS

CONFIG=ROOT/'scripts/emotic/configs/d9_task_patch_20261128.json'
FIT=dict(seed=1,steps=300,lr=.001,l2=.001,amplitude=.25,folds=3,
         permutations=100,permutation_seed=20261128,patch_swaps=10,patch_swap_seed=20261129,
         spatial_seed=20261130)


def check(cfg):
    if (cfg['schema']!=1 or cfg['seed']!=1 or cfg['tasks']!=list(range(8)) or cfg['fit']!=FIT
        or cfg['candidates']!=candidates() or cfg['methods']!=list(METHODS)
        or cfg['extraction']!=dict(batch_size=64,workers=2)
        or cfg['selection']!='own-task methods: Val average then final then declared order'
        or set(cfg['sources'])!={'equal90','full'}):
        raise ValueError('Declared task-head comparison changed')
    for k in ('output','logs'):
        p=Path(cfg[k])
        if p.is_absolute() or '..' in p.parts or p.parts[0]!=k:raise ValueError('Use relative artifact paths')


def protocol(cfg):
    check(cfg);parent=Path(cfg['parent']);old=read(parent/'protocol.json');complete=read(parent/'complete.json')
    if old['digest']!=complete['digest']:raise ValueError('Parent run not complete')
    require_digest(parent/'comparison.json',complete['comparison_sha256'])
    source=source_check(cfg)
    if source!=old['sources']:raise ValueError('Branch source changed')
    allowed=set(CAPTURE_FILES[:4]);changes={}
    for p,h in old['code_sha256'].items():
        actual=sha256_file(ROOT/p)
        if actual!=h:
            if p not in allowed:raise ValueError('Unexpected parent dependency change: '+p)
            changes[p]=dict(old=h,new=actual)
    for name in cfg['sources']:
        verify_validation(parent/name)
        report=read(parent/name/'test_results.json')
        if not report['target_comparable'] or len(report['stages'])!=8:raise ValueError('Incomplete parent evaluation')
    code=code_record()
    return dict(config=cfg,code_sha256=code,sources=source,
        digest=hashlib.sha256(json.dumps(dict(config=cfg,code=code),sort_keys=True).encode()).hexdigest(),
        parent_protocol_sha256=sha256_file(parent/'protocol.json'),
        parent_complete_sha256=sha256_file(parent/'complete.json'),capture_changes=changes,
        parent_fits={n:sha256_file(parent/n/'fit.json') for n in cfg['sources']},
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip())


def run(cfg,phase):
    record=protocol(cfg)
    if phase=='check':
        print(json.dumps({k:record[k] for k in ('digest','head','branch')},indent=2));return
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Run committed code only')
    if not torch.cuda.is_available():raise RuntimeError('Linux CUDA required')
    smoke=phase=='preflight';base=Path(cfg['output'])/phase;base.mkdir(parents=True,exist_ok=True)
    if not smoke:
        pre=read(Path(cfg['output'])/'preflight/complete.json')
        if pre['digest']!=record['digest']:raise ValueError('Matching complete preflight required')
    record=lock_protocol(base,record)
    fitting=dict(cfg['fit']);tasks=[0,7] if smoke else cfg['tasks']
    if smoke:fitting.update(steps=5,permutations=2,patch_swaps=2)
    for name in cfg['sources']:
        train(cfg,name,base/name,fitting,smoke,'cuda')
    for split in ('val','test'):
        if split=='test':
            for name in cfg['sources']:verify_validation(base/name)
        for name in cfg['sources']:
            for task in tasks:extract(cfg,name,base/name,split,task,smoke)
            evaluate(cfg,name,base/name,split,tasks,fitting,smoke,'cuda')
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
    parser.add_argument('--phase',required=True,choices=('check','preflight','formal'))
    args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:parser.error('Run from experiment worktree root')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    cfg=read(CONFIG);check(cfg);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    state=dict(phase=args.phase,status='running',started=datetime.now(timezone.utc).isoformat());start=time.monotonic()
    write_json(out/'run_state.json',state)
    try:run(cfg,args.phase)
    except BaseException as e:
        state.update(status='failed',error=repr(e),elapsed_seconds=time.monotonic()-start)
        write_json(out/'run_state.json',state);raise
    state.update(status='complete',finished=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
    write_json(out/'run_state.json',state);print(json.dumps(state),flush=True)

if __name__=='__main__':main()

