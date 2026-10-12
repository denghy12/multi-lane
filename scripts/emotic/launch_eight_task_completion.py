"""Complete report gaps with fixed eight-task protocols, never test selection."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def gpu_layout(inventory):
    idle=[]
    free_by_gpu={}
    for line in inventory.splitlines():
        gpu,used,free,util=map(int,line.split(','))
        if used<512 and free>=18000 and util<=5:
            idle.append(gpu)
            free_by_gpu[gpu]=free
    if not idle:
        raise RuntimeError('No idle GPU has 18GiB free; other tasks are untouched')
    if len(idle)>=7:
        return idle[:7],{gpu:1 for gpu in idle[:7]}
    # Same-user workers may share an idle card, with a conservative 4GiB slot
    # plus 3GiB reserve; at most four inference/training pipelines per card.
    layout=[idle[i%len(idle)] for i in range(7)]
    limits={gpu:min(4,(free_by_gpu[gpu]-3072)//4096) for gpu in set(layout)}
    return layout,limits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch', required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    output = Path('output/emotic_eight_task_completion') / args.batch
    logs = Path('logs/emotic_eight_task_completion') / args.batch
    if output.exists() or logs.exists():
        raise FileExistsError('Never overwrite a batch')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Clean worktree required')
    py = sys.executable
    data = '../multi-lane-main/datasets/EMOTIC'
    clip = '../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt'
    face = '../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909'
    base = Path('../multi-lane-main-parax-freezing-eight-task-relaunch/output/emotic_protected_parax_full/protected_parax_freezing_8task_seed012_valtest_20261011_01')
    fitted = Path('../multi-lane-main-parax-fixed-baseline-cached-routing/output/emotic_fixed_baseline_routes/fixed_baseline_cached_routes_20261011_01/cached_fit')
    for seed in (0,1,2):
        for arm in ('dynamic','static'):
            for task in range(8):
                if not (fitted / f'seed{seed}' / arm / 'compact_checkpoints' / f'task{task}.pth').is_file():
                    raise FileNotFoundError(f'Missing fitted seed{seed}/{arm}/task{task}')
    # Only genuinely idle cards are claimed. GPU0's other worker is untouched.
    inventory = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu',
                                         '--format=csv,noheader,nounits'],text=True)
    gpus,limits=gpu_layout(inventory)
    reservations={gpu:threading.Semaphore(count) for gpu,count in limits.items()}
    output.mkdir(parents=True); logs.mkdir(parents=True)
    manifest=dict(batch=args.batch, git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  gpu_inventory=inventory, allocated_gpus=gpus, concurrent_pipeline_limits=limits, smoke=args.smoke,
                  priority='existing eight-task dynamic/static test first',
                  tasks=8, final_decision='task7 test 26-class mAP; validation diagnostic only',
                  cached_seeds=[0,1,2], extension_seed=0,
                  base_epochs=1 if args.smoke else 30, cache_epochs=5, fusion_epochs=1 if args.smoke else 30,
                  hyperparameters='batch64 Adam main_lr0.0125 route_lr0.0004 wd0 cosine; rank32; no Image-token Adapter/Projector',
                  data=data, clip=clip, face=face, baseline=str(base), fitted=str(fitted))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    common=['--data-root',data,'--clip-checkpoint',clip,'--face-manifest-root',face]
    def call(name,command,gpu,extra=None):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1')
        env.update(extra or {})
        (output/f'{name}.started.json').write_text(json.dumps(dict(command=command,gpu=gpu,time=time.time()))+'\n')
        with (logs/f'{name}.log').open('x') as stream:
            code=subprocess.call(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        (output/f'{name}.exit_code').write_text(str(code)+'\n')
        if code:
            raise RuntimeError(f'{name} failed with exit {code}; see {logs/name}.log')
    def test_cached_unlocked(seed,gpu):
        for arm in ('dynamic','static'):
            cmd=[py,'-u','-m','multi_lane.track_a.complete_eight_task_test','--kind','cached',
                 '--source',str(fitted/f'seed{seed}'),'--baseline-source',str(base/f'seed{seed}'/'baseline'),
                 '--arm',arm,'--output',str(output/'cached_test'/f'seed{seed}'/arm),*common]
            if args.smoke: cmd.append('--smoke')
            call(f'cached_seed{seed}_{arm}',cmd,gpu)
    def test_cached(seed,gpu):
        with reservations[gpu]: test_cached_unlocked(seed,gpu)
    def train(method,gpu,heldout=False):
        run_root=output/('heldout_train' if heldout else 'eight_task_train')
        extra=dict(METHOD=method,GPU=str(gpu),RUN_ID='paired',PYTHON=py,MAX_TASKS='8',SEED='0',
                   EPOCHS='1' if args.smoke else '30', DATA_ROOT=data,CLIP_CHECKPOINT=clip,FACE_MANIFEST_ROOT=face,
                   OUTPUT_ROOT=str(run_root),LOG_DIR=str(logs/'training'),SAVE_COMPACT='1',EXPORT_VIEW_SCORES='1')
        if heldout: extra['EXPORT_HELDOUT_VIEWS']='1'
        if args.smoke: extra['UPDATES_PER_TASK']='2'
        name='heldout_base' if heldout else method
        call(name,['bash','scripts/emotic/run_multilane_track_a_post_task_router_val.sh'],gpu,extra)
        return run_root/'paired'/method
    def test_compact(source,name,gpu):
        destination=output/'extended_test'/name
        cmd=[py,'-u','-m','multi_lane.track_a.complete_eight_task_test','--kind','compact',
             '--source',str(source),'--output',str(destination),*common]
        if args.smoke: cmd.append('--preflight')
        # Even smoke training produces all 8 checkpoints, but full score export
        # is needed to exercise frozen fusion; its batch is explicitly smoke.
        call(name+'_test',cmd,gpu)
        return destination
    def model_pipeline(method,gpu):
        with reservations[gpu]:
            source=train(method,gpu)
            test_compact(source,method,gpu)
    def fusion_pipeline_unlocked(gpu):
        source=train('B0',gpu,True)
        fusion=source.parent/'fusion'
        epochs='1' if args.smoke else '30'
        call('fusion_original',[py,'-u','-m','multi_lane.track_a.frozen_view_ranking_fusion',
             '--source',str(source),'--output',str(fusion),'--tasks','8','--epochs',epochs,
             '--batch-size','256','--device','cuda:0'],gpu)
        background=output/'background_calibration'
        call('fusion_background',[py,'-u','-m','multi_lane.track_a.frozen_fusion_calibration_support',
             '--source',str(source),'--output',str(background),'--tasks','8','--epochs',epochs,*common],gpu)
        exported=test_compact(source,'heldout_base',gpu)
        for name,gates in (('original',fusion),('background',background/'fusion')):
            cmd=[py,'-u','-m','multi_lane.track_a.complete_eight_task_test',
                 '--kind','fusion','--source',str(gates),'--test-source',str(exported),
                 '--output',str(output/'fusion_test'/name)]
            if args.smoke: cmd.append('--preflight')
            call('fusion_test_'+name,cmd,gpu)
    def fusion_pipeline(gpu):
        with reservations[gpu]: fusion_pipeline_unlocked(gpu)
    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        # Test-only submissions happen first, before new training submissions.
        futures={pool.submit(test_cached,seed,gpus[seed]):f'cached_seed{seed}' for seed in (0,1,2)}
        methods=('B0','POST_TASK_ROUTER','POST_TASK_STAGED','POST_TASK_STAGED_STATIC')
        futures.update({pool.submit(model_pipeline,method,gpus[3+i]):method for i,method in enumerate(methods)})
        # After its test-only work, seed0's GPU runs heldout calibration pipeline.
        def after_seed0():
            for f,name in list(futures.items()):
                if name=='cached_seed0': f.result(); break
            fusion_pipeline(gpus[0])
        futures[pool.submit(after_seed0)]='fusion_pipeline'
        for future in concurrent.futures.as_completed(futures):
            try: future.result()
            except Exception as error:
                failures.append(dict(job=futures[future],error=str(error)))
    (output/'completion.json').write_text(json.dumps(dict(status='failed' if failures else 'complete',
          failures=failures,smoke=args.smoke,finished=time.time()),indent=2)+'\n')
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
