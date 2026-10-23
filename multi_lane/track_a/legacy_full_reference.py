"""Run immutable historical source and retain task boundaries without altering RNG."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--legacy-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--clip-checkpoint',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    from .strong_full_asl import LEGACY_COMMIT
    from .runner import git_metadata
    wrapper_git=git_metadata(Path(__file__).resolve().parents[2])
    legacy=a.legacy_root.resolve();out=a.output.resolve();data=a.data_root.resolve();clip=a.clip_checkpoint.resolve()
    if subprocess.check_output(['git','-C',str(legacy),'rev-parse','HEAD'],text=True).strip()!=LEGACY_COMMIT:
        raise ValueError('Historical source commit differs')
    if subprocess.check_output(['git','-C',str(legacy),'status','--porcelain'],text=True).strip():raise ValueError('Historical source dirty')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    # Load the old package in this isolated child, never copy or modify its code.
    for name in list(sys.modules):
        if name=='multi_lane' or name.startswith('multi_lane.'):del sys.modules[name]
    sys.path.insert(0,str(legacy));os.chdir(legacy)
    from multi_lane.track_a import runner as old
    import torch
    import numpy as np
    import random
    original=old.train_task
    captured=[]
    def train(*args,**kwargs):
        holders={};adam=torch.optim.Adam;cosine=torch.optim.lr_scheduler.CosineAnnealingLR;scaler=torch.cuda.amp.GradScaler
        def wrap(key,fn):
            def create(*x,**y):
                obj=fn(*x,**y);holders[key]=obj;return obj
            return create
        torch.optim.Adam=wrap('optimizer',adam);torch.optim.lr_scheduler.CosineAnnealingLR=wrap('scheduler',cosine)
        torch.cuda.amp.GradScaler=wrap('scaler',scaler)
        try:history=original(*args,**kwargs)
        finally:torch.optim.Adam=adam;torch.optim.lr_scheduler.CosineAnnealingLR=cosine;torch.cuda.amp.GradScaler=scaler
        model,loader=args[0],args[1];task=args[4]
        names={id(v):n for n,v in model.named_parameters()}
        snapshot=dict(schema_version=1,kind='historical_task_boundary_before_reporting',task_id=task,
            source_commit=LEGACY_COMMIT,wrapper_git=wrapper_git,model=old.compact_model_state_dict(model),
            optimizer=holders['optimizer'].state_dict(),scheduler=holders['scheduler'].state_dict(),
            scaler=holders['scaler'].state_dict(),optimizer_parameter_names=[
                [names[id(v)] for v in g['params']] for g in holders['optimizer'].param_groups],
            requires_grad={n:v.requires_grad for n,v in model.named_parameters()},
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all()),sampler_generator=None,history=history)
        folder=out/'historical_recovery_checkpoints';folder.mkdir(exist_ok=True)
        path=folder/f'task{task}_before_reporting.pth';temporary=path.with_suffix('.tmp')
        torch.save(snapshot,temporary);temporary.replace(path);captured.append(task)
        return history
    old.train_task=train
    flags=['--seed','0','--data-root',str(data),'--clip-checkpoint',str(clip),'--output-root',str(out/'train'),
        '--epochs','30','--train-batch-size','64','--eval-batch-size','64','--workers','2',
        '--person-transform-mode','letterbox',
        '--source-learning-rate','.05','--source-reference-batch-size','256','--weight-decay','0',
        '--scheduler-mode','cosine','--scheduler-min-lr-ratio','0','--scheduler-warmup-ratio','0',
        '--training-loss-mode','legacy_full_zero','--loss-routing','adapter_asl',
        '--asl-gamma-neg','9.8','--asl-gamma-pos','0','--asl-clip','.05','--asl-eps','1e-8',
        '--input-mode','full','--input-normalization','clip','--train-crop-scale','.05','1',
        '--full-crop-mode','legacy','--selector-conditioning','disabled','--adapter-mode','image_token',
        '--adapter-bottleneck-dim','32','--adapter-layer-indices','1','--adapter-residual-scale','.1',
        '--adapter-activation','relu','--adapter-task-init','independent','--adapter-learning-rate','.0004',
        '--adapter-weight-decay','0','--calibration-fraction','0','--save-evaluation-scores',
        '--reporting-split','val' if a.smoke else 'test','--evaluation-score-purpose','validation_search' if a.smoke else 'fixed_test_fusion',
        '--save-compact-checkpoints','--no-save-checkpoints','--max-tasks','2' if a.smoke else '8']
    if a.smoke:flags+=['--optimizer-updates-per-task','2']
    sys.argv=['historical_runner',*flags];old.main()
    config=json.loads((out/'train/config.json').read_text());summary=json.loads((out/'train/seed_summary.json').read_text())
    assert captured==list(range(2 if a.smoke else 8)) and summary['status']=='complete'
    assert summary['completed_optimizer_updates']==(4 if a.smoke else 13950)
    history=json.loads((out/'train/training_history.json').read_text())
    assert not any(row['skipped_optimizer_steps'] for rows in history.values() for row in rows)
    export=out/'export';export.mkdir()
    converted=copy.deepcopy(config)
    converted.update(legacy_full_reference=True,strong_full_asl=False,module_stacking=False,
        view_fusion='disabled',view_classifier_mode='shared_post_fusion',selector_mode='shared',prompt_mode='shared',
        parax_mode='disabled',cross_view_mode='disabled',spatial_summary_mode='disabled',joint_summary_mode='disabled',
        person_color_jitter_strength=0.,person_color_jitter_probability=0.,
        historical_export=dict(wrapper_git=wrapper_git,original_config_sha256=hashlib.sha256((out/'train/config.json').read_bytes()).hexdigest(),
            original_source_run=str(out/'train'),checkpoint_policy='hardlink_immutable_historical_compact'))
    (export/'config.json').write_text(json.dumps(converted,indent=2)+'\n')
    summary['config']=converted;(export/'seed_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (export/'compact_checkpoints').mkdir()
    for t in captured:os.link(out/'train/compact_checkpoints'/f'task{t}.pth',export/'compact_checkpoints'/f'task{t}.pth')
    (out/'reference_verification.json').write_text(json.dumps(dict(status='verified',source_commit=LEGACY_COMMIT,
        wrapper_git=wrapper_git,updates=summary['completed_optimizer_updates'],skipped=0,tasks=captured,
        original_metrics=summary['metrics'],configuration=flags,completed_unix=time.time()),indent=2)+'\n')
    print('HISTORICAL_FULL_REFERENCE_COMPLETE',flush=True)

if __name__=='__main__':main()
