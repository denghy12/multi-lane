"""Read-only zero-output identity and eight-lane resource check after smoke.

Later Routers in a two-task smoke are untrained. Restoring task7 below is ONLY
for tensor-shape/resource testing, never a reported validation/test score.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .model import MultiLaneModel
from .openai_clip_loader import load_openai_clip_visual
from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _torch_load, _sha256, restore_compact_model_state
from .paired_transforms import move_model_inputs
from .runner import TASK_SIZES, LabelView, seen_indices, set_seed, tensor_state_digest


def run(args):
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    config=json.loads((args.source_run/'config.json').read_text())
    assert config['adapter_mode']=='moe_frozen_ffn_activate_freeze' and config['max_tasks']==2
    set_seed(0,tf32=False);device=torch.device('cuda')
    model=MultiLaneModel(load_openai_clip_visual(args.clip_checkpoint),TASK_SIZES,
        num_selectors=10,num_prompts=10,num_prompt_layers=5,
        view_fusion='fixed_three_view',normalize='pre-head',
        adapter_mode=config['adapter_mode'],adapter_layer_indices=config['adapter_layer_indices'],
        adapter_bottleneck_dim=64,adapter_residual_scale=.1,moe_num_experts=22,
        moe_top_k=3,moe_freeze_count=3,moe_freeze_metric='gate_mass',moe_dropout=.1,
    ).float().to(device)
    checkpoint=args.source_run/'compact_checkpoints/task1.pth';source_sha=_sha256(checkpoint)
    restore_compact_model_state(model,_torch_load(checkpoint),1,config['git']);model.eval()
    source=make_source(config,args.data_root,args.face_manifest_root,'val')
    inputs,_=next(iter(DataLoader(LabelView(source,list(range(64)),seen_indices(1)),batch_size=64,num_workers=0)))
    inputs=move_model_inputs(inputs,device)
    before=tensor_state_digest(model.state_dict());clip=tensor_state_digest(model.visual_encoder.state_dict())
    # Directly check zero-up initialization on real pixels, then restore all weights.
    backup={n:p.detach().clone() for n,p in model.adapter_bank.named_parameters() if '.up.' in n}
    with torch.no_grad():
        for n,p in model.adapter_bank.named_parameters():
            if n in backup:p.zero_()
        on=model.seen_logits(inputs)
        model.set_adapter_runtime_enabled(False);off=model.seen_logits(inputs)
        identity=float((on-off).abs().max())
        model.set_adapter_runtime_enabled(True)
        for n,p in model.adapter_bank.named_parameters():
            if n in backup:p.copy_(backup[n])
    assert identity<=1e-6,identity
    assert tensor_state_digest(model.state_dict())==before
    # Explicitly untrained later Routers: shapes and resource cost ONLY.
    model.restore_task(7);model.eval();model.adapter_bank.reset_route_statistics()
    torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True;start=time.perf_counter()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):logits=model.seen_logits(inputs)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    assert logits.shape==(64,26) and bool(torch.isfinite(logits).all())
    routes=model.adapter_bank.route_statistics()
    assert len(routes)==8
    for task in routes.values():
        for layer in task.values():
            assert all(layer[v]['samples']==64 for v in ('full','person','face'))
    model.restore_task(1)
    assert tensor_state_digest(model.state_dict())==before
    assert tensor_state_digest(model.visual_encoder.state_dict())==clip
    assert _sha256(checkpoint)==source_sha
    result=dict(status='passed',scope='shape_and_resource_only_later_routers_untrained_not_accuracy',
                validation_batch=64,test_accessed=False,optimizer_updates=0,
                zero_output_fp32_max_logit_difference=identity,eight_lane_logits_shape=list(logits.shape),
                eight_lane_seconds=elapsed,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                frozen_visual_and_source_checkpoint_unchanged=True,source_git=config['git'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source-run','output','data-root','clip-checkpoint','face-manifest-root']:
        p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
