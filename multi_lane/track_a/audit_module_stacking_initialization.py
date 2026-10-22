"""Real CLIP initialization checks for eight cells; validation forward only."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .module_stacking import ARMS, modes
from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import build_model
from .openai_clip_loader import load_openai_clip_visual
from .paired_transforms import move_model_inputs
from .runner import LabelView, set_seed, tensor_state_digest


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    config=json.loads((args.reference_run/'config.json').read_text())
    config.update(module_stacking=True,adapter_bottleneck_dim=32,adapter_layer_indices=[1],
        adapter_residual_scale=.1,adapter_activation='relu',adapter_task_initialization='independent',
        adapter_residual_gate_mode='fixed',adapter_view_mode='shared',adapter_view_bottleneck_dim=0,
        adapter_bottleneck_dims_per_task=[32]*8,adapter_regularization='none',
        cross_view_layer=5,cross_view_dim=64,cross_view_heads=4,cross_view_scale=.1)
    source=make_source(dict(config,spatial_summary_mode='target'),args.data_root,args.face_manifest_root,'val')
    batch,_,_=next(iter(DataLoader(LabelView(source,range(len(source)),range(5),include_sample_id=True),batch_size=64,num_workers=0)))
    device=torch.device('cuda');images=move_model_inputs(batch,device)
    visual=load_openai_clip_visual(args.clip_checkpoint)
    def fresh(arm):
        set_seed(0,True)
        model=build_model(dict(config,**modes(arm)),visual).float().to(device)
        model.activate_task(0);model.eval();return model
    base=fresh('F000');rng=torch.get_rng_state();frozen=tensor_state_digest(base.visual_encoder.state_dict())
    with torch.no_grad(),torch.cuda.amp.autocast():expected,ev=base.seen_logits_with_views(images)
    checks={}
    for arm in ARMS:
        model=fresh(arm)
        assert torch.equal(rng,torch.get_rng_state()),arm
        for name,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[name]),(arm,name)
        with torch.no_grad(),torch.cuda.amp.autocast():actual,views=model.seen_logits_with_views(images)
        assert torch.equal(actual,expected) and all(torch.equal(ev[v],views[v]) for v in ev),arm
        checks[arm]=dict(initial_outputs_bitwise_equal=True,common_parameters_equal=True,initialization_rng_equal=True,
            added_parameters=(model.adapter_bank.per_task_parameter_count() if model.adapter_bank else 0)
            +(model.cross_view_bank.parameter_count_per_task() if model.cross_view_bank else 0)
            +(model.spatial_summary_bank.parameter_count_per_task() if model.spatial_summary_bank else 0))
        del model
    assert frozen==tensor_state_digest(base.visual_encoder.state_dict())
    result=dict(status='verified',checks=checks,training_performed=False,batch_size=64,
                frozen_visual_unchanged=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('reference-run','data-root','clip-checkpoint','face-manifest-root','output'):parser.add_argument('--'+key,type=Path,required=True)
    run(parser.parse_args())
