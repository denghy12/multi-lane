"""Real CLIP/AMP zero-output and initialization audit, separate from training RNG."""
import argparse
from pathlib import Path
import torch
from .model import MultiLaneModel
from .runner import load_openai_clip_visual, set_seed, tensor_state_digest
from .historical_b32_modules import ARMS, modes
from .run_module_stacking import write


def run(args):
    set_seed(1, True)
    visual = load_openai_clip_visual(args.clip_checkpoint)
    state = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(20261025)
    images = {k: torch.randn(2, 3, 224, 224, generator=generator).cuda() for k in ('full', 'person', 'face')}
    images.update(face_reliable=torch.tensor([True, False], device='cuda'),
        spatial_bbox=torch.tensor([[.1, .1, .7, .9], [.4, .1, .9, .8]], device='cuda'),
        spatial_visibility=torch.ones(2, device='cuda'))
    baseline, initial, rng = None, None, None
    result = {}
    for arm in ('LEGACY', *ARMS):
        torch.set_rng_state(state)
        options = modes('BASE' if arm == 'LEGACY' else arm)
        model = MultiLaneModel(visual, (5, 3, 3, 3, 3, 3, 3, 3),
            adapter_mode='image_token', adapter_bottleneck_dim=32, adapter_layer_indices=(1,),
            adapter_residual_scale=.03, view_fusion='fixed_three_view',
            historical_b32_modules=arm != 'LEGACY', cross_view_layer=5,
            cross_view_dim=64, cross_view_heads=4, cross_view_scale=.1, **options).float().cuda()
        digest = tensor_state_digest({k:v for k,v in model.state_dict().items()
            if not k.startswith(('cross_view_bank.', 'spatial_summary_bank.'))})
        after = torch.get_rng_state().clone()
        model.activate_task(0); model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            scores = model.current_all_logits(images).detach().clone()
        if arm == 'LEGACY':
            baseline, initial, rng = scores, digest, after
        else:
            assert digest == initial and torch.equal(after, rng), arm
            assert torch.equal(scores, baseline), (arm, float((scores-baseline).abs().max()))
        result[arm] = dict(common_state_sha256=digest, zero_output_equals_legacy=True,
            initialization_rng_equal=True)
        del model
        torch.cuda.empty_cache()
    write(args.output, dict(status='verified', real_clip_amp=True, synthetic_batch_only=True,
        seed=1, arms=result, no_training=True))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip-checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
