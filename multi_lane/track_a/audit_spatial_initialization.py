"""Real CLIP/validation initialization and historical replay, without training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import build_model, _torch_load, _sha256, restore_compact_model_state
from .openai_clip_loader import load_openai_clip_visual
from .paired_transforms import move_model_inputs
from .runner import LabelView, set_seed, tensor_state_digest


def audit(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    config_path = args.reference_run / 'config.json'
    original = json.loads(config_path.read_text())
    checkpoints = [args.reference_run/'compact_checkpoints'/f'task{task}.pth' for task in (0,2,7)]
    hashes = {str(p): _sha256(p) for p in [config_path, *checkpoints]}
    config = dict(original, spatial_summary_mode='target', cross_view_mode='disabled')
    source = make_source(config, args.data_root, args.face_manifest_root, 'val')
    batch, _, _ = next(iter(DataLoader(LabelView(source, range(len(source)), range(5), include_sample_id=True),
                                      batch_size=64, shuffle=False, num_workers=0)))
    device = torch.device('cuda'); images = move_model_inputs(batch, device)
    visual = load_openai_clip_visual(args.clip_checkpoint)
    def fresh(mode):
        set_seed(0, True)
        model = build_model(dict(config, spatial_summary_mode=mode), visual).float().to(device)
        model.activate_task(0); model.eval()
        return model
    base = fresh('disabled')
    before = tensor_state_digest(base.visual_encoder.state_dict())
    with torch.no_grad(), torch.cuda.amp.autocast():
        expected, reference_views = base.seen_logits_with_views(images)
    initial = {}
    for mode in ('absolute', 'target'):
        model = fresh(mode)
        for name, value in base.state_dict().items():
            assert torch.equal(value, model.state_dict()[name]), name
        with torch.no_grad(), torch.cuda.amp.autocast():
            actual, views = model.seen_logits_with_views(images)
        assert torch.equal(expected, actual) and all(torch.equal(reference_views[k], views[k]) for k in views)
        initial[mode] = 'bitwise_equal_to_baseline'
        del model
    target = fresh('target')
    with torch.no_grad():
        target.spatial_summary_bank.tasks[0].normal_(0, .5)
        with torch.cuda.amp.autocast():
            normal, views = target.seen_logits_with_views(images)
            changed, alternate = target.seen_logits_with_views(dict(images, spatial_bbox=images['spatial_bbox'].roll(1,0)))
        assert all(torch.equal(views[k], alternate[k]) for k in ('person','face'))
        assert not torch.equal(normal, changed)
    del target, base, expected, reference_views, actual, views, alternate, normal, changed
    torch.cuda.empty_cache()
    replay = {}
    old = build_model(original, visual).float().to(device)
    for task, checkpoint in zip((0,2,7), checkpoints):
        restore_compact_model_state(old, _torch_load(checkpoint), task, original['git'])
        old.eval().requires_grad_(False)
        with torch.no_grad(), torch.cuda.amp.autocast():
            actual = old.seen_logits(images).float().cpu().numpy()
        archive = args.reference_run.parent/'full_val'/'scores'/f'task{task}.npz'
        with np.load(archive) as saved:
            assert np.array_equal(actual, saved['logits'][:len(actual)]), ('Historical replay differs', task)
        replay[str(task)] = 'bitwise_equal_to_archived'
    assert before == tensor_state_digest(old.visual_encoder.state_dict())
    assert all(_sha256(Path(p)) == sha for p,sha in hashes.items())
    result = dict(status='verified', real_validation_batch=64, initialization=initial,
                  historical_replay=replay, swap_only_full=True, source_unchanged=True,
                  source_sha256=hashes, training_performed=False,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-run','output','data-root','clip-checkpoint','face-manifest-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    audit(parser.parse_args())
