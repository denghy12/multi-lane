"""Verify actual v4 expert inputs and zero-output identity on real validation images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .evaluate_joint26_checkpoint import make_source
from .model import MultiLaneModel
from .openai_clip_loader import load_openai_clip_visual
from .paired_transforms import move_model_inputs
from .runner import TASK_SIZES, git_metadata, set_seed


def _variation(vectors: torch.Tensor) -> dict:
    normalized = F.normalize(
        F.layer_norm(vectors.float(), (vectors.shape[-1],)), dim=-1
    )
    centered = normalized - normalized.mean(dim=0, keepdim=True)
    adjacent = (normalized[1:] - normalized[:-1]).norm(dim=-1)
    return {
        "samples": len(normalized),
        "centered_rms": float(centered.square().mean().sqrt()),
        "adjacent_distance_mean": float(adjacent.mean()),
        "adjacent_distance_min": float(adjacent.min()),
        "varying_coordinates": int((centered.std(dim=0) > 1e-4).sum()),
    }


def probe(data_root: Path, clip_checkpoint: Path, face_manifest_root: Path,
          output: Path, samples: int, batch_size: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the image-evidence probe")
    if samples < 2 or batch_size < 1:
        raise ValueError("Probe needs at least two images and positive batch size")
    set_seed(0, tf32=True)
    config = {
        "input_normalization": "clip", "train_crop_scale": [0.05, 1.0],
        "view_fusion": "fixed_three_view", "person_crop_margin": 0.15,
        "person_color_jitter_strength": 0.0,
        "person_color_jitter_probability": 0.0,
        "full_crop_mode": "legacy",
    }
    source = make_source(config, data_root, face_manifest_root, "val")
    sample_count = min(samples, len(source))
    loader = DataLoader(Subset(source, range(sample_count)), batch_size=batch_size,
                        shuffle=False, num_workers=2, drop_last=False)
    model = MultiLaneModel(
        load_openai_clip_visual(clip_checkpoint), TASK_SIZES,
        num_selectors=10, num_prompts=10, num_prompt_layers=5,
        selector_mode="shared", prompt_mode="shared", normalize="pre-head",
        adapter_mode="moe_task_ffn_v4", adapter_bottleneck_dim=64,
        adapter_layer_indices=(1,), adapter_residual_scale=0.1,
        moe_num_experts=4, moe_top_k=2, moe_dropout=0.1,
        moe_route_mode="dynamic", moe_shared_fraction=0.5,
        moe_router_logit_scale=2, moe_router_temperature=1,
        moe_noise_std=0, view_fusion="fixed_three_view",
        view_classifier_mode="shared_post_fusion",
    ).float().cuda()
    model.activate_task(0)
    model.eval()
    model.assert_visual_frozen()
    captured = {view: {"evidence": [], "raw_cls": [], "expert_input": []}
                for view in ("full", "person", "face")}
    bank = model.adapter_bank
    original = bank.delta_for_layer
    current_view = [None]
    def capture_input(module, args):
        captured[current_view[0]]["expert_input"].append(args[0][:, 0].detach().float().cpu())
    hook = bank.tasks[0][0].experts[0].down.register_forward_pre_hook(capture_input)

    def record(layer_id, task_lane_tokens, lane_ids, image_view, router_evidence):
        if layer_id == 1:
            captured[image_view]["evidence"].append(
                router_evidence[0].detach().float().cpu()
            )
            captured[image_view]["raw_cls"].append(
                task_lane_tokens[0, :, 0].detach().float().cpu()
            )
        current_view[0] = image_view
        delta = original(layer_id, task_lane_tokens, lane_ids, image_view, router_evidence)
        if torch.count_nonzero(delta).item():
            raise RuntimeError("Zero-initialized v4 residual is not zero")
        return delta

    bank.delta_for_layer = record
    with torch.inference_mode():
        for images, *_ in loader:
            inputs = move_model_inputs(images, torch.device("cuda"))
            logits = model.current_all_logits(inputs)
            model.set_adapter_runtime_enabled(False)
            baseline_logits = model.current_all_logits(inputs)
            model.set_adapter_runtime_enabled(True)
            if not torch.equal(logits, baseline_logits):
                raise RuntimeError("V4 zero-output identity failed")
    hook.remove()
    result = {
        "git": git_metadata(Path(__file__).resolve().parents[2]),
        "split": "val", "training_updates": 0, "zero_output_identity": True,
        "expert_input_interface": "layer_norm_mean_selected_image_evidence",
        "requested_samples": samples, "sample_count": sample_count,
        "batch_size": batch_size,
        "views": {
            view: {
                name: _variation(torch.cat(captured[view][name], dim=0))
                for name in ("evidence", "raw_cls", "expert_input")
            } for view in captured
        },
        "route_statistics": bank.route_statistics(),
    }
    if any(row["expert_input"]["adjacent_distance_mean"] <= 0.01
           or row["expert_input"]["varying_coordinates"] < 8
           for row in result["views"].values()):
        raise RuntimeError("Normalized image evidence did not vary sufficiently")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = probe(args.data_root, args.clip_checkpoint,
                   args.face_manifest_root, args.output,
                   args.samples, args.batch_size)
    print(json.dumps(result["views"], indent=2), flush=True)


if __name__ == "__main__":
    main()
