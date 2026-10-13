"""Validation-only residual-strength ablation for fixed static-MoE checkpoints.

The compact checkpoints are never modified.  The runtime multiplier is applied
after restoring each task checkpoint and is therefore an inference-only probe of
the jointly trained Adapter/Selector/Prompt/head state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .compare_joint26 import require
from .evaluate_static_moe import METHODS, audit_pairing, build_static_model, read
from .evaluate_whole_parax_eight_task import metrics_at_task, seen_payload
from .export_compact_test_scores import _sha256, _torch_load, restore_compact_model_state
from .openai_clip_loader import load_openai_clip_visual
from .paired_transforms import move_model_inputs
from .runner import LabelView, TASK_SIZES, git_metadata, set_seed, tensor_state_digest, validate_classes
from .snapshot_validation import infer, replay
from .evaluate_static_moe import make_source


DEFAULT_MULTIPLIERS = (0.0, 0.25, 0.5, 1.0)
SCOPES = ("all", "full", "auxiliary")
ADAPTER_METHODS = ("SHARED", "MOE_UNIFORM", "MOE_OVERLAP")


def evaluate_method_scope(batch: Path, output: Path, method: str, scope: str,
                          multiplier: float, data_root: Path, clip_checkpoint: Path,
                          face_manifest_root: Path, split: str = "val") -> dict:
    require(method in METHODS and scope in SCOPES and method in ADAPTER_METHODS,
            "Ablation requires an Adapter method and a valid scope")
    require(not output.exists(), f"Output exists: {output}")
    config = read(batch / method / "config.json")
    set_seed(config["seed"], config["tf32"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = batch / method
    model = build_static_model(config, load_openai_clip_visual(clip_checkpoint)).float().to(device)
    model.enable_adapter_ablation_diagnostics(True)
    validation = make_source(config, data_root, face_manifest_root, "val")
    validate_classes(validation)
    require(len(validation) == 2397, "Expected complete validation cohort")
    cohort = validation if split == "val" else make_source(config, data_root, face_manifest_root, "test")
    validate_classes(cohort)
    require(len(cohort) == (2397 if split == "val" else 5368), f"Expected complete {split} cohort")
    loader = DataLoader(
        LabelView(cohort, range(len(cohort)), range(26), include_sample_id=True),
        batch_size=config["eval_batch_size"], num_workers=config["workers"], shuffle=False,
    )
    source_hashes = {
        str(path): _sha256(path)
        for path in [source / "config.json"] + [source / "compact_checkpoints" / f"task{t}.pth" for t in range(8)]
    }
    visual_hash = tensor_state_digest(model.visual_encoder.state_dict())
    rows = {}
    output.mkdir(parents=True)
    for task in range(8):
        payload = _torch_load(source / "compact_checkpoints" / f"task{task}.pth")
        restore_compact_model_state(model, payload, task, config["git"])
        if model.adapter_bank is None:
            raise RuntimeError(f"{method} did not restore an Adapter bank")
        model.adapter_bank.set_residual_ablation("all", 1.0)
        model.reset_adapter_ablation_diagnostics()
        model.requires_grad_(False).eval()
        replay_difference = replay(model, validation, source, config, task, device)
        model.adapter_bank.set_residual_ablation(scope, multiplier)
        model.reset_adapter_ablation_diagnostics()
        block, full = infer(model, loader, device, config["amp"], task, TASK_SIZES)
        data = seen_payload(full, task)
        metrics = metrics_at_task(data, config["threshold"], task)
        diagnostics = model.adapter_ablation_diagnostics()
        rows[str(task)] = {
            "samples": int(len(data["sample_ids"])),
            "replay_max_logit_difference": replay_difference,
            "metrics": metrics,
            "selector_attention_entropy": diagnostics["selector_attention_entropy"],
            "selector_attention_max": diagnostics["selector_attention_max"],
            "expert_output_correlation": diagnostics["expert_output_correlation"],
        }
        np.savez_compressed(output / f"task{task}.npz", **data)
        print(f"STATIC_MOE_ABLATION method={method} scope={scope} multiplier={multiplier} task={task}", flush=True)
    require(tensor_state_digest(model.visual_encoder.state_dict()) == visual_hash, "Inference changed CLIP")
    require(all(_sha256(Path(path)) == digest for path, digest in source_hashes.items()),
            "Inference changed source checkpoint/config")
    final = rows["7"]["metrics"]
    result = {
        "status": "complete", "split": split, "method": method, "scope": scope,
        "multiplier": float(multiplier), "training_performed": False,
        "optimizer_updates": 0, "source": str(source.resolve()),
        "source_sha256": source_hashes, "tasks": rows,
        "final_metrics": final, "samples": len(cohort),
        "evaluator_git": git_metadata(Path(__file__).resolve().parents[2]),
    }
    output.joinpath("summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def run(batch: Path, output_root: Path, data_root: Path, clip_checkpoint: Path,
        face_manifest_root: Path, methods=ADAPTER_METHODS,
        multipliers=DEFAULT_MULTIPLIERS, scopes=SCOPES, split="val") -> dict:
    require(not output_root.exists(), "Ablation output root already exists")
    require(not git_metadata(Path(__file__).resolve().parents[2])["dirty"], "Evaluator worktree is dirty")
    audit_pairing(batch)
    output_root.mkdir(parents=True)
    all_results = {}
    for method in methods:
        for scope in scopes:
            for multiplier in multipliers:
                key = f"{method}__{scope}__m{multiplier:g}"
                all_results[key] = evaluate_method_scope(
                    batch, output_root / key, method, scope, float(multiplier),
                    data_root, clip_checkpoint, face_manifest_root, split,
                )
    result = {"status": "complete", "split": split, "methods": list(methods),
              "scopes": list(scopes), "multipliers": list(map(float, multipliers)),
              "results": all_results}
    output_root.joinpath("comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--methods", nargs="+", default=list(ADAPTER_METHODS), choices=ADAPTER_METHODS)
    parser.add_argument("--scopes", nargs="+", default=list(SCOPES), choices=SCOPES)
    parser.add_argument("--multipliers", nargs="+", type=float, default=list(DEFAULT_MULTIPLIERS))
    args = parser.parse_args()
    run(**vars(args))
