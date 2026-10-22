"""Evaluate frozen Task-Forward MoE checkpoints or a paired baseline on one split."""

from __future__ import annotations
from .spatial_resume import checkpoint_source_git

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _sha256, _torch_load, restore_compact_model_state
from .model import MultiLaneModel
from .openai_clip_loader import load_openai_clip_visual
from .runner import (TASK_SIZES, LabelView, dataset_view, evaluate,
                     evaluate_view_diagnostics, seen_indices, set_seed, summarize_tasks, git_metadata)


def evaluation_loader(source, batch_size, config, workers=None):
    """Override evaluation workers without editing training provenance."""
    resolved = config["workers"] if workers is None else workers
    if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < 0:
        raise ValueError("Evaluation workers must be a nonnegative integer")
    return DataLoader(source, batch_size=batch_size, shuffle=False,
                      num_workers=resolved, pin_memory=False, drop_last=False)


def evaluate_split(source_run: Path, output: Path, split: str,
                   data_root: Path, clip_checkpoint: Path,
                   face_manifest_root: Path, batch_size: int,
                   allow_smoke: bool = False,
                   full_cohort: bool = False,
                   disable_moe_runtime: bool = False,
                   disable_specialist_runtime: bool = False,
                   expected_tasks: int = 8, workers: int = None,
                   checkpoint_tasks=None, disable_joint_residual: bool = False,
                   disable_stacking_modules=()) -> dict:
    if split not in {"val", "test"}:
        raise ValueError("Split must be val or test")
    expected_tasks = 1 if allow_smoke else expected_tasks
    task_ids = list(range(expected_tasks)) if checkpoint_tasks is None else list(checkpoint_tasks)
    if (not task_ids or len(set(task_ids)) != len(task_ids)
            or any(t < 0 or t >= expected_tasks for t in task_ids)):
        raise ValueError("Invalid checkpoint selection")
    if (checkpoint_tasks is not None or disable_joint_residual) and split != "val":
        raise ValueError("Joint residual attribution is validation-only")
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((source_run / "config.json").read_text())
    if config.get("module_stacking"):
        from .module_stacking import validate_config
        validate_config(config)
    if disable_stacking_modules and (not config.get("module_stacking") or split != "val"):
        raise ValueError("Stacking bypass is validation-only and requires the factorial protocol")
    if set(disable_stacking_modules) - {"A", "B", "C"}:
        raise ValueError("Unknown stacking runtime module")
    summary = json.loads((source_run / "seed_summary.json").read_text())
    if not 1 <= expected_tasks <= len(TASK_SIZES):
        raise ValueError("Invalid checkpoint task count")
    if split == "test" and (allow_smoke or expected_tasks != len(TASK_SIZES)):
        raise ValueError("Partial-task evaluation is validation-only")
    expected_tasks = 1 if allow_smoke else expected_tasks
    if (summary.get("status") != "complete" or summary.get("config") != config
        or config.get("git", {}).get("dirty") is not False
        or config.get("adapter_mode") not in ({"disabled", "image_token"} if config.get("module_stacking") else {"moe_task_ffn", "moe_task_ffn_activate_freeze", "moe_frozen_ffn_activate_freeze", "moe_task_ffn_v2", "moe_task_ffn_v3", "moe_task_ffn_v4", "disabled"})
        or (config.get("adapter_mode") == "disabled" and not config.get("moe_paired_audit"))
        or config.get("parax_mode") != "disabled"
        or not config.get("save_compact_checkpoints")
        or config.get("max_tasks") != expected_tasks
        or len(summary.get("task_metrics", [])) != expected_tasks):
        raise ValueError("Source run lacks complete clean MoE provenance")
    if disable_specialist_runtime and (
        config.get("adapter_mode") not in {"moe_task_ffn_v3", "moe_task_ffn_v4"} or disable_moe_runtime
    ):
        raise ValueError("Specialist-only ablation requires enabled v3/v4 MoE")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal evaluation")
    set_seed(config["seed"], tf32=True)
    device = torch.device("cuda")
    model = MultiLaneModel(
        visual_encoder=load_openai_clip_visual(clip_checkpoint),
        task_sizes=TASK_SIZES,
        num_selectors=config["num_selectors"],
        num_prompts=config["num_prompts"],
        num_prompt_layers=config["num_prompt_layers"],
        selector_mode=config["selector_mode"],
        prompt_mode=config["prompt_mode"],
        normalize=config["normalize"],
        adapter_mode=config["adapter_mode"],
        adapter_bottleneck_dim=config["adapter_bottleneck_dim"],
        adapter_layer_indices=config["adapter_layer_indices"],
        adapter_residual_scale=config["adapter_residual_scale"],
        adapter_activation=config["adapter_activation"],
        adapter_task_initialization=config["adapter_task_initialization"],
        adapter_view_mode=config.get("adapter_view_mode", "shared"),
        adapter_view_bottleneck_dim=config.get("adapter_view_bottleneck_dim", 0),
        moe_num_experts=config.get("moe_num_experts") if config.get("moe_num_experts") is not None else 4,
        moe_top_k=config.get("moe_top_k") if config.get("moe_top_k") is not None else 2,
        moe_dropout=config.get("moe_dropout") if config.get("moe_dropout") is not None else 0.1,
        moe_noisy_gating=config.get("moe_noisy_gating") if config.get("moe_noisy_gating") is not None else True,
        moe_freeze_metric=config.get("moe_freeze_metric") or "hard_count",
        moe_freeze_count=config.get("moe_freeze_count") if config.get("moe_freeze_count") is not None else 2,
        moe_route_mode=config.get("moe_route_mode") or "dynamic",
        moe_shared_fraction=config.get("moe_shared_fraction") or 0.5,
        moe_router_logit_scale=config.get("moe_router_logit_scale") or 2.0,
        moe_router_temperature=config.get("moe_router_temperature") or 1.0,
        moe_noise_std=config.get("moe_noise_std") if config.get("moe_noise_std") is not None else 0.5,
        spatial_summary_mode=config.get("spatial_summary_mode", "disabled"),
        module_stacking=config.get("module_stacking", False),
        cross_view_mode=config.get("cross_view_mode", "disabled"),
        cross_view_layer=config.get("cross_view_layer", 5),
        cross_view_dim=config.get("cross_view_dim", 64),
        cross_view_heads=config.get("cross_view_heads", 4),
        cross_view_scale=config.get("cross_view_scale", 0.1),
        joint_summary_mode=config.get("joint_summary_mode", "disabled"),
        joint_summary_bottleneck=config.get("joint_summary_bottleneck", 16),
        joint_summary_experts=config.get("joint_summary_experts", 4),
        joint_summary_top_k=config.get("joint_summary_top_k", 2),
        joint_summary_scale=config.get("joint_summary_scale", 0.1),
        view_fusion=config["view_fusion"],
        view_classifier_mode=config["view_classifier_mode"],
    ).float().to(device)
    model.assert_visual_frozen()
    if disable_joint_residual:
        if config.get("joint_summary_mode") not in {"dense", "moe"}:
            raise ValueError("Residual intervention requires learned joint summaries")
        for task in model.joint_summary_bank.tasks:
            for layer in task:
                layer.scale = 0.0
    source = make_source(config, data_root, face_manifest_root, split)
    output.mkdir(parents=True)
    (output / "scores").mkdir()
    (output / "view_scores").mkdir()
    rows = []
    views = {}
    routing = {}
    hashes = {}
    cross_view_statistics = {}
    spatial_statistics = {}
    timings = {}
    torch.cuda.reset_peak_memory_stats(device)
    for task_id in task_ids:
        checkpoint = source_run / "compact_checkpoints" / f"task{task_id}.pth"
        before_hash = _sha256(checkpoint)
        payload = _torch_load(checkpoint)
        restore_compact_model_state(model, payload, task_id, checkpoint_source_git(config, task_id))
        model.eval()
        if model.adapter_bank is not None:
            model.set_adapter_runtime_enabled(not disable_moe_runtime)
        if config.get("adapter_mode") in {"moe_task_ffn_v3", "moe_task_ffn_v4"}:
            model.adapter_bank.specialist_runtime_enabled = not disable_specialist_runtime
        if model.adapter_bank is not None and hasattr(model.adapter_bank, "reset_route_statistics"):
            model.adapter_bank.reset_route_statistics()
        if disable_stacking_modules:
            from .module_stacking import runtime_bypass
            runtime_bypass(model, disable_stacking_modules)
        if model.joint_summary_bank is not None:
            model.joint_summary_bank.reset_statistics()
        if model.cross_view_bank is not None:
            model.cross_view_bank.reset_statistics()
        if model.spatial_summary_bank is not None:
            model.spatial_summary_bank.reset_statistics()
        view = (
            LabelView(source, range(len(source)), seen_indices(task_id),
                      include_sample_id=True)
            if full_cohort else
            dataset_view(source, seen_indices(task_id), include_sample_id=True)
        )
        loader = evaluation_loader(view, batch_size, config, workers)
        torch.cuda.synchronize(device)
        started = time.monotonic()
        row = evaluate(model, loader, device, task_id, config["threshold"],
                       amp=True, score_output_path=output / "scores" / f"task{task_id}.npz")
        torch.cuda.synchronize(device)
        timings[str(task_id)] = dict(fused_evaluation_seconds=time.monotonic()-started,
                                    people=len(view))
        routing[str(task_id)] = model.adapter_bank.route_statistics() if hasattr(model.adapter_bank, "route_statistics") else {}
        rows.append(row)
        if model.cross_view_bank is not None:
            cross_view_statistics[str(task_id)] = model.cross_view_bank.statistics()
        if model.joint_summary_bank is not None:
            routing[str(task_id)] = model.joint_summary_bank.statistics()
            views[str(task_id)] = {"status": "not_applicable", "reason": "single_joint_task_path_has_no_independent_view_endpoints"}
        else:
            views[str(task_id)] = evaluate_view_diagnostics(
                model, loader, device, task_id, config["threshold"], amp=True,
                score_output_path=output / "view_scores" / f"task{task_id}.npz",
            )
        if model.spatial_summary_bank is not None:
            spatial_statistics[str(task_id)] = model.spatial_summary_bank.statistics()
        hashes[str(task_id)] = _sha256(checkpoint)
        if hashes[str(task_id)] != before_hash:
            raise RuntimeError("Source checkpoint changed during attribution")
        print(f"split={split} task={task_id} mAP={row.mAP:.6f}", flush=True)
    result = {
        "split": split,
        "evaluation_workers": config["workers"] if workers is None else workers,
        "evaluation_git": git_metadata(Path(__file__).resolve().parents[2]),
        "source_run": str(source_run.resolve()),
        "source_git": config["git"],
        "checkpoint_sha256": hashes,
        "clip_checkpoint_sha256": _sha256(clip_checkpoint),
        "task_metrics": [asdict(row) for row in rows],
        "summary": summarize_tasks(rows),
        "view_diagnostics": views,
        "route_statistics": routing,
        "spatial_summary_mode": model.spatial_summary_mode,
        "spatial_summary_statistics": spatial_statistics,
        "cross_view_mode": model.cross_view_mode,
        "cross_view_statistics": cross_view_statistics,
        "person_endpoint_semantics": config.get("cross_view_person_endpoint", "person_only"),
        "joint_summary_mode": model.joint_summary_mode,
        "checkpoint_tasks": task_ids,
        "joint_residual_runtime_enabled": not disable_joint_residual,
        "summary_is_selected_boundaries_only": checkpoint_tasks is not None,
        "test_used_for_selection": False,
        "smoke_validation_replay": bool(allow_smoke),
        "full_cohort": bool(full_cohort),
        "cohort_samples": len(source) if full_cohort else None,
        "moe_runtime_enabled": model.adapter_bank is not None and not disable_moe_runtime,
        "disabled_stacking_modules": list(disable_stacking_modules),
        "module_stacking": config.get("module_stacking", False),
        "inference_timings": timings,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device)/1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device)/1024**2,
        "specialist_runtime_enabled": model.adapter_bank is not None and not disable_specialist_runtime,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=None,
                        help="Evaluation-only override; 0 avoids shared-memory worker queues.")
    parser.add_argument("--expected-tasks", type=int, default=8,
                        help="Partial task counts are allowed only on validation.")
    parser.add_argument("--allow-smoke", action="store_true",
                        help="Replay one-task smoke checkpoint on validation only.")
    parser.add_argument("--full-cohort", action="store_true",
                        help="Use every person at every task for comparable continual curves.")
    parser.add_argument("--disable-moe-runtime", action="store_true",
                        help="Evaluate the same checkpoints with the MoE residual off.")
    parser.add_argument("--disable-specialist-runtime", action="store_true",
                        help="V3 validation ablation: retain half-weight shared expert only.")
    parser.add_argument("--checkpoint-tasks", type=int, nargs="+", default=None,
                        help="Validation-only selected checkpoint boundaries.")
    parser.add_argument("--disable-joint-residual", action="store_true",
                        help="Zero every task-local joint residual scale in memory only.")
    parser.add_argument("--disable-stacking-modules", nargs="+", choices=("A", "B", "C"), default=[])
    args = parser.parse_args()
    if args.allow_smoke and args.split != "val":
        parser.error("Smoke replay is validation-only")
    if args.disable_specialist_runtime and args.split != "val":
        parser.error("Specialist ablation is validation-only")
    evaluate_split(args.source_run, args.output, args.split, args.data_root,
                   args.clip_checkpoint, args.face_manifest_root, args.batch_size,
                   allow_smoke=args.allow_smoke,
                   full_cohort=args.full_cohort,
                   disable_moe_runtime=args.disable_moe_runtime,
                   disable_specialist_runtime=args.disable_specialist_runtime,
                   expected_tasks=args.expected_tasks, workers=args.workers,
                   checkpoint_tasks=args.checkpoint_tasks,
                   disable_joint_residual=args.disable_joint_residual,
                   disable_stacking_modules=args.disable_stacking_modules)


if __name__ == "__main__":
    main()
