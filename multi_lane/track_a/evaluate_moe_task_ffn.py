"""Evaluate all eight frozen Task-Forward MoE checkpoints on one split."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .evaluate_joint26_checkpoint import make_source
from .export_compact_test_scores import _sha256, _torch_load, restore_compact_model_state
from .model import MultiLaneModel
from .openai_clip_loader import load_openai_clip_visual
from .runner import TASK_SIZES, dataset_view, evaluate, evaluate_view_diagnostics, seen_indices, set_seed, summarize_tasks


def evaluate_split(source_run: Path, output: Path, split: str,
                   data_root: Path, clip_checkpoint: Path,
                   face_manifest_root: Path, batch_size: int,
                   allow_smoke: bool = False) -> dict:
    if split not in {"val", "test"}:
        raise ValueError("Split must be val or test")
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((source_run / "config.json").read_text())
    summary = json.loads((source_run / "seed_summary.json").read_text())
    expected_tasks = 1 if allow_smoke else len(TASK_SIZES)
    if (summary.get("status") != "complete" or summary.get("config") != config
        or config.get("git", {}).get("dirty") is not False
        or config.get("adapter_mode") != "moe_task_ffn"
        or config.get("parax_mode") != "disabled"
        or not config.get("save_compact_checkpoints")
        or config.get("max_tasks") != expected_tasks
        or len(summary.get("task_metrics", [])) != expected_tasks):
        raise ValueError("Source run lacks complete clean MoE provenance")
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
        adapter_mode="moe_task_ffn",
        adapter_bottleneck_dim=config["adapter_bottleneck_dim"],
        adapter_layer_indices=config["adapter_layer_indices"],
        adapter_residual_scale=config["adapter_residual_scale"],
        moe_num_experts=config["moe_num_experts"],
        moe_top_k=config["moe_top_k"],
        moe_dropout=config["moe_dropout"],
        moe_noisy_gating=config["moe_noisy_gating"],
        view_fusion=config["view_fusion"],
        view_classifier_mode=config["view_classifier_mode"],
    ).float().to(device)
    model.assert_visual_frozen()
    source = make_source(config, data_root, face_manifest_root, split)
    output.mkdir(parents=True)
    (output / "scores").mkdir()
    (output / "view_scores").mkdir()
    rows = []
    views = {}
    routing = {}
    hashes = {}
    for task_id in range(expected_tasks):
        checkpoint = source_run / "compact_checkpoints" / f"task{task_id}.pth"
        payload = _torch_load(checkpoint)
        restore_compact_model_state(model, payload, task_id, config["git"])
        model.eval()
        model.adapter_bank.reset_route_statistics()
        loader = DataLoader(
            dataset_view(source, seen_indices(task_id), include_sample_id=True),
            batch_size=batch_size, shuffle=False, num_workers=config["workers"],
            pin_memory=False, drop_last=False,
        )
        row = evaluate(model, loader, device, task_id, config["threshold"],
                       amp=True, score_output_path=output / "scores" / f"task{task_id}.npz")
        routing[str(task_id)] = model.adapter_bank.route_statistics()
        rows.append(row)
        views[str(task_id)] = evaluate_view_diagnostics(
            model, loader, device, task_id, config["threshold"], amp=True,
            score_output_path=output / "view_scores" / f"task{task_id}.npz",
        )
        hashes[str(task_id)] = _sha256(checkpoint)
        print(f"split={split} task={task_id} mAP={row.mAP:.6f}", flush=True)
    result = {
        "split": split,
        "source_run": str(source_run.resolve()),
        "source_git": config["git"],
        "checkpoint_sha256": hashes,
        "clip_checkpoint_sha256": _sha256(clip_checkpoint),
        "task_metrics": [asdict(row) for row in rows],
        "summary": summarize_tasks(rows),
        "view_diagnostics": views,
        "route_statistics": routing,
        "test_used_for_selection": False,
        "smoke_validation_replay": bool(allow_smoke),
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-smoke", action="store_true",
                        help="Replay one-task smoke checkpoint on validation only.")
    args = parser.parse_args()
    if args.allow_smoke and args.split != "val":
        parser.error("Smoke replay is validation-only")
    evaluate_split(args.source_run, args.output, args.split, args.data_root,
                   args.clip_checkpoint, args.face_manifest_root, args.batch_size,
                   allow_smoke=args.allow_smoke)


if __name__ == "__main__":
    main()
