"""Export locked test scores from validation-trained compact model states.

This module deliberately performs no optimization or model selection.  It is
only valid after a validation artifact has locked one learned reliability gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
from torch.utils.data import DataLoader

from multi_lane.continual_datasets.continual_datasets import EMOTIC

from .model import MultiLaneModel
from .paired_transforms import PairedFullPersonTransform
from .openai_clip_loader import OPENAI_VIT_B16_SHA256, load_openai_clip_visual
from .runner import (
    CLASS_ORDER,
    TASK_SIZES,
    build_transforms,
    dataset_view,
    evaluate,
    git_metadata,
    resolve_dataset_parent,
    seen_indices,
    set_seed,
    summarize_tasks,
    validate_classes,
)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_load(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_locked_selection(path: Path) -> Tuple[Dict[str, Any], str]:
    selection = _read_json(path)
    if (
        selection.get("selection_split") != "val"
        or selection.get("test_accessed") is not False
        or selection.get("advance_to_locked_test") is not True
        or not selection.get("winner")
    ):
        raise ValueError("Validation did not lock one eligible learned gate")
    return selection, _sha256(path)


def restore_compact_model_state(
    model: MultiLaneModel, payload: Mapping[str, Any], task_id: int,
    source_git: Mapping[str, Any],
) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("task_id") != task_id
        or payload.get("source_git") != source_git
        or not isinstance(payload.get("model"), Mapping)
    ):
        raise ValueError("Compact checkpoint provenance mismatch")
    result = model.load_state_dict(payload["model"], strict=False)
    if result.unexpected_keys:
        raise ValueError(
            "Compact checkpoint has unexpected keys: "
            + ", ".join(result.unexpected_keys[:3])
        )
    if not result.missing_keys or any(
        not key.startswith("visual_encoder.") for key in result.missing_keys
    ):
        raise ValueError("Compact checkpoint is missing method state")
    model.restore_task(task_id)


def build_model(config: Mapping[str, Any], visual: torch.nn.Module) -> MultiLaneModel:
    return MultiLaneModel(
        visual_encoder=visual,
        task_sizes=TASK_SIZES,
        num_selectors=int(config["num_selectors"]),
        num_prompts=int(config["num_prompts"]),
        num_prompt_layers=int(config["num_prompt_layers"]),
        normalize=str(config["normalize"]),
        adapter_mode=str(config["adapter_mode"]),
        adapter_bottleneck_dim=int(config["adapter_bottleneck_dim"]),
        adapter_layer_indices=tuple(config["adapter_layer_indices"]),
        adapter_residual_scale=float(config["adapter_residual_scale"]),
        adapter_activation=str(config["adapter_activation"]),
        adapter_task_initialization=str(config["adapter_task_initialization"]),
        adapter_bottleneck_dims_per_task=tuple(
            config["adapter_bottleneck_dims_per_task"]
        ),
        adapter_residual_gate_mode=str(config["adapter_residual_gate_mode"]),
        adapter_auxiliary_metric_mode=str(config["adapter_regularization"]),
        selector_conditioning=str(config.get("selector_conditioning", "disabled")),
        selector_condition_layers=tuple(config.get("selector_condition_layers", (1,))),
        selector_condition_hidden_dim=int(config.get("selector_condition_hidden_dim", 32)),
        selector_condition_scale=float(config.get("selector_condition_scale", 0.1)),
    )


def audit_source_run(run: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = _read_json(run / "config.json")
    summary = _read_json(run / "seed_summary.json")
    if summary.get("config") != config or summary.get("status") != "complete":
        raise ValueError("Source validation run is incomplete")
    if (
        config.get("reporting_split") != "val"
        or config.get("evaluation_score_purpose") != "validation_search"
        or config.get("save_compact_checkpoints") is not True
        or config.get("compact_checkpoint_excludes_frozen_visual") is not True
        or config.get("calibration_training_exclusion") is not True
        or config.get("calibration_fraction") != 0.10
        or config.get("git", {}).get("dirty") is not False
        or len(summary.get("task_metrics", [])) != len(TASK_SIZES)
    ):
        raise ValueError("Source validation run has invalid locked-test provenance")
    return config, summary


def export_scores(
    source_run: Path,
    selection_path: Path,
    data_root: Path,
    clip_checkpoint: Path,
    output_root: Path,
    device_name: str,
) -> Dict[str, Any]:
    _, selection_sha = validate_locked_selection(selection_path)
    source_config, source_summary = audit_source_run(source_run)
    if output_root.exists():
        raise FileExistsError(output_root)
    if not clip_checkpoint.is_file():
        raise FileNotFoundError(clip_checkpoint)
    dataset_parent = resolve_dataset_parent(data_root)
    if Path(source_config["data_root"]).resolve() != (dataset_parent / "EMOTIC").resolve():
        raise ValueError("Test data root differs from the validation source")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Locked test export requires CUDA")
    set_seed(int(source_config["seed"]), bool(source_config["tf32"]))

    root = Path(__file__).resolve().parents[2]
    metadata = git_metadata(root)
    if metadata["dirty"]:
        raise RuntimeError("Locked test export requires a clean Git worktree")
    visual = load_openai_clip_visual(clip_checkpoint)
    model = build_model(source_config, visual).float().to(device)
    model.visual_encoder.requires_grad_(False)
    model.assert_visual_frozen()
    _, eval_transform = build_transforms(
        source_config["input_normalization"],
        source_config["train_crop_scale"],
        input_mode=source_config["input_mode"],
        person_transform_mode=source_config["person_transform_mode"],
        person_color_jitter_strength=source_config["person_color_jitter_strength"],
        person_color_jitter_probability=source_config[
            "person_color_jitter_probability"
        ],
    )
    test_source = EMOTIC(
        str(dataset_parent),
        train=False,
        transform=eval_transform,
        eval_splits=("test",),
        input_mode=source_config["input_mode"],
        person_crop_margin=float(source_config["person_crop_margin"]),
        paired_transform=(
            PairedFullPersonTransform(
                train=False, normalization=source_config["input_normalization"],
                crop_scale=source_config["train_crop_scale"],
                margin=float(source_config["person_crop_margin"]),
                jitter_strength=float(source_config["person_color_jitter_strength"]),
                jitter_probability=float(source_config["person_color_jitter_probability"]),
            ) if source_config.get("paired_full_person", False) else None
        ),
    )
    validate_classes(test_source)

    output_root.mkdir(parents=True)
    score_root = output_root / "test_scores"
    score_root.mkdir()
    task_rows = []
    checkpoint_hashes = {}
    amp = bool(source_config["amp"])
    for task_id in range(len(TASK_SIZES)):
        checkpoint = source_run / "compact_checkpoints" / f"task{task_id}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = _torch_load(checkpoint)
        restore_compact_model_state(
            model, payload, task_id, source_config["git"]
        )
        loader = DataLoader(
            dataset_view(
                test_source, seen_indices(task_id), include_sample_id=True
            ),
            batch_size=int(source_config["eval_batch_size"]),
            shuffle=False,
            num_workers=int(source_config["workers"]),
            pin_memory=False,
            drop_last=False,
        )
        task_rows.append(
            evaluate(
                model,
                loader,
                device,
                task_id,
                float(source_config["threshold"]),
                amp,
                score_output_path=score_root / f"task{task_id}.npz",
            )
        )
        checkpoint_hashes[str(task_id)] = _sha256(checkpoint)

    config = dict(source_config)
    config.update(
        {
            "reporting_split": "test",
            "evaluation_score_purpose": "learned_reliability_gate_locked_test",
            "validation_selection": str(selection_path.resolve()),
            "validation_selection_sha256": selection_sha,
            "source_validation_run": str(source_run.resolve()),
            "source_validation_config_sha256": _sha256(source_run / "config.json"),
            "source_validation_summary_sha256": _sha256(
                source_run / "seed_summary.json"
            ),
            "source_compact_checkpoint_sha256": checkpoint_hashes,
            "test_training_performed": False,
            "save_evaluation_scores": True,
            "save_calibration_scores": False,
            "save_compact_checkpoints": False,
            "git": metadata,
            "clip_checkpoint": str(clip_checkpoint.resolve()),
            "clip_checkpoint_sha256": OPENAI_VIT_B16_SHA256,
        }
    )
    result = {
        "config": config,
        "status": "complete",
        "elapsed_seconds": None,
        "metrics": summarize_tasks(task_rows),
        "task_metrics": [asdict(row) for row in task_rows],
        "source_validation_metrics": source_summary["metrics"],
    }
    (output_root / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "task_metrics.json").write_text(
        json.dumps(result["task_metrics"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "training_history.json").write_text(
        json.dumps(
            {
                "optimization_performed": False,
                "source_validation_run": str(source_run.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "seed_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--validation-selection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = export_scores(
        args.source_run,
        args.validation_selection,
        args.data_root,
        args.clip_checkpoint,
        args.output_root,
        args.device,
    )
    print(
        "LOCKED_COMPACT_TEST_EXPORT_COMPLETE",
        json.dumps(
            {
                "seed": result["config"]["seed"],
                "view": result["config"]["input_mode"],
                "final_mAP": result["metrics"]["final_mAP"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
