"""Summarize the seed0 shared-head versus Full-private-head diagnostic."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch

from .p0_r1_gap_diagnostic import (
    _bootstrap_final_map_differences,
    fuse_probabilities,
    load_view_scores,
)
from .runner import TASK_SIZES, compute_metrics, summarize_tasks


METHODS = ("H0", "HF")
MODES = {"H0": "shared_per_view", "HF": "full_private_per_view"}
P0_REFERENCE = 43.075407096617674
R1_REFERENCE = 43.5811932162358


def _read(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_run(method: str, run: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = _read(run / "config.json")
    summary = _read(run / "seed_summary.json")
    history = _read(run / "training_history.json")
    diagnostics = _read(run / "view_diagnostics.json")
    expected = {
        "seed": 0,
        "reporting_split": "val",
        "max_tasks": len(TASK_SIZES),
        "training_budget_mode": "epochs",
        "epochs_per_task": 30,
        "train_batch_size": 64,
        "eval_batch_size": 64,
        "view_fusion": "fixed_three_view",
        "view_classifier_mode": MODES[method],
        "view_auxiliary_loss_weight": 0.1,
        "view_gradient_routing": "joint",
        "adapter_mode": "image_token",
        "adapter_bottleneck_dim": 32,
        "adapter_view_bottleneck_dim": 0,
        "save_checkpoints": False,
        "save_compact_checkpoints": True,
        "save_evaluation_scores": True,
        "save_view_evaluation_scores": True,
    }
    for field, expected_value in expected.items():
        if config.get(field) != expected_value:
            raise ValueError(f"{method} config differs on {field}")
    if summary.get("status") != "complete" or summary.get("config") != config:
        raise ValueError(f"{method} summary is incomplete")
    if summary.get("completed_epochs") != 240:
        raise ValueError(f"{method} did not complete 240 epochs")
    if (
        set(history) != {str(task) for task in range(len(TASK_SIZES))}
        or any(len(history[str(task)]) != 30 for task in range(len(TASK_SIZES)))
        or sum(
            row["skipped_optimizer_steps"]
            for rows in history.values() for row in rows
        ) != 0
        or set(diagnostics) != set(history)
    ):
        raise ValueError(f"{method} training/diagnostic history is incomplete")
    expected_classifier = 13_338 if method == "H0" else 26_676
    if config.get("classifier_parameters") != expected_classifier:
        raise ValueError(f"{method} classifier parameter count differs")
    return config, summary


def _probability_rows(view_dumps: Sequence[Any]):
    rows = []
    for task_id, dump in enumerate(view_dumps):
        scores = fuse_probabilities(dump.probabilities, dump.face_reliable)
        rows.append(compute_metrics(
            task_id, torch.from_numpy(scores), torch.from_numpy(dump.targets), 0.5
        ))
    return rows


def _head_divergence(run: Path) -> Dict[str, float]:
    checkpoint = torch.load(
        run / "compact_checkpoints" / "task7.pth", map_location="cpu"
    )["model"]
    shared_weight = checkpoint["head.weight"].float()
    full_weight = checkpoint["full_view_head.weight"].float()
    shared_bias = checkpoint["head.bias"].float()
    full_bias = checkpoint["full_view_head.bias"].float()
    weight_delta = full_weight - shared_weight
    bias_delta = full_bias - shared_bias
    return {
        "weight_delta_l2": float(torch.linalg.vector_norm(weight_delta)),
        "weight_delta_relative_to_shared": float(
            torch.linalg.vector_norm(weight_delta)
            / torch.linalg.vector_norm(shared_weight).clamp_min(1e-12)
        ),
        "weight_cosine": float(torch.nn.functional.cosine_similarity(
            full_weight.flatten(), shared_weight.flatten(), dim=0
        )),
        "bias_delta_l2": float(torch.linalg.vector_norm(bias_delta)),
    }


def summarize(runs: Mapping[str, Path], bootstrap_replicates: int = 2000):
    if set(runs) != set(METHODS):
        raise ValueError("H0 and HF runs are required")
    records: Dict[str, Any] = {}
    view_dumps = {}
    configs = {}
    for method in METHODS:
        config, summary = _validate_run(method, runs[method])
        configs[method] = config
        dumps = [
            load_view_scores(runs[method] / "view_val_scores" / f"task{task}.npz")
            for task in range(len(TASK_SIZES))
        ]
        view_dumps[method] = dumps
        probability_rows = _probability_rows(dumps)
        records[method] = {
            "run": str(runs[method].resolve()),
            "training_logit_fusion": summary["metrics"],
            "training_logit_task_metrics": summary["task_metrics"],
            "offline_probability_fusion": summarize_tasks(probability_rows),
            "offline_probability_task_metrics": [asdict(row) for row in probability_rows],
            "final_view_metrics": _read(runs[method] / "view_diagnostics.json")["7"]["metrics"],
            "classifier_parameters": config["classifier_parameters"],
        }
    if configs["H0"]["git"]["commit"] != configs["HF"]["git"]["commit"]:
        raise ValueError("H0 and HF were not trained from one commit")
    for task_id, (h0, hf) in enumerate(zip(view_dumps["H0"], view_dumps["HF"])):
        if (
            h0.task_id != task_id or hf.task_id != task_id
            or not np.array_equal(h0.sample_ids, hf.sample_ids)
            or not np.array_equal(h0.targets, hf.targets)
            or not np.array_equal(h0.face_reliable, hf.face_reliable)
        ):
            raise ValueError(f"H0/HF task{task_id} evaluation samples differ")

    h0_logit = records["H0"]["training_logit_fusion"]["final_mAP"]
    hf_logit = records["HF"]["training_logit_fusion"]["final_mAP"]
    h0_average = records["H0"]["training_logit_fusion"]["average_mAP"]
    hf_average = records["HF"]["training_logit_fusion"]["average_mAP"]
    h0_full = records["H0"]["final_view_metrics"]["full"]["mAP"]
    hf_full = records["HF"]["final_view_metrics"]["full"]["mAP"]
    checks = {
        "H0_within_0p10_of_historical_P0": abs(h0_logit - P0_REFERENCE) <= 0.10,
        "HF_final_at_least_H0_plus_0p10": hf_logit >= h0_logit + 0.10,
        "HF_average_not_lower_than_H0": hf_average >= h0_average,
        "HF_Full_standalone_not_lower_than_H0": hf_full >= h0_full,
        "HF_gap_to_R1_smaller_than_H0": (
            abs(R1_REFERENCE - hf_logit) < abs(R1_REFERENCE - h0_logit)
        ),
    }
    final_h0 = view_dumps["H0"][-1]
    final_hf = view_dumps["HF"][-1]
    bootstrap = _bootstrap_final_map_differences(
        final_h0.sample_ids,
        final_h0.targets,
        {
            "H0_logit": final_h0.fused_probabilities,
            "HF_logit": final_hf.fused_probabilities,
            "H0_probability": fuse_probabilities(
                final_h0.probabilities, final_h0.face_reliable
            ),
            "HF_probability": fuse_probabilities(
                final_hf.probabilities, final_hf.face_reliable
            ),
        },
        (
            ("HF_logit", "H0_logit"),
            ("HF_probability", "H0_probability"),
        ),
        bootstrap_replicates,
    )
    return {
        "schema_version": 1,
        "stage": "seed0_full_private_classifier_validation_only",
        "selection_uses_test": False,
        "git_commit": configs["H0"]["git"]["commit"],
        "historical_references": {
            "P0_final_mAP": P0_REFERENCE,
            "independent_R1_final_mAP": R1_REFERENCE,
        },
        "methods": records,
        "paired_differences": {
            "HF_minus_H0_final_mAP_logit": hf_logit - h0_logit,
            "HF_minus_H0_average_mAP_logit": hf_average - h0_average,
            "HF_minus_H0_Full_standalone_mAP": hf_full - h0_full,
            "HF_minus_H0_final_mAP_probability": (
                records["HF"]["offline_probability_fusion"]["final_mAP"]
                - records["H0"]["offline_probability_fusion"]["final_mAP"]
            ),
        },
        "HF_final_head_divergence": _head_divergence(runs["HF"]),
        "paired_image_group_bootstrap": bootstrap,
        "advance_checks": checks,
        "full_private_head_effect_supported": all(checks.values()),
        "run_seed1_seed2": all(checks.values()),
        "test_permitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h0-run", type=Path, required=True)
    parser.add_argument("--hf-run", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(
        {"H0": args.h0_run, "HF": args.hf_run}, args.bootstrap_replicates
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "paired_differences": result["paired_differences"],
        "advance_checks": result["advance_checks"],
        "effect_supported": result["full_private_head_effect_supported"],
    }, indent=2), flush=True)
    print("FULL_PRIVATE_HEAD_DIAGNOSTIC_COMPLETE test_accessed=false", flush=True)


if __name__ == "__main__":
    main()
