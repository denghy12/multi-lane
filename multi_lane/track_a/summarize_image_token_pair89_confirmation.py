"""Validate the fresh BCE-only single8/single9/pair8_9 confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .summarize_image_token_layer_search import (
    _close,
    _comparison,
    _focus_metrics,
    _gates,
    _load_complete_run,
    _summary_metrics,
    _validate_disabled,
)


STRUCTURES = {
    "single8_bce": (8,),
    "single9_bce": (9,),
    "pair8_9_bce": (8, 9),
}
PER_LAYER_PER_TASK_PARAMETERS = 49952
BASE_TRAINABLE_PARAMETERS = 689178
NUM_TASKS = 8


def _validate_bce_candidate(
    run: Dict[str, Any], layers: Sequence[int]
) -> None:
    config = run["config"]
    per_task = len(layers) * PER_LAYER_PER_TASK_PARAMETERS
    valid = (
        config.get("adapter_mode") == "image_token"
        and config.get("adapter_layer_indices") == list(layers)
        and config.get("adapter_target") == "frozen_image_tokens_for_selector"
        and config.get("adapter_image_token_scope")
        == "block_ln1_cls_plus_patch_tokens"
        and config.get("adapter_initialization_rng") == "forked_global_state"
        and _close(config.get("adapter_learning_rate"), 0.0004)
        and config.get("parameter_group_loss_routing") == "joint_bce"
        and config.get("model_parameter_objective") == "bce"
        and config.get("adapter_parameter_objective") == "bce"
        and config.get("asl") is None
        and config.get("adapter_parameters_per_task") == per_task
        and config.get("adapter_parameters") == per_task * NUM_TASKS
        and config.get("trainable_parameters")
        == BASE_TRAINABLE_PARAMETERS + per_task
    )
    if not valid:
        raise ValueError(f"Run {run['label']} has invalid BCE Adapter metadata")


def _run_result(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": run["label"],
        "root": run["root"],
        "summary_metrics": _summary_metrics(run),
        "task_curve": [
            {key: row[key] for key in ("task_id", "mAP", "cF1", "oF1")}
            for row in run["rows"]
        ],
        "task6_new_classes": _focus_metrics(run),
    }


def _strict_pair_vs_single9_gates(
    comparison: Dict[str, Any]
) -> Dict[str, bool]:
    summary = comparison["summary_deltas"]
    classes = comparison["task6_new_class_deltas"]
    return {
        "final_mAP_non_decreasing": summary["final_mAP"] >= 0.0,
        "average_mAP_non_decreasing": summary["average_mAP"] >= 0.0,
        "task6_mAP_non_decreasing": comparison["task6_mAP_delta"] >= 0.0,
        "sadness_ap_non_decreasing": classes["Sadness"] >= 0.0,
        "sensitivity_ap_non_decreasing": classes["Sensitivity"] >= 0.0,
        "suffering_ap_non_decreasing": classes["Suffering"] >= 0.0,
        "final_cF1_within_half_point": summary["final_cF1"] >= -0.5,
        "final_oF1_within_half_point": summary["final_oF1"] >= -0.5,
        "forgetting_non_increasing": summary["forgetting"] <= 0.0,
    }


def summarize_pair89_confirmation(
    labeled_roots: Sequence[Tuple[str, Path]],
) -> Dict[str, Any]:
    expected_labels = {"disabled_bce", *STRUCTURES}
    labels = [label for label, _ in labeled_roots]
    if set(labels) != expected_labels or len(labels) != len(expected_labels):
        raise ValueError(
            "Pair8/9 confirmation requires disabled, single8, single9, and pair8_9"
        )
    runs = {
        label: _load_complete_run(label, root) for label, root in labeled_roots
    }
    commits = {
        (run["config"]["git"].get("commit"), run["config"]["git"].get("tree"))
        for run in runs.values()
    }
    if len(commits) != 1:
        raise ValueError("Pair8/9 confirmation runs do not share one commit/tree")

    disabled = runs["disabled_bce"]
    _validate_disabled(disabled)
    for label, layers in STRUCTURES.items():
        _validate_bce_candidate(runs[label], layers)

    candidate_results = {}
    for label in STRUCTURES:
        run = runs[label]
        comparison = _comparison(run, disabled)
        gates = _gates(comparison)
        candidate_results[label] = {
            **_run_result(run),
            "vs_disabled": comparison,
            "eligibility_gates": gates,
            "eligible_vs_disabled": all(gates.values()),
        }

    pair = runs["pair8_9_bce"]
    pair_vs_single8 = _comparison(pair, runs["single8_bce"])
    pair_vs_single9 = _comparison(pair, runs["single9_bce"])
    pair_vs_disabled_gates = candidate_results["pair8_9_bce"][
        "eligibility_gates"
    ]
    pair_vs_single9_gates = _strict_pair_vs_single9_gates(pair_vs_single9)
    confirmation_gates = {
        **{
            f"vs_disabled_{key}": value
            for key, value in pair_vs_disabled_gates.items()
        },
        **{
            f"vs_single9_{key}": value
            for key, value in pair_vs_single9_gates.items()
        },
    }
    best_single_label = max(
        ("single8_bce", "single9_bce"),
        key=lambda label: _summary_metrics(runs[label])["final_mAP"],
    )
    commit, tree = next(iter(commits))
    return {
        "schema_version": 1,
        "status": "complete",
        "comparison": "image_token_adapter_fp32_seed0_pair8_9_bce_confirmation",
        "git": {"commit": commit, "tree": tree},
        "protocol": {
            "structures": {
                "disabled_bce": [],
                **{label: list(layers) for label, layers in STRUCTURES.items()},
            },
            "loss": "joint_bce",
            "selection": (
                "pair8_9 passes all gates vs disabled and strictly preserves "
                "final/average/task6 mAP, task6 classes, F1, and forgetting vs single9"
            ),
        },
        "disabled": _run_result(disabled),
        "candidates": candidate_results,
        "best_single_label": best_single_label,
        "pair8_9": {
            **candidate_results["pair8_9_bce"],
            "vs_single8": pair_vs_single8,
            "vs_single9": pair_vs_single9,
            "confirmation_gates": confirmation_gates,
            "confirmed_for_formal_test": all(confirmation_gates.values()),
        },
        "recommended_validation_structure": (
            "pair8_9_bce"
            if all(confirmation_gates.values())
            else best_single_label
        ),
    }


def _labeled_root(value: str) -> Tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("Run must have LABEL=PATH form")
    return label, Path(path).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=_labeled_root, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarize_pair89_confirmation(args.run)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
