"""Validate and rank the paired Image-token Adapter multilayer screen."""

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


STRUCTURES = (
    ("single3", (3,), False),
    ("single11", (11,), False),
    ("pair2_3", (2, 3), True),
    ("pair3_7", (3, 7), True),
    ("pair3_11", (3, 11), True),
    ("late8_9", (8, 9), True),
    ("late_blocks8_12", (7, 8, 9, 10, 11), True),
)
PER_LAYER_PER_TASK_PARAMETERS = 49952
BASE_TRAINABLE_PARAMETERS = 689178
NUM_TASKS = 8


def _validate_candidate(
    run: Dict[str, Any], layers: Sequence[int], routing: str
) -> None:
    config = run["config"]
    layer_count = len(layers)
    expected_per_task = layer_count * PER_LAYER_PER_TASK_PARAMETERS
    if config.get("adapter_mode") != "image_token":
        raise ValueError(f"Run {run['label']} is not an Image-token Adapter")
    if config.get("adapter_layer_indices") != list(layers):
        raise ValueError(f"Run {run['label']} has the wrong Adapter layers")
    metadata_valid = (
        config.get("adapter_target") == "frozen_image_tokens_for_selector"
        and config.get("adapter_image_token_scope")
        == "block_ln1_cls_plus_patch_tokens"
        and _close(config.get("adapter_learning_rate"), 0.0004)
        and config.get("adapter_initialization_rng") == "forked_global_state"
        and config.get("adapter_parameters_per_task") == expected_per_task
        and config.get("adapter_parameters") == expected_per_task * NUM_TASKS
        and config.get("trainable_parameters")
        == BASE_TRAINABLE_PARAMETERS + expected_per_task
    )
    if not metadata_valid:
        raise ValueError(f"Run {run['label']} has invalid Adapter metadata")
    if config.get("parameter_group_loss_routing") != routing:
        raise ValueError(f"Run {run['label']} has the wrong loss routing")
    if config.get("model_parameter_objective") != "bce":
        raise ValueError(f"Run {run['label']} must retain model BCE")
    if routing == "joint_bce":
        if (
            config.get("adapter_parameter_objective") != "bce"
            or config.get("asl") is not None
        ):
            raise ValueError(f"Run {run['label']} has invalid BCE metadata")
        return
    asl = config.get("asl") or {}
    expected_asl = {
        "gamma_neg": 9.8,
        "gamma_pos": 0.0,
        "clip": 0.05,
        "eps": 1e-8,
        "detach_focal_weight": True,
        "reduction": "mean_over_training_loss_view",
    }
    if (
        config.get("adapter_parameter_objective") != "asl"
        or any(not _close(asl.get(key), value) for key, value in expected_asl.items())
    ):
        raise ValueError(f"Run {run['label']} has invalid ASL metadata")


def _prefixed_gates(prefix: str, gates: Dict[str, bool]) -> Dict[str, bool]:
    return {f"{prefix}_{key}": value for key, value in gates.items()}


def _task6_map(run: Dict[str, Any]) -> float:
    return float(run["rows"][6]["mAP"])


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


def summarize_multilayer_search(
    labeled_roots: Sequence[Tuple[str, Path]],
) -> Dict[str, Any]:
    expected_labels = {"disabled_bce"}
    for structure, _, _ in STRUCTURES:
        expected_labels.add(f"{structure}_bce")
        expected_labels.add(f"{structure}_asl")
    labels = [label for label, _ in labeled_roots]
    if set(labels) != expected_labels or len(labels) != len(expected_labels):
        raise ValueError(
            "Multilayer screen requires disabled plus paired BCE/ASL structures"
        )
    runs = {
        label: _load_complete_run(label, root) for label, root in labeled_roots
    }
    commits = {
        (run["config"]["git"].get("commit"), run["config"]["git"].get("tree"))
        for run in runs.values()
    }
    if len(commits) != 1:
        raise ValueError("Multilayer runs do not share one Git commit and tree")

    disabled = runs["disabled_bce"]
    _validate_disabled(disabled)
    for structure, layers, _ in STRUCTURES:
        _validate_candidate(runs[f"{structure}_bce"], layers, "joint_bce")
        _validate_candidate(runs[f"{structure}_asl"], layers, "adapter_asl")

    controls: Dict[str, Any] = {}
    for structure in ("single3", "single11"):
        bce = runs[f"{structure}_bce"]
        asl = runs[f"{structure}_asl"]
        bce_vs_disabled = _comparison(bce, disabled)
        asl_vs_bce = _comparison(asl, bce)
        asl_vs_disabled = _comparison(asl, disabled)
        bce_gates = _gates(bce_vs_disabled)
        asl_gates = {
            **_prefixed_gates("vs_paired_bce", _gates(asl_vs_bce)),
            **_prefixed_gates("vs_disabled", _gates(asl_vs_disabled)),
        }
        controls[structure] = {
            "layers": list(next(row[1] for row in STRUCTURES if row[0] == structure)),
            "bce": {
                **_run_result(bce),
                "vs_disabled": bce_vs_disabled,
                "eligibility_gates": bce_gates,
                "eligible": all(bce_gates.values()),
            },
            "asl": {
                **_run_result(asl),
                "vs_paired_bce": asl_vs_bce,
                "vs_disabled": asl_vs_disabled,
                "eligibility_gates": asl_gates,
                "eligible": all(asl_gates.values()),
            },
        }

    best_single = {}
    for route in ("bce", "asl"):
        single_runs = [runs[f"single3_{route}"], runs[f"single11_{route}"]]
        best_single[route] = {
            "final_mAP": max(_summary_metrics(run)["final_mAP"] for run in single_runs),
            "task6_mAP": max(_task6_map(run) for run in single_runs),
            "best_final_label": max(
                single_runs, key=lambda run: _summary_metrics(run)["final_mAP"]
            )["label"],
            "best_task6_label": max(single_runs, key=_task6_map)["label"],
        }

    candidates = []
    for structure, layers, is_multilayer in STRUCTURES:
        if not is_multilayer:
            continue
        bce = runs[f"{structure}_bce"]
        asl = runs[f"{structure}_asl"]
        bce_vs_disabled = _comparison(bce, disabled)
        asl_vs_bce = _comparison(asl, bce)
        asl_vs_disabled = _comparison(asl, disabled)
        bce_gates = {
            **_prefixed_gates("vs_disabled", _gates(bce_vs_disabled)),
            "final_mAP_not_below_best_single": (
                _summary_metrics(bce)["final_mAP"] >= best_single["bce"]["final_mAP"]
            ),
            "task6_mAP_not_below_best_single": (
                _task6_map(bce) >= best_single["bce"]["task6_mAP"]
            ),
        }
        asl_gates = {
            **_prefixed_gates("vs_paired_bce", _gates(asl_vs_bce)),
            **_prefixed_gates("vs_disabled", _gates(asl_vs_disabled)),
            "final_mAP_not_below_best_single": (
                _summary_metrics(asl)["final_mAP"] >= best_single["asl"]["final_mAP"]
            ),
            "task6_mAP_not_below_best_single": (
                _task6_map(asl) >= best_single["asl"]["task6_mAP"]
            ),
        }
        candidates.append({
            "structure": structure,
            "layers": list(layers),
            "bce": {
                **_run_result(bce),
                "vs_disabled": bce_vs_disabled,
                "vs_single3": _comparison(bce, runs["single3_bce"]),
                "vs_single11": _comparison(bce, runs["single11_bce"]),
                "eligibility_gates": bce_gates,
                "eligible": all(bce_gates.values()),
            },
            "asl": {
                **_run_result(asl),
                "vs_paired_bce": asl_vs_bce,
                "vs_disabled": asl_vs_disabled,
                "vs_single3": _comparison(asl, runs["single3_asl"]),
                "vs_single11": _comparison(asl, runs["single11_asl"]),
                "eligibility_gates": asl_gates,
                "eligible": all(asl_gates.values()),
            },
        })

    def rank(route: str) -> list:
        rows = [row for row in candidates if row[route]["eligible"]]
        rows.sort(
            key=lambda row: (
                row[route]["summary_metrics"]["final_mAP"],
                row[route]["task6_new_classes"]["mean_ap"],
                row[route]["summary_metrics"]["average_mAP"],
                row[route]["summary_metrics"]["final_cF1"],
                row[route]["summary_metrics"]["final_oF1"],
                -row[route]["summary_metrics"]["forgetting"],
            ),
            reverse=True,
        )
        return rows

    eligible_bce = rank("bce")
    eligible_asl = rank("asl")
    best_final_bce = max(
        candidates, key=lambda row: row["bce"]["summary_metrics"]["final_mAP"]
    )
    best_final_asl = max(
        candidates, key=lambda row: row["asl"]["summary_metrics"]["final_mAP"]
    )
    commit, tree = next(iter(commits))
    return {
        "schema_version": 1,
        "status": "complete",
        "comparison": "image_token_adapter_fp32_seed0_multilayer_validation_screen",
        "git": {"commit": commit, "tree": tree},
        "protocol": {
            "structures": {
                structure: list(layers) for structure, layers, _ in STRUCTURES
            },
            "losses": ["joint_bce", "model_bce_adapter_asl"],
            "asl": {"gamma_neg": 9.8, "gamma_pos": 0.0, "clip": 0.05},
            "selection": (
                "all hard gates vs disabled; ASL also vs paired BCE; multilayer "
                "final/task6 mAP not below the best fresh single3/single11 control"
            ),
        },
        "disabled": _run_result(disabled),
        "single_controls": controls,
        "best_single_thresholds": best_single,
        "multilayer_candidates": candidates,
        "eligible_bce_structures": [row["structure"] for row in eligible_bce],
        "eligible_asl_structures": [row["structure"] for row in eligible_asl],
        "winner_bce_structure": (
            eligible_bce[0]["structure"] if eligible_bce else None
        ),
        "winner_asl_structure": (
            eligible_asl[0]["structure"] if eligible_asl else None
        ),
        "best_final_mAP_bce_structure": best_final_bce["structure"],
        "best_final_mAP_asl_structure": best_final_asl["structure"],
        "continue_with_bce": bool(eligible_bce),
        "continue_with_asl": bool(eligible_asl),
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
    result = summarize_multilayer_search(args.run)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
