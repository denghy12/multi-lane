"""Compare locked OOF-R1 teacher D3 against the existing fresh Full D0 anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from .compare_oof_distillation_validation import MINIMUM_FINAL_GAIN, _named_path, _row


LOCKED_CONFIG_KEYS = (
    "seed", "class_order", "task_sizes", "reporting_split", "max_tasks",
    "epochs_per_task", "train_batch_size", "eval_batch_size", "threshold",
    "training_loss_mode", "parameter_group_loss_routing", "learning_rate",
    "scheduler", "scheduler_mode", "scheduler_min_lr_ratio",
    "scheduler_warmup_ratio", "input_mode", "input_normalization",
    "train_crop_scale", "full_crop_mode", "amp", "tf32", "adapter_mode",
    "adapter_bottleneck_dim", "adapter_layer_indices", "adapter_residual_scale",
    "adapter_residual_gate_mode", "adapter_activation",
    "adapter_task_initialization", "adapter_learning_rate",
    "adapter_weight_decay", "adapter_regularization", "clip_checkpoint_sha256",
)


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(
    anchor: Tuple[str, Path],
    candidate: Tuple[str, Path],
    person: Path,
    face: Path,
    manifest: Path,
) -> Dict[str, Any]:
    if anchor[0] != "D0" or candidate[0] != "D3":
        raise ValueError("Expected D0 anchor and D3 candidate")
    anchor_config = _read(anchor[1] / "config.json")
    candidate_config = _read(candidate[1] / "config.json")
    mismatches = {
        key: (anchor_config.get(key), candidate_config.get(key))
        for key in LOCKED_CONFIG_KEYS
        if anchor_config.get(key) != candidate_config.get(key)
    }
    if mismatches:
        raise ValueError(f"D0/D3 locked configuration differs: {mismatches}")
    anchor_row = _row("D0", anchor[1], person, face, manifest)
    candidate_row = _row("D3", candidate[1], person, face, manifest)
    if anchor_row["distillation"] is not None:
        raise ValueError("D0 must not contain OOF distillation")
    distillation = candidate_row["distillation"] or {}
    if distillation.get("mode") != "r1" or float(distillation.get("mix", -1)) != 0.20:
        raise ValueError("D3 must use the locked OOF R1 teacher at mix 0.20")
    gains = {
        "full_final_mAP": float(
            candidate_row["full_metrics"]["final_mAP"]
            - anchor_row["full_metrics"]["final_mAP"]
        ),
        "full_average_mAP": float(
            candidate_row["full_metrics"]["average_mAP"]
            - anchor_row["full_metrics"]["average_mAP"]
        ),
        "locked_R1_final_mAP": float(
            candidate_row["locked_R1_metrics"]["final_mAP"]
            - anchor_row["locked_R1_metrics"]["final_mAP"]
        ),
    }
    candidate_row["gain_vs_D0"] = gains
    candidate_row["eligible"] = (
        gains["full_final_mAP"] >= MINIMUM_FINAL_GAIN
        and gains["full_average_mAP"] >= 0
        and gains["locked_R1_final_mAP"] >= MINIMUM_FINAL_GAIN
    )
    return {
        "schema_version": 1,
        "comparison": "oof_locked_R1_teacher_distillation_of_full_lane",
        "selection_split": "val",
        "test_accessed": False,
        "locked": {
            "distillation_mix": 0.20,
            "teacher_R1_reliable_weights": [0.64, 0.16, 0.20],
            "teacher_R1_invalid_face_weights": [0.80, 0.20, 0.0],
            "minimum_full_and_R1_final_gain": MINIMUM_FINAL_GAIN,
            "minimum_full_average_gain": 0.0,
            "weight_search": False,
        },
        "anchor": anchor_row,
        "candidate": candidate_row,
        "decision": {
            "advance_to_seed1_seed2_validation": bool(candidate_row["eligible"]),
            "run_test": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=_named_path, required=True)
    parser.add_argument("--candidate", type=_named_path, required=True)
    parser.add_argument("--person-run", type=Path, required=True)
    parser.add_argument("--face-run", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        args.anchor, args.candidate, args.person_run, args.face_run,
        args.face_manifest_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    candidate = result["candidate"]
    print(json.dumps({
        "candidate": candidate["name"],
        "eligible": candidate["eligible"],
        **candidate["gain_vs_D0"],
        **result["decision"],
    }, indent=2), flush=True)
    print("OOF_R1_TEACHER_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
