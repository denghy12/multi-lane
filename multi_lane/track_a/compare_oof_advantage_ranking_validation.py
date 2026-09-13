"""Compare OOF-advantage ranking E1 with the existing fresh Full D0 anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from .compare_oof_distillation_validation import MINIMUM_FINAL_GAIN, _named_path, _row
from .compare_oof_r1_teacher_validation import LOCKED_CONFIG_KEYS


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(
    anchor: Tuple[str, Path], candidate: Tuple[str, Path],
    person: Path, face: Path, manifest: Path,
) -> Dict[str, Any]:
    if anchor[0] != "D0" or candidate[0] != "E1":
        raise ValueError("Expected D0 anchor and E1 candidate")
    anchor_config = _read(anchor[1] / "config.json")
    candidate_config = _read(candidate[1] / "config.json")
    mismatches = {
        key: (anchor_config.get(key), candidate_config.get(key))
        for key in LOCKED_CONFIG_KEYS
        if anchor_config.get(key) != candidate_config.get(key)
    }
    if mismatches:
        raise ValueError(f"D0/E1 locked configuration differs: {mismatches}")
    anchor_row = _row("D0", anchor[1], person, face, manifest)
    candidate_row = _row(
        "E1",
        candidate[1],
        person,
        face,
        manifest,
        allow_full_training_objective_difference=True,
    )
    protocol = candidate_config.get("oof_advantage_ranking", {})
    if (
        candidate_row["distillation"] is not None
        or protocol.get("loss") != "softplus(negative_logit-positive_logit)"
        or float(protocol.get("loss_weight", -1)) != 0.05
        or float(protocol.get("hard_bce_weight", -1)) != 1.0
        or protocol.get("adapter_receives_ranking_gradient") is not False
    ):
        raise ValueError("E1 ranking protocol differs")
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
        "comparison": "oof_advantage_limited_pairwise_ranking_distillation",
        "selection_split": "val",
        "test_accessed": False,
        "locked": {
            "hard_bce_weight": 1.0,
            "ranking_loss_weight": 0.05,
            "ranking_batch_size": 16,
            "minimum_full_and_R1_final_gain": MINIMUM_FINAL_GAIN,
            "minimum_full_average_gain": 0.0,
            "hyperparameter_search": False,
        },
        "anchor": anchor_row,
        "candidate": candidate_row,
        "decision": {
            "advance_to_seed1_seed2_validation": bool(candidate_row["eligible"]),
            "run_test": False,
            "end_oof_distillation_if_ineligible": True,
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
    print("OOF_ADVANTAGE_RANKING_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
