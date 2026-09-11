"""Compare locked-R1 fusion for a small predeclared Face expert validation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .fuse_face_endpoint_validation import fuse_face_endpoint_validation


LOCKED_BETA = 0.20
MINIMUM_GAIN = 0.05


def _fixed_candidate(result: Dict[str, Any]) -> Dict[str, Any]:
    return next(
        row for row in result["candidates"]
        if abs(float(row["beta"]) - LOCKED_BETA) < 1e-12
    )


def _summary(name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    fixed = _fixed_candidate(result)
    return {
        "name": name,
        "face_run": result["runs"]["face"],
        "face_all_metrics": result["source_metrics"]["face_all_recomputed"],
        "face_reliable_metrics": result["source_metrics"][
            "face_reliable_subset_diagnostic_only"
        ],
        "locked_R1_beta": LOCKED_BETA,
        "locked_R1_metrics": fixed["metrics"],
        "locked_R1_task_metrics": fixed["task_metrics"],
        "reliability_counts": result["reliability_counts"],
    }


def compare_face_experts(
    full_run: Path,
    person_run: Path,
    manifest_root: Path,
    anchor: Tuple[str, Path],
    candidates: Sequence[Tuple[str, Path]],
) -> Dict[str, Any]:
    if not candidates or len({name for name, _ in candidates}) != len(candidates):
        raise ValueError("Face candidate names must be nonempty and unique")
    anchor_row = _summary(
        anchor[0],
        fuse_face_endpoint_validation(
            full_run, person_run, anchor[1], manifest_root
        ),
    )
    rows = [
        _summary(
            name,
            fuse_face_endpoint_validation(
                full_run, person_run, run, manifest_root
            ),
        )
        for name, run in candidates
    ]
    anchor_fusion = float(anchor_row["locked_R1_metrics"]["final_mAP"])
    anchor_face = float(anchor_row["face_reliable_metrics"]["final_mAP"])
    for row in rows:
        row["gain_vs_anchor"] = {
            "locked_R1_final_mAP": float(
                row["locked_R1_metrics"]["final_mAP"] - anchor_fusion
            ),
            "face_reliable_final_mAP": float(
                row["face_reliable_metrics"]["final_mAP"] - anchor_face
            ),
        }
        row["passes_both_minimum_gains"] = all(
            gain >= MINIMUM_GAIN for gain in row["gain_vs_anchor"].values()
        )
    winner = max(
        rows,
        key=lambda row: (
            row["passes_both_minimum_gains"],
            row["locked_R1_metrics"]["final_mAP"],
            row["face_reliable_metrics"]["final_mAP"],
        ),
    )
    return {
        "schema_version": 1,
        "selection_split": "val",
        "test_accessed": False,
        "comparison": "independent_face_quality_with_locked_R1",
        "locked": {
            "full_weight_before_face": 0.80,
            "person_weight_before_face": 0.20,
            "face_beta": LOCKED_BETA,
            "effective_reliable_weights": [0.64, 0.16, 0.20],
            "invalid_face_weights": [0.80, 0.20, 0.0],
        },
        "minimum_gain_each_final_mAP": MINIMUM_GAIN,
        "anchor": anchor_row,
        "candidates": rows,
        "winner": winner,
        "decision": {
            "advance_winner_to_seed1_seed2_validation": bool(
                winner["passes_both_minimum_gains"]
            ),
            "run_test": False,
        },
    }


def _named_path(value: str) -> Tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Expected NAME=/path/to/run")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--person-run", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--anchor", type=_named_path, required=True)
    parser.add_argument("--candidate", type=_named_path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_face_experts(
        args.full_run, args.person_run, args.face_manifest_root,
        args.anchor, args.candidate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "winner": result["winner"]["name"],
        "locked_R1_final_mAP": result["winner"]["locked_R1_metrics"]["final_mAP"],
        "face_reliable_final_mAP": result["winner"]["face_reliable_metrics"]["final_mAP"],
        **result["decision"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
