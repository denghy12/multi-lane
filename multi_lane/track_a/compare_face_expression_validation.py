"""Compare frozen expression Face experts against the old CLIP Face anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .compare_face_expert_quality import LOCKED_BETA, MINIMUM_GAIN, _fixed_candidate
from .fuse_face_endpoint_validation import fuse_face_endpoint_validation


def _named_path(value: str) -> Tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Expected NAME=/path/to/run")
    return name, Path(path)


def _row(
    name: str, full: Path, person: Path, face: Path, manifest: Path,
    representation_difference: bool,
) -> Dict[str, Any]:
    result = fuse_face_endpoint_validation(
        full, person, face, manifest,
        betas=(LOCKED_BETA,),
        allow_face_training_budget_difference=True,
        allow_face_representation_difference=representation_difference,
        require_stage1_grid=False,
    )
    fixed = _fixed_candidate(result)
    return {
        "name": name,
        "face_run": str(face.resolve()),
        "face_all_metrics": result["source_metrics"]["face_all_recomputed"],
        "face_reliable_metrics": result["source_metrics"][
            "face_reliable_subset_diagnostic_only"
        ],
        "locked_R1_metrics": fixed["metrics"],
        "locked_R1_task_metrics": fixed["task_metrics"],
        "reliability_counts": result["reliability_counts"],
    }


def compare(
    full: Path, person: Path, manifest: Path, anchor: Tuple[str, Path],
    candidates: Sequence[Tuple[str, Path]],
) -> Dict[str, Any]:
    anchor_row = _row(anchor[0], full, person, anchor[1], manifest, False)
    rows = [_row(name, full, person, path, manifest, True) for name, path in candidates]
    anchor_face = float(anchor_row["face_reliable_metrics"]["final_mAP"])
    anchor_r1 = float(anchor_row["locked_R1_metrics"]["final_mAP"])
    for row in rows:
        row["gain_vs_anchor"] = {
            "face_reliable_final_mAP": float(
                row["face_reliable_metrics"]["final_mAP"] - anchor_face
            ),
            "locked_R1_final_mAP": float(
                row["locked_R1_metrics"]["final_mAP"] - anchor_r1
            ),
        }
        row["passes_both_minimum_gains"] = all(
            gain >= MINIMUM_GAIN for gain in row["gain_vs_anchor"].values()
        )
    winner = max(
        rows,
        key=lambda item: (
            item["passes_both_minimum_gains"],
            item["locked_R1_metrics"]["final_mAP"],
            item["face_reliable_metrics"]["final_mAP"],
        ),
    )
    return {
        "schema_version": 1,
        "comparison": "frozen_expression_face_representation_with_locked_R1",
        "selection_split": "val",
        "test_accessed": False,
        "locked": {
            "face_beta": LOCKED_BETA,
            "effective_reliable_weights": [0.64, 0.16, 0.20],
            "invalid_face_weights": [0.80, 0.20, 0.0],
            "weight_search": False,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--person-run", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--anchor", type=_named_path, required=True)
    parser.add_argument("--candidate", type=_named_path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        args.full_run, args.person_run, args.face_manifest_root,
        args.anchor, args.candidate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "winner": result["winner"]["name"],
        **result["winner"]["gain_vs_anchor"],
        **result["decision"],
    }, indent=2), flush=True)
    print("FACE_EXPRESSION_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
