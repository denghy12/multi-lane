"""Compare Full-lane OOF distillation candidates and their locked R1 ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .fuse_face_endpoint_validation import fuse_face_endpoint_validation


MINIMUM_FINAL_GAIN = 0.05


def _named_path(value: str) -> Tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("Expected NAME=/path/to/run")
    return name, Path(path)


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(
    name: str,
    full: Path,
    person: Path,
    face: Path,
    manifest: Path,
    allow_full_training_objective_difference: bool = False,
) -> Dict[str, Any]:
    summary = _read(full / "seed_summary.json")
    config = _read(full / "config.json")
    if summary.get("config") != config or summary.get("status") != "complete":
        raise ValueError(f"Incomplete Full candidate: {name}")
    fused = fuse_face_endpoint_validation(
        full,
        person,
        face,
        manifest,
        betas=(0.20,),
        allow_full_training_objective_difference=(
            allow_full_training_objective_difference
        ),
        require_stage1_grid=False,
    )
    return {
        "name": name,
        "run": str(full.resolve()),
        "distillation": config.get("oof_distillation"),
        "full_metrics": summary["metrics"],
        "full_task_metrics": summary["task_metrics"],
        "locked_R1_metrics": fused["winner"]["metrics"],
        "locked_R1_task_metrics": fused["winner"]["task_metrics"],
    }


def compare(
    anchor: Tuple[str, Path],
    candidates: Sequence[Tuple[str, Path]],
    person: Path,
    face: Path,
    manifest: Path,
) -> Dict[str, Any]:
    if anchor[0] != "D0":
        raise ValueError("Fresh Full anchor must be named D0")
    if {name for name, _ in candidates} != {"D1", "D2"}:
        raise ValueError("Distillation candidates must be exactly D1 and D2")
    anchor_row = _row(anchor[0], anchor[1], person, face, manifest)
    if anchor_row["distillation"] is not None:
        raise ValueError("D0 must not contain OOF distillation")
    rows = [_row(name, path, person, face, manifest) for name, path in candidates]
    expected_modes = {"D1": "person", "D2": "person_face"}
    for row in rows:
        distillation = row["distillation"] or {}
        if (
            distillation.get("mode") != expected_modes[row["name"]]
            or float(distillation.get("mix", -1)) != 0.20
        ):
            raise ValueError(f"{row['name']} distillation protocol differs")
        row["gain_vs_D0"] = {
            "full_final_mAP": float(
                row["full_metrics"]["final_mAP"]
                - anchor_row["full_metrics"]["final_mAP"]
            ),
            "full_average_mAP": float(
                row["full_metrics"]["average_mAP"]
                - anchor_row["full_metrics"]["average_mAP"]
            ),
            "locked_R1_final_mAP": float(
                row["locked_R1_metrics"]["final_mAP"]
                - anchor_row["locked_R1_metrics"]["final_mAP"]
            ),
        }
        row["eligible"] = (
            row["gain_vs_D0"]["full_final_mAP"] >= MINIMUM_FINAL_GAIN
            and row["gain_vs_D0"]["full_average_mAP"] >= 0
            and row["gain_vs_D0"]["locked_R1_final_mAP"] >= MINIMUM_FINAL_GAIN
        )
    winner = max(
        rows,
        key=lambda row: (
            row["eligible"],
            row["locked_R1_metrics"]["final_mAP"],
            row["full_metrics"]["final_mAP"],
            row["full_metrics"]["average_mAP"],
        ),
    )
    return {
        "schema_version": 1,
        "comparison": "oof_cross_view_teacher_distillation_of_full_lane",
        "selection_split": "val",
        "test_accessed": False,
        "locked": {
            "distillation_mix": 0.20,
            "R1_reliable_weights": [0.64, 0.16, 0.20],
            "R1_invalid_face_weights": [0.80, 0.20, 0.0],
            "minimum_full_and_R1_final_gain": MINIMUM_FINAL_GAIN,
            "minimum_full_average_gain": 0.0,
            "weight_search": False,
        },
        "anchor": anchor_row,
        "candidates": rows,
        "winner": winner,
        "decision": {
            "advance_to_seed1_seed2_validation": bool(winner["eligible"]),
            "run_test": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=_named_path, required=True)
    parser.add_argument("--candidate", type=_named_path, action="append", required=True)
    parser.add_argument("--person-run", type=Path, required=True)
    parser.add_argument("--face-run", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        args.anchor,
        args.candidate,
        args.person_run,
        args.face_run,
        args.face_manifest_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "winner": result["winner"]["name"],
        "eligible": result["winner"]["eligible"],
        **result["winner"]["gain_vs_D0"],
        **result["decision"],
    }, indent=2), flush=True)
    print("OOF_DISTILLATION_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
