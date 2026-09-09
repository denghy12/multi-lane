"""Apply the validation-locked Full/Person/Face rule once on three test seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .fuse_face_endpoint_validation import load_face_reliability
from .fuse_fixed_test_scores import _metrics_close, _validate_test_runs
from .fuse_validation_scores import COMMON_CONFIG_FIELDS, validated_run_scores
from .runner import (
    TASK_SIZES,
    TaskMetrics,
    compute_metrics,
    load_face_manifest_provenance,
    summarize_tasks,
)
from .three_view_router import _align_three


LOCKED_VALIDATION_SHA256 = (
    "683e611f883a515fd39b4915f80375a72aa98c91f9c0a9b44e38e5fca39501ea"
)
LOCKED_BETA = 0.20
LOCKED_RELIABLE_WEIGHTS = np.asarray((0.64, 0.16, 0.20), dtype=np.float32)
LOCKED_FALLBACK_WEIGHTS = np.asarray((0.80, 0.20, 0.00), dtype=np.float32)
METRIC_NAMES = ("final_mAP", "average_mAP", "final_cF1", "final_oF1", "forgetting")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_lock(path: Path) -> Dict[str, Any]:
    if _sha256(path) != LOCKED_VALIDATION_SHA256:
        raise ValueError("Three-view validation lock hash differs")
    record = _read_json(path)
    r1 = record.get("R1_fixed_face_beta0.20", {})
    if (
        record.get("selection_split") != "val"
        or record.get("test_accessed") is not False
        or not math.isclose(float(r1.get("beta", -1)), LOCKED_BETA, abs_tol=1e-12)
        or record.get("decision", {}).get("R3_beats_R1") is not False
    ):
        raise ValueError("Validation record does not lock fixed R1")
    return record


def _validate_face_test_run(
    full_run: Path, face_run: Path, manifest_root: Path
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    full_config = _read_json(full_run / "config.json")
    face_config = _read_json(face_run / "config.json")
    face_summary = _read_json(face_run / "seed_summary.json")
    history = _read_json(face_run / "training_history.json")
    if face_summary.get("status") != "complete" or face_summary.get("config") != face_config:
        raise ValueError("Face formal test source is incomplete")
    if (
        face_config.get("input_mode") != "face_crop"
        or face_config.get("reporting_split") != "test"
        or face_config.get("evaluation_score_purpose") != "fixed_test_fusion"
        or face_config.get("save_evaluation_scores") is not True
        or face_config.get("calibration_fraction") != 0.0
        or face_config.get("save_checkpoints") is not False
    ):
        raise ValueError("Face formal test protocol differs")
    if (
        len(face_summary.get("task_metrics", ())) != len(TASK_SIZES)
        or face_summary.get("completed_epochs") != 240
        or len(history) != len(TASK_SIZES)
        or any(len(history.get(str(task_id), ())) != 30 for task_id in range(len(TASK_SIZES)))
        or sum(
            row["skipped_optimizer_steps"]
            for rows in history.values()
            for row in rows
        )
        != 0
    ):
        raise ValueError("Face formal test training budget is incomplete")
    git = face_config.get("git", {})
    if git.get("dirty") is not False or not git.get("commit"):
        raise ValueError("Face source did not record a clean Git commit")
    provenance = load_face_manifest_provenance(manifest_root)
    if "test_manifest" not in provenance.get("artifact_sha256", {}):
        raise ValueError("Face manifest provenance does not cover test")
    if face_config.get("face_manifest") != provenance:
        raise ValueError("Face test manifest provenance differs")
    for field in COMMON_CONFIG_FIELDS:
        if full_config.get(field) != face_config.get(field):
            raise ValueError(f"Full and Face formal test differ on {field}")
    return face_config, face_summary


def _rows_from_scores(
    task_arrays: Sequence[Tuple[np.ndarray, ...]],
    reliability: Dict[str, bool],
) -> Tuple[List[TaskMetrics], List[TaskMetrics], List[TaskMetrics], List[TaskMetrics], List[Dict[str, int]]]:
    full_rows: List[TaskMetrics] = []
    person_rows: List[TaskMetrics] = []
    face_rows: List[TaskMetrics] = []
    fusion_rows: List[TaskMetrics] = []
    counts = []
    for task_id, (ids, targets, full, person, face) in enumerate(task_arrays):
        missing = [sample_id for sample_id in ids if sample_id not in reliability]
        if missing:
            raise ValueError(f"Test Face reliability missing ID {missing[0]}")
        reliable = np.asarray([reliability[sample_id] for sample_id in ids], dtype=np.bool_)
        weights = np.tile(LOCKED_FALLBACK_WEIGHTS, (len(ids), 1))
        weights[reliable] = LOCKED_RELIABLE_WEIGHTS
        fused = (
            weights[:, 0, None] * full
            + weights[:, 1, None] * person
            + weights[:, 2, None] * face
        )
        rows = []
        for scores in (full, person, face, fused):
            rows.append(
                compute_metrics(
                    task_id, torch.from_numpy(scores), torch.from_numpy(targets), 0.5
                )
            )
        full_rows.append(rows[0])
        person_rows.append(rows[1])
        face_rows.append(rows[2])
        fusion_rows.append(rows[3])
        counts.append({
            "task_id": task_id,
            "samples": len(ids),
            "reliable_face_samples": int(reliable.sum()),
        })
    return full_rows, person_rows, face_rows, fusion_rows, counts


def evaluate_seed(
    full_run: Path,
    person_run: Path,
    face_run: Path,
    manifest_root: Path,
) -> Dict[str, Any]:
    full_summary, person_summary = _validate_test_runs(full_run, person_run)
    face_config, face_summary = _validate_face_test_run(full_run, face_run, manifest_root)
    seed = int(face_config["seed"])
    if int(full_summary["config"]["seed"]) != seed or int(person_summary["config"]["seed"]) != seed:
        raise ValueError("Full/Person/Face seeds differ")
    full_dumps, _ = validated_run_scores(full_run, "test")
    person_dumps, _ = validated_run_scores(person_run, "test")
    face_dumps, _ = validated_run_scores(face_run, "test")
    task_arrays = [
        _align_three(full, person, face)
        for full, person, face in zip(full_dumps, person_dumps, face_dumps)
    ]
    rows = _rows_from_scores(task_arrays, load_face_reliability(manifest_root, "test"))
    full_rows, person_rows, face_rows, fusion_rows, counts = rows
    for label, calculated, recorded in (
        ("full", full_rows, full_summary),
        ("person", person_rows, person_summary),
        ("face", face_rows, face_summary),
    ):
        for row, expected in zip(calculated, recorded["task_metrics"]):
            if not _metrics_close(asdict(row), expected):
                raise ValueError(f"{label} scores do not reproduce recorded test metrics")
    metrics = {
        "full": summarize_tasks(full_rows),
        "person": summarize_tasks(person_rows),
        "face": summarize_tasks(face_rows),
        "R0_full_person": summarize_tasks([
            compute_metrics(
                task_id,
                torch.from_numpy(0.8 * arrays[2] + 0.2 * arrays[3]),
                torch.from_numpy(arrays[1]),
                0.5,
            )
            for task_id, arrays in enumerate(task_arrays)
        ]),
        "R1_fixed_three_view": summarize_tasks(fusion_rows),
    }
    return {
        "seed": seed,
        "sources": {"full": str(full_run.resolve()), "person": str(person_run.resolve()), "face": str(face_run.resolve())},
        "metrics": metrics,
        "task_metrics": {"R1_fixed_three_view": [asdict(row) for row in fusion_rows]},
        "reliability_counts": counts,
    }


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    values = [float(value) for value in values]
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
    }


def fixed_three_view_test(
    full_runs: Sequence[Path],
    person_runs: Sequence[Path],
    face_runs: Sequence[Path],
    manifest_root: Path,
    validation_selection: Path,
) -> Dict[str, Any]:
    lock = _validate_lock(validation_selection)
    if not (len(full_runs) == len(person_runs) == len(face_runs) == 3):
        raise ValueError("Exactly three Full/Person/Face sources are required")
    seeds = [
        evaluate_seed(full, person, face, manifest_root)
        for full, person, face in zip(full_runs, person_runs, face_runs)
    ]
    seeds.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in seeds] != [0, 1, 2]:
        raise ValueError("Exactly seed0/1/2 are required")
    groups = {}
    for group in ("full", "person", "face", "R0_full_person", "R1_fixed_three_view"):
        groups[group] = {
            metric: _stats([row["metrics"][group][metric] for row in seeds])
            for metric in METRIC_NAMES
        }
    differences = {
        metric: _stats([
            row["metrics"]["R1_fixed_three_view"][metric]
            - row["metrics"]["R0_full_person"][metric]
            for row in seeds
        ])
        for metric in METRIC_NAMES
    }
    return {
        "schema_version": 1,
        "evaluation_split": "test",
        "search_performed_on_test": False,
        "evaluated_fusion_rule_count": 1,
        "locked_rule": {
            "reliable_weights": LOCKED_RELIABLE_WEIGHTS.tolist(),
            "invalid_face_fallback": LOCKED_FALLBACK_WEIGHTS.tolist(),
            "threshold": 0.5,
        },
        "validation_lock": {
            "path": str(validation_selection.resolve()),
            "sha256": LOCKED_VALIDATION_SHA256,
            "R1_final_mAP": lock["R1_fixed_face_beta0.20"]["metrics"]["final_mAP"],
        },
        "seeds": seeds,
        "groups": groups,
        "paired_R1_minus_R0": differences,
        "positive_final_mAP_seeds": sum(value > 0 for value in differences["final_mAP"]["values"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--person-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--face-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--validation-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fixed_three_view_test(
        args.full_runs,
        args.person_runs,
        args.face_runs,
        args.face_manifest_root,
        args.validation_selection,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "R1_final_mAP": result["groups"]["R1_fixed_three_view"]["final_mAP"],
        "paired_R1_minus_R0": result["paired_R1_minus_R0"]["final_mAP"],
        "positive_seeds": result["positive_final_mAP_seeds"],
    }, indent=2), flush=True)
    print("FIXED_THREE_VIEW_SEED012_TEST_COMPLETE search_on_test=false", flush=True)


if __name__ == "__main__":
    main()
