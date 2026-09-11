"""Validation-only Full/Person/Face fusion with a locked Face quality mask."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from .evaluation_scores import align_evaluation_scores
from .fuse_validation_scores import (
    COMMON_CONFIG_FIELDS,
    _load_json,
    _validate_runs,
    validated_run_scores,
)
from .runner import (
    TASK_SIZES,
    TaskMetrics,
    compute_metrics,
    load_face_manifest_provenance,
    summarize_tasks,
)


FULL_WEIGHT = 0.80
PERSON_WEIGHT = 0.20
BETAS = (0.0, 0.05, 0.10, 0.20)
MIN_FACE_SHORT_SIDE = 24.0
MIN_FACE_SCORE = 0.6


def load_face_reliability(
    manifest_root: Path, split: str = "val"
) -> Dict[str, bool]:
    if split not in ("val", "test"):
        raise ValueError("Face fusion reliability is restricted to val or test")
    provenance = load_face_manifest_provenance(manifest_root)
    records: Dict[str, bool] = {}
    path = Path(provenance["root"]) / "manifests" / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("sample_id", ""))
            if not sample_id.startswith(f"{split}:") or sample_id in records:
                raise ValueError(
                    f"Invalid {split} Face manifest ID at line {line_number}"
                )
            reliable = (
                bool(record.get("valid_face"))
                and not bool(record.get("ambiguous_match"))
                and float(record.get("face_short_side", 0.0)) >= MIN_FACE_SHORT_SIDE
                and float(record.get("face_detection_score", 0.0)) >= MIN_FACE_SCORE
            )
            records[sample_id] = reliable
    if not records:
        raise ValueError("Empty val Face manifest")
    return records


def masked_face_fusion(
    full_probabilities: np.ndarray,
    person_probabilities: np.ndarray,
    face_probabilities: np.ndarray,
    reliable: np.ndarray,
    beta: float,
) -> np.ndarray:
    if beta not in BETAS:
        raise ValueError(f"Face beta must be one of {BETAS}")
    if not (
        full_probabilities.shape == person_probabilities.shape == face_probabilities.shape
    ):
        raise ValueError("Full, Person, and Face probability shapes differ")
    if reliable.shape != (len(full_probabilities),) or reliable.dtype != np.bool_:
        raise ValueError("Face reliability mask shape/dtype is invalid")
    fp = FULL_WEIGHT * full_probabilities + PERSON_WEIGHT * person_probabilities
    if beta == 0:
        return fp
    fused = fp.copy()
    fused[reliable] = (
        (1.0 - beta) * fp[reliable] + beta * face_probabilities[reliable]
    )
    return fused


def _validate_face_run(
    full_run: Path,
    face_run: Path,
    manifest_root: Path,
    allow_face_training_budget_difference: bool = False,
) -> Mapping[str, Any]:
    full_config = _load_json(full_run / "config.json")
    face_config = _load_json(face_run / "config.json")
    summary = _load_json(face_run / "seed_summary.json")
    if summary.get("status") != "complete" or summary.get("config") != face_config:
        raise ValueError("Face validation run is incomplete or config provenance differs")
    if face_config.get("input_mode") != "face_crop":
        raise ValueError("Face endpoint must use face_crop input")
    if face_config.get("reporting_split") != "val" or not face_config.get("save_evaluation_scores"):
        raise ValueError("Face endpoint must be validation-only with score dumps")
    provenance = load_face_manifest_provenance(manifest_root)
    if face_config.get("face_manifest") != provenance:
        raise ValueError("Face run manifest provenance differs from the supplied audit")
    for field in COMMON_CONFIG_FIELDS:
        if allow_face_training_budget_difference and field in {
            "training_budget_mode", "epochs_per_task", "scheduler"
        }:
            continue
        if full_config.get(field) != face_config.get(field):
            raise ValueError(f"Face and Full runs differ on fixed config field {field}")
    return summary


def fuse_face_endpoint_validation(
    full_run: Path,
    person_run: Path,
    face_run: Path,
    manifest_root: Path,
    betas: Sequence[float] = BETAS,
    allow_face_training_budget_difference: bool = False,
) -> Dict[str, Any]:
    if tuple(float(value) for value in betas) != BETAS:
        raise ValueError(f"Stage-1 beta grid is locked to {BETAS}")
    full_summary, person_summary = _validate_runs(full_run, person_run)
    face_summary = _validate_face_run(
        full_run,
        face_run,
        manifest_root,
        allow_face_training_budget_difference,
    )
    full_dumps, _ = validated_run_scores(full_run, "val")
    person_dumps, _ = validated_run_scores(person_run, "val")
    face_dumps, _ = validated_run_scores(face_run, "val")
    reliability = load_face_reliability(manifest_root)

    candidates: List[Dict[str, Any]] = [
        {"beta": float(beta), "rows": []} for beta in betas
    ]
    face_all_rows: List[TaskMetrics] = []
    face_reliable_rows: List[TaskMetrics] = []
    task_arrays = []
    reliability_counts = []
    for task_id, (full_dump, person_dump, face_dump) in enumerate(
        zip(full_dumps, person_dumps, face_dumps)
    ):
        ids, _, _, targets, full_probs, person_probs = align_evaluation_scores(
            full_dump, person_dump
        )
        face_ids, _, _, face_targets, _, face_probs = align_evaluation_scores(
            full_dump, face_dump
        )
        if not np.array_equal(ids, face_ids) or not np.array_equal(targets, face_targets):
            raise ValueError("Face score alignment differs from Full/Person")
        missing = [sample_id for sample_id in ids if sample_id not in reliability]
        if missing:
            raise ValueError(f"Face reliability metadata is missing IDs: {missing[:3]}")
        mask = np.asarray([reliability[sample_id] for sample_id in ids], dtype=np.bool_)
        task_arrays.append((targets, full_probs, person_probs, face_probs, mask))
        reliability_counts.append({
            "task_id": task_id,
            "samples": len(ids),
            "reliable_face_samples": int(mask.sum()),
            "reliable_face_rate": float(mask.mean()),
        })
        face_all_rows.append(
            compute_metrics(task_id, torch.from_numpy(face_probs), torch.from_numpy(targets), 0.5)
        )
        if not mask.any():
            raise ValueError(f"Task {task_id} has no reliable Face validation samples")
        face_reliable_rows.append(
            compute_metrics(
                task_id,
                torch.from_numpy(face_probs[mask]),
                torch.from_numpy(targets[mask]),
                0.5,
            )
        )
        for candidate in candidates:
            scores = masked_face_fusion(
                full_probs, person_probs, face_probs, mask, candidate["beta"]
            )
            candidate["rows"].append(
                compute_metrics(
                    task_id, torch.from_numpy(scores), torch.from_numpy(targets), 0.5
                )
            )

    candidate_rows = []
    for candidate in candidates:
        rows = candidate.pop("rows")
        candidate_rows.append({
            "beta": candidate["beta"],
            "metrics": summarize_tasks(rows),
            "task_metrics": [asdict(row) for row in rows],
        })
    anchor = candidate_rows[0]
    winner = max(
        candidate_rows,
        key=lambda row: (
            row["metrics"]["final_mAP"],
            row["metrics"]["average_mAP"],
            -row["beta"],
        ),
    )
    subgroup_rows = {
        "anchor_reliable": [],
        "winner_reliable": [],
        "anchor_unreliable": [],
        "winner_unreliable": [],
    }
    for task_id, (targets, full_probs, person_probs, face_probs, mask) in enumerate(
        task_arrays
    ):
        anchor_scores = masked_face_fusion(
            full_probs, person_probs, face_probs, mask, 0.0
        )
        winner_scores = masked_face_fusion(
            full_probs, person_probs, face_probs, mask, float(winner["beta"])
        )
        for label, scores in (
            ("anchor_reliable", anchor_scores),
            ("winner_reliable", winner_scores),
        ):
            subgroup_rows[label].append(
                compute_metrics(
                    task_id,
                    torch.from_numpy(scores[mask]),
                    torch.from_numpy(targets[mask]),
                    0.5,
                )
            )
        if (~mask).any():
            for label, scores in (
                ("anchor_unreliable", anchor_scores),
                ("winner_unreliable", winner_scores),
            ):
                subgroup_rows[label].append(
                    compute_metrics(
                        task_id,
                        torch.from_numpy(scores[~mask]),
                        torch.from_numpy(targets[~mask]),
                        0.5,
                    )
                )
    gain = float(winner["metrics"]["final_mAP"] - anchor["metrics"]["final_mAP"])
    return {
        "schema_version": 1,
        "comparison": "full_person_anchor_plus_quality_masked_face_validation",
        "selection_split": "val",
        "test_accessed": False,
        "runs": {
            "full": str(full_run.resolve()),
            "person": str(person_run.resolve()),
            "face": str(face_run.resolve()),
        },
        "anchor_rule": {"mode": "probability", "full_weight": 0.8, "person_weight": 0.2},
        "face_rule": {
            "mode": "probability",
            "beta_grid": list(BETAS),
            "invalid_face_fallback": "exact_full0.8_person0.2",
            "minimum_face_short_side": MIN_FACE_SHORT_SIDE,
            "minimum_detection_score": MIN_FACE_SCORE,
            "requires_valid_face": True,
            "rejects_ambiguous_match": True,
        },
        "source_metrics": {
            "full": full_summary["metrics"],
            "person": person_summary["metrics"],
            "face_all_samples_placeholder_included": face_summary["metrics"],
            "face_all_recomputed": summarize_tasks(face_all_rows),
            "face_reliable_subset_diagnostic_only": summarize_tasks(face_reliable_rows),
        },
        "reliability_counts": reliability_counts,
        "subgroup_diagnostics": {
            label: summarize_tasks(rows) for label, rows in subgroup_rows.items()
        },
        "anchor": anchor,
        "winner": winner,
        "candidates": candidate_rows,
        "decision": {
            "beats_fp_anchor": gain > 1e-9,
            "final_mAP_gain": gain,
            "advance_to_dynamic_three_view_router": gain > 1e-9,
            "run_test": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--person-run", type=Path, required=True)
    parser.add_argument("--face-run", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fuse_face_endpoint_validation(
        args.full_run, args.person_run, args.face_run, args.face_manifest_root
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2), flush=True)
    print(
        "FACE_ENDPOINT_VALIDATION_COMPLETE "
        f"beta={result['winner']['beta']:.2f} "
        f"final_mAP={result['winner']['metrics']['final_mAP']:.6f} "
        f"gain={result['decision']['final_mAP_gain']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
