"""Evaluate one validation-locked class-aware OOF stacker on three test seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from .class_aware_oof_stacking import _evaluate
from .evaluate_taskwise_three_view_router import (
    _candidate_states,
    _fixed_rows_for_tasks,
    _load_test_descriptors,
    _load_test_tasks,
)
from .runner import load_face_manifest_provenance, resolve_dataset_parent, summarize_tasks
from .search_constrained_gated_fusion import load_geometry
from .three_view_router import load_face_metadata


METRICS = ("final_mAP", "average_mAP", "final_cF1", "final_oF1", "forgetting")


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


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    numbers = [float(value) for value in values]
    return {
        "values": numbers,
        "mean": statistics.mean(numbers),
        "sample_std": statistics.stdev(numbers),
    }


def _select_locked_c2(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = selection.get("best_C2", {})
    if (
        selection.get("selection_split") != "val"
        or selection.get("test_accessed") is not False
        or candidate.get("family") != "C2"
        or int(candidate.get("rank", -1)) != 2
        or len(candidate.get("tasks", ())) != 8
    ):
        raise ValueError("C2 was not locked by the validation-only protocol")
    return candidate


def evaluate(
    selection_path: Path,
    expected_selection_sha256: str,
    data_root: Path,
    test_face_manifest: Path,
    test_descriptor_root: Path,
    fixed_test_summary: Path,
    output: Path,
    device: torch.device,
) -> Dict[str, Any]:
    actual_hash = _sha256(selection_path)
    if actual_hash != expected_selection_sha256:
        raise ValueError("Validation C2 selection hash differs")
    selection = _read_json(selection_path)
    candidate = _select_locked_c2(selection)
    states = _candidate_states(selection_path.parent, candidate)
    fixed = _read_json(fixed_test_summary)
    if fixed.get("search_performed_on_test") is not False:
        raise ValueError("Fixed R1 test provenance differs")

    dataset_parent = resolve_dataset_parent(data_root)
    geometry = load_geometry(dataset_parent, "test")
    face = load_face_metadata(test_face_manifest, "test")
    descriptors = _load_test_descriptors(test_descriptor_root, test_face_manifest)
    provenance = load_face_manifest_provenance(test_face_manifest)
    if "test_manifest" not in provenance.get("artifact_sha256", {}):
        raise ValueError("Face manifest does not cover test")

    seeds = []
    for seed_record in fixed.get("seeds", ()):
        paths = seed_record["sources"]
        tasks = _load_test_tasks(
            Path(paths["full"]), Path(paths["person"]), Path(paths["face"]),
            test_face_manifest,
        )
        r1_metrics = summarize_tasks(_fixed_rows_for_tasks(tasks, face))
        recorded = seed_record["metrics"]["R1_fixed_three_view"]["final_mAP"]
        if not math.isclose(r1_metrics["final_mAP"], recorded, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("Fixed R1 test anchor does not reproduce")
        metrics, task_metrics, diagnostics = _evaluate(
            tasks, states, geometry, face, descriptors, device
        )
        seeds.append({
            "seed": int(seed_record["seed"]),
            "R1_fixed": r1_metrics,
            "C2_class_aware": metrics,
            "paired_C2_minus_R1": {
                metric: float(metrics[metric] - r1_metrics[metric]) for metric in METRICS
            },
            "task_metrics": task_metrics,
            "diagnostics": diagnostics,
            "sources": paths,
        })
    seeds.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in seeds] != [0, 1, 2]:
        raise ValueError("C2 test requires exactly seed0/1/2 expert triplets")

    aggregate = {
        method: {
            metric: _stats([row[method][metric] for row in seeds])
            for metric in METRICS
        }
        for method in ("R1_fixed", "C2_class_aware")
    }
    aggregate["paired_C2_minus_R1"] = {
        metric: _stats([row["paired_C2_minus_R1"][metric] for row in seeds])
        for metric in METRICS
    }
    result = {
        "schema_version": 1,
        "evaluation_split": "test",
        "selection_split": "val",
        "test_weight_search": False,
        "evaluated_C2_rule_count": 1,
        "test_evaluation_scope": "exploratory_horizontal_comparison_per_explicit_user_request",
        "state_training": "seed0 three-fold image-group OOF; reused unchanged for all test seeds",
        "locked_candidate": {
            "candidate_id": candidate["candidate_id"],
            "rank": candidate["rank"],
            "interaction_prior_strength": candidate["interaction_prior_strength"],
            "validation_final_mAP": candidate["metrics"]["final_mAP"],
        },
        "seeds": seeds,
        "aggregate": aggregate,
        "sources": {
            "selection": str(selection_path.resolve()),
            "selection_sha256": actual_hash,
            "fixed_test_summary": str(fixed_test_summary.resolve()),
            "test_descriptors": str(test_descriptor_root.resolve()),
            "test_face_manifest": str(test_face_manifest.resolve()),
        },
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--test-face-manifest", type=Path, required=True)
    parser.add_argument("--test-descriptor-root", type=Path, required=True)
    parser.add_argument("--fixed-test-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = evaluate(
        args.selection, args.expected_selection_sha256, args.data_root,
        args.test_face_manifest, args.test_descriptor_root, args.fixed_test_summary,
        args.output, torch.device(args.device),
    )
    print(json.dumps({
        "C2_final_mAP": result["aggregate"]["C2_class_aware"]["final_mAP"],
        "C2_minus_R1": result["aggregate"]["paired_C2_minus_R1"]["final_mAP"],
    }, indent=2), flush=True)
    print("CLASS_AWARE_OOF_LOCKED_TEST_COMPLETE search_on_test=false", flush=True)


if __name__ == "__main__":
    main()
