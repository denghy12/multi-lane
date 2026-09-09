"""Re-evaluate saved three-view routers with task-lane-preserving semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .export_three_view_descriptors import FEATURE_NAMES as VISUAL_FEATURE_NAMES
from .fuse_fixed_three_view_test import _validate_face_test_run
from .fuse_fixed_test_scores import _validate_test_runs
from .fuse_validation_scores import validated_run_scores
from .runner import (
    TASK_SIZES,
    TaskMetrics,
    compute_metrics,
    load_face_manifest_provenance,
    resolve_dataset_parent,
    summarize_tasks,
)
from .search_constrained_gated_fusion import Geometry, load_geometry
from .three_view_router import (
    EndpointTriple,
    FaceMetadata,
    R2_FEATURE_NAMES,
    ThreeViewRouter,
    _align_three,
    _fixed_rows,
    _probability_features,
    _select_visual_features,
    load_endpoint_triple,
    load_face_metadata,
    load_visual_descriptors,
)


LOCKED_SELECTION_SHA256 = "683e611f883a515fd39b4915f80375a72aa98c91f9c0a9b44e38e5fca39501ea"
METHODS = ("R1_fixed", "R2_taskwise", "R3_taskwise")
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


def _load_state(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _candidate_states(
    selection_dir: Path, candidate: Mapping[str, Any]
) -> Sequence[Dict[str, Any]]:
    states = []
    tasks = candidate.get("tasks", ())
    if len(tasks) != len(TASK_SIZES):
        raise ValueError("Router candidate does not contain eight task states")
    for task_id, record in enumerate(tasks):
        if int(record.get("task_id", -1)) != task_id:
            raise ValueError("Router task state order differs")
        state_path = selection_dir / record["state_path"]
        if _sha256(state_path) != record["state_sha256"]:
            raise ValueError("Router task state hash differs")
        state = _load_state(state_path)
        if (
            int(state.get("task_id", -1)) != task_id
            or state.get("candidate_id") != candidate["candidate_id"]
            or tuple(state.get("feature_names", ())) != tuple(candidate["feature_names"])
        ):
            raise ValueError("Router task state provenance differs")
        states.append(state)
    return states


def _load_test_descriptors(
    root: Path, face_manifest_root: Path
) -> Dict[str, Tuple[np.ndarray, np.bool_]]:
    manifest = _read_json(root / "descriptor_manifest.json")
    artifact = root / "test_descriptors.npz"
    if (
        manifest.get("evaluation_split") != "test"
        or manifest.get("used_for_selection") is not False
        or manifest.get("test_weight_search") is not False
        or tuple(manifest.get("feature_names", ())) != VISUAL_FEATURE_NAMES
        or manifest.get("face_manifest") != load_face_manifest_provenance(face_manifest_root)
        or manifest.get("artifact_sha256") != _sha256(artifact)
    ):
        raise ValueError("Test descriptor provenance differs")
    with np.load(artifact, allow_pickle=False) as data:
        ids = data["sample_ids"].astype(str)
        features = data["features"].astype(np.float32)
        reliable = data["face_reliable"].astype(np.bool_)
        if (
            int(data["schema_version"]) != 1
            or str(data["split"]) != "test"
            or tuple(data["feature_names"].astype(str)) != VISUAL_FEATURE_NAMES
            or len(set(ids.tolist())) != len(ids)
            or features.shape != (len(ids), len(VISUAL_FEATURE_NAMES))
            or reliable.shape != (len(ids),)
            or not np.isfinite(features).all()
        ):
            raise ValueError("Invalid test descriptor artifact")
    return {
        sample_id: (features[index], reliable[index])
        for index, sample_id in enumerate(ids)
    }


def assemble_task_lanes(
    views: Sequence[np.ndarray], lane_weights: Sequence[np.ndarray]
) -> np.ndarray:
    if len(views) != 3 or not (views[0].shape == views[1].shape == views[2].shape):
        raise ValueError("Three aligned probability views are required")
    total_classes = views[0].shape[1]
    expected_tasks = next(
        (task + 1 for task in range(len(TASK_SIZES)) if sum(TASK_SIZES[:task + 1]) == total_classes),
        None,
    )
    if expected_tasks is None or len(lane_weights) != expected_tasks:
        raise ValueError("Lane weights do not cover the seen task partition")
    fused = np.empty_like(views[0], dtype=np.float32)
    start = 0
    for task_id, weights in enumerate(lane_weights):
        end = start + TASK_SIZES[task_id]
        if weights.shape != (len(fused), 3):
            raise ValueError("Invalid task-lane router weight shape")
        if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("Task-lane weights do not sum to one")
        fused[:, start:end] = sum(
            weights[:, view, None] * views[view][:, start:end]
            for view in range(3)
        )
        start = end
    return fused


def _router_weights(
    task_id: int,
    state: Mapping[str, Any],
    sample_ids: Sequence[str],
    views: Sequence[np.ndarray],
    geometry: Mapping[str, Geometry],
    face_metadata: Mapping[str, FaceMetadata],
    descriptors: Optional[Mapping[str, Tuple[np.ndarray, np.bool_]]],
) -> np.ndarray:
    prefix = sum(TASK_SIZES[:task_id + 1])
    features, reliable = _probability_features(
        sample_ids,
        views[0][:, :prefix],
        views[1][:, :prefix],
        views[2][:, :prefix],
        geometry,
        face_metadata,
    )
    feature_names = tuple(state["feature_names"])
    if descriptors is not None:
        features = np.column_stack((
            features,
            _select_visual_features(sample_ids, descriptors, reliable),
        )).astype(np.float32)
    if feature_names not in (tuple(R2_FEATURE_NAMES), tuple(R2_FEATURE_NAMES) + VISUAL_FEATURE_NAMES):
        raise ValueError("Unexpected saved Router feature schema")
    if features.shape[1] != len(feature_names):
        raise ValueError("Router feature family and state differ")
    normalization = state["normalization"]
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    scale = np.asarray(normalization["scale"], dtype=np.float32)
    if mean.shape != (features.shape[1],) or scale.shape != mean.shape or np.any(scale <= 0):
        raise ValueError("Invalid saved Router normalization")
    normalized = ((features - mean) / scale).astype(np.float32)
    router = ThreeViewRouter(len(feature_names), initialization_seed=0).float()
    router.load_state_dict(state["model"], strict=True)
    router.eval()
    with torch.no_grad():
        weights = router(
            torch.from_numpy(normalized), torch.from_numpy(reliable)
        ).numpy()
    if not np.array_equal(weights[~reliable, 2], np.zeros(int((~reliable).sum()), dtype=np.float32)):
        raise RuntimeError("Invalid Face received nonzero task-lane weight")
    return weights


def taskwise_rows(
    tasks: Sequence[Tuple[np.ndarray, ...]],
    states: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Geometry],
    face_metadata: Mapping[str, FaceMetadata],
    descriptors: Optional[Mapping[str, Tuple[np.ndarray, np.bool_]]],
) -> Tuple[Sequence[TaskMetrics], Sequence[Dict[str, Any]]]:
    rows = []
    diagnostics = []
    for evaluation_task, (ids, targets, full, person, face) in enumerate(tasks):
        views = (full, person, face)
        weights = [
            _router_weights(
                lane, states[lane], ids, views, geometry, face_metadata, descriptors
            )
            for lane in range(evaluation_task + 1)
        ]
        fused = assemble_task_lanes(views, weights)
        rows.append(compute_metrics(
            evaluation_task,
            torch.from_numpy(fused),
            torch.from_numpy(targets),
            0.5,
        ))
        diagnostics.append({
            "task_id": evaluation_task,
            "samples": len(ids),
            "lane_weight_mean": [value.mean(axis=0).tolist() for value in weights],
            "lane_weight_std": [value.std(axis=0).tolist() for value in weights],
        })
    return rows, diagnostics


def _fixed_rows_for_tasks(
    tasks: Sequence[Tuple[np.ndarray, ...]], face_metadata: Mapping[str, FaceMetadata]
) -> Sequence[TaskMetrics]:
    rows = []
    for task_id, (ids, targets, full, person, face) in enumerate(tasks):
        reliable = np.asarray([face_metadata[sample_id].reliable for sample_id in ids])
        fused = 0.8 * full + 0.2 * person
        fused[reliable] = 0.8 * fused[reliable] + 0.2 * face[reliable]
        rows.append(compute_metrics(
            task_id, torch.from_numpy(fused), torch.from_numpy(targets), 0.5
        ))
    return rows


def _load_test_tasks(
    full_run: Path, person_run: Path, face_run: Path, manifest_root: Path
) -> Sequence[Tuple[np.ndarray, ...]]:
    _validate_test_runs(full_run, person_run)
    _validate_face_test_run(full_run, face_run, manifest_root)
    full, _ = validated_run_scores(full_run, "test")
    person, _ = validated_run_scores(person_run, "test")
    face, _ = validated_run_scores(face_run, "test")
    return [_align_three(*triple) for triple in zip(full, person, face)]


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    numbers = [float(value) for value in values]
    return {"values": numbers, "mean": statistics.mean(numbers), "sample_std": statistics.stdev(numbers)}


def evaluate(
    selection_path: Path,
    data_root: Path,
    validation_face_manifest: Path,
    test_face_manifest: Path,
    test_descriptor_root: Path,
    fixed_test_summary: Path,
) -> Dict[str, Any]:
    if _sha256(selection_path) != LOCKED_SELECTION_SHA256:
        raise ValueError("Validation Router selection lock differs")
    selection = _read_json(selection_path)
    if selection.get("selection_split") != "val" or selection.get("test_accessed") is not False:
        raise ValueError("Router states were not selected on validation only")
    sources = selection["sources"]
    endpoint: EndpointTriple = load_endpoint_triple(
        Path(sources["full"]),
        Path(sources["person"]),
        Path(sources["face"]),
        validation_face_manifest,
    )
    dataset_parent = resolve_dataset_parent(data_root)
    val_geometry = load_geometry(dataset_parent, "val")
    val_face = load_face_metadata(validation_face_manifest, "val")
    validation_descriptors = load_visual_descriptors(
        Path(sources["descriptors"]), validation_face_manifest
    )["val"]

    validation_candidates: Dict[str, Sequence[Dict[str, Any]]] = {"R2": [], "R3": []}
    state_cache: Dict[str, Sequence[Dict[str, Any]]] = {}
    for candidate in selection["candidates"]:
        family = candidate["family"]
        states = _candidate_states(selection_path.parent, candidate)
        state_cache[candidate["candidate_id"]] = states
        rows, diagnostics = taskwise_rows(
            endpoint.validation_tasks,
            states,
            val_geometry,
            val_face,
            validation_descriptors if family == "R3" else None,
        )
        validation_candidates[family].append({
            "candidate_id": candidate["candidate_id"],
            "family": family,
            "prior_strength": candidate["prior_strength"],
            "metrics": summarize_tasks(rows),
            "task_metrics": [asdict(row) for row in rows],
            "diagnostics": diagnostics,
        })
    best = {
        family: max(
            validation_candidates[family],
            key=lambda row: (
                row["metrics"]["final_mAP"],
                row["metrics"]["average_mAP"],
                row["prior_strength"],
            ),
        )
        for family in ("R2", "R3")
    }
    r1_val = _fixed_rows(endpoint, val_face, beta=0.20)

    test_descriptor = _load_test_descriptors(test_descriptor_root, test_face_manifest)
    test_geometry = load_geometry(dataset_parent, "test")
    test_face = load_face_metadata(test_face_manifest, "test")
    prior_test = _read_json(fixed_test_summary)
    if prior_test.get("search_performed_on_test") is not False:
        raise ValueError("Fixed test source record differs")
    test_seed_rows = []
    for seed_record in prior_test["seeds"]:
        paths = seed_record["sources"]
        tasks = _load_test_tasks(
            Path(paths["full"]), Path(paths["person"]), Path(paths["face"]), test_face_manifest
        )
        method_rows: Dict[str, Sequence[TaskMetrics]] = {
            "R1_fixed": _fixed_rows_for_tasks(tasks, test_face),
        }
        method_diagnostics = {}
        for family in ("R2", "R3"):
            candidate = best[family]
            rows, diagnostics = taskwise_rows(
                tasks,
                state_cache[candidate["candidate_id"]],
                test_geometry,
                test_face,
                test_descriptor if family == "R3" else None,
            )
            method_rows[f"{family}_taskwise"] = rows
            method_diagnostics[f"{family}_taskwise"] = diagnostics
        recorded_r1 = seed_record["metrics"]["R1_fixed_three_view"]["final_mAP"]
        calculated_r1 = summarize_tasks(method_rows["R1_fixed"])["final_mAP"]
        if not math.isclose(recorded_r1, calculated_r1, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("Fixed R1 test anchor does not reproduce")
        test_seed_rows.append({
            "seed": int(seed_record["seed"]),
            "methods": {
                name: {
                    "metrics": summarize_tasks(rows),
                    "task_metrics": [asdict(row) for row in rows],
                }
                for name, rows in method_rows.items()
            },
            "diagnostics": method_diagnostics,
        })
    test_seed_rows.sort(key=lambda row: row["seed"])
    if [row["seed"] for row in test_seed_rows] != [0, 1, 2]:
        raise ValueError("Diagnostic test requires seed0/1/2")
    test_aggregate = {
        method: {
            metric: _stats([row["methods"][method]["metrics"][metric] for row in test_seed_rows])
            for metric in METRICS
        }
        for method in METHODS
    }
    decision = {
        "R3_beats_R1_on_validation": best["R3"]["metrics"]["final_mAP"] > r1_val["metrics"]["final_mAP"],
        "R3_beats_R2_on_validation": best["R3"]["metrics"]["final_mAP"] > best["R2"]["metrics"]["final_mAP"],
    }
    decision["advance_taskwise_router"] = all(decision.values())
    return {
        "schema_version": 1,
        "method": "task_lane_preserving_three_view_router_reassembly",
        "router_training_reused": True,
        "expert_training_reused": True,
        "validation_used_for_candidate_selection": True,
        "test_used_for_candidate_selection": False,
        "test_evaluation_scope": "exploratory_all_three_methods_per_explicit_user_request",
        "lane_semantics": "router_k_weights_only_task_k_classes_and_is_frozen_after_task_k",
        "validation": {
            "R1_fixed": r1_val,
            "candidates": validation_candidates,
            "best_R2": best["R2"],
            "best_R3": best["R3"],
            "decision": decision,
        },
        "test": {"seeds": test_seed_rows, "aggregate": test_aggregate},
        "sources": {
            "selection": str(selection_path.resolve()),
            "selection_sha256": LOCKED_SELECTION_SHA256,
            "validation_face_manifest": str(validation_face_manifest.resolve()),
            "test_face_manifest": str(test_face_manifest.resolve()),
            "test_descriptors": str(test_descriptor_root.resolve()),
            "fixed_test_summary": str(fixed_test_summary.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--validation-face-manifest", type=Path, required=True)
    parser.add_argument("--test-face-manifest", type=Path, required=True)
    parser.add_argument("--test-descriptor-root", type=Path, required=True)
    parser.add_argument("--fixed-test-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.selection,
        args.data_root,
        args.validation_face_manifest,
        args.test_face_manifest,
        args.test_descriptor_root,
        args.fixed_test_summary,
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "validation": {
            "R1": result["validation"]["R1_fixed"]["metrics"]["final_mAP"],
            "R2": result["validation"]["best_R2"]["metrics"]["final_mAP"],
            "R3": result["validation"]["best_R3"]["metrics"]["final_mAP"],
            "decision": result["validation"]["decision"],
        },
        "test_final_mAP": {
            name: values["final_mAP"]
            for name, values in result["test"]["aggregate"].items()
        },
    }, indent=2), flush=True)
    print("TASKWISE_THREE_VIEW_ROUTER_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
