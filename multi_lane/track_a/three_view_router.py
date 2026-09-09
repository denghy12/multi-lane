"""Validation-only sample-wise Full/Person/Face router selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .evaluation_scores import align_evaluation_scores
from .export_three_view_descriptors import FEATURE_NAMES as VISUAL_FEATURE_NAMES
from .fuse_validation_scores import COMMON_CONFIG_FIELDS
from .learned_reliability_gate import _audit_source_run
from .runner import TASK_SIZES, TaskMetrics, compute_metrics, summarize_tasks, task_indices
from .search_constrained_gated_fusion import Geometry, load_geometry
from .runner import load_face_manifest_provenance, resolve_dataset_parent


CALIBRATION_FRACTION = 0.10
PRIOR_STRENGTHS = (0.0, 0.1, 1.0)
HIDDEN_DIM = 16
EPOCHS_PER_TASK = 80
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
VALID_PRIOR = np.asarray((0.64, 0.16, 0.20), dtype=np.float32)
INVALID_PRIOR = np.asarray((0.80, 0.20, 0.00), dtype=np.float32)
R2_FEATURE_NAMES = (
    "bbox_log_area",
    "bbox_log_aspect",
    "people_scaled",
    "is_multi_person",
    "face_training_valid",
    "face_reliable",
    "face_short_side_scaled",
    "face_detection_score",
    "full_confidence",
    "person_confidence",
    "face_confidence",
    "full_entropy",
    "person_entropy",
    "face_entropy",
    "mean_disagreement_full_person",
    "mean_disagreement_full_face",
    "mean_disagreement_person_face",
    "max_disagreement_full_person",
    "max_disagreement_full_face",
    "max_disagreement_person_face",
)


@dataclass(frozen=True)
class FaceMetadata:
    training_valid: bool
    reliable: bool
    short_side: float
    detection_score: float


@dataclass
class EndpointTriple:
    full_run: Path
    person_run: Path
    face_run: Path
    validation_tasks: List[Tuple[np.ndarray, ...]]
    calibration_tasks: List[Tuple[np.ndarray, ...]]


class ThreeViewRouter(nn.Module):
    """One task-local three-way weight vector per sample."""

    def __init__(self, feature_dim: int, initialization_seed: int) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("Router feature dimension must be positive")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.network = nn.Sequential(
                nn.Linear(feature_dim, HIDDEN_DIM),
                nn.GELU(),
                nn.Linear(HIDDEN_DIM, 3),
            )
            nn.init.xavier_uniform_(self.network[0].weight)
            nn.init.zeros_(self.network[0].bias)
            nn.init.zeros_(self.network[2].weight)
            nn.init.constant_(self.network[2].bias[0], math.log(float(VALID_PRIOR[0])))
            nn.init.constant_(self.network[2].bias[1], math.log(float(VALID_PRIOR[1])))
            nn.init.constant_(self.network[2].bias[2], math.log(float(VALID_PRIOR[2])))

    def forward(self, features: torch.Tensor, face_reliable: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or face_reliable.shape != (len(features),):
            raise ValueError("Invalid router inputs")
        logits = self.network(features)
        logits = logits.masked_fill(~face_reliable[:, None] & torch.tensor(
            [False, False, True], device=logits.device
        ), torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        if not torch.isfinite(weights).all():
            raise FloatingPointError("Non-finite router weights")
        return weights


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


def load_face_metadata(manifest_root: Path, split: str) -> Dict[str, FaceMetadata]:
    load_face_manifest_provenance(manifest_root)
    if split not in ("train", "val"):
        raise ValueError("Router metadata is restricted to train and val")
    records: Dict[str, FaceMetadata] = {}
    manifest = manifest_root / "manifests" / f"{split}.jsonl"
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if not sample_id.startswith(f"{split}:") or sample_id in records:
                raise ValueError(f"Invalid Face metadata ID at line {line_number}")
            training_valid = bool(row.get("valid_face")) and not bool(
                row.get("ambiguous_match")
            )
            raw_short_side = row.get("face_short_side")
            raw_detection_score = row.get("face_detection_score")
            short_side = 0.0 if raw_short_side is None else float(raw_short_side)
            detection_score = (
                0.0 if raw_detection_score is None else float(raw_detection_score)
            )
            reliable = (
                training_valid and short_side >= 24.0 and detection_score >= 0.6
            )
            records[sample_id] = FaceMetadata(
                training_valid, reliable, short_side, detection_score
            )
    if not records:
        raise ValueError("Empty Face metadata")
    return records


def _align_three(full_dump, person_dump, face_dump) -> Tuple[np.ndarray, ...]:
    ids, _, _, targets, full_probs, person_probs = align_evaluation_scores(
        full_dump, person_dump
    )
    face_ids, _, _, face_targets, _, face_probs = align_evaluation_scores(
        full_dump, face_dump
    )
    if not np.array_equal(ids, face_ids) or not np.array_equal(targets, face_targets):
        raise ValueError("Full/Person/Face score alignment differs")
    return ids, targets, full_probs, person_probs, face_probs


def load_endpoint_triple(
    full_run: Path,
    person_run: Path,
    face_run: Path,
    face_manifest_root: Path,
) -> EndpointTriple:
    full_config, full_val, _, full_cal = _audit_source_run(full_run, "full")
    person_config, person_val, _, person_cal = _audit_source_run(
        person_run, "person_crop"
    )
    face_config, face_val, _, face_cal = _audit_source_run(face_run, "face_crop")
    for other in (person_config, face_config):
        for field in COMMON_CONFIG_FIELDS:
            if full_config.get(field) != other.get(field):
                raise ValueError(f"Three-view source configuration drift: {field}")
    if any(int(config.get("seed", -1)) != 0 for config in (
        full_config, person_config, face_config
    )):
        raise ValueError("Stage-2 router selection is locked to seed0")
    provenance = load_face_manifest_provenance(face_manifest_root)
    if face_config.get("face_manifest") != provenance:
        raise ValueError("Face source manifest provenance differs")
    validation = [
        _align_three(full, person, face)
        for full, person, face in zip(full_val, person_val, face_val)
    ]
    calibration = [
        _align_three(full, person, face)
        for full, person, face in zip(full_cal, person_cal, face_cal)
    ]
    return EndpointTriple(full_run, person_run, face_run, validation, calibration)


def _probability_features(
    sample_ids: Sequence[str],
    full: np.ndarray,
    person: np.ndarray,
    face: np.ndarray,
    geometry: Mapping[str, Geometry],
    face_metadata: Mapping[str, FaceMetadata],
) -> Tuple[np.ndarray, np.ndarray]:
    if not (full.shape == person.shape == face.shape):
        raise ValueError("Three-view probability shapes differ")
    missing = [
        sample_id for sample_id in sample_ids
        if sample_id not in geometry or sample_id not in face_metadata
    ]
    if missing:
        raise ValueError(f"Router metadata missing ID {missing[0]}")
    geom = [geometry[sample_id] for sample_id in sample_ids]
    faces = [face_metadata[sample_id] for sample_id in sample_ids]
    reliable = np.asarray([item.reliable for item in faces], dtype=np.bool_)
    area = np.asarray([item.bbox_area_ratio for item in geom], dtype=np.float32)
    aspect = np.asarray([item.absolute_aspect_ratio for item in geom], dtype=np.float32)
    people = np.asarray([item.people_in_image for item in geom], dtype=np.float32)
    short_side = np.asarray([item.short_side for item in faces], dtype=np.float32)
    score = np.asarray([item.detection_score for item in faces], dtype=np.float32)
    valid = np.asarray([item.training_valid for item in faces], dtype=np.float32)
    epsilon = np.float32(1e-6)
    probabilities = [
        np.clip(values.astype(np.float32), epsilon, 1.0 - epsilon)
        for values in (full, person, face)
    ]
    confidence = [np.mean(np.abs(values - 0.5) * 2.0, axis=1) for values in probabilities]
    entropy = [
        np.mean(-(
            values * np.log(values) + (1.0 - values) * np.log(1.0 - values)
        ) / np.log(2.0), axis=1)
        for values in probabilities
    ]
    differences = [
        np.abs(probabilities[left] - probabilities[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    ]
    features = np.column_stack((
        np.clip((np.log10(np.maximum(area, 1e-4)) + 4.0) / 4.0, 0.0, 1.0),
        np.clip(np.log2(np.maximum(aspect, 1.0)) / 4.0, 0.0, 1.0),
        np.clip(people, 1.0, 5.0) / 5.0,
        (people > 1).astype(np.float32),
        valid,
        reliable.astype(np.float32),
        np.clip(np.log2(np.maximum(short_side, 1.0) / 8.0) / 5.0, 0.0, 1.0),
        np.clip(score, 0.0, 1.0),
        *confidence,
        *entropy,
        *(np.mean(values, axis=1) for values in differences),
        *(np.max(values, axis=1) for values in differences),
    )).astype(np.float32)
    # Placeholder Face predictions are not evidence when Face is unavailable.
    features[~reliable, 10] = 0.0
    features[~reliable, 13] = 0.0
    features[~reliable, 15:17] = 0.0
    features[~reliable, 18:20] = 0.0
    if features.shape != (len(sample_ids), len(R2_FEATURE_NAMES)):
        raise ValueError("Unexpected R2 feature shape")
    if not np.isfinite(features).all():
        raise FloatingPointError("Non-finite R2 features")
    return features, reliable


def load_visual_descriptors(
    descriptor_root: Path,
    face_manifest_root: Path,
) -> Dict[str, Dict[str, np.ndarray]]:
    manifest = _read_json(descriptor_root / "descriptor_manifest.json")
    if (
        manifest.get("selection_only") is not True
        or manifest.get("test_accessed") is not False
        or tuple(manifest.get("feature_names", ())) != VISUAL_FEATURE_NAMES
        or manifest.get("calibration_fraction") != CALIBRATION_FRACTION
        or manifest.get("face_manifest") != load_face_manifest_provenance(face_manifest_root)
    ):
        raise ValueError("Visual descriptor provenance differs")
    output = {}
    for split in ("train", "val"):
        artifact = descriptor_root / f"{split}_descriptors.npz"
        if _sha256(artifact) != manifest["splits"][split]["sha256"]:
            raise ValueError("Visual descriptor hash differs")
        with np.load(artifact, allow_pickle=False) as data:
            ids = data["sample_ids"].astype(str)
            features = data["features"].astype(np.float32)
            reliable = data["face_reliable"].astype(np.bool_)
            if (
                int(data["schema_version"]) != 1
                or str(data["split"]) != split
                or tuple(data["feature_names"].astype(str)) != VISUAL_FEATURE_NAMES
                or len(set(ids.tolist())) != len(ids)
                or features.shape != (len(ids), len(VISUAL_FEATURE_NAMES))
                or reliable.shape != (len(ids),)
                or not np.isfinite(features).all()
            ):
                raise ValueError("Invalid visual descriptor artifact")
        output[split] = {
            sample_id: (features[index], reliable[index])
            for index, sample_id in enumerate(ids)
        }
    return output


def _select_visual_features(
    sample_ids: Sequence[str],
    descriptors: Mapping[str, Tuple[np.ndarray, np.bool_]],
    reliable: np.ndarray,
) -> np.ndarray:
    missing = [sample_id for sample_id in sample_ids if sample_id not in descriptors]
    if missing:
        raise ValueError(f"Visual descriptors missing ID {missing[0]}")
    rows = np.stack([descriptors[sample_id][0] for sample_id in sample_ids])
    descriptor_reliable = np.asarray(
        [descriptors[sample_id][1] for sample_id in sample_ids], dtype=np.bool_
    )
    if not np.array_equal(descriptor_reliable, reliable):
        raise ValueError("Visual descriptor Face mask differs from manifest")
    return rows.astype(np.float32)


def _standardize_fit_apply(
    fit: np.ndarray, apply: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[float]]]:
    mean = fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = fit.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return (
        ((fit - mean) / scale).astype(np.float32),
        ((apply - mean) / scale).astype(np.float32),
        {"mean": mean.tolist(), "scale": scale.tolist()},
    )


def _fuse(probabilities: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if weights.shape != (len(probabilities[0]), 3):
        raise ValueError("Invalid three-view weights")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("Router weights do not sum to one")
    return sum(weights[:, index, None] * values for index, values in enumerate(probabilities))


def _fixed_rows(
    endpoint: EndpointTriple,
    face_metadata: Mapping[str, FaceMetadata],
    beta: float,
) -> Dict[str, Any]:
    rows = []
    for task_id, (ids, targets, full, person, face) in enumerate(endpoint.validation_tasks):
        reliable = np.asarray([face_metadata[sample_id].reliable for sample_id in ids])
        weights = np.tile(INVALID_PRIOR, (len(ids), 1))
        if beta:
            weights[reliable] = np.asarray(
                ((1.0 - beta) * 0.8, (1.0 - beta) * 0.2, beta),
                dtype=np.float32,
            )
        scores = _fuse((full, person, face), weights)
        rows.append(compute_metrics(
            task_id, torch.from_numpy(scores), torch.from_numpy(targets), THRESHOLD
        ))
    return {"beta": beta, "metrics": summarize_tasks(rows), "task_metrics": [asdict(row) for row in rows]}


def _diagnostics(
    targets: np.ndarray,
    probabilities: Sequence[np.ndarray],
    weights: np.ndarray,
    current: Sequence[int],
) -> Dict[str, Any]:
    scores = _fuse(probabilities, weights)
    selected = np.clip(scores[:, current], 1e-6, 1.0 - 1e-6)
    target = targets[:, current]
    return {
        "samples": len(targets),
        "current_positive_support": target.sum(axis=0).tolist(),
        "current_BCE": float(np.mean(
            -target * np.log(selected) - (1.0 - target) * np.log(1.0 - selected)
        )),
        "weight_mean": weights.mean(axis=0).tolist(),
        "weight_std": weights.std(axis=0).tolist(),
        "weight_min": weights.min(axis=0).tolist(),
        "weight_max": weights.max(axis=0).tolist(),
        "weight_quantiles_05_50_95": np.quantile(
            weights, (0.05, 0.5, 0.95), axis=0
        ).tolist(),
    }


def _train_candidate(
    endpoint: EndpointTriple,
    family: str,
    prior_strength: float,
    calibration_geometry: Mapping[str, Geometry],
    validation_geometry: Mapping[str, Geometry],
    calibration_face: Mapping[str, FaceMetadata],
    validation_face: Mapping[str, FaceMetadata],
    visual_descriptors: Mapping[str, Mapping[str, Tuple[np.ndarray, np.bool_]]],
    state_root: Path,
) -> Dict[str, Any]:
    if family not in ("R2", "R3") or prior_strength not in PRIOR_STRENGTHS:
        raise ValueError("Unexpected router candidate")
    candidate_id = f"{family}_prior{str(prior_strength).replace('.', 'p')}"
    candidate_root = state_root / candidate_id
    candidate_root.mkdir(parents=True, exist_ok=False)
    rows: List[TaskMetrics] = []
    tasks = []
    for task_id in range(len(TASK_SIZES)):
        cal_ids, cal_targets, cal_full, cal_person, cal_face = endpoint.calibration_tasks[task_id]
        val_ids, val_targets, val_full, val_person, val_face = endpoint.validation_tasks[task_id]
        cal_features, cal_reliable = _probability_features(
            cal_ids, cal_full, cal_person, cal_face, calibration_geometry, calibration_face
        )
        val_features, val_reliable = _probability_features(
            val_ids, val_full, val_person, val_face, validation_geometry, validation_face
        )
        feature_names = list(R2_FEATURE_NAMES)
        if family == "R3":
            cal_features = np.column_stack((
                cal_features,
                _select_visual_features(cal_ids, visual_descriptors["train"], cal_reliable),
            ))
            val_features = np.column_stack((
                val_features,
                _select_visual_features(val_ids, visual_descriptors["val"], val_reliable),
            ))
            feature_names.extend(VISUAL_FEATURE_NAMES)
        cal_features, val_features, normalization = _standardize_fit_apply(
            cal_features, val_features
        )
        router = ThreeViewRouter(
            len(feature_names), 42000 + task_id + (100 if family == "R3" else 0)
        ).float()
        optimizer = torch.optim.AdamW(
            router.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        generator = torch.Generator().manual_seed(
            43000 + task_id + (100 if family == "R3" else 0)
        )
        features_tensor = torch.from_numpy(cal_features)
        reliable_tensor = torch.from_numpy(cal_reliable)
        probability_tensors = [
            torch.from_numpy(values) for values in (cal_full, cal_person, cal_face)
        ]
        targets_tensor = torch.from_numpy(cal_targets)
        prior = torch.from_numpy(np.where(
            cal_reliable[:, None], VALID_PRIOR[None], INVALID_PRIOR[None]
        ).astype(np.float32))
        current = list(task_indices(task_id))
        final_objective = None
        for _ in range(EPOCHS_PER_TASK):
            order = torch.randperm(len(features_tensor), generator=generator)
            for start in range(0, len(order), BATCH_SIZE):
                indices = order[start:start + BATCH_SIZE]
                weights = router(features_tensor[indices], reliable_tensor[indices])
                fused = sum(
                    weights[:, view, None] * probability_tensors[view][indices]
                    for view in range(3)
                ).clamp(1e-6, 1.0 - 1e-6)
                data_loss = F.binary_cross_entropy(
                    fused[:, current], targets_tensor[indices][:, current]
                )
                prior_loss = torch.mean(torch.sum(
                    (weights - prior[indices]) ** 2, dim=1
                ))
                objective = data_loss + float(prior_strength) * prior_loss
                if not torch.isfinite(objective):
                    raise FloatingPointError("Non-finite three-view router objective")
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()
                final_objective = float(objective.detach())
        router.eval()
        with torch.no_grad():
            calibration_weights = router(features_tensor, reliable_tensor).numpy()
            validation_weights = router(
                torch.from_numpy(val_features), torch.from_numpy(val_reliable)
            ).numpy()
        if not np.array_equal(validation_weights[~val_reliable, 2], np.zeros(
            int((~val_reliable).sum()), dtype=np.float32
        )):
            raise RuntimeError("Invalid Face received nonzero router weight")
        validation_scores = _fuse(
            (val_full, val_person, val_face), validation_weights
        )
        row = compute_metrics(
            task_id,
            torch.from_numpy(validation_scores),
            torch.from_numpy(val_targets),
            THRESHOLD,
        )
        rows.append(row)
        state_path = candidate_root / f"task{task_id}.pth"
        torch.save({
            "schema_version": 1,
            "candidate_id": candidate_id,
            "task_id": task_id,
            "feature_names": feature_names,
            "normalization": normalization,
            "model": router.state_dict(),
        }, state_path)
        tasks.append({
            "task_id": task_id,
            "calibration_samples": len(cal_ids),
            "validation_samples": len(val_ids),
            "calibration_reliable_face_samples": int(cal_reliable.sum()),
            "validation_reliable_face_samples": int(val_reliable.sum()),
            "final_objective": final_objective,
            "normalization": normalization,
            "calibration": _diagnostics(
                cal_targets, (cal_full, cal_person, cal_face), calibration_weights, current
            ),
            "validation": _diagnostics(
                val_targets, (val_full, val_person, val_face), validation_weights, current
            ),
            "state_path": str(state_path.relative_to(state_root.parent)),
            "state_sha256": _sha256(state_path),
        })
    return {
        "candidate_id": candidate_id,
        "family": family,
        "prior_strength": prior_strength,
        "feature_names": feature_names,
        "metrics": summarize_tasks(rows),
        "task_metrics": [asdict(row) for row in rows],
        "tasks": tasks,
    }


def select_three_view_router(
    endpoint: EndpointTriple,
    data_root: Path,
    face_manifest_root: Path,
    descriptor_root: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    state_root = output_dir / "router_states"
    state_root.mkdir()
    dataset_parent = resolve_dataset_parent(data_root)
    calibration_geometry = load_geometry(dataset_parent, "train")
    validation_geometry = load_geometry(dataset_parent, "val")
    calibration_face = load_face_metadata(face_manifest_root, "train")
    validation_face = load_face_metadata(face_manifest_root, "val")
    descriptors = load_visual_descriptors(descriptor_root, face_manifest_root)
    r0 = _fixed_rows(endpoint, validation_face, beta=0.0)
    r1 = _fixed_rows(endpoint, validation_face, beta=0.20)
    candidates = [
        _train_candidate(
            endpoint,
            family,
            prior,
            calibration_geometry,
            validation_geometry,
            calibration_face,
            validation_face,
            descriptors,
            state_root,
        )
        for family in ("R2", "R3")
        for prior in PRIOR_STRENGTHS
    ]
    best = {}
    for family in ("R2", "R3"):
        best[family] = max(
            (row for row in candidates if row["family"] == family),
            key=lambda row: (
                row["metrics"]["final_mAP"],
                row["metrics"]["average_mAP"],
                row["prior_strength"],
            ),
        )
    advance = (
        best["R3"]["metrics"]["final_mAP"] > r1["metrics"]["final_mAP"]
        and best["R3"]["metrics"]["final_mAP"]
        > best["R2"]["metrics"]["final_mAP"]
    )
    result = {
        "schema_version": 1,
        "selection_split": "val",
        "test_accessed": False,
        "method": "task_local_sample_wise_three_view_soft_router",
        "calibration": {
            "fraction": CALIBRATION_FRACTION,
            "split": "stable_sha256_image_group_v1",
            "training_exclusion": True,
            "loss_scope": "current_task_classes",
            "epochs_per_task": EPOCHS_PER_TASK,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "router": {
            "task_parameters_shared": False,
            "architecture_shared": True,
            "hidden_dim": HIDDEN_DIM,
            "output": "one_three_view_weight_vector_per_sample_and_task",
            "normalization": "fit_on_current_task_calibration_then_frozen",
            "face_invalid_weight": 0.0,
            "valid_initial_weights": VALID_PRIOR.tolist(),
            "invalid_initial_weights": INVALID_PRIOR.tolist(),
        },
        "R0_fixed_full_person": r0,
        "R1_fixed_face_beta0.20": r1,
        "candidates": candidates,
        "best_by_family": best,
        "decision": {
            "R3_beats_R1": best["R3"]["metrics"]["final_mAP"] > r1["metrics"]["final_mAP"],
            "R3_beats_R2": best["R3"]["metrics"]["final_mAP"] > best["R2"]["metrics"]["final_mAP"],
            "advance_to_seed1_seed2_validation": advance,
            "run_test": False,
        },
        "sources": {
            "full": str(endpoint.full_run.resolve()),
            "person": str(endpoint.person_run.resolve()),
            "face": str(endpoint.face_run.resolve()),
            "descriptors": str(descriptor_root.resolve()),
            "full_config_sha256": _sha256(endpoint.full_run / "config.json"),
            "person_config_sha256": _sha256(endpoint.person_run / "config.json"),
            "face_config_sha256": _sha256(endpoint.face_run / "config.json"),
        },
    }
    output = output_dir / "validation_selection.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2), flush=True)
    print("THREE_VIEW_ROUTER_VALIDATION_COMPLETE", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--person-run", type=Path, required=True)
    parser.add_argument("--face-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--descriptor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    endpoint = load_endpoint_triple(
        args.full_run, args.person_run, args.face_run, args.face_manifest_root
    )
    select_three_view_router(
        endpoint,
        args.data_root,
        args.face_manifest_root,
        args.descriptor_root,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
