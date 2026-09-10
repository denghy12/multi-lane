"""Select a taskwise three-view router from image-group out-of-fold predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .evaluation_scores import load_evaluation_scores
from .export_three_view_descriptors import FEATURE_NAMES as VISUAL_FEATURE_NAMES
from .fuse_validation_scores import COMMON_CONFIG_FIELDS, validated_run_scores
from .runner import TASK_SIZES, TaskMetrics, compute_metrics, summarize_tasks, task_indices
from .search_constrained_gated_fusion import Geometry, load_geometry
from .three_view_router import (
    BATCH_SIZE,
    EPOCHS_PER_TASK,
    INVALID_PRIOR,
    LEARNING_RATE,
    PRIOR_STRENGTHS,
    R2_FEATURE_NAMES,
    THRESHOLD,
    VALID_PRIOR,
    WEIGHT_DECAY,
    FaceMetadata,
    _align_three,
    _diagnostics,
    _fixed_rows,
    _fuse,
    _probability_features,
    _select_visual_features,
    _sha256,
    _standardize_fit_apply,
    load_face_metadata,
)
from .runner import load_face_manifest_provenance, resolve_dataset_parent


FOLDS = 3
HIDDEN_DIM = 16
OOF_SPLIT = "stable_sha256_image_group_3fold_oof_v1"


class SharedTaskBiasRouter(nn.Module):
    """Shared reliability trunk with one three-value bias per incremental task."""

    def __init__(self, feature_dim: int, tasks: int, initialization_seed: int) -> None:
        super().__init__()
        if feature_dim <= 0 or tasks <= 0:
            raise ValueError("Router dimensions must be positive")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(initialization_seed))
            self.trunk = nn.Sequential(nn.Linear(feature_dim, HIDDEN_DIM), nn.GELU())
            self.output = nn.Linear(HIDDEN_DIM, 3)
            self.task_bias = nn.Embedding(tasks, 3)
            nn.init.xavier_uniform_(self.trunk[0].weight)
            nn.init.zeros_(self.trunk[0].bias)
            nn.init.zeros_(self.output.weight)
            nn.init.constant_(self.output.bias[0], math.log(float(VALID_PRIOR[0])))
            nn.init.constant_(self.output.bias[1], math.log(float(VALID_PRIOR[1])))
            nn.init.constant_(self.output.bias[2], math.log(float(VALID_PRIOR[2])))
            nn.init.zeros_(self.task_bias.weight)

    def forward(
        self,
        features: torch.Tensor,
        task_ids: torch.Tensor,
        face_reliable: torch.Tensor,
    ) -> torch.Tensor:
        if (
            features.ndim != 2
            or task_ids.shape != (len(features),)
            or face_reliable.shape != (len(features),)
        ):
            raise ValueError("Invalid shared Router inputs")
        logits = self.output(self.trunk(features)) + self.task_bias(task_ids)
        logits = logits.masked_fill(
            ~face_reliable[:, None]
            & torch.tensor([False, False, True], device=logits.device),
            torch.finfo(logits.dtype).min,
        )
        weights = torch.softmax(logits, dim=-1)
        if not torch.isfinite(weights).all():
            raise FloatingPointError("Non-finite shared Router weights")
        return weights


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_oof_source(run: Path, view: str, fold: int):
    config = _read_json(run / "config.json")
    summary = _read_json(run / "seed_summary.json")
    history = _read_json(run / "training_history.json")
    if summary.get("config") != config or summary.get("status") != "complete":
        raise ValueError("OOF source is incomplete or its config provenance differs")
    expected = {
        "input_mode": view,
        "reporting_split": "val",
        "evaluation_score_purpose": "validation_search",
        "calibration_fraction": 0.0,
        "calibration_split": OOF_SPLIT,
        "crossfit_folds": FOLDS,
        "crossfit_held_out_fold": fold,
        "crossfit_split_salt": "emotic-three-view-oof-v1",
        "calibration_training_exclusion": True,
        "save_calibration_scores": True,
        "save_checkpoints": False,
        "seed": 0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"OOF source provenance differs: {key}")
    if config.get("git", {}).get("dirty") is not False:
        raise ValueError("OOF source was trained from a dirty worktree")
    if set(history) != {str(task) for task in range(len(TASK_SIZES))}:
        raise ValueError("OOF source training history is incomplete")
    if any(
        row["skipped_optimizer_steps"] != 0
        for task_history in history.values()
        for row in task_history
    ):
        raise ValueError("OOF source skipped optimizer updates")
    dumps = []
    for task in range(len(TASK_SIZES)):
        dump = load_evaluation_scores(
            run / "calibration_scores" / f"task{task}.npz",
            config.get("eval_batch_size"),
        )
        if dump.task_id != task or dump.logits.shape[1] != sum(TASK_SIZES[: task + 1]):
            raise ValueError("Unexpected OOF score layout")
        dumps.append(dump)
    return config, dumps


def load_oof_tasks(oof_root: Path) -> Tuple[List[Tuple[np.ndarray, ...]], Dict[str, Any]]:
    by_fold = []
    configs = []
    for fold in range(FOLDS):
        runs = [oof_root / f"fold{fold}_{view}" for view in ("full", "person", "face")]
        audited = [
            _audit_oof_source(run, view, fold)
            for run, view in zip(runs, ("full", "person_crop", "face_crop"))
        ]
        configs.extend(item[0] for item in audited)
        by_fold.append([
            _align_three(full, person, face)
            for full, person, face in zip(audited[0][1], audited[1][1], audited[2][1])
        ])
    reference = configs[0]
    for config in configs[1:]:
        for field in COMMON_CONFIG_FIELDS:
            if reference.get(field) != config.get(field):
                raise ValueError(f"OOF source fixed configuration differs: {field}")
    tasks = []
    fold_counts: Dict[str, List[int]] = {}
    for task in range(len(TASK_SIZES)):
        parts = [by_fold[fold][task] for fold in range(FOLDS)]
        all_ids = np.concatenate([part[0] for part in parts])
        if len(set(all_ids.tolist())) != len(all_ids):
            raise ValueError(f"OOF task{task} contains duplicate image-person samples")
        tasks.append(tuple(np.concatenate([part[column] for part in parts]) for column in range(5)))
        fold_counts[str(task)] = [len(part[0]) for part in parts]
    return tasks, {"fold_counts": fold_counts, "source_git": reference.get("git")}


def load_validation_tasks(
    full_run: Path, person_run: Path, face_run: Path
) -> Tuple[List[Tuple[np.ndarray, ...]], Dict[str, str]]:
    runs = (full_run, person_run, face_run)
    configs = [_read_json(run / "config.json") for run in runs]
    expected_views = ("full", "person_crop", "face_crop")
    for config, view in zip(configs, expected_views):
        if config.get("input_mode") != view or config.get("reporting_split") != "val":
            raise ValueError("Validation endpoint view differs")
        if float(config.get("calibration_fraction", 0.0)) != 0.0:
            raise ValueError("OOF validation endpoints must use the complete train split")
    for other in configs[1:]:
        for field in COMMON_CONFIG_FIELDS:
            if configs[0].get(field) != other.get(field):
                raise ValueError(f"Validation endpoint configuration differs: {field}")
    dumps = [validated_run_scores(run, "val")[0] for run in runs]
    tasks = [
        _align_three(full, person, face)
        for full, person, face in zip(dumps[0], dumps[1], dumps[2])
    ]
    return tasks, {
        "full": str(full_run.resolve()),
        "person": str(person_run.resolve()),
        "face": str(face_run.resolve()),
    }


def load_oof_descriptors(
    root: Path, face_manifest_root: Path
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.bool_]]]:
    manifest = _read_json(root / "descriptor_manifest.json")
    if (
        manifest.get("selection_only") is not True
        or manifest.get("test_accessed") is not False
        or manifest.get("train_scope") != "all"
        or tuple(manifest.get("feature_names", ())) != VISUAL_FEATURE_NAMES
        or manifest.get("face_manifest") != load_face_manifest_provenance(face_manifest_root)
    ):
        raise ValueError("OOF descriptor provenance differs")
    output = {}
    for split in ("train", "val"):
        artifact = root / f"{split}_descriptors.npz"
        if manifest["splits"][split]["sha256"] != _sha256(artifact):
            raise ValueError("OOF descriptor hash differs")
        with np.load(artifact, allow_pickle=False) as data:
            ids = data["sample_ids"].astype(str)
            features = data["features"].astype(np.float32)
            reliable = data["face_reliable"].astype(np.bool_)
        if (
            len(set(ids.tolist())) != len(ids)
            or features.shape != (len(ids), len(VISUAL_FEATURE_NAMES))
            or reliable.shape != (len(ids),)
            or not np.isfinite(features).all()
        ):
            raise ValueError("Invalid OOF descriptors")
        output[split] = {
            sample_id: (features[index], reliable[index])
            for index, sample_id in enumerate(ids)
        }
    return output


def _raw_features(
    task: Tuple[np.ndarray, ...],
    geometry: Mapping[str, Geometry],
    face: Mapping[str, FaceMetadata],
    descriptors: Optional[Mapping[str, Tuple[np.ndarray, np.bool_]]],
) -> Tuple[np.ndarray, np.ndarray]:
    ids, _, full, person, face_prob = task
    features, reliable = _probability_features(
        ids, full, person, face_prob, geometry, face
    )
    if descriptors is not None:
        features = np.column_stack((
            features,
            _select_visual_features(ids, descriptors, reliable),
        )).astype(np.float32)
    return features, reliable


def _state_weights(
    state: Mapping[str, Any],
    lane: int,
    task: Tuple[np.ndarray, ...],
    geometry: Mapping[str, Geometry],
    face: Mapping[str, FaceMetadata],
    descriptors: Optional[Mapping[str, Tuple[np.ndarray, np.bool_]]],
) -> np.ndarray:
    raw, reliable = _raw_features(task, geometry, face, descriptors)
    mean = np.asarray(state["normalization"]["mean"], dtype=np.float32)
    scale = np.asarray(state["normalization"]["scale"], dtype=np.float32)
    normalized = ((raw - mean) / scale).astype(np.float32)
    router = SharedTaskBiasRouter(len(mean), len(TASK_SIZES), 0).float()
    router.load_state_dict(state["model"], strict=True)
    router.eval()
    with torch.no_grad():
        weights = router(
            torch.from_numpy(normalized),
            torch.full((len(normalized),), lane, dtype=torch.long),
            torch.from_numpy(reliable),
        ).numpy()
    if not np.array_equal(weights[~reliable, 2], np.zeros(int((~reliable).sum()))):
        raise RuntimeError("Invalid Face received nonzero OOF Router weight")
    return weights


def _evaluate_taskwise(
    validation_tasks: Sequence[Tuple[np.ndarray, ...]],
    states: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Geometry],
    face: Mapping[str, FaceMetadata],
    descriptors: Optional[Mapping[str, Tuple[np.ndarray, np.bool_]]],
) -> Tuple[List[TaskMetrics], List[Dict[str, Any]]]:
    rows, diagnostics = [], []
    for evaluation_task, task in enumerate(validation_tasks):
        ids, targets, full, person, face_prob = task
        fused = np.empty_like(full, dtype=np.float32)
        means = []
        start = 0
        for lane in range(evaluation_task + 1):
            end = start + TASK_SIZES[lane]
            lane_task = (ids, targets[:, :end], full[:, :end], person[:, :end], face_prob[:, :end])
            weights = _state_weights(
                states[lane], lane, lane_task, geometry, face, descriptors
            )
            fused[:, start:end] = _fuse(
                (full[:, start:end], person[:, start:end], face_prob[:, start:end]),
                weights,
            )
            means.append(weights.mean(axis=0).tolist())
            start = end
        rows.append(compute_metrics(
            evaluation_task, torch.from_numpy(fused), torch.from_numpy(targets), THRESHOLD
        ))
        diagnostics.append({"task_id": evaluation_task, "lane_weight_mean": means})
    return rows, diagnostics


def _train_candidate(
    family: str,
    prior_strength: float,
    oof_tasks: Sequence[Tuple[np.ndarray, ...]],
    validation_tasks: Sequence[Tuple[np.ndarray, ...]],
    train_geometry: Mapping[str, Geometry],
    val_geometry: Mapping[str, Geometry],
    train_face: Mapping[str, FaceMetadata],
    val_face: Mapping[str, FaceMetadata],
    descriptors: Mapping[str, Mapping[str, Tuple[np.ndarray, np.bool_]]],
    state_root: Path,
) -> Dict[str, Any]:
    use_visual = family == "R3"
    feature_names = list(R2_FEATURE_NAMES) + (list(VISUAL_FEATURE_NAMES) if use_visual else [])
    prepared = []
    for task_id, task in enumerate(oof_tasks):
        raw, reliable = _raw_features(
            task, train_geometry, train_face,
            descriptors["train"] if use_visual else None,
        )
        prepared.append({
            "raw": raw,
            "reliable": reliable,
            "probabilities": task[2:5],
            "targets": task[1],
            "current": list(task_indices(task_id)),
        })
    candidate_id = f"OOF_{family}_prior{str(prior_strength).replace('.', 'p')}"
    candidate_root = state_root / candidate_id
    candidate_root.mkdir(parents=True, exist_ok=False)
    router = SharedTaskBiasRouter(len(feature_names), len(TASK_SIZES), 52000).float()
    states, task_records = [], []
    generator = torch.Generator().manual_seed(53000)
    for task_id in range(len(TASK_SIZES)):
        accumulated = np.concatenate([item["raw"] for item in prepared[: task_id + 1]])
        _, _, normalization = _standardize_fit_apply(accumulated, accumulated)
        mean = np.asarray(normalization["mean"], dtype=np.float32)
        scale = np.asarray(normalization["scale"], dtype=np.float32)
        optimizer = torch.optim.AdamW(
            router.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        final_objective = None
        for _ in range(EPOCHS_PER_TASK):
            for source_task, item in enumerate(prepared[: task_id + 1]):
                features = torch.from_numpy(((item["raw"] - mean) / scale).astype(np.float32))
                reliable = torch.from_numpy(item["reliable"])
                targets = torch.from_numpy(item["targets"])
                probabilities = [torch.from_numpy(values) for values in item["probabilities"]]
                prior = torch.from_numpy(np.where(
                    item["reliable"][:, None], VALID_PRIOR[None], INVALID_PRIOR[None]
                ).astype(np.float32))
                order = torch.randperm(len(features), generator=generator)
                for start in range(0, len(order), BATCH_SIZE):
                    indices = order[start : start + BATCH_SIZE]
                    task_ids = torch.full((len(indices),), source_task, dtype=torch.long)
                    weights = router(features[indices], task_ids, reliable[indices])
                    fused = sum(
                        weights[:, view, None] * probabilities[view][indices]
                        for view in range(3)
                    ).clamp(1e-6, 1.0 - 1e-6)
                    current = item["current"]
                    data_loss = F.binary_cross_entropy(
                        fused[:, current], targets[indices][:, current]
                    )
                    prior_loss = torch.mean(torch.sum((weights - prior[indices]) ** 2, dim=1))
                    objective = data_loss + float(prior_strength) * prior_loss
                    if not torch.isfinite(objective):
                        raise FloatingPointError("Non-finite OOF Router objective")
                    optimizer.zero_grad(set_to_none=True)
                    objective.backward()
                    optimizer.step()
                    final_objective = float(objective.detach())
        state = {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "task_id": task_id,
            "feature_names": feature_names,
            "normalization": normalization,
            "model": {key: value.detach().clone() for key, value in router.state_dict().items()},
        }
        state_path = candidate_root / f"task{task_id}.pth"
        torch.save(state, state_path)
        states.append(state)
        item = prepared[task_id]
        train_weights = _state_weights(
            state,
            task_id,
            oof_tasks[task_id],
            train_geometry,
            train_face,
            descriptors["train"] if use_visual else None,
        )
        task_records.append({
            "task_id": task_id,
            "oof_samples": len(item["raw"]),
            "final_objective": final_objective,
            "state_path": str(state_path.relative_to(state_root.parent)),
            "state_sha256": _sha256(state_path),
            "oof": _diagnostics(
                item["targets"], item["probabilities"], train_weights, item["current"]
            ),
        })
    rows, validation_diagnostics = _evaluate_taskwise(
        validation_tasks, states, val_geometry, val_face,
        descriptors["val"] if use_visual else None,
    )
    return {
        "candidate_id": candidate_id,
        "family": family,
        "prior_strength": prior_strength,
        "feature_names": feature_names,
        "metrics": summarize_tasks(rows),
        "task_metrics": [asdict(row) for row in rows],
        "tasks": task_records,
        "validation_diagnostics": validation_diagnostics,
    }


def select_oof_router(
    oof_root: Path,
    full_run: Path,
    person_run: Path,
    face_run: Path,
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
    oof_tasks, oof_provenance = load_oof_tasks(oof_root)
    validation_tasks, validation_sources = load_validation_tasks(
        full_run, person_run, face_run
    )
    dataset_parent = resolve_dataset_parent(data_root)
    train_geometry = load_geometry(dataset_parent, "train")
    val_geometry = load_geometry(dataset_parent, "val")
    train_face = load_face_metadata(face_manifest_root, "train")
    val_face = load_face_metadata(face_manifest_root, "val")
    descriptors = load_oof_descriptors(descriptor_root, face_manifest_root)
    endpoint = type("ValidationEndpoint", (), {"validation_tasks": validation_tasks})()
    r1 = _fixed_rows(endpoint, val_face, beta=0.20)
    candidates = [
        _train_candidate(
            family, prior, oof_tasks, validation_tasks, train_geometry, val_geometry,
            train_face, val_face, descriptors, state_root,
        )
        for family in ("R2", "R3")
        for prior in PRIOR_STRENGTHS
    ]
    best = {
        family: max(
            (row for row in candidates if row["family"] == family),
            key=lambda row: (
                row["metrics"]["final_mAP"], row["metrics"]["average_mAP"],
                row["prior_strength"],
            ),
        )
        for family in ("R2", "R3")
    }
    advance = (
        best["R3"]["metrics"]["final_mAP"] > r1["metrics"]["final_mAP"]
        and best["R3"]["metrics"]["final_mAP"] > best["R2"]["metrics"]["final_mAP"]
    )
    result = {
        "schema_version": 1,
        "selection_split": "val",
        "test_accessed": False,
        "method": "three_fold_image_group_oof_shared_trunk_task_bias_router",
        "crossfit": {
            "folds": FOLDS,
            "group": "source_image_before_person_suffix",
            "training_coverage": "complete_current_task_train_pool",
            "every_router_training_prediction_is_out_of_fold": True,
            **oof_provenance,
        },
        "router": {
            "hidden_dim": HIDDEN_DIM,
            "shared_reliability_trunk": True,
            "task_specific_parameters": "three_value_bias",
            "incremental_semantics": "lane_k_uses_snapshot_saved_after_task_k",
            "epochs_per_task": EPOCHS_PER_TASK,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "invalid_face_weight": 0.0,
        },
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
            "oof_root": str(oof_root.resolve()),
            "validation": validation_sources,
            "descriptors": str(descriptor_root.resolve()),
        },
    }
    output = output_dir / "validation_selection.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["decision"], indent=2), flush=True)
    print("THREE_VIEW_OOF_ROUTER_VALIDATION_COMPLETE", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-root", type=Path, required=True)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--person-run", type=Path, required=True)
    parser.add_argument("--face-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--descriptor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    select_oof_router(
        args.oof_root, args.full_run, args.person_run, args.face_run,
        args.data_root, args.face_manifest_root, args.descriptor_root, args.output_dir,
    )


if __name__ == "__main__":
    main()
