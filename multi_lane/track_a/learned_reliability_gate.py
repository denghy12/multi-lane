"""Train a shared sample-level Full/Person reliability gate on held-out train data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .evaluation_scores import align_evaluation_scores, load_evaluation_scores
from .fuse_validation_scores import COMMON_CONFIG_FIELDS, validated_run_scores
from .runner import (
    CLASS_ORDER,
    TASK_SIZES,
    TaskMetrics,
    compute_metrics,
    average_precision,
    resolve_dataset_parent,
    summarize_tasks,
    task_indices,
)
from .search_constrained_gated_fusion import Geometry, load_geometry


ANCHOR_PERSON_WEIGHT = 0.20
MIN_PERSON_WEIGHT = 0.10
MAX_PERSON_WEIGHT = 0.35
CALIBRATION_FRACTION = 0.10
GATE_HIDDEN_DIMS = (8, 16)
PRIOR_STRENGTHS = (0.1, 0.3, 1.0, 3.0)
GATE_EPOCHS_PER_TASK = 80
GATE_BATCH_SIZE = 64
GATE_LEARNING_RATE = 1e-3
GATE_WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
FEATURE_NAMES = (
    "bbox_log_area",
    "bbox_log_aspect",
    "people_scaled",
    "is_multi_person",
    "full_confidence",
    "person_confidence",
    "full_entropy",
    "person_entropy",
    "mean_probability_disagreement",
    "max_probability_disagreement",
)
METRICS = ("final_mAP", "average_mAP", "final_cF1", "final_oF1", "forgetting")


@dataclass
class EndpointPair:
    seed: int
    full_run: Path
    person_run: Path
    validation_tasks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    calibration_tasks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    full_rows: List[TaskMetrics]


class ReliabilityGate(nn.Module):
    """A tiny task-shared MLP that emits one bounded Person weight per sample."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("Reliability gate hidden dimension must be positive")
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.xavier_uniform_(self.network[0].weight)
        nn.init.zeros_(self.network[0].bias)
        nn.init.zeros_(self.network[2].weight)
        initial_fraction = (
            (ANCHOR_PERSON_WEIGHT - MIN_PERSON_WEIGHT)
            / (MAX_PERSON_WEIGHT - MIN_PERSON_WEIGHT)
        )
        nn.init.constant_(
            self.network[2].bias,
            math.log(initial_fraction / (1.0 - initial_fraction)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.network(features)).squeeze(-1)
        return MIN_PERSON_WEIGHT + (MAX_PERSON_WEIGHT - MIN_PERSON_WEIGHT) * raw


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: Sequence[float]) -> Dict[str, Any]:
    values = [float(value) for value in values]
    return {
        "values": values,
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _aggregate(seed_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        metric: _stats([row["metrics"][metric] for row in seed_rows])
        for metric in METRICS
    }


def _candidate_id(hidden_dim: int, prior_strength: float) -> str:
    return f"h{hidden_dim}_prior{str(prior_strength).replace('.', 'p')}"


def gate_features(
    sample_ids: Sequence[str],
    full_probabilities: np.ndarray,
    person_probabilities: np.ndarray,
    geometry: Mapping[str, Geometry],
) -> np.ndarray:
    if full_probabilities.shape != person_probabilities.shape:
        raise ValueError("Full and Person probabilities have different shapes")
    if full_probabilities.ndim != 2 or len(sample_ids) != len(full_probabilities):
        raise ValueError("Invalid reliability feature inputs")
    missing = [sample_id for sample_id in sample_ids if sample_id not in geometry]
    if missing:
        raise ValueError(f"Missing geometry for {missing[0]}")
    geom = [geometry[sample_id] for sample_id in sample_ids]
    area = np.asarray([item.bbox_area_ratio for item in geom], dtype=np.float32)
    aspect = np.asarray([item.absolute_aspect_ratio for item in geom], dtype=np.float32)
    people = np.asarray([item.people_in_image for item in geom], dtype=np.float32)
    epsilon = np.float32(1e-6)
    full = np.clip(full_probabilities.astype(np.float32), epsilon, 1.0 - epsilon)
    person = np.clip(person_probabilities.astype(np.float32), epsilon, 1.0 - epsilon)
    binary_entropy = lambda values: -(
        values * np.log(values) + (1.0 - values) * np.log(1.0 - values)
    ) / np.log(2.0)
    difference = np.abs(full - person)
    features = np.column_stack(
        (
            np.clip((np.log10(np.maximum(area, 1e-4)) + 4.0) / 4.0, 0.0, 1.0),
            np.clip(np.log2(np.maximum(aspect, 1.0)) / 4.0, 0.0, 1.0),
            np.clip(people, 1.0, 5.0) / 5.0,
            (people > 1).astype(np.float32),
            np.mean(np.abs(full - 0.5) * 2.0, axis=1),
            np.mean(np.abs(person - 0.5) * 2.0, axis=1),
            np.mean(binary_entropy(full), axis=1),
            np.mean(binary_entropy(person), axis=1),
            np.mean(difference, axis=1),
            np.max(difference, axis=1),
        )
    ).astype(np.float32)
    if features.shape != (len(sample_ids), len(FEATURE_NAMES)) or not np.isfinite(features).all():
        raise FloatingPointError("Invalid reliability gate features")
    return features


def _aligned_arrays(full_dump, person_dump):
    ids, _, _, targets, full_probs, person_probs = align_evaluation_scores(
        full_dump, person_dump
    )
    return ids, targets, full_probs, person_probs


def _audit_source_run(run: Path, view: str):
    config = _read_json(run / "config.json")
    summary = _read_json(run / "seed_summary.json")
    history = _read_json(run / "training_history.json")
    if summary.get("config") != config or summary.get("status") != "complete":
        raise ValueError("Source run is incomplete or config provenance differs")
    if config.get("reporting_split") != "val" or config.get("input_mode") != view:
        raise ValueError("Learned-gate source must be a matching validation run")
    if config.get("evaluation_score_purpose") != "validation_search":
        raise ValueError("Learned-gate source scores must be validation-only")
    if not math.isclose(
        float(config.get("calibration_fraction", -1)),
        CALIBRATION_FRACTION,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Unexpected calibration fraction")
    if (
        config.get("calibration_split") != "stable_sha256_image_group_v1"
        or config.get("calibration_training_exclusion") is not True
        or config.get("save_calibration_scores") is not True
        or config.get("save_compact_checkpoints") is not True
        or config.get("save_checkpoints") is not False
    ):
        raise ValueError("Missing calibration/checkpoint provenance")
    if view == "person_crop" and config.get("person_transform_mode") != "letterbox":
        raise ValueError("Person source must use letterbox")
    if config.get("git", {}).get("dirty") is not False:
        raise ValueError("Source training worktree was dirty")
    if set(history) != {str(task) for task in range(len(TASK_SIZES))}:
        raise ValueError("Source training history is incomplete")
    if any(
        row["skipped_optimizer_steps"] != 0
        for task_history in history.values()
        for row in task_history
    ):
        raise ValueError("Source training skipped optimizer updates")
    validation_dumps, validation_rows = validated_run_scores(run, "val")
    calibration_records = summary.get("calibration_metrics", [])
    counts = summary.get("calibration_counts", {})
    if len(calibration_records) != len(TASK_SIZES) or set(counts) != {
        str(task) for task in range(len(TASK_SIZES))
    }:
        raise ValueError("Calibration metrics/counts are incomplete")
    calibration_dumps = []
    for task in range(len(TASK_SIZES)):
        dump = load_evaluation_scores(
            run / "calibration_scores" / f"task{task}.npz",
            int(config["eval_batch_size"]),
        )
        if dump.task_id != task or dump.probabilities.shape[1] != sum(TASK_SIZES[: task + 1]):
            raise ValueError("Unexpected calibration score layout")
        row = compute_metrics(
            task,
            torch.from_numpy(dump.probabilities),
            torch.from_numpy(dump.targets),
            THRESHOLD,
        )
        for key, value in asdict(row).items():
            if not np.allclose(
                value, calibration_records[task][key], rtol=0.0, atol=1e-10
            ):
                raise ValueError(f"Calibration score does not reproduce {key}")
        checkpoint = run / "compact_checkpoints" / f"task{task}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        calibration_dumps.append(dump)
    return config, validation_dumps, validation_rows, calibration_dumps


def load_endpoint_pairs(
    full_runs: Sequence[Path], person_runs: Sequence[Path]
) -> Dict[int, EndpointPair]:
    if len(full_runs) != 3 or len(person_runs) != 3:
        raise ValueError("Exactly three Full and three Person source runs are required")
    indexed: Dict[Tuple[str, int], Tuple] = {}
    signatures: Dict[str, Dict[str, Any]] = {}
    for view, runs in (("full", full_runs), ("person_crop", person_runs)):
        for run in runs:
            config, validation, rows, calibration = _audit_source_run(run, view)
            seed = int(config.get("seed", -1))
            if seed not in (0, 1, 2) or (view, seed) in indexed:
                raise ValueError("Source runs need unique seeds 0, 1 and 2")
            fields = [field for field in COMMON_CONFIG_FIELDS if field != "seed"]
            fields.extend(
                (
                    "calibration_fraction",
                    "calibration_split",
                    "person_transform_mode",
                    "person_crop_margin",
                    "train_crop_scale",
                )
            )
            signature = {field: config.get(field) for field in fields}
            if view in signatures and signature != signatures[view]:
                raise ValueError(f"Cross-seed source configuration drift for {view}")
            signatures.setdefault(view, signature)
            indexed[(view, seed)] = (run, config, validation, rows, calibration)
    pairs = {}
    for seed in range(3):
        full_run, full_config, full_val, full_rows, full_cal = indexed[("full", seed)]
        person_run, person_config, person_val, _, person_cal = indexed[("person_crop", seed)]
        for field in COMMON_CONFIG_FIELDS:
            if full_config.get(field) != person_config.get(field):
                raise ValueError(f"Full/Person source configuration drift: {field}")
        validation_tasks = [
            _aligned_arrays(full, person) for full, person in zip(full_val, person_val)
        ]
        calibration_tasks = [
            _aligned_arrays(full, person) for full, person in zip(full_cal, person_cal)
        ]
        pairs[seed] = EndpointPair(
            seed,
            full_run,
            person_run,
            validation_tasks,
            calibration_tasks,
            list(full_rows),
        )
    return pairs


def _fused_metrics(
    task_id: int,
    targets: np.ndarray,
    full_probabilities: np.ndarray,
    person_probabilities: np.ndarray,
    weights: np.ndarray,
) -> TaskMetrics:
    probabilities = (
        (1.0 - weights[:, None]) * full_probabilities
        + weights[:, None] * person_probabilities
    )
    return compute_metrics(
        task_id,
        torch.from_numpy(probabilities),
        torch.from_numpy(targets),
        THRESHOLD,
    )


def _train_one_seed_candidate(
    pair: EndpointPair,
    calibration_geometry: Mapping[str, Geometry],
    validation_geometry: Mapping[str, Geometry],
    hidden_dim: int,
    prior_strength: float,
    state_root: Path,
) -> Dict[str, Any]:
    candidate_id = _candidate_id(hidden_dim, prior_strength)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23000 + hidden_dim)
        gate = ReliabilityGate(hidden_dim).float()
    rows, training = [], []
    candidate_root = state_root / f"seed{pair.seed}" / candidate_id
    candidate_root.mkdir(parents=True, exist_ok=False)
    for task_id in range(len(TASK_SIZES)):
        sample_ids, targets, full_probs, person_probs = pair.calibration_tasks[task_id]
        features = torch.from_numpy(
            gate_features(sample_ids, full_probs, person_probs, calibration_geometry)
        )
        full_tensor = torch.from_numpy(full_probs)
        person_tensor = torch.from_numpy(person_probs)
        targets_tensor = torch.from_numpy(targets)
        current = list(task_indices(task_id))
        optimizer = torch.optim.AdamW(
            gate.parameters(), lr=GATE_LEARNING_RATE, weight_decay=GATE_WEIGHT_DECAY
        )
        generator = torch.Generator().manual_seed(
            31000 + pair.seed * 1000 + hidden_dim * 10 + task_id
        )
        final_loss = None
        for _ in range(GATE_EPOCHS_PER_TASK):
            order = torch.randperm(len(features), generator=generator)
            for start in range(0, len(order), GATE_BATCH_SIZE):
                indices = order[start : start + GATE_BATCH_SIZE]
                weights = gate(features[indices])
                fused = (
                    (1.0 - weights[:, None]) * full_tensor[indices]
                    + weights[:, None] * person_tensor[indices]
                ).clamp(1e-6, 1.0 - 1e-6)
                data_loss = F.binary_cross_entropy(
                    fused[:, current], targets_tensor[indices][:, current]
                )
                prior_loss = torch.mean(
                    ((weights - ANCHOR_PERSON_WEIGHT) / (MAX_PERSON_WEIGHT - MIN_PERSON_WEIGHT)) ** 2
                )
                loss = data_loss + float(prior_strength) * prior_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite reliability gate objective")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                final_loss = float(loss.detach())
        state_path = candidate_root / f"task{task_id}.pth"
        torch.save(
            {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "seed": pair.seed,
                "task_id": task_id,
                "hidden_dim": hidden_dim,
                "prior_strength": prior_strength,
                "model": gate.state_dict(),
            },
            state_path,
        )
        val_ids, val_targets, val_full, val_person = pair.validation_tasks[task_id]
        val_features = torch.from_numpy(
            gate_features(val_ids, val_full, val_person, validation_geometry)
        )
        with torch.no_grad():
            val_weights = gate(val_features).numpy()
        row = _fused_metrics(
            task_id, val_targets, val_full, val_person, val_weights
        )
        rows.append(row)
        with torch.no_grad():
            calibration_weights = gate(features).numpy()
        training.append(
            {
                "task_id": task_id,
                "samples": len(features),
                "final_loss": final_loss,
                "calibration_weight_mean": float(calibration_weights.mean()),
                "calibration_weight_std": float(calibration_weights.std()),
                "calibration_weight_min": float(calibration_weights.min()),
                "calibration_weight_max": float(calibration_weights.max()),
                "diagnostics": {
                    "calibration": gate_diagnostics(
                        targets, full_probs, person_probs, calibration_weights, current
                    ),
                    "validation": gate_diagnostics(
                        val_targets, val_full, val_person, val_weights, current
                    ),
                },
                "state_path": str(state_path.relative_to(state_root.parent)),
                "state_sha256": _sha256(state_path),
            }
        )
    return {
        "seed": pair.seed,
        "metrics": summarize_tasks(rows),
        "task_metrics": [asdict(row) for row in rows],
        "training": training,
    }


def _fixed_anchor(pair: EndpointPair) -> Dict[str, Any]:
    rows = []
    for task_id, (_, targets, full, person) in enumerate(pair.validation_tasks):
        weights = np.full(len(targets), ANCHOR_PERSON_WEIGHT, dtype=np.float32)
        rows.append(_fused_metrics(task_id, targets, full, person, weights))
    return {
        "seed": pair.seed,
        "metrics": summarize_tasks(rows),
        "task_metrics": [asdict(row) for row in rows],
    }


def gate_diagnostics(targets, full, person, weights, current):
    """Reporting only: these values never influence optimization or selection."""
    fused = (1.0 - weights[:, None]) * full + weights[:, None] * person
    anchor = (1.0 - ANCHOR_PERSON_WEIGHT) * full + ANCHOR_PERSON_WEIGHT * person
    target = targets[:, current]
    def metrics(probabilities):
        selected = probabilities[:, current]
        clipped = np.clip(selected, 1e-6, 1.0 - 1e-6)
        return {
            "current_BCE": float(np.mean(-target * np.log(clipped)
                                         - (1.0 - target) * np.log(1.0 - clipped))),
            "current_mAP": float(np.mean([
                100.0 * average_precision(selected[:, j], target[:, j])
                for j in range(len(current))
            ])),
        }
    return {
        "samples": len(targets),
        "current_positive_support": target.sum(axis=0).tolist(),
        "weight_mean": float(weights.mean()),
        "weight_std": float(weights.std()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_quantiles_05_50_95": np.quantile(weights, [0.05, 0.50, 0.95]).tolist(),
        "near_constant_std_lt_0.001": bool(weights.std() < 0.001),
        "mean_absolute_distance_from_0.20": float(np.abs(weights - ANCHOR_PERSON_WEIGHT).mean()),
        "gate": metrics(fused),
        "fixed_0.20": metrics(anchor),
    }


def load_selection_geometry(data_root: Path):
    """Load feature metadata only for the two explicitly allowed selection splits."""
    parent = resolve_dataset_parent(data_root)
    return load_geometry(parent, "train"), load_geometry(parent, "val")


def validate_geometry_coverage(pairs, calibration_geometry, validation_geometry):
    for pair in pairs.values():
        for split, tasks, geometry in (
            ("train", pair.calibration_tasks, calibration_geometry),
            ("val", pair.validation_tasks, validation_geometry),
        ):
            if len(tasks) != len(TASK_SIZES):
                raise ValueError(f"Incomplete {split} task scores")
            for ids, targets, full, person in tasks:
                if any(not str(sample_id).startswith(split + ":") for sample_id in ids):
                    raise ValueError(f"Unexpected sample split in {split} scores")
                # Check all IDs and finite features before creating any gate states.
                gate_features(ids, full, person, geometry)


def select_gate(
    pairs: Mapping[int, EndpointPair],
    calibration_geometry: Mapping[str, Geometry],
    validation_geometry: Mapping[str, Geometry],
    output_dir: Path,
) -> Dict[str, Any]:
    validate_geometry_coverage(pairs, calibration_geometry, validation_geometry)
    state_root = output_dir / "gate_states"
    state_root.mkdir(parents=True, exist_ok=False)
    anchors = [_fixed_anchor(pairs[seed]) for seed in range(3)]
    anchor_by_seed = {
        row["seed"]: row["metrics"]["final_mAP"] for row in anchors
    }
    candidates = []
    for hidden_dim in GATE_HIDDEN_DIMS:
        for prior_strength in PRIOR_STRENGTHS:
            seed_rows = [
                _train_one_seed_candidate(
                    pairs[seed], calibration_geometry, validation_geometry,
                    hidden_dim, prior_strength, state_root
                )
                for seed in range(3)
            ]
            eligible = all(
                row["metrics"]["final_mAP"] >= anchor_by_seed[row["seed"]]
                for row in seed_rows
            )
            candidates.append(
                {
                    "candidate_id": _candidate_id(hidden_dim, prior_strength),
                    "hidden_dim": hidden_dim,
                    "prior_strength": prior_strength,
                    "eligible_each_seed_vs_fixed_0.20": eligible,
                    "seeds": seed_rows,
                    "aggregate": _aggregate(seed_rows),
                }
            )
            print(
                "GATE_CANDIDATE_COMPLETE",
                _candidate_id(hidden_dim, prior_strength),
                [row["metrics"]["final_mAP"] for row in seed_rows],
                f"eligible={eligible}",
                flush=True,
            )
    eligible = [
        candidate
        for candidate in candidates
        if candidate["eligible_each_seed_vs_fixed_0.20"]
        and candidate["aggregate"]["final_mAP"]["mean"]
        > _aggregate(anchors)["final_mAP"]["mean"]
    ]
    winner = max(
        eligible,
        key=lambda row: (
            row["aggregate"]["final_mAP"]["mean"],
            row["aggregate"]["average_mAP"]["mean"],
            -row["hidden_dim"],
            row["prior_strength"],
        ),
        default=None,
    )
    return {
        "schema_version": 1,
        "selection_split": "val",
        "test_accessed": False,
        "geometry_splits": {"calibration": "train", "validation": "val"},
        "diagnostics_affect_selection": False,
        "method": "shared_sample_level_full_person_reliability_gate",
        "feature_names": list(FEATURE_NAMES),
        "person_weight_bounds": [MIN_PERSON_WEIGHT, MAX_PERSON_WEIGHT],
        "anchor_person_weight": ANCHOR_PERSON_WEIGHT,
        "calibration": {
            "fraction": CALIBRATION_FRACTION,
            "source": "stable held-out train samples excluded from base-model fitting",
            "replay": False,
            "loss_scope": "current task classes only",
            "epochs_per_task": GATE_EPOCHS_PER_TASK,
            "batch_size": GATE_BATCH_SIZE,
            "learning_rate": GATE_LEARNING_RATE,
            "weight_decay": GATE_WEIGHT_DECAY,
        },
        "candidate_grid": {
            "hidden_dims": list(GATE_HIDDEN_DIMS),
            "prior_strengths": list(PRIOR_STRENGTHS),
        },
        "selection_rule": (
            "candidate must beat or tie fixed 0.20 final mAP in every seed; "
            "rank by mean final mAP then mean average mAP"
        ),
        "fixed_0.20_anchor": {
            "seeds": anchors,
            "aggregate": _aggregate(anchors),
        },
        "candidates": candidates,
        "winner": winner,
        "advance_to_locked_test": winner is not None,
        "sources": [
            {
                "seed": seed,
                "full_run": str(pairs[seed].full_run.resolve()),
                "person_run": str(pairs[seed].person_run.resolve()),
                "full_config_sha256": _sha256(pairs[seed].full_run / "config.json"),
                "person_config_sha256": _sha256(pairs[seed].person_run / "config.json"),
            }
            for seed in range(3)
        ],
    }


def _load_exported_pair(full_run: Path, person_run: Path, selection_sha256: str):
    configs, dumps, rows = [], [], []
    for run, view in ((full_run, "full"), (person_run, "person_crop")):
        config = _read_json(run / "config.json")
        summary = _read_json(run / "seed_summary.json")
        if summary.get("config") != config or summary.get("status") != "complete":
            raise ValueError("Test score export is incomplete")
        if (
            config.get("reporting_split") != "test"
            or config.get("input_mode") != view
            or config.get("evaluation_score_purpose")
            != "learned_reliability_gate_locked_test"
            or config.get("validation_selection_sha256") != selection_sha256
        ):
            raise ValueError("Test score export provenance mismatch")
        run_dumps = [
            load_evaluation_scores(
                run / "test_scores" / f"task{task}.npz",
                int(config["eval_batch_size"]),
            )
            for task in range(len(TASK_SIZES))
        ]
        run_rows = []
        for task, dump in enumerate(run_dumps):
            row = compute_metrics(
                task,
                torch.from_numpy(dump.probabilities),
                torch.from_numpy(dump.targets),
                THRESHOLD,
            )
            for key, value in asdict(row).items():
                if not np.allclose(value, summary["task_metrics"][task][key], rtol=0, atol=1e-10):
                    raise ValueError(f"Test score export does not reproduce {key}")
            run_rows.append(row)
        configs.append(config)
        dumps.append(run_dumps)
        rows.append(run_rows)
    if configs[0]["seed"] != configs[1]["seed"]:
        raise ValueError("Full/Person test export seed mismatch")
    return configs[0]["seed"], dumps[0], dumps[1], rows[0]


def evaluate_locked_test(
    selection_path: Path,
    full_runs: Sequence[Path],
    person_runs: Sequence[Path],
    data_root: Path,
) -> Dict[str, Any]:
    selection = _read_json(selection_path)
    winner = selection.get("winner")
    if (
        selection.get("selection_split") != "val"
        or selection.get("test_accessed") is not False
        or not selection.get("advance_to_locked_test")
        or not winner
    ):
        raise ValueError("Validation did not lock an eligible learned gate")
    if len(full_runs) != 3 or len(person_runs) != 3:
        raise ValueError("Exactly three Full and three Person test exports are required")
    selection_sha = _sha256(selection_path)
    indexed = {}
    for full_run, person_run in zip(full_runs, person_runs):
        seed, full_dumps, person_dumps, full_rows = _load_exported_pair(
            full_run, person_run, selection_sha
        )
        if seed in indexed:
            raise ValueError("Duplicate test seed")
        indexed[seed] = (full_dumps, person_dumps, full_rows)
    if set(indexed) != {0, 1, 2}:
        raise ValueError("Test exports must cover seeds 0, 1 and 2")
    geometry = load_geometry(resolve_dataset_parent(data_root), "test")
    winner_by_seed = {row["seed"]: row for row in winner["seeds"]}
    seed_rows, full_anchor = [], []
    for seed in range(3):
        full_dumps, person_dumps, full_rows = indexed[seed]
        task_rows = []
        training = winner_by_seed[seed]["training"]
        for task, (full_dump, person_dump) in enumerate(zip(full_dumps, person_dumps)):
            sample_ids, targets, full_probs, person_probs = _aligned_arrays(
                full_dump, person_dump
            )
            features = torch.from_numpy(
                gate_features(sample_ids, full_probs, person_probs, geometry)
            )
            gate = ReliabilityGate(int(winner["hidden_dim"]))
            state_path = selection_path.parent / training[task]["state_path"]
            if _sha256(state_path) != training[task]["state_sha256"]:
                raise ValueError("Selected gate state hash mismatch")
            state = torch.load(state_path, map_location="cpu")
            if (
                state.get("candidate_id") != winner["candidate_id"]
                or state.get("seed") != seed
                or state.get("task_id") != task
            ):
                raise ValueError("Selected gate state provenance mismatch")
            gate.load_state_dict(state["model"], strict=True)
            gate.eval()
            with torch.no_grad():
                weights = gate(features).numpy()
            task_rows.append(
                _fused_metrics(task, targets, full_probs, person_probs, weights)
            )
        seed_rows.append(
            {
                "seed": seed,
                "metrics": summarize_tasks(task_rows),
                "task_metrics": [asdict(row) for row in task_rows],
            }
        )
        full_anchor.append({"seed": seed, "metrics": summarize_tasks(full_rows)})
    return {
        "schema_version": 1,
        "evaluation_split": "test",
        "search_performed_on_test": False,
        "evaluated_gate_count": 1,
        "selection": {
            "path": str(selection_path.resolve()),
            "sha256": selection_sha,
            "candidate_id": winner["candidate_id"],
        },
        "gate": {
            "hidden_dim": winner["hidden_dim"],
            "prior_strength": winner["prior_strength"],
            "person_weight_bounds": [MIN_PERSON_WEIGHT, MAX_PERSON_WEIGHT],
        },
        "seeds": seed_rows,
        "aggregate": _aggregate(seed_rows),
        "full_anchor": {"seeds": full_anchor, "aggregate": _aggregate(full_anchor)},
        "paired_final_mAP_gain": _stats(
            [
                seed_rows[seed]["metrics"]["final_mAP"]
                - full_anchor[seed]["metrics"]["final_mAP"]
                for seed in range(3)
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--full-runs", type=Path, nargs=3, required=True)
    select.add_argument("--person-runs", type=Path, nargs=3, required=True)
    select.add_argument("--data-root", type=Path, required=True)
    select.add_argument("--output-dir", type=Path, required=True)
    test = subparsers.add_parser("evaluate-test")
    test.add_argument("--selection", type=Path, required=True)
    test.add_argument("--full-runs", type=Path, nargs=3, required=True)
    test.add_argument("--person-runs", type=Path, nargs=3, required=True)
    test.add_argument("--data-root", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "select":
        if args.output_dir.exists():
            raise FileExistsError(args.output_dir)
        args.output_dir.mkdir(parents=True)
        pairs = load_endpoint_pairs(args.full_runs, args.person_runs)
        calibration_geometry, validation_geometry = load_selection_geometry(args.data_root)
        result = select_gate(pairs, calibration_geometry, validation_geometry, args.output_dir)
        output = args.output_dir / "validation_selection.json"
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            "LEARNED_GATE_VALIDATION_COMPLETE",
            json.dumps(
                {
                    "advance": result["advance_to_locked_test"],
                    "winner": result["winner"]["candidate_id"] if result["winner"] else None,
                }
            ),
            flush=True,
        )
        return

    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate_locked_test(
        args.selection, args.full_runs, args.person_runs, args.data_root
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result["aggregate"], indent=2), flush=True)
    print("LEARNED_GATE_LOCKED_TEST_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
