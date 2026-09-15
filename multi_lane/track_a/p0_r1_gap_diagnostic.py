"""Diagnose the seed0 validation gap between shared P0 and independent R1."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from .evaluation_scores import EvaluationScores, load_evaluation_scores
from .fuse_face_endpoint_validation import load_face_reliability
from .fuse_validation_scores import validated_run_scores
from .runner import (
    CLASS_ORDER,
    TASK_SIZES,
    average_precision,
    compute_metrics,
    summarize_tasks,
    task_indices,
)


VIEWS = ("full", "person", "face")
RELIABLE_WEIGHTS = np.asarray((0.64, 0.16, 0.20), dtype=np.float32)
FALLBACK_WEIGHTS = np.asarray((0.80, 0.20, 0.00), dtype=np.float32)
BOOTSTRAP_SEED = 20260915
HISTORICAL_I_PROB_FINAL_MAP = 43.581193
HISTORICAL_P0_FINAL_MAP = 43.075407


@dataclass(frozen=True)
class ViewScores:
    task_id: int
    sample_ids: np.ndarray
    targets: np.ndarray
    face_reliable: np.ndarray
    fused_logits: np.ndarray
    fused_probabilities: np.ndarray
    logits: Mapping[str, np.ndarray]
    probabilities: Mapping[str, np.ndarray]


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_view_scores(path: Path) -> ViewScores:
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema_version"]) != 1:
            raise ValueError("Unsupported view score schema")
        task_id = int(data["task_id"])
        ids = data["sample_ids"].astype(str)
        targets = data["targets"].astype(np.float32)
        reliable = data["face_reliable"].astype(np.bool_)
        classes = data["class_indices"]
        lengths = data["batch_lengths"]
        fused_logits = data["fused_logits"].astype(np.float32)
        fused_probabilities = data["fused_probabilities"].astype(np.float32)
        logits = {name: data[f"{name}_logits"].astype(np.float32) for name in VIEWS}
        probabilities = {
            name: data[f"{name}_probabilities"].astype(np.float32) for name in VIEWS
        }
        provenance = (
            str(data["probability_device"]),
            str(data["probability_dtype"]),
            str(data["probability_operation"]),
            str(data["torch_version"]),
        )
    expected = targets.shape
    arrays = [fused_logits, fused_probabilities, *logits.values(), *probabilities.values()]
    if targets.ndim != 2 or not len(targets) or any(value.shape != expected for value in arrays):
        raise ValueError("Invalid view score shapes")
    if ids.shape != (len(targets),) or len(set(ids.tolist())) != len(ids):
        raise ValueError("Invalid or duplicate view score IDs")
    if reliable.shape != (len(targets),) or reliable.dtype != np.bool_:
        raise ValueError("Invalid Face reliability mask")
    if not np.array_equal(classes, np.arange(targets.shape[1])):
        raise ValueError("Unexpected view score class indices")
    if (lengths.ndim != 1 or not len(lengths) or (lengths <= 0).any()
            or int(lengths.sum()) != len(targets)):
        raise ValueError("Invalid view score batch lengths")
    if provenance[:3] != ("cpu", "float32", "torch.sigmoid") or not provenance[3]:
        raise ValueError("Unsupported view probability provenance")
    if not np.isin(targets, (0, 1)).all() or not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Invalid view scores or targets")
    if any((value < 0).any() or (value > 1).any() for value in [fused_probabilities, *probabilities.values()]):
        raise ValueError("View probabilities are outside [0, 1]")
    return ViewScores(
        task_id, ids, targets, reliable, fused_logits, fused_probabilities,
        logits, probabilities,
    )


def _align(reference_ids: np.ndarray, reference_targets: np.ndarray, dump: EvaluationScores):
    if set(reference_ids.tolist()) != set(dump.sample_ids.tolist()):
        raise ValueError("Independent and shared sources contain different samples")
    positions = {sample_id: index for index, sample_id in enumerate(dump.sample_ids)}
    order = np.asarray([positions[sample_id] for sample_id in reference_ids], dtype=np.int64)
    if not np.array_equal(reference_targets, dump.targets[order]):
        raise ValueError("Independent and shared targets differ after alignment")
    return dump.logits[order], dump.probabilities[order]


def _weights(reliable: np.ndarray) -> np.ndarray:
    weights = np.tile(FALLBACK_WEIGHTS, (len(reliable), 1))
    weights[reliable] = RELIABLE_WEIGHTS
    return weights


def fuse_probabilities(endpoints: Mapping[str, np.ndarray], reliable: np.ndarray) -> np.ndarray:
    if set(endpoints) != set(VIEWS):
        raise ValueError("Probability fusion requires exactly Full/Person/Face")
    weights = _weights(reliable)
    return sum(weights[:, index, None] * endpoints[name] for index, name in enumerate(VIEWS))


def fuse_logits(endpoints: Mapping[str, np.ndarray], reliable: np.ndarray) -> np.ndarray:
    if set(endpoints) != set(VIEWS):
        raise ValueError("Logit fusion requires exactly Full/Person/Face")
    weights = _weights(reliable)
    combined = sum(weights[:, index, None] * endpoints[name] for index, name in enumerate(VIEWS))
    return torch.sigmoid(torch.from_numpy(combined)).numpy()


def _rows(task_scores: Sequence[Tuple[np.ndarray, np.ndarray]]):
    return [
        compute_metrics(task_id, torch.from_numpy(scores), torch.from_numpy(targets), 0.5)
        for task_id, (scores, targets) in enumerate(task_scores)
    ]


def _record(task_scores: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    rows = _rows(task_scores)
    group_metrics = []
    for task_id, (scores, targets) in enumerate(task_scores):
        current = np.asarray(task_indices(task_id), dtype=np.int64)
        old = np.arange(current[0], dtype=np.int64)
        current_row = compute_metrics(
            task_id, torch.from_numpy(scores[:, current]), torch.from_numpy(targets[:, current]), 0.5
        )
        group_metrics.append({
            "task_id": task_id,
            "current_classes_mAP": current_row.mAP,
            "old_classes_mAP": (
                compute_metrics(
                    task_id, torch.from_numpy(scores[:, old]), torch.from_numpy(targets[:, old]), 0.5
                ).mAP if len(old) else None
            ),
        })
    return {
        "metrics": summarize_tasks(rows),
        "task_metrics": [asdict(row) for row in rows],
        "current_old_class_metrics": group_metrics,
    }


def _subset_metrics(task_id: int, scores: np.ndarray, targets: np.ndarray, mask: np.ndarray):
    if not mask.any():
        return None
    return asdict(compute_metrics(
        task_id, torch.from_numpy(scores[mask]), torch.from_numpy(targets[mask]), 0.5
    ))


def _rank_flips(reference: np.ndarray, candidate: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    corrected_total = damaged_total = pair_total = 0
    rows = []
    for class_id in range(targets.shape[1]):
        positive = targets[:, class_id] > 0.5
        negative = ~positive
        ref_margin = reference[positive, class_id, None] - reference[negative, class_id][None, :]
        new_margin = candidate[positive, class_id, None] - candidate[negative, class_id][None, :]
        corrected = int(np.logical_and(ref_margin <= 0, new_margin > 0).sum())
        damaged = int(np.logical_and(ref_margin > 0, new_margin <= 0).sum())
        pairs = int(ref_margin.size)
        corrected_total += corrected
        damaged_total += damaged
        pair_total += pairs
        rows.append({
            "class_index": class_id,
            "class_name": CLASS_ORDER[class_id],
            "pairs": pairs,
            "corrected": corrected,
            "damaged": damaged,
            "net": corrected - damaged,
        })
    return {
        "pairs": pair_total,
        "corrected": corrected_total,
        "damaged": damaged_total,
        "net": corrected_total - damaged_total,
        "per_class": rows,
    }


def _distribution_diagnostics(endpoints: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"views": {}, "pairwise_mean_absolute_disagreement": {}}
    for name, probabilities in endpoints.items():
        result["views"][name] = {
            "mean": float(probabilities.mean()),
            "std": float(probabilities.std()),
            "mean_absolute_margin_from_0.5": float(np.abs(probabilities - 0.5).mean()),
            "fraction_below_0.01": float((probabilities < 0.01).mean()),
            "fraction_above_0.99": float((probabilities > 0.99).mean()),
        }
    for left, right in itertools.combinations(VIEWS, 2):
        result["pairwise_mean_absolute_disagreement"][f"{left}_{right}"] = float(
            np.abs(endpoints[left] - endpoints[right]).mean()
        )
    return result


def _bootstrap_final_map_differences(
    sample_ids: np.ndarray,
    targets: np.ndarray,
    methods: Mapping[str, np.ndarray],
    contrasts: Sequence[Tuple[str, str]],
    replicates: int,
) -> Dict[str, Any]:
    if replicates <= 0:
        raise ValueError("Bootstrap replicates must be positive")
    group_names = np.asarray([sample_id.rsplit("#person=", 1)[0] for sample_id in sample_ids])
    unique_groups = np.unique(group_names)
    group_rows = {name: np.flatnonzero(group_names == name) for name in unique_groups}
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    distributions = {f"{left}_minus_{right}": [] for left, right in contrasts}
    required_methods = sorted({name for contrast in contrasts for name in contrast})
    for _ in range(replicates):
        sampled = generator.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_rows[name] for name in sampled])
        maps = {
            name: float(np.mean([
                100.0 * average_precision(scores[indices, class_id], targets[indices, class_id])
                for class_id in range(targets.shape[1])
            ]))
            for name, scores in methods.items() if name in required_methods
        }
        for left, right in contrasts:
            distributions[f"{left}_minus_{right}"].append(maps[left] - maps[right])
    result = {}
    for name, values in distributions.items():
        array = np.asarray(values, dtype=np.float64)
        result[name] = {
            "mean": float(array.mean()),
            "ci95_percentile": [float(value) for value in np.percentile(array, (2.5, 97.5))],
            "positive_fraction": float((array > 0).mean()),
        }
    return {
        "unit": "original_image_group",
        "group_count": int(len(unique_groups)),
        "replicates": replicates,
        "seed": BOOTSTRAP_SEED,
        "contrasts": result,
    }


def _audit_shared_run(shared_run: Path) -> Tuple[List[EvaluationScores], List[ViewScores], Dict[str, Any]]:
    config = _read_json(shared_run / "config.json")
    summary = _read_json(shared_run / "seed_summary.json")
    expected = {
        "protocol_id": "emotic_b5c3_v0.1",
        "seed": 0,
        "reporting_split": "val",
        "max_tasks": len(TASK_SIZES),
        "training_budget_mode": "epochs",
        "epochs_per_task": 30,
        "train_batch_size": 64,
        "eval_batch_size": 64,
        "workers": 2,
        "threshold": 0.5,
        "training_loss_mode": "legacy_full_zero",
        "parameter_group_loss_routing": "adapter_asl",
        "model_parameter_objective": "bce",
        "adapter_parameter_objective": "asl",
        "learning_rate": 0.0125,
        "optimizer": "Adam_reset_per_task",
        "weight_decay": 0.0,
        "scheduler_mode": "cosine",
        "scheduler_min_lr_ratio": 0.0,
        "scheduler_warmup_ratio": 0.0,
        "input_mode": "full",
        "input_normalization": "clip",
        "full_crop_mode": "legacy",
        "person_crop_margin": 0.15,
        "person_transform_mode": "letterbox",
        "save_evaluation_scores": True,
        "save_view_evaluation_scores": True,
        "save_compact_checkpoints": True,
        "view_fusion": "fixed_three_view",
        "view_auxiliary_loss_weight": 0.1,
        "view_gradient_routing": "joint",
        "adapter_mode": "image_token",
        "adapter_bottleneck_dim": 32,
        "adapter_view_bottleneck_dim": 0,
        "adapter_layer_indices": [1],
        "adapter_residual_scale": 0.1,
        "adapter_residual_gate_mode": "fixed",
        "adapter_activation": "relu",
        "adapter_task_initialization": "independent",
        "adapter_learning_rate": 0.0004,
        "adapter_weight_decay": 0.0,
        "adapter_regularization": "none",
        "amp": True,
        "tf32": True,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"Shared P0 diagnostic config differs on {field}")
    asl = config.get("asl", {})
    if any(not math.isclose(float(asl.get(field, float("nan"))), value, rel_tol=0, abs_tol=1e-12)
           for field, value in {"gamma_neg": 9.8, "gamma_pos": 0.0, "clip": 0.05, "eps": 1e-8}.items()):
        raise ValueError("Shared P0 diagnostic ASL config differs")
    if summary.get("status") != "complete" or summary.get("config") != config:
        raise ValueError("Shared P0 diagnostic run is incomplete")
    if summary.get("completed_epochs") != 240:
        raise ValueError("Shared P0 diagnostic did not complete 240 epochs")
    history = _read_json(shared_run / "training_history.json")
    if (len(history) != len(TASK_SIZES)
            or any(len(history.get(str(task), ())) != 30 for task in range(len(TASK_SIZES)))
            or sum(row["skipped_optimizer_steps"] for rows in history.values() for row in rows) != 0):
        raise ValueError("Shared P0 training history is incomplete or contains skipped updates")
    standard, _ = validated_run_scores(shared_run, "val")
    views = [load_view_scores(shared_run / "view_val_scores" / f"task{task}.npz")
             for task in range(len(TASK_SIZES))]
    checks = []
    for task_id, (ordinary, view) in enumerate(zip(standard, views)):
        if view.task_id != task_id or not np.array_equal(ordinary.sample_ids, view.sample_ids):
            raise ValueError("Shared fused/view score ID order differs")
        if not np.array_equal(ordinary.targets, view.targets):
            raise ValueError("Shared fused/view targets differ")
        fused_logit_error = float(np.max(np.abs(ordinary.logits - view.fused_logits)))
        fused_probability_error = float(np.max(np.abs(
            ordinary.probabilities - view.fused_probabilities
        )))
        algebra = sum(
            _weights(view.face_reliable)[:, index, None] * view.logits[name]
            for index, name in enumerate(VIEWS)
        )
        algebra_error = float(np.max(np.abs(algebra - view.fused_logits)))
        if fused_logit_error > 5e-4 or fused_probability_error > 1e-6 or algebra_error > 5e-3:
            raise ValueError("Shared P0 score consistency check failed")
        manifest_mask = load_face_reliability(
            Path(config["face_manifest"]["root"]), "val"
        )
        expected_mask = np.asarray([manifest_mask[sample_id] for sample_id in view.sample_ids])
        if not np.array_equal(expected_mask, view.face_reliable):
            raise ValueError("Exported Face mask differs from audited manifest")
        checks.append({
            "task_id": task_id,
            "ordinary_vs_diagnostic_max_abs_logit": fused_logit_error,
            "ordinary_vs_diagnostic_max_abs_probability": fused_probability_error,
            "weighted_branch_vs_fused_max_abs_logit": algebra_error,
        })
    compact = shared_run / "compact_checkpoints"
    compact_files = [compact / f"task{task}.pth" for task in range(len(TASK_SIZES))]
    if any(not path.is_file() for path in compact_files):
        raise FileNotFoundError("Shared P0 compact checkpoint series is incomplete")
    provenance = {
        "path": str(shared_run.resolve()),
        "config_sha256": _sha256(shared_run / "config.json"),
        "seed_summary_sha256": _sha256(shared_run / "seed_summary.json"),
        "compact_checkpoint_sha256": [_sha256(path) for path in compact_files],
        "score_consistency": checks,
    }
    return standard, views, provenance


def _audit_independent_run(run: Path, expected_input: str) -> List[EvaluationScores]:
    config = _read_json(run / "config.json")
    if int(config.get("seed", -1)) != 0 or config.get("input_mode") != expected_input:
        raise ValueError(f"Independent {expected_input} source protocol differs")
    if expected_input == "person_crop" and config.get("person_transform_mode") != "letterbox":
        raise ValueError("Independent Person source is not letterbox")
    return validated_run_scores(run, "val")[0]


def diagnose_gap(
    shared_run: Path,
    independent_runs: Mapping[str, Path],
    face_manifest_root: Path,
    bootstrap_replicates: int = 2000,
) -> Dict[str, Any]:
    if set(independent_runs) != set(VIEWS):
        raise ValueError("Exactly three independent source runs are required")
    standard, shared_dumps, shared_provenance = _audit_shared_run(shared_run)
    independent_dumps = {
        "full": _audit_independent_run(independent_runs["full"], "full"),
        "person": _audit_independent_run(independent_runs["person"], "person_crop"),
        "face": _audit_independent_run(independent_runs["face"], "face_crop"),
    }
    manifest = load_face_reliability(face_manifest_root, "val")
    face_config = _read_json(independent_runs["face"] / "config.json")
    configured_manifest = face_config.get("face_manifest", {})
    if Path(configured_manifest.get("root", "")).resolve() != face_manifest_root.resolve():
        raise ValueError("Independent Face source uses a different manifest root")
    task_data = []
    for task_id, shared in enumerate(shared_dumps):
        expected_mask = np.asarray([manifest[sample_id] for sample_id in shared.sample_ids], dtype=np.bool_)
        if not np.array_equal(expected_mask, shared.face_reliable):
            raise ValueError("Supplied Face manifest differs from shared score mask")
        independent = {}
        for name in VIEWS:
            logits, probabilities = _align(
                shared.sample_ids, shared.targets, independent_dumps[name][task_id]
            )
            independent[name] = {"logits": logits, "probabilities": probabilities}
        task_data.append({"shared": shared, "independent": independent})

    cells: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
        "I_logit": [], "I_probability": [], "S_logit": [], "S_probability": []
    }
    combinations: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {
        "".join(source): [] for source in itertools.product("IS", repeat=3)
    }
    for data in task_data:
        shared = data["shared"]
        independent = data["independent"]
        i_logits = {name: independent[name]["logits"] for name in VIEWS}
        i_probs = {name: independent[name]["probabilities"] for name in VIEWS}
        cells["I_logit"].append((fuse_logits(i_logits, shared.face_reliable), shared.targets))
        cells["I_probability"].append((fuse_probabilities(i_probs, shared.face_reliable), shared.targets))
        cells["S_logit"].append((shared.fused_probabilities, shared.targets))
        cells["S_probability"].append((fuse_probabilities(
            shared.probabilities, shared.face_reliable
        ), shared.targets))
        for code in combinations:
            endpoints = {
                name: (independent[name]["probabilities"] if source == "I"
                       else shared.probabilities[name])
                for name, source in zip(VIEWS, code)
            }
            combinations[code].append((
                fuse_probabilities(endpoints, shared.face_reliable), shared.targets
            ))

    cell_records = {name: _record(values) for name, values in cells.items()}
    combination_records = {name: _record(values) for name, values in combinations.items()}
    maps = {name: row["metrics"]["final_mAP"] for name, row in cell_records.items()}
    observed = maps["I_probability"] - maps["S_logit"]
    remaining = maps["I_probability"] - maps["S_probability"]
    operation = maps["S_probability"] - maps["S_logit"]

    marginal = {}
    shapley = {}
    for index, view in enumerate(VIEWS):
        context_rows = []
        contribution = 0.0
        for other in itertools.product("IS", repeat=2):
            base = list(other)
            base.insert(index, "S")
            changed = list(base)
            changed[index] = "I"
            base_code, changed_code = "".join(base), "".join(changed)
            delta = (
                combination_records[changed_code]["metrics"]["final_mAP"]
                - combination_records[base_code]["metrics"]["final_mAP"]
            )
            independent_count = sum(value == "I" for position, value in enumerate(base) if position != index)
            coefficient = 1.0 / 3.0 if independent_count in (0, 2) else 1.0 / 6.0
            contribution += coefficient * delta
            context_rows.append({
                "from": base_code, "to": changed_code,
                "final_mAP_delta": delta,
                "average_mAP_delta": (
                    combination_records[changed_code]["metrics"]["average_mAP"]
                    - combination_records[base_code]["metrics"]["average_mAP"]
                ),
            })
        marginal[view] = context_rows
        shapley[view] = contribution

    final = task_data[-1]
    final_shared = final["shared"]
    final_targets = final_shared.targets
    final_methods = {name: values[-1][0] for name, values in cells.items()}
    final_methods.update({f"B_{name}": values[-1][0] for name, values in combinations.items()})
    subgroup = {}
    for name in ("I_probability", "S_logit", "S_probability", "B_III", "B_SSS"):
        subgroup[name] = {
            "reliable_face": _subset_metrics(
                len(TASK_SIZES) - 1, final_methods[name], final_targets, final_shared.face_reliable
            ),
            "unreliable_face": _subset_metrics(
                len(TASK_SIZES) - 1, final_methods[name], final_targets, ~final_shared.face_reliable
            ),
        }
    rank_swaps = {}
    for index, view in enumerate(VIEWS):
        code = list("SSS")
        code[index] = "I"
        rank_swaps[f"SSS_to_{''.join(code)}_{view}"] = _rank_flips(
            final_methods["B_SSS"], final_methods[f"B_{''.join(code)}"], final_targets
        )
    standalone = {"independent": {}, "shared": {}}
    for name in VIEWS:
        standalone["independent"][name] = _record([
            (data["independent"][name]["probabilities"], data["shared"].targets)
            for data in task_data
        ])
        standalone["shared"][name] = _record([
            (data["shared"].probabilities[name], data["shared"].targets)
            for data in task_data
        ])

    bootstrap = _bootstrap_final_map_differences(
        final_shared.sample_ids,
        final_targets,
        final_methods,
        (
            ("I_probability", "S_logit"),
            ("I_probability", "S_probability"),
            ("S_probability", "S_logit"),
            ("B_III", "B_SSS"),
        ),
        bootstrap_replicates,
    )
    return {
        "schema_version": 1,
        "experiment": "P0_reproduction_plus_A_fusion_operator_plus_B_endpoint_source",
        "selection_split": "val",
        "seed": 0,
        "test_accessed": False,
        "search_performed": False,
        "fixed_rules": {
            "reliable_weights": RELIABLE_WEIGHTS.tolist(),
            "fallback_weights": FALLBACK_WEIGHTS.tolist(),
            "threshold": 0.5,
        },
        "sources": {
            "shared_P0": shared_provenance,
            "independent": {
                name: {"path": str(path.resolve()), "config_sha256": _sha256(path / "config.json")}
                for name, path in independent_runs.items()
            },
            "face_manifest_root": str(face_manifest_root.resolve()),
        },
        "historical_anchor": {
            "I_probability_final_mAP": HISTORICAL_I_PROB_FINAL_MAP,
            "P0_final_mAP": HISTORICAL_P0_FINAL_MAP,
            "gap": HISTORICAL_I_PROB_FINAL_MAP - HISTORICAL_P0_FINAL_MAP,
        },
        "P0_reproduction": {
            "recorded_fused": _record([
                (dump.probabilities, dump.targets) for dump in standard
            ]),
            "historical_minus_reproduced_final_mAP": (
                HISTORICAL_P0_FINAL_MAP - maps["S_logit"]
            ),
        },
        "experiment_A": {
            "cells": cell_records,
            "reproduction_gap_decomposition": {
                "I_probability_minus_S_logit": observed,
                "model_source_at_probability_I_minus_S": remaining,
                "fusion_operator_on_shared_S_probability_minus_logit": operation,
                "sum_check": remaining + operation,
            },
        },
        "experiment_B": {
            "source_code_order": list(VIEWS),
            "source_code_meaning": {"I": "independent_expert", "S": "shared_P0_branch"},
            "combinations": combination_records,
            "marginal_replacements": marginal,
            "shapley_final_mAP_I_over_S": shapley,
            "shapley_sum": float(sum(shapley.values())),
            "endpoint_III_minus_SSS": (
                combination_records["III"]["metrics"]["final_mAP"]
                - combination_records["SSS"]["metrics"]["final_mAP"]
            ),
        },
        "why_standalone_does_not_imply_fusion_gain": {
            "standalone": standalone,
            "final_task_probability_distribution": {
                "independent": _distribution_diagnostics({
                    name: final["independent"][name]["probabilities"] for name in VIEWS
                }),
                "shared": _distribution_diagnostics(final_shared.probabilities),
            },
            "final_task_rank_flips_when_replacing_one_shared_branch_with_independent": rank_swaps,
            "final_task_face_subgroups": subgroup,
        },
        "paired_image_group_bootstrap": bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-run", type=Path, required=True)
    parser.add_argument("--independent-full-run", type=Path, required=True)
    parser.add_argument("--independent-person-run", type=Path, required=True)
    parser.add_argument("--independent-face-run", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose_gap(
        args.shared_run,
        {
            "full": args.independent_full_run,
            "person": args.independent_person_run,
            "face": args.independent_face_run,
        },
        args.face_manifest_root,
        args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "P0_reproduced_final_mAP": result["experiment_A"]["cells"]["S_logit"]["metrics"]["final_mAP"],
        "I_probability_final_mAP": result["experiment_A"]["cells"]["I_probability"]["metrics"]["final_mAP"],
        "decomposition": result["experiment_A"]["reproduction_gap_decomposition"],
        "shapley": result["experiment_B"]["shapley_final_mAP_I_over_S"],
    }, indent=2), flush=True)
    print("P0_R1_GAP_DIAGNOSTIC_COMPLETE test_accessed=false search=false", flush=True)


if __name__ == "__main__":
    main()
