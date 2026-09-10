"""Validation-only hierarchical class-aware stacking from three-view OOF scores."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .runner import (
    CLASS_ORDER,
    TASK_SIZES,
    compute_metrics,
    resolve_dataset_parent,
    summarize_tasks,
    task_indices,
)
from .search_constrained_gated_fusion import Geometry, load_geometry
from .three_view_oof_router import (
    _raw_features,
    load_oof_descriptors,
    load_oof_tasks,
    load_validation_tasks,
)
from .three_view_router import (
    INVALID_PRIOR,
    R2_FEATURE_NAMES,
    VALID_PRIOR,
    _fixed_rows,
    _sha256,
    _standardize_fit_apply,
    load_face_metadata,
)
from .export_three_view_descriptors import FEATURE_NAMES as VISUAL_FEATURE_NAMES


RANK = 2
EPOCHS_PER_TASK = 120
BATCH_SIZE = 512
LEARNING_RATE = 1e-2
PARAMETER_REGULARIZATION = 1e-4
BIAS_PRIOR_STRENGTHS = (1.0, 3.0, 10.0)
INTERACTION_PRIOR_STRENGTHS = (1.0, 3.0, 10.0)
ADVANCE_MARGIN_MAP = 0.05
POSITIVE_CLASS_EPS = 0.01
MAX_SINGLE_CLASS_POSITIVE_SHARE = 0.80
THRESHOLD = 0.5


def _named_final_ap_gains(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> Dict[str, float]:
    candidate_ap = candidate["task_metrics"][-1]["per_class_ap"]
    reference_ap = reference["task_metrics"][-1]["per_class_ap"]
    if len(candidate_ap) != len(CLASS_ORDER) or len(reference_ap) != len(CLASS_ORDER):
        raise ValueError("Final per-class AP does not match the protocol class order")
    return {
        name: float(candidate_ap[index] - reference_ap[index])
        for index, name in enumerate(CLASS_ORDER)
    }


class ClassAwareStacker(nn.Module):
    """R1-anchored class bias with an optional rank-k sample interaction."""

    def __init__(
        self,
        classes: int,
        feature_dim: int = 0,
        rank: int = 0,
        initialization_seed: int = 0,
    ) -> None:
        super().__init__()
        if classes <= 0 or feature_dim < 0 or rank < 0:
            raise ValueError("Invalid class-aware stacker dimensions")
        if (rank == 0) != (feature_dim == 0):
            raise ValueError("Features and rank must either both be enabled or both be disabled")
        self.classes = int(classes)
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.class_bias = nn.Parameter(torch.zeros(classes, 3))
        if rank:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(initialization_seed))
                self.sample_projection = nn.Linear(feature_dim, rank)
                nn.init.xavier_uniform_(self.sample_projection.weight)
                nn.init.zeros_(self.sample_projection.bias)
            self.class_interaction = nn.Parameter(torch.zeros(classes, rank, 3))
        else:
            self.sample_projection = None
            self.register_parameter("class_interaction", None)

    @staticmethod
    def _center(values: torch.Tensor) -> torch.Tensor:
        return values - values.mean(dim=-1, keepdim=True)

    def forward(
        self,
        features: Optional[torch.Tensor],
        class_ids: torch.Tensor,
        face_reliable: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if class_ids.ndim != 1 or class_ids.dtype != torch.long:
            raise ValueError("class_ids must be a one-dimensional long tensor")
        if face_reliable.ndim != 1:
            raise ValueError("face_reliable must be one-dimensional")
        samples = len(face_reliable)
        if class_ids.numel() == 0 or int(class_ids.min()) < 0 or int(class_ids.max()) >= self.classes:
            raise ValueError("Invalid class_ids")
        bias = self._center(self.class_bias[class_ids])
        if self.rank:
            if features is None or features.shape != (samples, self.feature_dim):
                raise ValueError("Invalid sample reliability features")
            sample_factors = torch.tanh(self.sample_projection(features))
            interaction = torch.einsum(
                "nr,crv->ncv", sample_factors, self.class_interaction[class_ids]
            )
            interaction = self._center(interaction)
        else:
            if features is not None and features.shape != (samples, 0):
                raise ValueError("Bias-only stacker does not accept reliability features")
            interaction = self.class_bias.new_zeros((samples, len(class_ids), 3))
        base = torch.log(torch.as_tensor(
            VALID_PRIOR, dtype=self.class_bias.dtype, device=self.class_bias.device
        ))
        logits = base[None, None, :] + bias[None, :, :] + interaction
        invalid_face = (~face_reliable.bool())[:, None]
        logits[:, :, 2] = logits[:, :, 2].masked_fill(
            invalid_face, torch.finfo(logits.dtype).min
        )
        weights = torch.softmax(logits, dim=-1)
        if not torch.isfinite(weights).all():
            raise FloatingPointError("Non-finite class-aware stacking weights")
        return weights, interaction


def _cpu_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _fuse_classwise(probabilities: Sequence[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    if len(probabilities) != 3 or weights.ndim != 3 or weights.shape[-1] != 3:
        raise ValueError("Invalid classwise fusion inputs")
    return sum(weights[:, :, view] * probabilities[view] for view in range(3)).clamp(
        1e-6, 1.0 - 1e-6
    )


def _prepare_tasks(
    tasks: Sequence[Tuple[np.ndarray, ...]],
    geometry: Mapping[str, Geometry],
    face: Mapping[str, Any],
    descriptors: Optional[Mapping[str, Tuple[np.ndarray, np.bool_]]],
) -> List[Dict[str, Any]]:
    prepared = []
    for task_id, task in enumerate(tasks):
        raw, reliable = _raw_features(task, geometry, face, descriptors)
        prepared.append({
            "raw": raw.astype(np.float32),
            "reliable": reliable.astype(np.bool_),
            "probabilities": tuple(value.astype(np.float32) for value in task[2:5]),
            "targets": task[1].astype(np.float32),
            "current": list(task_indices(task_id)),
        })
    return prepared


def _batch_order(samples: int, generator: torch.Generator) -> Sequence[torch.Tensor]:
    order = torch.randperm(samples, generator=generator)
    return [order[start : start + BATCH_SIZE] for start in range(0, samples, BATCH_SIZE)]


def _train_c1(
    prior_strength: float,
    prepared: Sequence[Mapping[str, Any]],
    device: torch.device,
    state_root: Path,
) -> Dict[str, Any]:
    candidate_id = f"C1_bias_prior{str(prior_strength).replace('.', 'p')}"
    candidate_root = state_root / candidate_id
    candidate_root.mkdir(parents=True, exist_ok=False)
    model = ClassAwareStacker(len(CLASS_ORDER)).float().to(device)
    generator = torch.Generator().manual_seed(61000)
    states, records = [], []
    for task_id, item in enumerate(prepared):
        current = torch.tensor(item["current"], dtype=torch.long, device=device)
        reliable = torch.from_numpy(item["reliable"]).to(device)
        targets = torch.from_numpy(item["targets"]).to(device)[:, current]
        probabilities = tuple(
            torch.from_numpy(value).to(device)[:, current] for value in item["probabilities"]
        )
        optimizer = torch.optim.Adam([model.class_bias], lr=LEARNING_RATE)
        final = None
        for _ in range(EPOCHS_PER_TASK):
            for cpu_indices in _batch_order(len(reliable), generator):
                indices = cpu_indices.to(device)
                weights, _ = model(None, current, reliable[indices])
                fused = _fuse_classwise(
                    tuple(value[indices] for value in probabilities), weights
                )
                data_loss = F.binary_cross_entropy(fused, targets[indices])
                centered = ClassAwareStacker._center(model.class_bias[current])
                prior_loss = centered.square().mean()
                objective = data_loss + float(prior_strength) * prior_loss
                if not torch.isfinite(objective):
                    raise FloatingPointError("Non-finite C1 objective")
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                if model.class_bias.grad is not None:
                    mask = torch.zeros_like(model.class_bias.grad)
                    mask[current] = 1
                    model.class_bias.grad.mul_(mask)
                optimizer.step()
                final = (float(data_loss.detach()), float(prior_loss.detach()))
        state = {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "task_id": task_id,
            "rank": 0,
            "feature_names": [],
            "normalization": None,
            "model": _cpu_state(model),
        }
        path = candidate_root / f"task{task_id}.pth"
        torch.save(state, path)
        states.append(state)
        records.append({
            "task_id": task_id,
            "oof_samples": len(reliable),
            "current_classes": [CLASS_ORDER[index] for index in item["current"]],
            "final_data_loss": final[0],
            "final_prior_loss": final[1],
            "state_path": str(path.relative_to(state_root.parent)),
            "state_sha256": _sha256(path),
        })
    return {
        "candidate_id": candidate_id,
        "family": "C1",
        "bias_prior_strength": prior_strength,
        "states": states,
        "tasks": records,
    }


def _train_c2(
    interaction_strength: float,
    c1_states: Sequence[Mapping[str, Any]],
    prepared: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    device: torch.device,
    state_root: Path,
) -> Dict[str, Any]:
    candidate_id = f"C2_rank{RANK}_prior{str(interaction_strength).replace('.', 'p')}"
    candidate_root = state_root / candidate_id
    candidate_root.mkdir(parents=True, exist_ok=False)
    model = ClassAwareStacker(
        len(CLASS_ORDER), len(feature_names), RANK, initialization_seed=62000
    ).float().to(device)
    with torch.no_grad():
        model.class_bias.copy_(c1_states[-1]["model"]["class_bias"].to(device))
    model.class_bias.requires_grad_(False)
    generator = torch.Generator().manual_seed(63000)
    states, records = [], []
    for task_id in range(len(TASK_SIZES)):
        accumulated = np.concatenate([item["raw"] for item in prepared[: task_id + 1]])
        _, _, normalization = _standardize_fit_apply(accumulated, accumulated)
        mean = np.asarray(normalization["mean"], dtype=np.float32)
        scale = np.asarray(normalization["scale"], dtype=np.float32)
        current_ids = list(prepared[task_id]["current"])
        current = torch.tensor(current_ids, dtype=torch.long, device=device)
        optimizer = torch.optim.Adam(
            [*model.sample_projection.parameters(), model.class_interaction],
            lr=LEARNING_RATE,
        )
        final = None
        for _ in range(EPOCHS_PER_TASK):
            for source_task, item in enumerate(prepared[: task_id + 1]):
                class_ids = torch.tensor(item["current"], dtype=torch.long, device=device)
                features = torch.from_numpy(((item["raw"] - mean) / scale).astype(np.float32)).to(device)
                reliable = torch.from_numpy(item["reliable"]).to(device)
                targets = torch.from_numpy(item["targets"]).to(device)[:, class_ids]
                probabilities = tuple(
                    torch.from_numpy(value).to(device)[:, class_ids]
                    for value in item["probabilities"]
                )
                for cpu_indices in _batch_order(len(reliable), generator):
                    indices = cpu_indices.to(device)
                    weights, interaction = model(
                        features[indices], class_ids, reliable[indices]
                    )
                    fused = _fuse_classwise(
                        tuple(value[indices] for value in probabilities), weights
                    )
                    data_loss = F.binary_cross_entropy(fused, targets[indices])
                    prior_loss = interaction.square().mean()
                    parameter_loss = PARAMETER_REGULARIZATION * (
                        model.sample_projection.weight.square().mean()
                        + model.sample_projection.bias.square().mean()
                        + model.class_interaction[current].square().mean()
                    )
                    objective = (
                        data_loss
                        + float(interaction_strength) * prior_loss
                        + parameter_loss
                    )
                    if not torch.isfinite(objective):
                        raise FloatingPointError("Non-finite C2 objective")
                    optimizer.zero_grad(set_to_none=True)
                    objective.backward()
                    if model.class_interaction.grad is not None:
                        mask = torch.zeros_like(model.class_interaction.grad)
                        mask[current] = 1
                        model.class_interaction.grad.mul_(mask)
                    optimizer.step()
                    final = (
                        float(data_loss.detach()),
                        float(prior_loss.detach()),
                        float(parameter_loss.detach()),
                    )
        state = {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "task_id": task_id,
            "rank": RANK,
            "feature_names": list(feature_names),
            "normalization": normalization,
            "model": _cpu_state(model),
        }
        path = candidate_root / f"task{task_id}.pth"
        torch.save(state, path)
        states.append(state)
        records.append({
            "task_id": task_id,
            "oof_samples_seen": int(sum(len(item["raw"]) for item in prepared[: task_id + 1])),
            "current_classes": [CLASS_ORDER[index] for index in current_ids],
            "final_data_loss": final[0],
            "final_interaction_prior_loss": final[1],
            "final_parameter_loss": final[2],
            "state_path": str(path.relative_to(state_root.parent)),
            "state_sha256": _sha256(path),
        })
    return {
        "candidate_id": candidate_id,
        "family": "C2",
        "rank": RANK,
        "interaction_prior_strength": interaction_strength,
        "states": states,
        "tasks": records,
    }


def _state_weights(
    state: Mapping[str, Any],
    lane: int,
    task: Tuple[np.ndarray, ...],
    geometry: Mapping[str, Geometry],
    face: Mapping[str, Any],
    descriptors: Mapping[str, Tuple[np.ndarray, np.bool_]],
    device: torch.device,
) -> np.ndarray:
    rank = int(state["rank"])
    raw, reliable = _raw_features(
        task, geometry, face, descriptors if rank else None
    )
    if rank:
        normalization = state["normalization"]
        mean = np.asarray(normalization["mean"], dtype=np.float32)
        scale = np.asarray(normalization["scale"], dtype=np.float32)
        features = torch.from_numpy(((raw - mean) / scale).astype(np.float32)).to(device)
        feature_dim = len(state["feature_names"])
    else:
        features = None
        feature_dim = 0
    model = ClassAwareStacker(
        len(CLASS_ORDER), feature_dim, rank, initialization_seed=0
    ).float().to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    class_ids = torch.tensor(list(task_indices(lane)), dtype=torch.long, device=device)
    with torch.no_grad():
        weights, _ = model(
            features, class_ids, torch.from_numpy(reliable).to(device)
        )
    result = weights.cpu().numpy()
    if not np.array_equal(result[~reliable, :, 2], np.zeros((int((~reliable).sum()), len(class_ids)))):
        raise RuntimeError("Invalid Face received nonzero class-aware stacking weight")
    return result


def _evaluate(
    tasks: Sequence[Tuple[np.ndarray, ...]],
    states: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Geometry],
    face: Mapping[str, Any],
    descriptors: Mapping[str, Tuple[np.ndarray, np.bool_]],
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows, diagnostics = [], []
    for evaluation_task, task in enumerate(tasks):
        ids, targets, full, person, face_prob = task
        fused = np.empty_like(full, dtype=np.float32)
        class_weight_means = {}
        start = 0
        for lane in range(evaluation_task + 1):
            end = start + TASK_SIZES[lane]
            lane_task = (
                ids, targets[:, :end], full[:, :end], person[:, :end], face_prob[:, :end]
            )
            weights = _state_weights(
                states[lane], lane, lane_task, geometry, face, descriptors, device
            )
            probabilities = np.stack(
                (full[:, start:end], person[:, start:end], face_prob[:, start:end]), axis=-1
            )
            fused[:, start:end] = np.sum(probabilities * weights, axis=-1)
            for offset, class_id in enumerate(range(start, end)):
                class_weight_means[CLASS_ORDER[class_id]] = weights[:, offset].mean(axis=0).tolist()
            start = end
        metric = compute_metrics(
            evaluation_task, torch.from_numpy(fused), torch.from_numpy(targets), THRESHOLD
        )
        rows.append(metric)
        diagnostics.append({
            "task_id": evaluation_task,
            "class_weight_mean": class_weight_means,
        })
    return summarize_tasks(rows), [asdict(row) for row in rows], diagnostics


def _attach_evaluation(
    candidate: Dict[str, Any],
    tasks: Sequence[Tuple[np.ndarray, ...]],
    geometry: Mapping[str, Geometry],
    face: Mapping[str, Any],
    descriptors: Mapping[str, Tuple[np.ndarray, np.bool_]],
    device: torch.device,
) -> Dict[str, Any]:
    metrics, task_metrics, diagnostics = _evaluate(
        tasks, candidate.pop("states"), geometry, face, descriptors, device
    )
    candidate["metrics"] = metrics
    candidate["task_metrics"] = task_metrics
    candidate["validation_diagnostics"] = diagnostics
    return candidate


def select_class_aware_stacking(
    oof_root: Path,
    full_run: Path,
    person_run: Path,
    face_run: Path,
    data_root: Path,
    face_manifest_root: Path,
    descriptor_root: Path,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    output_dir.mkdir(parents=True)
    state_root = output_dir / "stacker_states"
    state_root.mkdir()
    torch.manual_seed(60000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(60000)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

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
    feature_names = list(R2_FEATURE_NAMES) + list(VISUAL_FEATURE_NAMES)
    prepared = _prepare_tasks(
        oof_tasks, train_geometry, train_face, descriptors["train"]
    )
    endpoint = type("ValidationEndpoint", (), {"validation_tasks": validation_tasks})()
    c0 = _fixed_rows(endpoint, val_face, beta=0.20)

    c1_candidates = []
    c1_state_sets = {}
    for strength in BIAS_PRIOR_STRENGTHS:
        trained = _train_c1(strength, prepared, device, state_root)
        states = trained["states"]
        c1_state_sets[trained["candidate_id"]] = states
        c1_candidates.append(_attach_evaluation(
            trained, validation_tasks, val_geometry, val_face,
            descriptors["val"], device,
        ))
    best_c1 = max(
        c1_candidates,
        key=lambda row: (
            row["metrics"]["final_mAP"], row["metrics"]["average_mAP"],
            row["bias_prior_strength"],
        ),
    )
    best_c1_states = c1_state_sets[best_c1["candidate_id"]]

    c2_candidates = []
    for strength in INTERACTION_PRIOR_STRENGTHS:
        trained = _train_c2(
            strength, best_c1_states, prepared, feature_names, device, state_root
        )
        c2_candidates.append(_attach_evaluation(
            trained, validation_tasks, val_geometry, val_face,
            descriptors["val"], device,
        ))
    best_c2 = max(
        c2_candidates,
        key=lambda row: (
            row["metrics"]["final_mAP"], row["metrics"]["average_mAP"],
            row["interaction_prior_strength"],
        ),
    )

    gains_c0 = _named_final_ap_gains(best_c2, c0)
    gains_c1 = _named_final_ap_gains(best_c2, best_c1)
    positive = {name: gain for name, gain in gains_c0.items() if gain > POSITIVE_CLASS_EPS}
    positive_total = float(sum(positive.values()))
    largest_positive_share = (
        max(positive.values()) / positive_total if positive_total > 0 else 1.0
    )
    distributed = (
        len(positive) >= 2
        and largest_positive_share <= MAX_SINGLE_CLASS_POSITIVE_SHARE
    )
    gain_over_c0 = float(best_c2["metrics"]["final_mAP"] - c0["metrics"]["final_mAP"])
    gain_over_c1 = float(
        best_c2["metrics"]["final_mAP"] - best_c1["metrics"]["final_mAP"]
    )
    advance = (
        gain_over_c0 >= ADVANCE_MARGIN_MAP
        and gain_over_c1 >= ADVANCE_MARGIN_MAP
        and distributed
    )
    result = {
        "schema_version": 1,
        "selection_split": "val",
        "test_accessed": False,
        "method": "hierarchical_class_aware_three_view_oof_stacking",
        "device": str(device),
        "crossfit": {
            "every_training_prediction_is_out_of_fold": True,
            **oof_provenance,
        },
        "protocol": {
            "C0": "fixed R1; reliable Face 0.64/0.16/0.20, otherwise 0.80/0.20/0",
            "C1": "one centered three-view bias per class, strongly shrunk to C0",
            "C2": "C1 plus rank-2 shared sample-reliability by class interaction",
            "rank": RANK,
            "feature_names": feature_names,
            "epochs_per_task": EPOCHS_PER_TASK,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "bias_prior_strengths": list(BIAS_PRIOR_STRENGTHS),
            "interaction_prior_strengths": list(INTERACTION_PRIOR_STRENGTHS),
            "incremental_semantics": "class lane k uses the snapshot saved when task k was learned",
            "invalid_face_weight": 0.0,
            "precision": "fp32",
        },
        "C0_fixed_R1": c0,
        "C1_candidates": c1_candidates,
        "best_C1": best_c1,
        "C2_candidates": c2_candidates,
        "best_C2": best_c2,
        "class_gain_C2_minus_C0": gains_c0,
        "class_gain_C2_minus_C1": gains_c1,
        "decision": {
            "required_margin_final_mAP": ADVANCE_MARGIN_MAP,
            "C2_gain_over_C0": gain_over_c0,
            "C2_gain_over_C1": gain_over_c1,
            "positive_class_epsilon_AP": POSITIVE_CLASS_EPS,
            "positive_class_count": len(positive),
            "largest_positive_class_share": largest_positive_share,
            "max_allowed_single_class_positive_share": MAX_SINGLE_CLASS_POSITIVE_SHARE,
            "gain_is_distributed": distributed,
            "advance_to_seed1_seed2_validation": advance,
            "run_test": False,
        },
        "sources": {
            "oof_root": str(oof_root.resolve()),
            "validation": validation_sources,
            "descriptors": str(descriptor_root.resolve()),
            "full_summary_sha256": _sha256(full_run / "seed_summary.json"),
            "person_summary_sha256": _sha256(person_run / "seed_summary.json"),
            "face_summary_sha256": _sha256(face_run / "seed_summary.json"),
        },
    }
    output = output_dir / "validation_selection.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "C0_final_mAP": c0["metrics"]["final_mAP"],
        "C1_final_mAP": best_c1["metrics"]["final_mAP"],
        "C2_final_mAP": best_c2["metrics"]["final_mAP"],
        **result["decision"],
    }, indent=2), flush=True)
    print("CLASS_AWARE_OOF_STACKING_VALIDATION_COMPLETE", flush=True)
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
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    select_class_aware_stacking(
        args.oof_root, args.full_run, args.person_run, args.face_run,
        args.data_root, args.face_manifest_root, args.descriptor_root,
        args.output_dir, torch.device(args.device),
    )


if __name__ == "__main__":
    main()
