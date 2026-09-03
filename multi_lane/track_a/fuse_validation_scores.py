"""Offline global fusion search for paired full/person validation scores."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .runner import TASK_SIZES, TaskMetrics, compute_metrics, summarize_tasks
from .evaluation_scores import align_evaluation_scores, load_evaluation_scores


COMMON_CONFIG_FIELDS = (
    "protocol_id", "seed", "task_sizes", "class_order", "reporting_split",
    "max_tasks", "save_checkpoints", "training_budget_mode", "epochs_per_task", "train_batch_size",
    "eval_batch_size", "threshold", "training_loss_mode",
    "parameter_group_loss_routing", "model_parameter_objective",
    "adapter_parameter_objective", "asl", "learning_rate", "optimizer",
    "weight_decay", "scheduler", "scheduler_mode", "scheduler_min_lr_ratio",
    "scheduler_warmup_ratio", "input_normalization", "amp", "tf32", "adapter_mode",
    "adapter_bottleneck_dim", "adapter_layer_indices",
    "adapter_residual_scale", "adapter_residual_gate_mode",
    "adapter_activation", "adapter_task_initialization",
    "adapter_learning_rate", "adapter_weight_decay", "adapter_regularization",
)


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation fusion artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_runs(full_run: Path, person_run: Path) -> Tuple[Dict, Dict]:
    full_config = _load_json(full_run / "config.json")
    person_config = _load_json(person_run / "config.json")
    full_summary = _load_json(full_run / "seed_summary.json")
    person_summary = _load_json(person_run / "seed_summary.json")
    for label, config, summary in (
        ("full", full_config, full_summary),
        ("person", person_config, person_summary),
    ):
        if summary.get("status") != "complete":
            raise ValueError(f"{label} validation run is incomplete")
        if config.get("reporting_split") != "val":
            raise ValueError(f"{label} fusion input must report validation")
        if not config.get("save_evaluation_scores"):
            raise ValueError(f"{label} run did not enable evaluation score dumps")
        if len(summary.get("task_metrics", [])) != len(TASK_SIZES):
            raise ValueError(f"{label} run does not contain all protocol tasks")
    if full_config.get("input_mode") != "full":
        raise ValueError("The full fusion input is not a full-image run")
    if person_config.get("input_mode") != "person_crop":
        raise ValueError("The person fusion input is not a person-crop run")
    if person_config.get("person_transform_mode") != "letterbox":
        raise ValueError("The person fusion input is not body-preserving letterbox")
    for field in COMMON_CONFIG_FIELDS:
        if full_config.get(field) != person_config.get(field):
            raise ValueError(f"Fusion runs differ on fixed config field {field}")
    return full_summary, person_summary


def load_aligned_task_scores(
    full_path: Path, person_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(full_path, allow_pickle=False) as full, np.load(
        person_path, allow_pickle=False
    ) as person:
        full_ids = full["sample_ids"].astype(str)
        person_ids = person["sample_ids"].astype(str)
        full_logits = full["logits"].astype(np.float32)
        person_logits = person["logits"].astype(np.float32)
        full_targets = full["targets"].astype(np.float32)
        person_targets = person["targets"].astype(np.float32)
    if len(set(full_ids.tolist())) != len(full_ids):
        raise ValueError("Full-view score dump has duplicate sample IDs")
    if len(set(person_ids.tolist())) != len(person_ids):
        raise ValueError("Person-view score dump has duplicate sample IDs")
    if set(full_ids.tolist()) != set(person_ids.tolist()):
        raise ValueError("Full and person score dumps contain different samples")
    positions = {sample_id: index for index, sample_id in enumerate(person_ids)}
    order = np.asarray([positions[sample_id] for sample_id in full_ids], dtype=np.int64)
    person_logits = person_logits[order]
    person_targets = person_targets[order]
    if full_logits.shape != person_logits.shape:
        raise ValueError("Full and person logits have different shapes")
    if full_targets.shape != person_targets.shape or not np.array_equal(
        full_targets, person_targets
    ):
        raise ValueError("Full and person targets differ after sample alignment")
    if not np.isfinite(full_logits).all() or not np.isfinite(person_logits).all():
        raise FloatingPointError("Fusion logits contain non-finite values")
    return full_ids, full_logits, person_logits, full_targets


def fused_scores(
    full_logits: np.ndarray,
    person_logits: np.ndarray,
    alpha: float,
    mode: str,
    full_probabilities: np.ndarray = None,
    person_probabilities: np.ndarray = None,
) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError("Fusion alpha must be in [0, 1]")
    if mode not in ('logit', 'probability'):
        raise ValueError('Unknown fusion mode')
    if (full_probabilities is None) != (person_probabilities is None):
        raise ValueError("Both endpoint probability arrays are required")
    if full_probabilities is not None:
        if alpha == 0:
            return full_probabilities.copy()
        if alpha == 1:
            return person_probabilities.copy()
        if mode == 'probability':
            return (1.0 - alpha) * full_probabilities + alpha * person_probabilities
    if mode == "logit":
        logits = (1.0 - alpha) * full_logits + alpha * person_logits
        return torch.sigmoid(torch.from_numpy(logits)).numpy()
    if mode == "probability":
        full_scores = torch.sigmoid(torch.from_numpy(full_logits)).numpy()
        person_scores = torch.sigmoid(torch.from_numpy(person_logits)).numpy()
        return (1.0 - alpha) * full_scores + alpha * person_scores
    raise ValueError("Fusion mode must be logit or probability")


def _metrics_close(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return all(
        math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1e-5)
        for key in ("mAP", "cF1", "oF1")
    )


def validated_run_scores(run: Path, split: str):
    """Audit every anchor field against the probabilities actually evaluated."""
    if split not in ('val', 'test'):
        raise ValueError('Expected val or test')
    config = _load_json(run / 'config.json')
    summary = _load_json(run / 'seed_summary.json')
    if summary.get('config') != config or config.get('reporting_split') != split:
        raise ValueError('Score/config reporting provenance mismatch')
    if summary.get('status') != 'complete' or len(summary.get('task_metrics', [])) != len(TASK_SIZES):
        raise ValueError('Incomplete score run')
    folder = 'validation_scores' if split == 'val' else 'test_scores'
    dumps, rows = [], []
    for task in range(len(TASK_SIZES)):
        dump = load_evaluation_scores(run / folder / f'task{task}.npz', config.get('eval_batch_size'))
        if dump.task_id != task or dump.logits.shape[1] != sum(TASK_SIZES[:task+1]):
            raise ValueError('Unexpected task/class score layout')
        row = compute_metrics(task, torch.from_numpy(dump.probabilities), torch.from_numpy(dump.targets), config['threshold'])
        for key, value in asdict(row).items():
            if not np.allclose(value, summary['task_metrics'][task][key], rtol=0, atol=1e-10):
                raise ValueError(f'{run.name} task{task} score dump does not reproduce {key}')
        dumps.append(dump)
        rows.append(row)
    return dumps, rows


def fuse_validation_runs(
    full_run: Path,
    person_run: Path,
    alphas: Sequence[float],
) -> Dict[str, Any]:
    full_summary, person_summary = _validate_runs(full_run, person_run)
    full_dumps, _ = validated_run_scores(full_run, 'val')
    person_dumps, _ = validated_run_scores(person_run, 'val')
    task_arrays = [align_evaluation_scores(a, b) for a, b in zip(full_dumps, person_dumps)]

    candidates: List[Dict[str, Any]] = []
    for mode in ("logit", "probability"):
        for alpha in alphas:
            rows: List[TaskMetrics] = []
            for task_id, (_, full_logits, person_logits, targets, full_probs, person_probs) in enumerate(
                task_arrays
            ):
                scores = fused_scores(full_logits, person_logits, float(alpha), mode, full_probs, person_probs)
                rows.append(
                    compute_metrics(
                        task_id,
                        torch.from_numpy(scores),
                        torch.from_numpy(targets),
                        float(full_summary["config"]["threshold"]),
                    )
                )
            candidates.append(
                {
                    "mode": mode,
                    "alpha": float(alpha),
                    "metrics": summarize_tasks(rows),
                    "task_metrics": [asdict(row) for row in rows],
                }
            )

    full_anchor = next(
        row for row in candidates if row["mode"] == "logit" and row["alpha"] == 0
    )
    person_anchor = next(
        row for row in candidates if row["mode"] == "logit" and row["alpha"] == 1
    )
    for label, candidate, summary in (
        ("full", full_anchor, full_summary),
        ("person", person_anchor, person_summary),
    ):
        for calculated, recorded in zip(
            candidate["task_metrics"], summary["task_metrics"]
        ):
            if not _metrics_close(calculated, recorded):
                raise ValueError(f"{label} score dump does not reproduce task metrics")

    winner = max(
        candidates,
        key=lambda row: (
            row["metrics"]["final_mAP"],
            row["metrics"]["average_mAP"],
            -row["alpha"],
            row["mode"] == "logit",
        ),
    )
    full_final = float(full_anchor["metrics"]["final_mAP"])
    gain = float(winner["metrics"]["final_mAP"] - full_final)
    return {
        "schema_version": 1,
        "comparison": "full_person_letterbox_global_validation_fusion",
        "selection_split": "val",
        "full_run": str(full_run.resolve()),
        "person_run": str(person_run.resolve()),
        "alpha_grid": [float(alpha) for alpha in alphas],
        "anchors": {"full": full_anchor, "person": person_anchor},
        "winner": winner,
        "decision": {
            "beats_full_anchor": gain > 1e-9,
            "final_mAP_gain": gain,
            "advance_to_formal_test": gain > 1e-9,
        },
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser("Fuse full/person validation score dumps")
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--person-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--alpha-step", type=float, default=0.05,
        help="Inclusive global alpha grid step between zero and one.",
    )
    args = parser.parse_args()
    if not 0 < args.alpha_step <= 1:
        raise ValueError("alpha-step must be in (0, 1]")
    count = int(round(1.0 / args.alpha_step))
    if not math.isclose(count * args.alpha_step, 1.0, abs_tol=1e-12):
        raise ValueError("alpha-step must divide one exactly")
    result = fuse_validation_runs(
        args.full_run,
        args.person_run,
        [index * args.alpha_step for index in range(count + 1)],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2), flush=True)
    print(
        "DUAL_VIEW_VALIDATION_FUSION_COMPLETE "
        f"mode={result['winner']['mode']} alpha={result['winner']['alpha']:.2f} "
        f"final_mAP={result['winner']['metrics']['final_mAP']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
