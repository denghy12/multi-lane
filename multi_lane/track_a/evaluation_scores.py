"""Read exact evaluation probabilities, including historical logits-only dumps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def batched_sigmoid(logits: np.ndarray, batch_size: int) -> np.ndarray:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("Legacy scores require a positive eval_batch_size")
    tensor = torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32))
    return torch.cat([torch.sigmoid(part) for part in tensor.split(batch_size)]).numpy()


@dataclass
class EvaluationScores:
    task_id: int
    sample_ids: np.ndarray
    class_indices: np.ndarray
    logits: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray
    probability_source: str


def load_evaluation_scores(path: Path, legacy_batch_size: Optional[int] = None) -> EvaluationScores:
    with np.load(path, allow_pickle=False) as data:
        version = int(data['schema_version'])
        if version not in (1, 2):
            raise ValueError(f"Unsupported score schema: {version}")
        ids = data['sample_ids'].astype(str)
        logits = data['logits'].astype(np.float32)
        targets = data['targets'].astype(np.float32)
        classes = data['class_indices']
        task = int(data['task_id'])
        if logits.ndim != 2 or not logits.shape[0] or targets.shape != logits.shape:
            raise ValueError("Invalid score/target shapes")
        if ids.shape != (len(logits),) or len(set(ids.tolist())) != len(ids):
            raise ValueError("Invalid or duplicate sample IDs")
        if not np.array_equal(classes, np.arange(logits.shape[1])):
            raise ValueError("Unexpected class indices")
        if not np.isfinite(logits).all() or not np.isfinite(targets).all():
            raise FloatingPointError("Non-finite score dump")
        if not np.isin(targets, (0, 1)).all():
            raise ValueError("Evaluation targets must be binary")
        if version == 2:
            probabilities = data['probabilities'].astype(np.float32)
            lengths = data['batch_lengths']
            if (lengths.ndim != 1 or not len(lengths) or lengths.dtype.kind not in 'iu'
                    or (lengths <= 0).any() or int(lengths.sum()) != len(logits)):
                raise ValueError("Invalid evaluation batch lengths")
            if (str(data['probability_device']) != 'cpu'
                    or str(data['probability_dtype']) != 'float32'
                    or str(data['probability_operation']) != 'torch.sigmoid'
                    or not str(data['torch_version'])):
                raise ValueError("Unsupported probability provenance")
            source = 'stored_evaluation_probabilities_v2'
        else:
            probabilities = batched_sigmoid(logits, legacy_batch_size)
            source = f'legacy_cpu_sigmoid_batch{legacy_batch_size}'
        if (probabilities.shape != logits.shape or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or (probabilities > 1).any()):
            raise ValueError("Invalid evaluation probabilities")
    return EvaluationScores(task, ids, classes, logits, targets, probabilities, source)


def align_evaluation_scores(full: EvaluationScores, auxiliary: EvaluationScores):
    if full.task_id != auxiliary.task_id or not np.array_equal(full.class_indices, auxiliary.class_indices):
        raise ValueError("Score dumps differ on task or class indices")
    if set(full.sample_ids.tolist()) != set(auxiliary.sample_ids.tolist()):
        raise ValueError("Score dumps contain different samples")
    positions = {sample_id: index for index, sample_id in enumerate(auxiliary.sample_ids)}
    order = np.asarray([positions[sample_id] for sample_id in full.sample_ids], dtype=np.int64)
    if not np.array_equal(full.targets, auxiliary.targets[order]):
        raise ValueError("Targets differ after sample alignment")
    # Reconstruct legacy probabilities BEFORE reordering: batch boundaries belong
    # to each run's original evaluation order, not the other run's order.
    return (full.sample_ids, full.logits, auxiliary.logits[order], full.targets,
            full.probabilities, auxiliary.probabilities[order])
