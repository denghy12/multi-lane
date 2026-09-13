"""Leakage-free cross-view soft teachers for incremental Full-lane training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .runner import TASK_SIZES, _intersects, task_indices
from .three_view_oof_router import load_oof_tasks
from .three_view_router import load_face_metadata


DISTILLATION_MODES = ("person", "person_face", "r1")
PERSON_FACE_WEIGHTS = (4.0 / 9.0, 5.0 / 9.0)
R1_RELIABLE_WEIGHTS = (0.64, 0.16, 0.20)
R1_INVALID_FACE_WEIGHTS = (0.80, 0.20, 0.0)


@dataclass(frozen=True)
class TaskTeacher:
    probabilities: Mapping[str, np.ndarray]
    targets: Mapping[str, np.ndarray]
    samples: int
    reliable_face_samples: int


class OOFTeacherBank:
    """Audited task-wise teachers pooled from three held-out image-group folds."""

    def __init__(
        self,
        oof_root,
        face_manifest_root,
        source,
        mode: str,
    ) -> None:
        if mode not in DISTILLATION_MODES:
            raise ValueError(f"OOF distillation mode must be one of {DISTILLATION_MODES}")
        tasks, provenance = load_oof_tasks(oof_root)
        if len(tasks) != len(TASK_SIZES):
            raise ValueError("OOF teacher bank does not contain all protocol tasks")
        face = load_face_metadata(face_manifest_root, "train")
        source_index_by_id = {
            str(sample_id): index for index, sample_id in enumerate(source.sample_ids)
        }
        if len(source_index_by_id) != len(source):
            raise ValueError("Full training source contains duplicate sample IDs")
        self.mode = mode
        self.provenance = provenance
        self.tasks: Tuple[TaskTeacher, ...] = tuple(
            self._prepare_task(task_id, task, source, source_index_by_id, face)
            for task_id, task in enumerate(tasks)
        )

    def _prepare_task(
        self, task_id, task, source, source_index_by_id, face
    ) -> TaskTeacher:
        sample_ids, targets, full, person, face_probabilities = task
        current = task_indices(task_id)
        expected_indices = [
            index for index, target in enumerate(source.targets)
            if _intersects(target, current)
        ]
        expected_ids = [str(source.sample_ids[index]) for index in expected_indices]
        if set(sample_ids.tolist()) != set(expected_ids) or len(sample_ids) != len(expected_ids):
            raise ValueError(f"OOF task{task_id} does not exactly cover the Full fit pool")
        current_targets = targets[:, list(current)].astype(np.float32)
        full = full[:, list(current)].astype(np.float32)
        person = person[:, list(current)].astype(np.float32)
        face_probabilities = face_probabilities[:, list(current)].astype(np.float32)
        if not (
            current_targets.shape == full.shape == person.shape == face_probabilities.shape
            and current_targets.shape == (len(sample_ids), len(current))
        ):
            raise ValueError(f"OOF task{task_id} teacher shapes differ")
        reliable = np.asarray(
            [face[str(sample_id)].reliable for sample_id in sample_ids], dtype=np.bool_
        )
        teacher = person.copy()
        if self.mode == "person_face":
            person_weight, face_weight = PERSON_FACE_WEIGHTS
            teacher[reliable] = (
                person_weight * person[reliable]
                + face_weight * face_probabilities[reliable]
            )
        elif self.mode == "r1":
            full_weight, person_weight, _ = R1_INVALID_FACE_WEIGHTS
            teacher = full_weight * full + person_weight * person
            full_weight, person_weight, face_weight = R1_RELIABLE_WEIGHTS
            teacher[reliable] = (
                full_weight * full[reliable]
                + person_weight * person[reliable]
                + face_weight * face_probabilities[reliable]
            )
        if not np.isfinite(teacher).all() or (teacher < 0).any() or (teacher > 1).any():
            raise FloatingPointError(f"OOF task{task_id} teacher probabilities are invalid")
        target_by_id: Dict[str, np.ndarray] = {}
        teacher_by_id: Dict[str, np.ndarray] = {}
        for index, sample_id in enumerate(sample_ids.astype(str)):
            expected = np.asarray(
                [float(label in source.targets[source_index_by_id[sample_id]])
                 for label in current],
                dtype=np.float32,
            )
            if not np.array_equal(current_targets[index], expected):
                raise ValueError(f"OOF task{task_id} labels differ for {sample_id}")
            target_by_id[sample_id] = current_targets[index]
            teacher_by_id[sample_id] = teacher[index]
        return TaskTeacher(
            probabilities=teacher_by_id,
            targets=target_by_id,
            samples=len(sample_ids),
            reliable_face_samples=int(reliable.sum()),
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "person_face_weights_on_reliable": (
                list(PERSON_FACE_WEIGHTS) if self.mode == "person_face" else None
            ),
            "R1_reliable_weights": (
                list(R1_RELIABLE_WEIGHTS) if self.mode == "r1" else None
            ),
            "R1_invalid_face_weights": (
                list(R1_INVALID_FACE_WEIGHTS) if self.mode == "r1" else None
            ),
            "invalid_face_fallback": (
                "locked_full_person" if self.mode == "r1" else "person_only"
            ),
            "tasks": [
                {
                    "task_id": task_id,
                    "samples": task.samples,
                    "reliable_face_samples": task.reliable_face_samples,
                }
                for task_id, task in enumerate(self.tasks)
            ],
            "provenance": self.provenance,
        }


class OOFTeacherLabelView(Dataset):
    """Full-image task view carrying an aligned current-class OOF soft target."""

    def __init__(
        self,
        source,
        indices: Sequence[int],
        class_indices: Sequence[int],
        teacher: TaskTeacher,
    ) -> None:
        self.source = source
        self.indices = tuple(int(value) for value in indices)
        self.class_indices = tuple(int(value) for value in class_indices)
        self.teacher = teacher
        ids = [str(source.sample_ids[index]) for index in self.indices]
        if len(ids) != teacher.samples or set(ids) != set(teacher.probabilities):
            raise ValueError("Training view and OOF teacher sample IDs differ")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = self.indices[index]
        image, target = self.source[source_index]
        sample_id = str(self.source.sample_ids[source_index])
        selected_target = target[list(self.class_indices)].float()
        expected = torch.from_numpy(self.teacher.targets[sample_id])
        if not torch.equal(selected_target, expected):
            raise RuntimeError("Runtime Full target differs from audited OOF target")
        soft_target = torch.from_numpy(self.teacher.probabilities[sample_id].copy())
        return image, selected_target, soft_target
