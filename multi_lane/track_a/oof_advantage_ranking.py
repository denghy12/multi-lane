"""Leakage-free OOF ranking corrections for deterministic Full-image views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .runner import CLASS_ORDER, TASK_SIZES, _intersects, average_precision, task_indices
from .three_view_oof_router import FOLDS, _align_three, _audit_oof_source, load_oof_tasks
from .three_view_router import load_face_metadata


ELIGIBLE_CLASS_NAMES = (
    "Affection", "Annoyance", "Confidence", "Disapproval", "Disconnection",
    "Engagement", "Esteem", "Excitement", "Fatigue", "Happiness", "Pain",
    "Pleasure", "Sympathy",
)
R1_RELIABLE_WEIGHTS = (0.64, 0.16, 0.20)
R1_INVALID_FACE_WEIGHTS = (0.80, 0.20, 0.0)


def _fixed_r1(
    sample_ids: np.ndarray,
    full: np.ndarray,
    person: np.ndarray,
    face: np.ndarray,
    metadata: Mapping[str, Any],
) -> np.ndarray:
    reliable = np.asarray(
        [metadata[str(sample_id)].reliable for sample_id in sample_ids],
        dtype=np.bool_,
    )
    result = (
        R1_INVALID_FACE_WEIGHTS[0] * full
        + R1_INVALID_FACE_WEIGHTS[1] * person
    )
    result[reliable] = (
        R1_RELIABLE_WEIGHTS[0] * full[reliable]
        + R1_RELIABLE_WEIGHTS[1] * person[reliable]
        + R1_RELIABLE_WEIGHTS[2] * face[reliable]
    )
    return result.astype(np.float32, copy=False)


@dataclass(frozen=True)
class TaskPairTable:
    class_indices: Tuple[int, ...]
    positive_indices: Tuple[np.ndarray, ...]
    negative_indices: Tuple[np.ndarray, ...]
    pair_counts: Tuple[int, ...]
    fold_ap_gains: Mapping[str, Tuple[float, ...]]


class OOFAdvantagePairBank:
    """Positive-negative pairs where OOF R1 repairs an OOF Full misordering."""

    def __init__(self, oof_root, face_manifest_root, deterministic_source) -> None:
        oof_root = oof_root.expanduser().resolve()
        metadata = load_face_metadata(face_manifest_root, "train")
        tasks, provenance = load_oof_tasks(oof_root)
        if len(tasks) != len(TASK_SIZES):
            raise ValueError("OOF pair bank does not contain all tasks")
        source_by_id = {
            str(sample_id): index
            for index, sample_id in enumerate(deterministic_source.sample_ids)
        }
        if len(source_by_id) != len(deterministic_source):
            raise ValueError("Deterministic Full source contains duplicate sample IDs")

        fold_tasks = []
        for fold in range(FOLDS):
            dumps = []
            for view, input_mode in (
                ("full", "full"),
                ("person", "person_crop"),
                ("face", "face_crop"),
            ):
                _, view_dumps = _audit_oof_source(
                    oof_root / f"fold{fold}_{view}", input_mode, fold
                )
                dumps.append(view_dumps)
            fold_tasks.append([
                _align_three(full, person, face)
                for full, person, face in zip(*dumps)
            ])

        self.provenance = provenance
        self.tasks = tuple(
            self._prepare_task(
                task_id, task, fold_tasks, metadata, deterministic_source,
                source_by_id,
            )
            for task_id, task in enumerate(tasks)
        )
        selected = tuple(
            CLASS_ORDER[index]
            for task in self.tasks
            for index in task.class_indices
        )
        if selected != ELIGIBLE_CLASS_NAMES:
            raise ValueError(
                f"Three-fold OOF eligibility drifted: expected {ELIGIBLE_CLASS_NAMES}, "
                f"got {selected}"
            )

    @staticmethod
    def _prepare_task(
        task_id,
        task,
        fold_tasks,
        metadata,
        source,
        source_by_id,
    ) -> TaskPairTable:
        sample_ids, targets, full, person, face = task
        current = tuple(task_indices(task_id))
        expected_indices = [
            index for index, target in enumerate(source.targets)
            if _intersects(target, current)
        ]
        expected_ids = [str(source.sample_ids[index]) for index in expected_indices]
        if len(sample_ids) != len(expected_ids) or set(sample_ids) != set(expected_ids):
            raise ValueError(f"OOF task{task_id} does not exactly cover the Full pool")
        for row, sample_id in enumerate(sample_ids.astype(str)):
            expected = np.asarray(
                [float(label in source.targets[source_by_id[sample_id]]) for label in current],
                dtype=np.float32,
            )
            if not np.array_equal(targets[row, list(current)], expected):
                raise ValueError(f"OOF labels differ for {sample_id}")

        fold_gains: Dict[int, Tuple[float, ...]] = {}
        for class_index in current:
            values = []
            for fold in range(FOLDS):
                ids, fold_targets, fold_full, fold_person, fold_face = (
                    fold_tasks[fold][task_id]
                )
                fold_r1 = _fixed_r1(
                    ids, fold_full, fold_person, fold_face, metadata
                )
                values.append(100.0 * (
                    average_precision(fold_r1[:, class_index], fold_targets[:, class_index])
                    - average_precision(fold_full[:, class_index], fold_targets[:, class_index])
                ))
            fold_gains[class_index] = tuple(float(value) for value in values)
        eligible = tuple(
            class_index for class_index in current
            if all(value > 0 for value in fold_gains[class_index])
        )

        r1 = _fixed_r1(sample_ids, full, person, face, metadata)
        positive_rows = []
        negative_rows = []
        pair_counts = []
        for class_index in eligible:
            positives = np.flatnonzero(targets[:, class_index] > 0.5)
            negatives = np.flatnonzero(targets[:, class_index] < 0.5)
            corrected_positive = []
            corrected_negative = []
            for start in range(0, len(positives), 64):
                chunk = positives[start:start + 64]
                corrected = (
                    (full[chunk, class_index, None] <= full[negatives, class_index][None, :])
                    & (r1[chunk, class_index, None] > r1[negatives, class_index][None, :])
                )
                positive_offset, negative_offset = np.nonzero(corrected)
                corrected_positive.append(chunk[positive_offset])
                corrected_negative.append(negatives[negative_offset])
            positive_oof = np.concatenate(corrected_positive)
            negative_oof = np.concatenate(corrected_negative)
            if not len(positive_oof):
                raise ValueError(f"Eligible class {CLASS_ORDER[class_index]} has no corrections")
            positive_rows.append(np.asarray(
                [source_by_id[str(sample_ids[index])] for index in positive_oof],
                dtype=np.int64,
            ))
            negative_rows.append(np.asarray(
                [source_by_id[str(sample_ids[index])] for index in negative_oof],
                dtype=np.int64,
            ))
            pair_counts.append(len(positive_oof))
        return TaskPairTable(
            class_indices=eligible,
            positive_indices=tuple(positive_rows),
            negative_indices=tuple(negative_rows),
            pair_counts=tuple(pair_counts),
            fold_ap_gains={
                CLASS_ORDER[index]: fold_gains[index] for index in current
            },
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "eligibility": "R1_minus_Full_OOF_AP_positive_in_all_3_folds",
            "pair_rule": "Full_positive_le_negative_and_R1_positive_gt_negative",
            "eligible_class_names": list(ELIGIBLE_CLASS_NAMES),
            "tasks": [
                {
                    "task_id": task_id,
                    "class_indices": list(task.class_indices),
                    "class_names": [CLASS_ORDER[index] for index in task.class_indices],
                    "pair_counts": {
                        CLASS_ORDER[index]: count
                        for index, count in zip(task.class_indices, task.pair_counts)
                    },
                    "fold_ap_gains": {
                        name: list(values)
                        for name, values in task.fold_ap_gains.items()
                    },
                }
                for task_id, task in enumerate(self.tasks)
            ],
            "provenance": self.provenance,
        }


class CorrectedPairDataset(Dataset):
    """Balanced deterministic pair stream without replacement within a run."""

    def __init__(
        self,
        deterministic_source,
        table: TaskPairTable,
        length: int,
        seed: int,
    ) -> None:
        if length <= 0 or not table.class_indices:
            raise ValueError("Pair dataset requires positive length and eligible classes")
        self.source = deterministic_source
        self.class_indices = table.class_indices
        self.length = int(length)
        rng = np.random.default_rng(int(seed))
        self.positive_indices = []
        self.negative_indices = []
        for slot, (positive, negative) in enumerate(zip(
            table.positive_indices, table.negative_indices
        )):
            required = (self.length + len(self.class_indices) - 1 - slot) // len(
                self.class_indices
            )
            if required > len(positive):
                raise ValueError(
                    f"Not enough unique corrected pairs for {CLASS_ORDER[self.class_indices[slot]]}: "
                    f"required={required}, available={len(positive)}"
                )
            order = rng.permutation(len(positive))[:required]
            self.positive_indices.append(positive[order])
            self.negative_indices.append(negative[order])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        slot = int(index) % len(self.class_indices)
        occurrence = int(index) // len(self.class_indices)
        positive_image, _ = self.source[self.positive_indices[slot][occurrence]]
        negative_image, _ = self.source[self.negative_indices[slot][occurrence]]
        return positive_image, negative_image, torch.tensor(
            self.class_indices[slot], dtype=torch.long
        )
