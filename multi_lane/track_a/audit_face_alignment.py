"""Audit five-point landmark coverage before Face-expression training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from multi_lane.continual_datasets.continual_datasets import EMOTIC

from .face_alignment import (
    canonical_five_point_template,
    estimate_similarity_transform,
    valid_five_point_landmarks,
)
from .runner import TASK_SIZES, resolve_dataset_parent, task_indices


def audit_source(source: EMOTIC) -> Dict[str, Any]:
    eligible = valid_landmarks = transform_failures = reliable = canonical_x_order = 0
    residuals = []
    task_rows = []
    for task_id in range(len(TASK_SIZES)):
        indices = set(task_indices(task_id))
        matching = [
            index for index, target in enumerate(source.targets)
            if any(label in indices for label in target)
        ]
        usable = [index for index in matching if source.face_training_valid[index]]
        positive_counts = {
            str(label): sum(label in source.targets[index] for index in usable)
            for label in sorted(indices)
        }
        task_rows.append({
            "task_id": task_id,
            "task_samples": len(matching),
            "valid_nonambiguous_samples": len(usable),
            "positive_counts_in_valid_pool": positive_counts,
        })
    destination = canonical_five_point_template(224, 0.15)
    for index, record in enumerate(source.face_records):
        if not source.face_training_valid[index]:
            continue
        eligible += 1
        reliable += int(bool(source.face_reliable[index]))
        image_size = (int(record["image_width"]), int(record["image_height"]))
        points = record.get("face_keypoints")
        if not valid_five_point_landmarks(points, image_size):
            continue
        valid_landmarks += 1
        points_array = np.asarray(points, dtype=np.float64)
        canonical_x_order += int(
            points_array[0, 0] < points_array[1, 0]
            and points_array[3, 0] < points_array[4, 0]
        )
        try:
            transform = estimate_similarity_transform(points, destination)
            source_points = np.asarray(points, dtype=np.float64)
            projected = source_points @ transform[:, :2].T + transform[:, 2]
            residuals.append(float(np.sqrt(np.mean((projected - destination) ** 2))))
        except (ValueError, np.linalg.LinAlgError):
            transform_failures += 1
    coverage = float(valid_landmarks / eligible) if eligible else 0.0
    return {
        "samples": len(source),
        "valid_nonambiguous_faces": eligible,
        "reliable_faces": reliable,
        "valid_five_point_landmarks": valid_landmarks,
        "fallback_letterbox_faces": eligible - valid_landmarks,
        "landmark_coverage": coverage,
        "canonical_x_order_rate_diagnostic_only": float(
            canonical_x_order / valid_landmarks
        ) if valid_landmarks else 0.0,
        "similarity_transform_failures": transform_failures,
        "alignment_rmse_mean": float(np.mean(residuals)) if residuals else None,
        "alignment_rmse_max": float(np.max(residuals)) if residuals else None,
        "tasks": task_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent = resolve_dataset_parent(args.data_root)
    train = EMOTIC(
        str(parent), train=True, input_mode="face_crop",
        face_manifest_root=args.face_manifest_root,
    )
    val = EMOTIC(
        str(parent), train=False, eval_splits=("val",), input_mode="face_crop",
        face_manifest_root=args.face_manifest_root,
    )
    splits = {"train": audit_source(train), "val": audit_source(val)}
    ready = all(
        row["landmark_coverage"] >= 0.99
        and row["similarity_transform_failures"] == 0
        for row in splits.values()
    )
    result = {
        "schema_version": 1,
        "protocol": "emotic-face-five-point-alignment-audit-v1",
        "alignment": "similarity_arcface_template_224_margin0.15",
        "training_policy": "valid_face_and_not_ambiguous",
        "minimum_landmark_coverage": 0.99,
        "splits": splits,
        "ready_for_full_validation": ready,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if not ready:
        raise RuntimeError("Five-point landmark audit did not pass the locked coverage gate")
    print("FACE_ALIGNMENT_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
