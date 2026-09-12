"""Deterministic five-point Face alignment for EMOTIC expression experts."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


ARCFACE_TEMPLATE_112 = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def valid_five_point_landmarks(
    landmarks: object,
    image_size: Optional[Tuple[int, int]] = None,
) -> bool:
    try:
        points = np.asarray(landmarks, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return False
    # SCRFD serializes left eye, right eye, nose, left mouth, right mouth.
    # Do not infer semantic order from x coordinates: strong yaw/roll can
    # legitimately reverse their image-plane ordering.
    if np.linalg.norm(points[0] - points[1]) <= 1e-6 or np.linalg.norm(
        points[3] - points[4]
    ) <= 1e-6:
        return False
    if image_size is not None:
        width, height = image_size
        if width <= 0 or height <= 0:
            return False
        if (
            (points[:, 0] < 0).any()
            or (points[:, 0] > width).any()
            or (points[:, 1] < 0).any()
            or (points[:, 1] > height).any()
        ):
            return False
    return True


def canonical_five_point_template(size: int = 224, margin: float = 0.15) -> np.ndarray:
    if size <= 0:
        raise ValueError("Face alignment size must be positive")
    if not np.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("Face alignment margin must be finite and in [0, 1]")
    template = ARCFACE_TEMPLATE_112 * (float(size) / 112.0)
    centre = np.asarray([size / 2.0, size / 2.0], dtype=np.float64)
    # Pull the canonical landmarks towards the centre so the inverse warp sees
    # 15% more head context on every side instead of producing a tighter face.
    return centre + (template - centre) / (1.0 + 2.0 * margin)


def estimate_similarity_transform(
    source: Sequence[Sequence[float]],
    destination: Sequence[Sequence[float]],
) -> np.ndarray:
    source_array = np.asarray(source, dtype=np.float64)
    destination_array = np.asarray(destination, dtype=np.float64)
    if source_array.shape != destination_array.shape or source_array.shape != (5, 2):
        raise ValueError("Similarity alignment requires matching five-point landmarks")
    rows = []
    values = []
    for (x_value, y_value), (u_value, v_value) in zip(
        source_array, destination_array
    ):
        rows.extend(
            ([x_value, -y_value, 1.0, 0.0], [y_value, x_value, 0.0, 1.0])
        )
        values.extend((u_value, v_value))
    solution, _, rank, _ = np.linalg.lstsq(
        np.asarray(rows, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        rcond=None,
    )
    if rank != 4 or not np.isfinite(solution).all():
        raise ValueError("Face landmarks do not define a finite similarity transform")
    a_value, b_value, tx_value, ty_value = solution.tolist()
    if a_value * a_value + b_value * b_value <= 1e-12:
        raise ValueError("Face landmark similarity transform has zero scale")
    return np.asarray(
        [[a_value, -b_value, tx_value], [b_value, a_value, ty_value]],
        dtype=np.float64,
    )


def align_face_five_points(
    image: Image.Image,
    record: Mapping[str, object],
    size: int = 224,
    margin: float = 0.15,
    fill: Tuple[int, int, int] = (124, 116, 104),
) -> Optional[Image.Image]:
    landmarks = record.get("face_keypoints")
    if not valid_five_point_landmarks(landmarks, image.size):
        return None
    forward = estimate_similarity_transform(
        np.asarray(landmarks, dtype=np.float64),
        canonical_five_point_template(size, margin),
    )
    linear = forward[:, :2]
    inverse_linear = np.linalg.inv(linear)
    inverse_offset = -inverse_linear @ forward[:, 2]
    coefficients = (
        float(inverse_linear[0, 0]),
        float(inverse_linear[0, 1]),
        float(inverse_offset[0]),
        float(inverse_linear[1, 0]),
        float(inverse_linear[1, 1]),
        float(inverse_offset[1]),
    )
    return image.convert("RGB").transform(
        (size, size),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )
