"""Offline EMOTIC face detection, target-person matching, and audit manifest.

This module deliberately stops before model training.  It turns a frozen face
detector's outputs into reproducible, target-person-specific records and emits
the coverage diagnostics needed to decide whether a Face branch is viable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from multi_lane.continual_datasets.continual_datasets import EMOTIC
from multi_lane.track_a.runner import CLASS_ORDER, TASK_SIZES, task_indices


SCHEMA_VERSION = 1
AUDIT_PROTOCOL = "emotic-face-manifest-v1"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def resolve_dataset_parent(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.name == "EMOTIC" and (path / "CVPR17_Annotations.mat").is_file():
        return path.parent
    if (path / "EMOTIC" / "CVPR17_Annotations.mat").is_file():
        return path
    raise FileNotFoundError(
        "Expected CVPR17_Annotations.mat under DATA_ROOT or DATA_ROOT/EMOTIC"
    )


def _finite_box(box: Sequence[float]) -> Optional[np.ndarray]:
    try:
        value = np.asarray(box, dtype=np.float64).reshape(-1)[:4]
    except (TypeError, ValueError):
        return None
    if value.size != 4 or not np.isfinite(value).all():
        return None
    if value[2] <= value[0] or value[3] <= value[1]:
        return None
    return value


def clip_box(box: Sequence[float], width: int, height: int) -> Optional[np.ndarray]:
    value = _finite_box(box)
    if value is None:
        return None
    value[[0, 2]] = np.clip(value[[0, 2]], 0.0, float(width))
    value[[1, 3]] = np.clip(value[[1, 3]], 0.0, float(height))
    return value if value[2] > value[0] and value[3] > value[1] else None


def expand_box(
    box: Sequence[float], width: int, height: int, margin: float
) -> Optional[List[int]]:
    value = clip_box(box, width, height)
    if value is None:
        return None
    box_width, box_height = value[2] - value[0], value[3] - value[1]
    value += np.asarray(
        [-margin * box_width, -margin * box_height, margin * box_width, margin * box_height]
    )
    value[[0, 2]] = np.clip(value[[0, 2]], 0.0, float(width))
    value[[1, 3]] = np.clip(value[[1, 3]], 0.0, float(height))
    result = [
        int(math.floor(value[0])), int(math.floor(value[1])),
        int(math.ceil(value[2])), int(math.ceil(value[3])),
    ]
    return result if result[2] > result[0] and result[3] > result[1] else None


def letterbox_square(image: Image.Image, size: int = 224, fill: int = 0) -> Image.Image:
    """Preserve all face pixels, pad to square, then resize."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Cannot letterbox an empty image")
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), color=(fill, fill, fill))
    canvas.paste(image.convert("RGB"), ((side - width) // 2, (side - height) // 2))
    return canvas.resize((size, size), resample=Image.Resampling.BICUBIC)


def _intersection_over_face(face: np.ndarray, body: np.ndarray) -> float:
    left_top = np.maximum(face[:2], body[:2])
    right_bottom = np.minimum(face[2:], body[2:])
    extent = np.maximum(0.0, right_bottom - left_top)
    intersection = float(extent[0] * extent[1])
    face_area = float((face[2] - face[0]) * (face[3] - face[1]))
    return intersection / face_area if face_area > 0 else 0.0


def person_face_score(
    body_box: Sequence[float], face_box: Sequence[float], detection_score: float
) -> Optional[float]:
    """Score an eligible face without using identity information.

    Eligibility is intentionally conservative: the face centre must lie in the
    target body and at least half the face must be contained by that body.  The
    soft score then favours a plausible upper-body location while retaining the
    detector confidence and containment evidence.
    """
    body = _finite_box(body_box)
    face = _finite_box(face_box)
    if body is None or face is None or not math.isfinite(float(detection_score)):
        return None
    centre = (face[:2] + face[2:]) / 2.0
    if not (body[0] <= centre[0] <= body[2] and body[1] <= centre[1] <= body[3]):
        return None
    containment = _intersection_over_face(face, body)
    if containment < 0.5:
        return None
    body_size = body[2:] - body[:2]
    relative = (centre - body[:2]) / body_size
    distance = math.sqrt(((relative[0] - 0.5) / 0.5) ** 2 + ((relative[1] - 0.18) / 0.45) ** 2)
    location = math.exp(-0.5 * distance * distance)
    face_area = float(np.prod(face[2:] - face[:2]))
    body_area = float(np.prod(body_size))
    ratio = face_area / body_area
    size_plausibility = math.exp(-abs(math.log(max(ratio, 1e-8) / 0.08)) / 2.0)
    confidence = min(1.0, max(0.0, float(detection_score)))
    return 0.45 * confidence + 0.25 * containment + 0.20 * location + 0.10 * size_plausibility


@dataclass(frozen=True)
class Match:
    face_index: Optional[int]
    score: Optional[float]
    ambiguous: bool
    candidate_count: int
    runner_up_gap: Optional[float]


def match_faces_to_people(
    body_boxes: Sequence[Sequence[float]],
    faces: Sequence[Mapping[str, object]],
    ambiguity_gap: float = 0.08,
) -> Tuple[List[Match], Dict[str, int]]:
    """Globally assign at most one detected face to each annotated person."""
    person_count, face_count = len(body_boxes), len(faces)
    scores = np.full((person_count, face_count), np.nan, dtype=np.float64)
    for person_index, body in enumerate(body_boxes):
        for face_index, face in enumerate(faces):
            value = person_face_score(body, face["bbox"], float(face["score"]))
            if value is not None:
                scores[person_index, face_index] = value

    # One dummy column per person means every person may remain unmatched.  A
    # large negative value prevents ineligible real assignments.
    utility = np.full((person_count, face_count + person_count), -1e6, dtype=np.float64)
    if face_count:
        utility[:, :face_count] = np.where(np.isfinite(scores), scores, -1e6)
    utility[:, face_count:] = 0.0
    if person_count:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(-utility)
        assigned = {int(row): int(column) for row, column in zip(rows, columns)}
    else:
        assigned = {}

    face_competitions = int(sum(np.isfinite(scores[:, index]).sum() > 1 for index in range(face_count)))
    matches: List[Match] = []
    for person_index in range(person_count):
        candidates = sorted(
            (float(scores[person_index, index]), index)
            for index in range(face_count)
            if np.isfinite(scores[person_index, index])
        )
        candidates.reverse()
        gap = candidates[0][0] - candidates[1][0] if len(candidates) >= 2 else None
        column = assigned.get(person_index, face_count)
        if column >= face_count or not np.isfinite(scores[person_index, column]):
            matches.append(Match(None, None, len(candidates) > 1 and gap <= ambiguity_gap, len(candidates), gap))
        else:
            matches.append(Match(
                column,
                float(scores[person_index, column]),
                len(candidates) > 1 and gap <= ambiguity_gap,
                len(candidates),
                gap,
            ))
    assigned_faces = [match.face_index for match in matches if match.face_index is not None]
    diagnostics = {
        "face_candidate_conflicts": face_competitions,
        "duplicate_face_assignments": len(assigned_faces) - len(set(assigned_faces)),
    }
    return matches, diagnostics


class InsightFaceDetector:
    """Load only InsightFace's frozen detection ONNX, never identity models."""

    def __init__(self, checkpoint: Path, det_size: int, threshold: float) -> None:
        import insightface

        self.checkpoint = checkpoint.resolve()
        self.model = insightface.model_zoo.get_model(
            str(self.checkpoint), providers=["CPUExecutionProvider"]
        )
        self.model.prepare(ctx_id=-1, input_size=(det_size, det_size), det_thresh=threshold)

    def detect(self, image_path: Path) -> Tuple[int, int, List[Dict[str, object]]]:
        import cv2

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        boxes, keypoints = self.model.detect(image, max_num=0, metric="default")
        faces = []
        for index, box in enumerate(np.asarray(boxes)):
            clipped = clip_box(box[:4], width, height)
            if clipped is None:
                continue
            face: Dict[str, object] = {
                "bbox": clipped.tolist(),
                "score": float(box[4]),
            }
            if keypoints is not None and index < len(keypoints):
                face["keypoints"] = np.asarray(keypoints[index], dtype=np.float64).tolist()
            faces.append(face)
        return width, height, faces


def detector_metadata(checkpoint: Path, det_size: int, threshold: float) -> Dict[str, object]:
    versions: Dict[str, Optional[str]] = {}
    for module_name in ("insightface", "onnxruntime", "cv2", "numpy", "scipy", "PIL"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[module_name] = None
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": AUDIT_PROTOCOL,
        "detector": "InsightFace SCRFD det_10g ONNX",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "provider": "CPUExecutionProvider",
        "det_size": [det_size, det_size],
        "det_threshold": threshold,
        "python": sys.version,
        "platform": platform.platform(),
        "versions": versions,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Invalid JSONL at {path}:{line_number}") from error
    return rows


def detect_images(
    image_paths: Sequence[Path],
    detector: InsightFaceDetector,
    cache_path: Path,
    resume: bool,
    max_images: Optional[int],
) -> Dict[str, Dict[str, object]]:
    cached_rows = _read_jsonl(cache_path) if resume else []
    cache = {str(row["image_path"]): row for row in cached_rows}
    remaining = [path for path in image_paths if str(path) not in cache]
    if max_images is not None:
        allowed = max(0, max_images - len(cache))
        remaining = remaining[:allowed]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    with cache_path.open(mode, encoding="utf-8") as handle:
        for number, path in enumerate(remaining, 1):
            width, height, faces = detector.detect(path)
            row = {
                "image_path": str(path),
                "width": width,
                "height": height,
                "faces": faces,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            cache[str(path)] = row
            if number % 250 == 0:
                print(f"detected {number}/{len(remaining)} new images", flush=True)
    return cache


def build_manifest_records(
    source: EMOTIC,
    split: str,
    detections: Mapping[str, Mapping[str, object]],
    face_margin: float,
    ambiguity_gap: float,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    grouped: MutableMapping[str, List[int]] = defaultdict(list)
    for index, path in enumerate(source.file_paths):
        grouped[str(Path(path).resolve())].append(index)
    records: List[Dict[str, object]] = []
    aggregate_diagnostics: Counter[str] = Counter()
    for image_path, indices in grouped.items():
        detection = detections.get(image_path)
        if detection is None:
            continue
        width, height = int(detection["width"]), int(detection["height"])
        faces = list(detection["faces"])
        bodies = [source.body_bboxes[index] for index in indices]
        matches, diagnostics = match_faces_to_people(bodies, faces, ambiguity_gap)
        aggregate_diagnostics.update(diagnostics)
        for person_order, (source_index, match) in enumerate(zip(indices, matches)):
            body = clip_box(source.body_bboxes[source_index], width, height)
            face = faces[match.face_index] if match.face_index is not None else None
            face_box = clip_box(face["bbox"], width, height) if face is not None else None
            if body is None:
                invalid_reason = "invalid_body_bbox"
            elif not faces:
                invalid_reason = "no_face_detected"
            elif face is None:
                invalid_reason = "no_eligible_target_face"
            elif face_box is None:
                invalid_reason = "invalid_face_bbox"
            else:
                invalid_reason = None
            target_indices = [int(value) for value in source.targets[source_index]]
            record: Dict[str, object] = {
                "schema_version": SCHEMA_VERSION,
                "sample_id": source.sample_ids[source_index],
                "split": split,
                "image_path": str(Path(image_path).relative_to(Path(source.path))),
                "image_absolute_path": image_path,
                "image_width": width,
                "image_height": height,
                "person_order_in_image": person_order,
                "people_in_image": len(indices),
                "target_indices": target_indices,
                "target_names": [source.classes[index] for index in target_indices],
                "body_bbox": body.tolist() if body is not None else None,
                "detected_face_count": len(faces),
                "detected_faces": faces,
                "matched_face_index": match.face_index,
                "face_bbox": face_box.tolist() if face_box is not None else None,
                "face_detection_score": float(face["score"]) if face is not None else None,
                "face_keypoints": face.get("keypoints") if face is not None else None,
                "match_score": match.score,
                "candidate_count": match.candidate_count,
                "runner_up_gap": match.runner_up_gap,
                "ambiguous_match": match.ambiguous,
                "valid_face": invalid_reason is None,
                "invalid_reason": invalid_reason,
            }
            if face_box is not None and body is not None:
                face_width, face_height = face_box[2] - face_box[0], face_box[3] - face_box[1]
                face_area = face_width * face_height
                body_area = (body[2] - body[0]) * (body[3] - body[1])
                record.update({
                    "face_width": float(face_width),
                    "face_height": float(face_height),
                    "face_short_side": float(min(face_width, face_height)),
                    "face_area": float(face_area),
                    "face_image_area_ratio": float(face_area / (width * height)),
                    "face_person_area_ratio": float(face_area / body_area),
                    "face_crop_bbox": expand_box(face_box, width, height, face_margin),
                })
            records.append(record)
    return records, dict(aggregate_diagnostics)


def _percentiles(values: Iterable[float]) -> Dict[str, Optional[float]]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {key: None for key in ("min", "p05", "p10", "p25", "median", "p75", "p90", "p95", "max")}
    levels = (0, 5, 10, 25, 50, 75, 90, 95, 100)
    names = ("min", "p05", "p10", "p25", "median", "p75", "p90", "p95", "max")
    return {name: float(value) for name, value in zip(names, np.percentile(array, levels))}


def summarize_records(
    records: Sequence[Mapping[str, object]], diagnostics: Mapping[str, int], partial: bool
) -> Dict[str, object]:
    valid = [record for record in records if record["valid_face"]]
    invalid_reasons = Counter(
        str(record["invalid_reason"]) for record in records if not record["valid_face"]
    )
    task_rows = []
    for task_id in range(len(TASK_SIZES)):
        current = set(task_indices(task_id))
        eligible = [record for record in records if current.intersection(record["target_indices"])]
        valid_eligible = [record for record in eligible if record["valid_face"]]
        class_rows = []
        for class_index in sorted(current):
            positive = [record for record in eligible if class_index in record["target_indices"]]
            valid_positive = sum(bool(record["valid_face"]) for record in positive)
            class_rows.append({
                "class_index": class_index,
                "class_name": CLASS_ORDER[class_index],
                "positive_samples": len(positive),
                "valid_face_positive_samples": valid_positive,
                "valid_face_positive_rate": valid_positive / len(positive) if positive else None,
            })
        task_rows.append({
            "task_id": task_id,
            "current_class_indices": sorted(current),
            "current_class_names": [CLASS_ORDER[index] for index in sorted(current)],
            "eligible_samples": len(eligible),
            "valid_face_samples": len(valid_eligible),
            "valid_face_rate": len(valid_eligible) / len(eligible) if eligible else None,
            "current_class_positive_samples": sum(row["positive_samples"] for row in class_rows),
            "valid_face_current_class_positive_samples": sum(row["valid_face_positive_samples"] for row in class_rows),
            "per_current_class": class_rows,
        })
    tiny_thresholds = {
        str(threshold): sum(float(record["face_short_side"]) < threshold for record in valid)
        for threshold in (16, 24, 32, 48, 64)
    }
    unique_images = len({record["image_path"] for record in records})
    multi_person = [record for record in records if int(record["people_in_image"]) > 1]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": AUDIT_PROTOCOL,
        "partial": partial,
        "samples": len(records),
        "unique_images": unique_images,
        "valid_faces": len(valid),
        "valid_face_rate": len(valid) / len(records) if records else None,
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "multi_person_samples": len(multi_person),
        "multi_person_valid_faces": sum(bool(record["valid_face"]) for record in multi_person),
        "multi_person_ambiguous_matches": sum(bool(record["ambiguous_match"]) for record in multi_person),
        "ambiguous_matches": sum(bool(record["ambiguous_match"]) for record in records),
        "face_candidate_conflicts": int(diagnostics.get("face_candidate_conflicts", 0)),
        "duplicate_face_assignments": int(diagnostics.get("duplicate_face_assignments", 0)),
        "face_width_percentiles": _percentiles(float(record["face_width"]) for record in valid),
        "face_height_percentiles": _percentiles(float(record["face_height"]) for record in valid),
        "face_short_side_percentiles": _percentiles(float(record["face_short_side"]) for record in valid),
        "face_person_area_ratio_percentiles": _percentiles(float(record["face_person_area_ratio"]) for record in valid),
        "tiny_face_counts_by_short_side_px": tiny_thresholds,
        "tasks": task_rows,
    }


def _visual_priority(record: Mapping[str, object]) -> str:
    digest = hashlib.sha256(str(record["sample_id"]).encode("utf-8")).hexdigest()
    return digest


def select_visual_records(
    records: Sequence[Mapping[str, object]], count: int
) -> List[Mapping[str, object]]:
    """Deterministically cover failure modes instead of sampling only easy faces."""
    buckets = (
        [record for record in records if record["ambiguous_match"]],
        [record for record in records if int(record["people_in_image"]) > 1],
        [record for record in records if not record["valid_face"]],
        [record for record in records if record["valid_face"] and float(record["face_short_side"]) < 32],
        [record for record in records if record["valid_face"]],
    )
    selected: List[Mapping[str, object]] = []
    selected_ids = set()
    quota = math.ceil(count / len(buckets)) if count else 0
    for bucket in buckets:
        taken = 0
        for record in sorted(bucket, key=_visual_priority):
            if record["sample_id"] in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(record["sample_id"])
            taken += 1
            if taken >= quota:
                break
    if len(selected) < count:
        for record in sorted(records, key=_visual_priority):
            if record["sample_id"] not in selected_ids:
                selected.append(record)
                selected_ids.add(record["sample_id"])
            if len(selected) >= count:
                break
    return selected[:count]


def write_visual_audit(
    records: Sequence[Mapping[str, object]], output_dir: Path, count: int
) -> List[Dict[str, object]]:
    selected = select_visual_records(records, count)
    tiles: List[Tuple[Image.Image, Dict[str, object]]] = []
    font = ImageFont.load_default()
    for record in selected:
        image = Image.open(str(record["image_absolute_path"])).convert("RGB")
        scale = min(1.0, 480.0 / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.BILINEAR,
            )
        draw = ImageDraw.Draw(image)
        def scaled(box: Sequence[float]) -> Tuple[float, float, float, float]:
            return tuple(float(value) * scale for value in box)  # type: ignore[return-value]
        for face in record["detected_faces"]:
            draw.rectangle(scaled(face["bbox"]), outline=(130, 130, 130), width=2)
        if record["body_bbox"] is not None:
            draw.rectangle(scaled(record["body_bbox"]), outline=(30, 220, 80), width=3)
        if record["face_bbox"] is not None:
            draw.rectangle(scaled(record["face_bbox"]), outline=(255, 210, 0), width=4)
        status = "VALID" if record["valid_face"] else str(record["invalid_reason"])
        label = f"{record['sample_id']} | {status} | amb={record['ambiguous_match']}"
        label_height = 28
        tile = Image.new("RGB", (image.width, image.height + label_height), "white")
        tile.paste(image, (0, label_height))
        ImageDraw.Draw(tile).text((4, 6), label[:100], fill="black", font=font)
        tiles.append((tile, dict(record)))

    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    per_sheet = 20
    for sheet_number in range(0, len(tiles), per_sheet):
        page = tiles[sheet_number:sheet_number + per_sheet]
        cell_width = max(tile.width for tile, _ in page)
        cell_height = max(tile.height for tile, _ in page)
        columns = 4
        rows = math.ceil(len(page) / columns)
        sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
        sheet_name = f"visual_audit_{sheet_number // per_sheet:03d}.jpg"
        for offset, (tile, record) in enumerate(page):
            x, y = (offset % columns) * cell_width, (offset // columns) * cell_height
            sheet.paste(tile, (x, y))
            index_rows.append({
                "sample_id": record["sample_id"],
                "sheet": sheet_name,
                "cell": offset,
                "valid_face": record["valid_face"],
                "ambiguous_match": record["ambiguous_match"],
            })
        sheet.save(output_dir / sheet_name, quality=90)
    _write_json(output_dir / "visual_review_index.json", index_rows)
    return index_rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--detector-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val"), default=("train", "val"))
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--det-threshold", type=float, default=0.5)
    parser.add_argument("--face-margin", type=float, default=0.15)
    parser.add_argument("--ambiguity-gap", type=float, default=0.08)
    parser.add_argument("--visual-samples", type=int, default=200)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.det_size <= 0 or not 0 <= args.det_threshold <= 1:
        parser.error("det-size must be positive and det-threshold must be in [0, 1]")
    if not 0 <= args.face_margin <= 1 or not 0 <= args.ambiguity_gap <= 1:
        parser.error("face-margin and ambiguity-gap must be in [0, 1]")
    if args.visual_samples < 0 or (args.max_images is not None and args.max_images <= 0):
        parser.error("visual-samples must be non-negative and max-images positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    checkpoint = args.detector_checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing face detector checkpoint: {checkpoint}")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = detector_metadata(checkpoint, args.det_size, args.det_threshold)
    metadata.update({
        "face_margin": args.face_margin,
        "ambiguity_gap": args.ambiguity_gap,
        "splits": list(args.splits),
    })
    metadata_path = output_root / "detector_config.json"
    if metadata_path.exists() and args.resume:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        stable_keys = ("checkpoint_sha256", "provider", "det_size", "det_threshold")
        if any(previous.get(key) != metadata.get(key) for key in stable_keys):
            raise RuntimeError("Resume requested with a different detector configuration")
    _write_json(metadata_path, metadata)

    parent = resolve_dataset_parent(args.data_root)
    sources = {
        split: EMOTIC(
            str(parent), train=(split == "train"), transform=lambda image: image,
            eval_splits=(split,) if split != "train" else ("val",),
        )
        for split in args.splits
    }
    for source in sources.values():
        if tuple(source.classes) != CLASS_ORDER:
            raise RuntimeError("EMOTIC class order differs from the frozen Track-A protocol")

    detector = InsightFaceDetector(checkpoint, args.det_size, args.det_threshold)
    all_summaries: Dict[str, object] = {}
    all_records: List[Dict[str, object]] = []
    for split, source in sources.items():
        image_paths = sorted({Path(path).resolve() for path in source.file_paths}, key=str)
        cache = detect_images(
            image_paths, detector, output_root / "detections" / f"{split}.jsonl",
            args.resume, args.max_images,
        )
        records, diagnostics = build_manifest_records(
            source, split, cache, args.face_margin, args.ambiguity_gap
        )
        partial = len(cache) < len(image_paths)
        summary = summarize_records(records, diagnostics, partial)
        summary["expected_unique_images"] = len(image_paths)
        summary["processed_unique_images"] = len(cache)
        write_jsonl(output_root / "manifests" / f"{split}.jsonl", records)
        _write_json(output_root / "summaries" / f"{split}.json", summary)
        all_summaries[split] = summary
        all_records.extend(records)
        print(
            f"split={split} samples={summary['samples']} images={summary['processed_unique_images']}/"
            f"{summary['expected_unique_images']} valid={summary['valid_faces']} "
            f"rate={summary['valid_face_rate']}",
            flush=True,
        )
    train_records = [record for record in all_records if record["split"] == "train"]
    review_source = train_records if train_records else all_records
    review = write_visual_audit(review_source, output_root / "visual_audit", args.visual_samples)
    overall = {
        "schema_version": SCHEMA_VERSION,
        "protocol": AUDIT_PROTOCOL,
        "detector_config": metadata,
        "splits": all_summaries,
        "visual_review_samples": len(review),
        "training_started": False,
    }
    _write_json(output_root / "audit_summary.json", overall)
    print(f"FACE_MANIFEST_AUDIT_COMPLETE output_root={output_root}", flush=True)


if __name__ == "__main__":
    main()
