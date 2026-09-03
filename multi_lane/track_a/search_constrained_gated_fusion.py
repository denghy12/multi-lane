"""Select a constrained full/person fusion rule on validation, then test it once."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from multi_lane.continual_datasets.continual_datasets import EMOTIC

from .compare_validation_ensembles import audit_validation_run
from .evaluation_scores import align_evaluation_scores
from .fuse_fixed_test_scores import _validate_test_runs
from .fuse_validation_scores import COMMON_CONFIG_FIELDS, validated_run_scores
from .runner import CLASS_ORDER, TaskMetrics, average_precision, compute_metrics, summarize_tasks


ALPHA_GRID = tuple(round(index / 100.0, 2) for index in range(101))
ANCHOR_PERSON_WEIGHT = 0.20
GATE_DELTAS = (0.0, 0.025, 0.05)
MIN_DIRECTION_AP = 0.05
MIN_CELL_SAMPLES = 50
AREA_SMALL_MAX = 0.10
AREA_MEDIUM_MAX = 0.30
ASPECT_EXTREME_MIN = 2.0
THRESHOLD = 0.5
METRIC_NAMES = ("final_mAP", "average_mAP", "final_cF1", "final_oF1", "forgetting")


@dataclass(frozen=True)
class Geometry:
    bbox_area_ratio: float
    absolute_aspect_ratio: float
    people_in_image: int
    cell: str


@dataclass
class ScorePair:
    seed: int
    full_run: Path
    person_run: Path
    tasks: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    full_rows: List[TaskMetrics]
    person_rows: List[TaskMetrics]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


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
        for metric in METRIC_NAMES
    }


def _config_signature(config: Mapping[str, Any]) -> Dict[str, Any]:
    fields = [field for field in COMMON_CONFIG_FIELDS if field != "seed"]
    fields.extend(
        (
            "input_mode",
            "person_crop_margin",
            "person_transform_mode",
            "person_color_jitter_strength",
            "person_color_jitter_probability",
            "train_crop_scale",
        )
    )
    return {field: config.get(field) for field in fields}


def load_score_pairs(
    full_runs: Sequence[Path], person_runs: Sequence[Path], split: str
) -> Dict[int, ScorePair]:
    if len(full_runs) != 3 or len(person_runs) != 3:
        raise ValueError("Exactly three full and three person runs are required")
    by_seed: Dict[Tuple[str, int], Path] = {}
    signatures: Dict[str, Dict[str, Any]] = {}
    for view, runs in (("full", full_runs), ("person_crop", person_runs)):
        for run in runs:
            config = _read_json(run / "config.json")
            seed = int(config.get("seed", -1))
            if seed not in (0, 1, 2) or (view, seed) in by_seed:
                raise ValueError("Runs must contain unique seeds 0, 1 and 2 per view")
            if config.get("input_mode") != view:
                raise ValueError("Input view does not match the supplied run group")
            signature = _config_signature(config)
            if view in signatures and signature != signatures[view]:
                raise ValueError(f"Cross-seed configuration drift for {view}")
            signatures.setdefault(view, signature)
            by_seed[(view, seed)] = run

    pairs: Dict[int, ScorePair] = {}
    for seed in range(3):
        full_run = by_seed[("full", seed)]
        person_run = by_seed[("person_crop", seed)]
        if split == "val":
            full_config, full_dumps, full_rows = audit_validation_run(full_run, "full")
            person_config, person_dumps, person_rows = audit_validation_run(person_run, "person_crop")
        elif split == "test":
            _validate_test_runs(full_run, person_run)
            full_config = _read_json(full_run / "config.json")
            person_config = _read_json(person_run / "config.json")
            full_dumps, full_rows = validated_run_scores(full_run, "test")
            person_dumps, person_rows = validated_run_scores(person_run, "test")
        else:
            raise ValueError("Split must be val or test")
        if full_config["seed"] != seed or person_config["seed"] != seed:
            raise ValueError("Full/person seed mismatch")
        for field in COMMON_CONFIG_FIELDS:
            if full_config.get(field) != person_config.get(field):
                raise ValueError(f"Full/person configuration drift: {field}")
        tasks = []
        for full, person in zip(full_dumps, person_dumps):
            ids, _, _, targets, full_probs, person_probs = align_evaluation_scores(full, person)
            tasks.append((ids, targets, np.stack((full_probs, person_probs))))
        pairs[seed] = ScorePair(
            seed, full_run, person_run, tasks, list(full_rows), list(person_rows)
        )
    return pairs


def geometry_cell(area_ratio: float, aspect_ratio: float, people: int) -> str:
    if area_ratio < AREA_SMALL_MAX:
        area = "small_lt_0.10"
    elif area_ratio < AREA_MEDIUM_MAX:
        area = "medium_0.10_0.30"
    else:
        area = "large_ge_0.30"
    aspect = "extreme_gt_2.0" if aspect_ratio > ASPECT_EXTREME_MIN else "regular_le_2.0"
    crowd = "multi" if people > 1 else "single"
    return f"area={area}|aspect={aspect}|people={crowd}"


def load_geometry(data_root: Path, split: str) -> Dict[str, Geometry]:
    dataset = EMOTIC(
        str(data_root), train=False, transform=None, download=False, eval_splits=(split,)
    )
    image_keys = [sample_id.rsplit("#person=", 1)[0] for sample_id in dataset.sample_ids]
    people_by_image = Counter(image_keys)
    image_sizes: Dict[str, Tuple[int, int]] = {}
    result: Dict[str, Geometry] = {}
    for sample_id, image_key, image_path, raw_bbox in zip(
        dataset.sample_ids, image_keys, dataset.file_paths, dataset.body_bboxes
    ):
        if image_path not in image_sizes:
            with Image.open(image_path) as image:
                image_sizes[image_path] = image.size
        image_width, image_height = image_sizes[image_path]
        bbox = np.asarray(raw_bbox, dtype=np.float64).ravel()
        if bbox.size < 4 or not np.isfinite(bbox[:4]).all():
            raise ValueError(f"Invalid bbox metadata for {sample_id}")
        x1, y1, x2, y2 = bbox[:4]
        x1, x2 = np.clip((x1, x2), 0.0, float(image_width))
        y1, y2 = np.clip((y1, y2), 0.0, float(image_height))
        width, height = float(x2 - x1), float(y2 - y1)
        if width <= 0 or height <= 0:
            raise ValueError(f"Degenerate bbox metadata for {sample_id}")
        area_ratio = width * height / float(image_width * image_height)
        aspect_ratio = max(width / height, height / width)
        people = people_by_image[image_key]
        result[sample_id] = Geometry(
            area_ratio, aspect_ratio, people, geometry_cell(area_ratio, aspect_ratio, people)
        )
    return result


def _rows_for_rule(
    pair: ScorePair,
    base_alpha: float,
    geometry: Optional[Mapping[str, Geometry]] = None,
    class_signs: Optional[Sequence[int]] = None,
    quality_signs: Optional[Mapping[str, int]] = None,
    class_delta: float = 0.0,
    quality_delta: float = 0.0,
) -> List[TaskMetrics]:
    rows: List[TaskMetrics] = []
    for task_id, (sample_ids, targets, endpoints) in enumerate(pair.tasks):
        full_probs, person_probs = endpoints
        weights = np.full(full_probs.shape, float(base_alpha), dtype=np.float32)
        if class_signs is not None:
            signs = np.asarray(class_signs[: full_probs.shape[1]], dtype=np.float32)
            weights += float(class_delta) * signs[None, :]
        if quality_signs is not None:
            if geometry is None:
                raise ValueError("Geometry is required for quality-aware fusion")
            missing = [sample_id for sample_id in sample_ids if sample_id not in geometry]
            if missing:
                raise ValueError(f"Missing geometry for {missing[0]}")
            signs = np.asarray(
                [quality_signs.get(geometry[sample_id].cell, 0) for sample_id in sample_ids],
                dtype=np.float32,
            )
            weights += float(quality_delta) * signs[:, None]
        weights = np.clip(weights, 0.0, 1.0)
        probabilities = (1.0 - weights) * full_probs + weights * person_probs
        rows.append(
            compute_metrics(
                task_id,
                torch.from_numpy(probabilities),
                torch.from_numpy(targets),
                THRESHOLD,
            )
        )
    return rows


def _record_rule(
    pairs: Mapping[int, ScorePair],
    base_alpha: float,
    geometry: Optional[Mapping[str, Geometry]] = None,
    class_signs: Optional[Sequence[int]] = None,
    quality_signs: Optional[Mapping[str, int]] = None,
    class_delta: float = 0.0,
    quality_delta: float = 0.0,
) -> Dict[str, Any]:
    seeds = []
    for seed in range(3):
        rows = _rows_for_rule(
            pairs[seed], base_alpha, geometry, class_signs, quality_signs,
            class_delta, quality_delta,
        )
        seeds.append(
            {
                "seed": seed,
                "metrics": summarize_tasks(rows),
                "task_metrics": [asdict(row) for row in rows],
            }
        )
    return {"seeds": seeds, "aggregate": _aggregate(seeds)}


def search_global_alpha(pairs: Mapping[int, ScorePair]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidates = []
    for alpha in ALPHA_GRID:
        record = _record_rule(pairs, alpha)
        candidates.append({"person_weight": alpha, **record})
    winner = max(
        candidates,
        key=lambda row: (
            row["aggregate"]["final_mAP"]["mean"],
            row["aggregate"]["average_mAP"]["mean"],
            -abs(row["person_weight"] - ANCHOR_PERSON_WEIGHT),
        ),
    )
    return candidates, winner


def _supported_map(scores: np.ndarray, targets: np.ndarray) -> Tuple[float, int]:
    supported = np.flatnonzero(targets.sum(axis=0) > 0)
    if not len(supported):
        return 0.0, 0
    values = [100.0 * average_precision(scores[:, index], targets[:, index]) for index in supported]
    return float(np.mean(values)), int(len(supported))


def geometry_diagnostics(
    pairs: Mapping[int, ScorePair], geometry: Mapping[str, Geometry]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    cells = sorted({item.cell for item in geometry.values()})
    records = []
    quality_signs: Dict[str, int] = {}
    anchor_record = _record_rule(pairs, ANCHOR_PERSON_WEIGHT)
    anchor_overall = {
        seed: anchor_record["seeds"][seed]["metrics"]["final_mAP"]
        - summarize_tasks(pairs[seed].full_rows)["final_mAP"]
        for seed in range(3)
    }
    # The final task contains all 26 classes and all annotated validation people.
    for cell in cells:
        seed_rows = []
        for seed in range(3):
            sample_ids, targets, endpoints = pairs[seed].tasks[-1]
            mask = np.asarray([geometry[sample_id].cell == cell for sample_id in sample_ids])
            full_probs, person_probs = endpoints[:, mask, :]
            cell_targets = targets[mask]
            fusion_probs = 0.8 * full_probs + 0.2 * person_probs
            full_map, supported = _supported_map(full_probs, cell_targets)
            person_map, _ = _supported_map(person_probs, cell_targets)
            fusion_map, _ = _supported_map(fusion_probs, cell_targets)
            gain = fusion_map - full_map
            seed_rows.append(
                {
                    "seed": seed,
                    "samples": int(mask.sum()),
                    "supported_classes": supported,
                    "full_mAP": full_map,
                    "person_mAP": person_map,
                    "fixed_0.20_fusion_mAP": fusion_map,
                    "fusion_minus_full": gain,
                    "relative_to_overall_fusion_gain": gain - anchor_overall[seed],
                }
            )
        relative = [row["relative_to_overall_fusion_gain"] for row in seed_rows]
        enough = min(row["samples"] for row in seed_rows) >= MIN_CELL_SAMPLES
        if enough and all(value >= MIN_DIRECTION_AP for value in relative):
            sign = 1
        elif enough and all(value <= -MIN_DIRECTION_AP for value in relative):
            sign = -1
        else:
            sign = 0
        quality_signs[cell] = sign
        records.append(
            {
                "cell": cell,
                "seed_results": seed_rows,
                "fusion_minus_full": _stats([row["fusion_minus_full"] for row in seed_rows]),
                "relative_gain": _stats(relative),
                "quality_direction": sign,
            }
        )
    return records, quality_signs


def consistent_class_directions(pairs: Mapping[int, ScorePair]) -> Tuple[List[Dict[str, Any]], List[int]]:
    deltas = []
    for seed in range(3):
        row = _rows_for_rule(pairs[seed], ANCHOR_PERSON_WEIGHT)[-1]
        full = pairs[seed].full_rows[-1]
        deltas.append(np.asarray(row.per_class_ap) - np.asarray(full.per_class_ap))
    matrix = np.stack(deltas)
    records, signs = [], []
    for index, name in enumerate(CLASS_ORDER):
        values = matrix[:, index].tolist()
        if all(value >= MIN_DIRECTION_AP for value in values):
            sign = 1
        elif all(value <= -MIN_DIRECTION_AP for value in values):
            sign = -1
        else:
            sign = 0
        signs.append(sign)
        records.append(
            {
                "class_index": index,
                "class_name": name,
                "fixed_0.20_fusion_minus_full": values,
                "mean": statistics.mean(values),
                "direction": sign,
            }
        )
    return records, signs


def search_constrained_gate(
    pairs: Mapping[int, ScorePair],
    geometry: Mapping[str, Geometry],
    class_signs: Sequence[int],
    quality_signs: Mapping[str, int],
    global_winner: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    global_by_seed = {
        row["seed"]: row["metrics"]["final_mAP"] for row in global_winner["seeds"]
    }
    candidates = []
    for class_delta in GATE_DELTAS:
        for quality_delta in GATE_DELTAS:
            record = _record_rule(
                pairs,
                ANCHOR_PERSON_WEIGHT,
                geometry,
                class_signs,
                quality_signs,
                class_delta,
                quality_delta,
            )
            per_seed = {row["seed"]: row["metrics"]["final_mAP"] for row in record["seeds"]}
            eligible = all(per_seed[seed] >= global_by_seed[seed] - 1e-12 for seed in range(3))
            candidates.append(
                {
                    "base_person_weight": ANCHOR_PERSON_WEIGHT,
                    "class_delta": class_delta,
                    "quality_delta": quality_delta,
                    "eligible_vs_global_each_seed": eligible,
                    **record,
                }
            )
    eligible = [
        row for row in candidates
        if row["eligible_vs_global_each_seed"]
        and row["aggregate"]["final_mAP"]["mean"]
        > global_winner["aggregate"]["final_mAP"]["mean"] + 1e-12
    ]
    winner = max(
        eligible,
        key=lambda row: (
            row["aggregate"]["final_mAP"]["mean"],
            row["aggregate"]["average_mAP"]["mean"],
            -(row["class_delta"] + row["quality_delta"]),
        ),
        default=None,
    )
    return candidates, winner


def select_validation_rule(
    pairs: Mapping[int, ScorePair], geometry: Mapping[str, Geometry]
) -> Dict[str, Any]:
    global_candidates, global_winner = search_global_alpha(pairs)
    class_records, class_signs = consistent_class_directions(pairs)
    geometry_records, quality_signs = geometry_diagnostics(pairs, geometry)
    gate_candidates, gate_winner = search_constrained_gate(
        pairs, geometry, class_signs, quality_signs, global_winner
    )
    if gate_winner is None:
        selected_rule = {
            "type": "global_probability_weight",
            "person_weight": global_winner["person_weight"],
            "class_delta": 0.0,
            "quality_delta": 0.0,
        }
        selected_metrics = global_winner
    else:
        selected_rule = {
            "type": "constrained_class_geometry_gate",
            "base_person_weight": ANCHOR_PERSON_WEIGHT,
            "class_delta": gate_winner["class_delta"],
            "quality_delta": gate_winner["quality_delta"],
            "class_signs": class_signs,
            "quality_signs": quality_signs,
        }
        selected_metrics = gate_winner
    return {
        "schema_version": 1,
        "selection_split": "val",
        "search_performed_on_test": False,
        "objective": "mean final mAP across seeds; average mAP tie-break",
        "global_search": {
            "alpha_grid": list(ALPHA_GRID),
            "anchor_person_weight": ANCHOR_PERSON_WEIGHT,
            "candidates": global_candidates,
            "winner": global_winner,
            "per_seed_independent_optima": [
                max(
                    (candidate for candidate in global_candidates),
                    key=lambda row, seed=seed: row["seeds"][seed]["metrics"]["final_mAP"],
                )["person_weight"]
                for seed in range(3)
            ],
        },
        "class_diagnostics": class_records,
        "geometry_diagnostics": geometry_records,
        "gate_definition": {
            "class_direction_threshold_ap": MIN_DIRECTION_AP,
            "quality_relative_direction_threshold_map": MIN_DIRECTION_AP,
            "minimum_cell_samples": MIN_CELL_SAMPLES,
            "candidate_deltas": list(GATE_DELTAS),
            "eligibility": "candidate final mAP must be >= global winner for every seed",
            "class_signs": class_signs,
            "quality_signs": quality_signs,
        },
        "gate_candidates": gate_candidates,
        "gate_winner": gate_winner,
        "selected_rule": selected_rule,
        "selected_validation_metrics": selected_metrics,
    }


def evaluate_locked_test(
    pairs: Mapping[int, ScorePair],
    geometry: Mapping[str, Geometry],
    selection: Mapping[str, Any],
    selection_path: Path,
) -> Dict[str, Any]:
    rule = selection["selected_rule"]
    seed_rows = []
    full_rows = []
    for seed in range(3):
        pair = pairs[seed]
        if rule["type"] == "global_probability_weight":
            rows = _rows_for_rule(pair, float(rule["person_weight"]))
        elif rule["type"] == "constrained_class_geometry_gate":
            rows = _rows_for_rule(
                pair,
                float(rule["base_person_weight"]),
                geometry,
                rule["class_signs"],
                rule["quality_signs"],
                float(rule["class_delta"]),
                float(rule["quality_delta"]),
            )
        else:
            raise ValueError("Unknown locked fusion rule")
        seed_rows.append(
            {
                "seed": seed,
                "metrics": summarize_tasks(rows),
                "task_metrics": [asdict(row) for row in rows],
            }
        )
        full_rows.append({"seed": seed, "metrics": summarize_tasks(pair.full_rows)})
    return {
        "schema_version": 1,
        "evaluation_split": "test",
        "search_performed_on_test": False,
        "evaluated_fusion_rule_count": 1,
        "locked_validation_selection": {
            "path": str(selection_path.resolve()),
            "sha256": _sha256(selection_path),
        },
        "locked_rule": rule,
        "seeds": seed_rows,
        "aggregate": _aggregate(seed_rows),
        "full_anchor": {"seeds": full_rows, "aggregate": _aggregate(full_rows)},
        "paired_final_mAP_gain": _stats(
            [
                seed_rows[seed]["metrics"]["final_mAP"]
                - full_rows[seed]["metrics"]["final_mAP"]
                for seed in range(3)
            ]
        ),
    }


def _source_records(pairs: Mapping[int, ScorePair]) -> List[Dict[str, Any]]:
    records = []
    for seed in range(3):
        pair = pairs[seed]
        for view, run in (("full", pair.full_run), ("person_crop", pair.person_run)):
            records.append(
                {
                    "seed": seed,
                    "view": view,
                    "run": str(run.resolve()),
                    "config_sha256": _sha256(run / "config.json"),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--validation-full-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--validation-person-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--test-full-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--test-person-runs", type=Path, nargs=3, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    validation_pairs = load_score_pairs(
        args.validation_full_runs, args.validation_person_runs, "val"
    )
    validation_geometry = load_geometry(args.data_root, "val")
    selection = select_validation_rule(validation_pairs, validation_geometry)
    selection["sources"] = _source_records(validation_pairs)
    selection_path = args.output_dir / "validation_selection.json"
    with selection_path.open("x", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        "VALIDATION_RULE_LOCKED",
        json.dumps(selection["selected_rule"], ensure_ascii=False),
        flush=True,
    )

    test_pairs = load_score_pairs(args.test_full_runs, args.test_person_runs, "test")
    test_geometry = load_geometry(args.data_root, "test")
    result = evaluate_locked_test(test_pairs, test_geometry, selection, selection_path)
    result["sources"] = _source_records(test_pairs)
    test_path = args.output_dir / "fixed_test_evaluation.json"
    with test_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result["aggregate"], indent=2), flush=True)
    print("LOCKED_TEST_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
