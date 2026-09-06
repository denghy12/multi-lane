"""Evaluate the validation-locked 80/20 rule with the initial Person crop."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

from .evaluation_scores import align_evaluation_scores
from .fuse_fixed_test_scores import (
    LOCKED_MODE,
    LOCKED_PERSON_ALPHA,
    LOCKED_THRESHOLD,
    LOCKED_VALIDATION_SUMMARY_SHA256,
    _compute_rows,
    _load_json,
    _metrics_close,
    _sha256,
    _validate_selection,
)
from .fuse_validation_scores import COMMON_CONFIG_FIELDS, validated_run_scores
from .runner import TASK_SIZES, summarize_tasks


INITIAL_PERSON_TRANSFORM = "legacy_crop"
INITIAL_PERSON_MARGIN = 0.15
INITIAL_PERSON_CROP_SCALE = (0.70, 1.0)


def _validate_initial_person_runs(
    full_run: Path, person_run: Path
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    full_config = _load_json(full_run / "config.json")
    person_config = _load_json(person_run / "config.json")
    full_summary = _load_json(full_run / "seed_summary.json")
    person_summary = _load_json(person_run / "seed_summary.json")
    for label, config, summary in (
        ("full", full_config, full_summary),
        ("person", person_config, person_summary),
    ):
        if summary.get("status") != "complete":
            raise ValueError(f"{label} fixed-fusion run is incomplete")
        if config.get("reporting_split") != "test":
            raise ValueError(f"{label} fixed-fusion input must report held-out test")
        if not config.get("save_evaluation_scores"):
            raise ValueError(f"{label} run did not export fixed-fusion scores")
        if config.get("evaluation_score_purpose") != "fixed_test_fusion":
            raise ValueError(f"{label} score export is not marked fixed_test_fusion")
        if not math.isclose(
            float(config.get("threshold", -1)), LOCKED_THRESHOLD, abs_tol=1e-12
        ):
            raise ValueError(f"{label} threshold differs from the locked value")
        if len(summary.get("task_metrics", [])) != len(TASK_SIZES):
            raise ValueError(f"{label} run does not contain all protocol tasks")
        git_metadata = config.get("git", {})
        if git_metadata.get("dirty") is not False or not git_metadata.get("commit"):
            raise ValueError(f"{label} run does not record a clean commit")
    if full_config.get("input_mode") != "full":
        raise ValueError("The full fixed-fusion input is not a full-image run")
    if person_config.get("input_mode") != "person_crop":
        raise ValueError("The Person fixed-fusion input is not a person-crop run")
    if person_config.get("person_transform_mode") != INITIAL_PERSON_TRANSFORM:
        raise ValueError("The Person input is not the initial legacy crop transform")
    if not math.isclose(
        float(person_config.get("person_crop_margin", -1)),
        INITIAL_PERSON_MARGIN,
        abs_tol=1e-12,
    ):
        raise ValueError("The initial Person crop margin must be 0.15")
    crop_scale = tuple(float(value) for value in person_config.get("train_crop_scale", ()))
    if crop_scale != INITIAL_PERSON_CROP_SCALE:
        raise ValueError("The initial Person train crop scale must be [0.70, 1.0]")
    if (
        float(person_config.get("person_color_jitter_strength", 0.0)) != 0.0
        or float(person_config.get("person_color_jitter_probability", 0.0)) != 0.0
    ):
        raise ValueError("The initial Person transform must not use ColorJitter")
    for field in COMMON_CONFIG_FIELDS:
        if full_config.get(field) != person_config.get(field):
            raise ValueError(f"Fixed-fusion runs differ on fixed field {field}")
    return full_config, person_config, full_summary, person_summary


def fuse_initial_person_fixed_test_runs(
    full_run: Path,
    person_run: Path,
    validation_fusion_summary: Path,
    expected_validation_sha256: str = LOCKED_VALIDATION_SUMMARY_SHA256,
) -> Dict[str, Any]:
    selection = _validate_selection(validation_fusion_summary, expected_validation_sha256)
    full_config, person_config, full_summary, person_summary = (
        _validate_initial_person_runs(full_run, person_run)
    )
    full_dumps, _ = validated_run_scores(full_run, "test")
    person_dumps, _ = validated_run_scores(person_run, "test")
    task_arrays = [
        align_evaluation_scores(full_dump, person_dump)
        for full_dump, person_dump in zip(full_dumps, person_dumps)
    ]
    full_rows = _compute_rows(task_arrays, "logit", 0.0)
    person_rows = _compute_rows(task_arrays, "logit", 1.0)
    fusion_rows = _compute_rows(task_arrays, LOCKED_MODE, LOCKED_PERSON_ALPHA)
    for label, calculated_rows, recorded in (
        ("full", full_rows, full_summary),
        ("person", person_rows, person_summary),
    ):
        for calculated, expected in zip(calculated_rows, recorded["task_metrics"]):
            if not _metrics_close(asdict(calculated), expected):
                raise ValueError(f"{label} test scores do not reproduce task metrics")

    full_metrics = summarize_tasks(full_rows)
    person_metrics = summarize_tasks(person_rows)
    fusion_metrics = summarize_tasks(fusion_rows)
    return {
        "schema_version": 1,
        "comparison": "fixed_full_initial_person_legacy_crop_test_fusion",
        "evaluation_split": "test",
        "search_performed_on_test": False,
        "evaluated_fusion_rule_count": 1,
        "seed": int(full_config["seed"]),
        "locked_rule": {
            "mode": LOCKED_MODE,
            "full_weight": 1.0 - LOCKED_PERSON_ALPHA,
            "person_weight": LOCKED_PERSON_ALPHA,
            "threshold": LOCKED_THRESHOLD,
            "source": "letterbox_validation_winner_transferred_without_search",
        },
        "validation_selection": {
            "path": str(validation_fusion_summary.resolve()),
            "sha256": _sha256(validation_fusion_summary),
            "validation_final_mAP": selection["winner"]["metrics"]["final_mAP"],
        },
        "sources": {
            "full_run": str(full_run.resolve()),
            "person_run": str(person_run.resolve()),
            "full_git": full_config["git"],
            "person_git": person_config["git"],
            "person_transform": {
                "mode": INITIAL_PERSON_TRANSFORM,
                "margin": INITIAL_PERSON_MARGIN,
                "train_crop_scale": list(INITIAL_PERSON_CROP_SCALE),
                "color_jitter_strength": 0.0,
                "color_jitter_probability": 0.0,
            },
        },
        "anchors": {
            "full": {
                "metrics": full_metrics,
                "task_metrics": [asdict(row) for row in full_rows],
            },
            "initial_person": {
                "metrics": person_metrics,
                "task_metrics": [asdict(row) for row in person_rows],
            },
        },
        "fixed_fusion": {
            "metrics": fusion_metrics,
            "task_metrics": [asdict(row) for row in fusion_rows],
        },
        "comparison_to_full": {
            key: float(fusion_metrics[key]) - float(full_metrics[key])
            for key in ("final_mAP", "average_mAP", "final_cF1", "final_oF1", "forgetting")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        "Apply the fixed 80/20 rule to Full and the initial Person crop on test"
    )
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--person-run", type=Path, required=True)
    parser.add_argument("--validation-fusion-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fuse_initial_person_fixed_test_runs(
        args.full_run, args.person_run, args.validation_fusion_summary
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "INITIAL_PERSON_FIXED_TEST_FUSION_COMPLETE "
        f"final_mAP={result['fixed_fusion']['metrics']['final_mAP']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
