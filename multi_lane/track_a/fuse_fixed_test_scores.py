"""Apply the validation-selected dual-view rule once on held-out test scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from .fuse_validation_scores import (
    COMMON_CONFIG_FIELDS,
    fused_scores,
    validated_run_scores,
)
from .evaluation_scores import align_evaluation_scores
from .runner import TASK_SIZES, TaskMetrics, compute_metrics, summarize_tasks


LOCKED_MODE = "probability"
LOCKED_PERSON_ALPHA = 0.20
LOCKED_THRESHOLD = 0.5
LOCKED_VALIDATION_SUMMARY_SHA256 = (
    "a03ff4da93e18bcfde4b4e8898639608e341dfff335d6f66fc355ee4df96b90f"
)
REFERENCE_CHAMPION_FINAL_MAP = 32.53651921448899


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing fixed-fusion artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics_close(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return all(
        math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1e-5)
        for key in ("mAP", "cF1", "oF1")
    )


def _validate_selection(
    summary_path: Path, expected_sha256: str
) -> Dict[str, Any]:
    if _sha256(summary_path) != expected_sha256:
        raise ValueError("Validation fusion summary does not match the locked artifact")
    selection = _load_json(summary_path)
    winner = selection.get("winner", {})
    decision = selection.get("decision", {})
    if selection.get("selection_split") != "val":
        raise ValueError("The locked fusion rule must originate from validation")
    if winner.get("mode") != LOCKED_MODE or not math.isclose(
        float(winner.get("alpha", -1)), LOCKED_PERSON_ALPHA, abs_tol=1e-12
    ):
        raise ValueError("Validation winner does not match the locked fusion rule")
    if not decision.get("advance_to_formal_test"):
        raise ValueError("Validation selection did not approve formal test")
    return selection


def _validate_test_runs(
    full_run: Path, person_run: Path
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    full_config = _load_json(full_run / "config.json")
    person_config = _load_json(person_run / "config.json")
    full_summary = _load_json(full_run / "seed_summary.json")
    person_summary = _load_json(person_run / "seed_summary.json")
    for label, config, summary in (
        ("full", full_config, full_summary),
        ("person", person_config, person_summary),
    ):
        if summary.get("status") != "complete":
            raise ValueError(f"{label} formal test run is incomplete")
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
            raise ValueError(f"{label} formal test run does not record a clean commit")
    if full_config.get("input_mode") != "full":
        raise ValueError("The full fixed-fusion input is not a full-image run")
    if person_config.get("input_mode") != "person_crop":
        raise ValueError("The person fixed-fusion input is not a person-crop run")
    if person_config.get("person_transform_mode") != "letterbox":
        raise ValueError("The person fixed-fusion input is not letterbox")
    for field in COMMON_CONFIG_FIELDS:
        if full_config.get(field) != person_config.get(field):
            raise ValueError(f"Formal fusion runs differ on fixed field {field}")
    if full_config["git"]["commit"] != person_config["git"]["commit"]:
        raise ValueError("Formal fusion runs were not produced by the same commit")
    return full_summary, person_summary


def _compute_rows(
    task_arrays,
    mode: str,
    alpha: float,
) -> List[TaskMetrics]:
    rows: List[TaskMetrics] = []
    for task_id, (_, full_logits, person_logits, targets, full_probs, person_probs) in enumerate(task_arrays):
        scores = fused_scores(full_logits, person_logits, alpha, mode, full_probs, person_probs)
        rows.append(
            compute_metrics(
                task_id,
                torch.from_numpy(scores),
                torch.from_numpy(targets),
                LOCKED_THRESHOLD,
            )
        )
    return rows


def fuse_fixed_test_runs(
    full_run: Path,
    person_run: Path,
    validation_fusion_summary: Path,
    expected_validation_sha256: str = LOCKED_VALIDATION_SUMMARY_SHA256,
) -> Dict[str, Any]:
    selection = _validate_selection(
        validation_fusion_summary, expected_validation_sha256
    )
    full_summary, person_summary = _validate_test_runs(full_run, person_run)
    full_dumps, _ = validated_run_scores(full_run, 'test')
    person_dumps, _ = validated_run_scores(person_run, 'test')
    task_arrays = [align_evaluation_scores(a, b) for a, b in zip(full_dumps, person_dumps)]

    full_rows = _compute_rows(task_arrays, "logit", 0.0)
    person_rows = _compute_rows(task_arrays, "logit", 1.0)
    fusion_rows = _compute_rows(
        task_arrays, LOCKED_MODE, LOCKED_PERSON_ALPHA
    )
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
    fusion_final = float(fusion_metrics["final_mAP"])
    full_final = float(full_metrics["final_mAP"])
    return {
        "schema_version": 1,
        "seed": full_summary["config"]["seed"],
        "comparison": "locked_full_person_letterbox_formal_test_fusion",
        "evaluation_split": "test",
        "search_performed_on_test": False,
        "probability_source": {
            "full": [dump.probability_source for dump in full_dumps],
            "person": [dump.probability_source for dump in person_dumps],
        },
        "locked_rule": {
            "mode": LOCKED_MODE,
            "full_weight": 1.0 - LOCKED_PERSON_ALPHA,
            "person_weight": LOCKED_PERSON_ALPHA,
            "threshold": LOCKED_THRESHOLD,
        },
        "validation_selection": {
            "path": str(validation_fusion_summary.resolve()),
            "sha256": _sha256(validation_fusion_summary),
            "validation_final_mAP": selection["winner"]["metrics"]["final_mAP"],
        },
        "full_run": str(full_run.resolve()),
        "person_run": str(person_run.resolve()),
        "anchors": {
            "full": {
                "metrics": full_metrics,
                "task_metrics": [asdict(row) for row in full_rows],
            },
            "person": {
                "metrics": person_metrics,
                "task_metrics": [asdict(row) for row in person_rows],
            },
        },
        "fixed_fusion": {
            "metrics": fusion_metrics,
            "task_metrics": [asdict(row) for row in fusion_rows],
        },
        "reference_champion_final_mAP": REFERENCE_CHAMPION_FINAL_MAP,
        "reference_champion_seed": 0,
        "decision": {
            "beats_same_run_full_anchor": fusion_final > full_final,
            "gain_over_same_run_full_anchor": fusion_final - full_final,
            "beats_reference_champion": fusion_final > REFERENCE_CHAMPION_FINAL_MAP,
            "gain_over_reference_champion": (
                fusion_final - REFERENCE_CHAMPION_FINAL_MAP
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        "Apply the validation-locked full/person rule once on held-out test"
    )
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--person-run", type=Path, required=True)
    parser.add_argument(
        "--validation-fusion-summary", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fuse_fixed_test_runs(
        args.full_run, args.person_run, args.validation_fusion_summary
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2), flush=True)
    print(
        "DUAL_VIEW_FIXED_TEST_FUSION_COMPLETE "
        f"mode={LOCKED_MODE} person_alpha={LOCKED_PERSON_ALPHA:.2f} "
        f"final_mAP={result['fixed_fusion']['metrics']['final_mAP']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
