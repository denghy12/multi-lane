"""Verify clean provenance, paired cohorts and successful v4 batch execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path, batch_id: str, phase: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    smoke = phase == "smoke"
    count = 1 if smoke else 8
    modes = ("baseline", "shared_fixed", "view_fixed", "dynamic")
    identity = None
    cohort = {}
    commits = set()
    rows = {}
    for mode in modes:
        run = root / f"{phase}_{mode}_{batch_id}"
        config = read(run / "train/config.json")
        summary = read(run / "train/seed_summary.json")
        protocol = read(run / "train/paired_protocol_audit.json")
        history = read(run / "train/training_history.json")
        assert summary["status"] == "complete" and summary["config"] == config
        assert config["git"]["dirty"] is False and config["seed"] == 0
        commits.add(config["git"]["commit"])
        assert config["max_tasks"] == count
        assert protocol["frozen_visual_unchanged"] is True
        assert protocol["frozen_visual_sha256_after"] == protocol["frozen_visual_sha256_before"]
        assert len(protocol["tasks"]) == count
        tasks = [protocol["tasks"][str(i)] for i in range(count)]
        assert len(summary["task_metrics"]) == count and len(history) == count
        current = {
            "initial": protocol["initial_selector_prompt_classifier_sha256"],
            "clip": protocol["frozen_visual_sha256_before"],
            "tasks": [{key: row[key] for key in (
                "train_sample_ids_sha256", "validation_sample_ids_sha256",
                "first_train_batch_ids_sha256", "sampler_seed", "train_instances",
            )} for row in tasks],
        }
        if identity is None:
            identity = current
        assert current == identity, f"Unpaired source initialization/samples: {mode}"
        assert all(row["test_loaded"] is False for row in tasks)
        expected_updates = 2 if smoke else sum(
            math.ceil(row["train_instances"] / config["train_batch_size"]) * config["epochs_per_task"]
            for row in tasks)
        assert summary["completed_optimizer_updates"] == expected_updates
        epochs = [row for task in history.values() for row in task]
        assert sum(row["skipped_optimizer_steps"] for row in epochs) == 0
        assert all(math.isfinite(float(value)) for row in epochs for value in row.values()
                   if isinstance(value, (int, float)))
        if mode != "baseline":
            assert config["moe_expert_input"] == "layer_norm_mean_selected_image_evidence"
            assert config["moe_residual_target"] == "cls_only"
            assert all(row["adapter_grad_finite"] == 1 for row in epochs)
            assert any(row["adapter_grad_total_norm"] > 0 for row in epochs)
            for view in ("full", "person", "face"):
                assert any(row[f"moe_{view}_evidence_centered_l2"] > 0 for row in epochs)
                assert any(row[f"moe_{view}_delta_cls_input_cls_ratio"] > 0 for row in epochs)
        directories = ["smoke_val", "replay_val"] if smoke else ["full_val", "full_test"]
        if not smoke and mode != "baseline":
            directories += ["full_val_moe_off"]
        if not smoke and mode in {"dynamic", "view_fixed"}:
            directories += ["full_val_specialist_off"]
        for directory in directories:
            result = read(run / directory / "result.json")
            assert len(result["task_metrics"]) == count
            for task_id in range(count):
                checkpoint = run / "train/compact_checkpoints" / f"task{task_id}.pth"
                assert result["checkpoint_sha256"][str(task_id)] == sha256(checkpoint)
                with np.load(run / directory / "scores" / f"task{task_id}.npz") as scores:
                    assert np.isfinite(scores["logits"]).all()
                    # Cohorts must align across methods and intervention arms.
                    key = ("test" if directory == "full_test" else "val", task_id,
                           directory == "replay_val")
                    sample = tuple(scores[k].copy() for k in ("sample_ids", "targets", "class_indices"))
                    if key not in cohort:
                        cohort[key] = sample
                    assert all(np.array_equal(a, b) for a, b in zip(sample, cohort[key]))
        if smoke:
            replay = read(run / "replay_val/result.json")
            assert abs(replay["task_metrics"][0]["mAP"] - summary["task_metrics"][0]["mAP"]) < 1e-7
        rows[mode] = {
            "updates": expected_updates, "zero_skips": True,
            "peak_reserved_mib": protocol["peak_reserved_mib"],
            "initial_main_sha256": current["initial"],
        }
    assert len(commits) == 1
    result = {"phase": phase, "batch_id": batch_id, "passed": True,
              "commits": sorted(commits), "methods": rows, "test_used_for_selection": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.batch_id, args.phase, args.output), indent=2))


if __name__ == "__main__":
    main()
