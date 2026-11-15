"""Score D on an EMOTIC benchmark protocol using the original fixed R1 logit fusion."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.evaluation_scores import align_evaluation_scores, load_evaluation_scores
from multi_lane.track_a.fuse_face_endpoint_validation import load_face_reliability
from multi_lane.track_a.fuse_fixed_three_view_test import LOCKED_FALLBACK_WEIGHTS, LOCKED_RELIABLE_WEIGHTS
from multi_lane.track_a.r1_crossvit_experiment import VIEWS, read, write
from multi_lane.track_a.runner import CLASS_ORDER, TASK_PROTOCOLS, compute_metrics, config_task_sizes, summarize_tasks
from scripts.emotic.r1_asymmetric_forward import fuse_logits
from scripts.emotic.run_r1_frozen_cross_view import input_paths, score_suffix

FINAL_PEOPLE = {"val": 2397, "test": 5368}
# Archived MULTI-LANE Test curves, people with a positive label among seen classes.
TEST_PEOPLE = {
    "b10c4": (4686, 5232, 5348, 5366, 5368),
    "b4c2": (3096, 4248, 4600, 4686, 5104, 5232, 5244, 5348, 5354, 5366, 5367, 5368),
    "joint26": (5368,),
}


def validate_sources(root, protocol, seed, smoke):
    sizes = TASK_PROTOCOLS[protocol]
    for view in VIEWS:
        train = root / view / "train"
        config = read(train / "config.json")
        expected = {
            "seed": seed, "protocol_id": f"emotic_{protocol}_v0.1",
            "class_order": list(CLASS_ORDER), "r1_receiver": view,
            "r1_kind": "native" if view == "full" else "attention",
            "reporting_split": "val" if smoke else "test",
        }
        if view != "full":
            expected.update(r1_frozen_mode="patch", r1_exchange_layer=8, r1_frozen_evidence="all")
        if not smoke:
            expected["max_tasks"] = len(sizes)
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"{view} {key}: expected {value!r}, got {config.get(key)!r}")
        if config_task_sizes(config) != sizes or read(train / "seed_summary.json")["status"] != "complete":
            raise ValueError(f"Wrong schedule or incomplete training: {train}")


def fused_stage(root, suffix, task, expected_classes, reliability):
    branches = [load_evaluation_scores(root / view / suffix, 64) for view in VIEWS]
    full, person, face = branches
    if full.task_id != task or full.logits.shape[1] != expected_classes:
        raise ValueError(f"Wrong task/seen class count in {suffix}")
    if not full.targets.any(axis=1).all():
        raise ValueError(f"Sample outside the seen-positive benchmark population in {suffix}")
    person_logits = align_evaluation_scores(full, person)[2]
    face_logits = align_evaluation_scores(full, face)[2]
    reliable = np.array([reliability[i] for i in full.sample_ids], dtype=bool)
    logits = fuse_logits(*(torch.from_numpy(z) for z in (full.logits, person_logits, face_logits, reliable)))
    metrics = compute_metrics(task, logits.sigmoid(), torch.from_numpy(full.targets), .5)
    return full, metrics, int(reliable.sum())


def check_population(stage, final, split):
    # Earlier cohorts consist of all final-split people positive for a seen label.
    width = stage.logits.shape[1]
    eligible = final.targets[:, :width].any(axis=1)
    if not np.array_equal(stage.sample_ids, final.sample_ids[eligible]):
        raise ValueError(f"{split} task{stage.task_id} population/order differs from the benchmark cohort")
    if not np.array_equal(stage.targets, final.targets[eligible, :width]):
        raise ValueError(f"{split} task{stage.task_id} labels disagree with final-stage labels")


def summarize(root, protocol, seed, manifest, smoke=False):
    validate_sources(root, protocol, seed, smoke)
    sizes = TASK_PROTOCOLS[protocol]
    result = {
        "status": "smoke_passed" if smoke else "complete", "protocol": protocol, "seed": seed,
        "task_sizes": sizes, "class_order": CLASS_ORDER, "threshold": .5,
        "fusion": "R1 weighted raw logits followed by one sigmoid",
        "weights": {"reliable": LOCKED_RELIABLE_WEIGHTS.tolist(), "fallback": LOCKED_FALLBACK_WEIGHTS.tolist()},
        "evaluation_scope": "intersects_seen_classes",
        "average_scope": "single stage equals Last" if len(sizes) == 1 else "unweighted average of stage mAP",
        "splits": {},
    }
    for split in (("val",) if smoke else ("val", "test")):
        reliability = load_face_reliability(manifest, split)
        stages = range(1 if smoke else len(sizes))
        final = None
        if not smoke:
            final = load_evaluation_scores(root / "full" / score_suffix(split, len(sizes)-1), 64)
            if len(final.sample_ids) != FINAL_PEOPLE[split] or final.logits.shape[1] != len(CLASS_ORDER):
                raise ValueError(f"Incomplete final {split} benchmark population or class list")
        rows, populations = [], []
        for task in stages:
            suffix = Path("train/val_scores/task0.npz") if smoke else score_suffix(split, task)
            branch, row, n_reliable = fused_stage(root, suffix, task, sum(sizes[:task+1]), reliability)
            if smoke:
                if row.samples != 64:
                    raise ValueError("Smoke requires the first 64 eligible validation people")
            else:
                check_population(branch, final, split)
                if split == "test" and row.samples != TEST_PEOPLE[protocol][task]:
                    raise ValueError(f"task{task} Test population differs from the registered benchmark")
            rows.append(row)
            populations.append({"task": task, "seen_positive": row.samples, "reliable_face": n_reliable,
                                "unreliable_face": row.samples-n_reliable})
        result["splits"][split] = {
            "metrics": [asdict(row) for row in rows], "populations": populations,
            "summary": None if smoke else summarize_tasks(rows, sizes),
        }
    if smoke:
        replay_files = [root / view / score_suffix("val", 0) for view in VIEWS]
        if any(path.exists() for path in replay_files):
            for view, path in zip(VIEWS, replay_files):
                original = load_evaluation_scores(root / view / "train/val_scores/task0.npz", 64)
                replay = load_evaluation_scores(path, 64)
                aligned = align_evaluation_scores(original, replay)[2]
                if not np.array_equal(original.logits, aligned):
                    raise ValueError(f"Restored {view} smoke logits do not replay training-time validation")
            result["validation_checkpoint_replay"] = "bitwise_equal"
    write(root / ("benchmark_smoke_summary.json" if smoke else "benchmark_summary.json"), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", choices=("b10c4", "b4c2", "joint26"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--face-manifest-root", type=Path, default=input_paths().face_manifest_root)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    summarize(args.root, args.protocol, args.seed, args.face_manifest_root, args.smoke)
