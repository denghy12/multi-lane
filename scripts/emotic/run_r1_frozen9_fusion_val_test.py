"""Evaluate Frozen9 fusion on crossfit Validation and locked Test in one run."""
import argparse
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.fuse_face_endpoint_validation import load_face_reliability
from multi_lane.track_a.r1_crossvit_experiment import read, sha, write
from multi_lane.track_a.runner import TASK_SIZES, task_indices
from scripts.emotic.r1_asymmetric_forward import verify_replay
from scripts.emotic.run_r1_frozen9_fusion_val_crossfit import (
    METHODS, ROOT as CROSSFIT_ROOT, VAL_MANIFEST, fit_fold, make_records, record,
    run as run_validation, summarize,
)
from scripts.emotic.run_r1_frozen9_learned_fusion import (
    ARCHIVE, archive_files, archived_rows, stage_logits, verify_hashes,
)
from scripts.emotic.run_r1_frozen_cross_view import input_paths, score_suffix


ROOT = Path("output/r1_frozen9_fusion_val_test/seed1_20261110")
PREVIOUS_VAL = (Path("../multi-lane-main-r1-frozen9-fusion-val-crossfit-20261110")
                / CROSSFIT_ROOT / "result.json")


def lock_test_weights(root, validation):
    reliable = load_face_reliability(VAL_MANIFEST, "val")
    weights, support = {}, {}
    for task in range(len(TASK_SIZES)):
        rows = archived_rows("val", task, reliable)
        matrix, support[task] = fit_fold(
            rows, np.ones(len(rows.ids), dtype=np.bool_), list(task_indices(task)),
        )
        weights[task] = {method: value.tolist() for method, value in matrix.items()}
    eligible = [name for name, comparison in validation["comparison"].items()
                if comparison["passes"]]
    write(root / "test_weights.json", {
        "fit_split": "full_val", "calibration_support": support, "weights": weights,
        "validation_eligible": eligible, "val_crossfit_sha256": sha(root / "result.json"),
        "reported_methods": METHODS,
    })


def evaluate_locked_test(root):
    lock_path = root / "test_weights.json"
    lock = read(lock_path)
    lock_hash = sha(lock_path)
    bank = {int(task): {method: np.asarray(values, dtype=np.float32)
                       for method, values in weights.items()}
            for task, weights in lock["weights"].items()}
    manifest = input_paths().face_manifest_root
    tasks = range(len(TASK_SIZES))
    paths = [*archive_files("test", tasks), manifest / "audit_summary.json",
             manifest / "manifests/test.jsonl"]
    hashes = {str(path): sha(path) for path in paths}
    reliable = load_face_reliability(manifest, "test")
    records, replays = make_records(), []
    for task in tasks:
        rows = archived_rows("test", task, reliable)
        fused = {name: stage_logits(rows, bank, task, name) for name in METHODS}
        reference = ARCHIVE / "fusion_scores/D_both_edges" / score_suffix("test", task)
        replays.append(verify_replay(reference, rows.ids, torch.from_numpy(rows.targets),
                                     torch.from_numpy(fused["fixed"])))
        record(records, task, fused, rows.targets, rows.reliable, list(task_indices(task)))
        print("LOCKED_TEST", task, len(rows.ids), int(rows.reliable.sum()), flush=True)
    verify_hashes(hashes)
    if sha(lock_path) != lock_hash:
        raise ValueError("Test weights changed after locking")
    return {"results": summarize(records, False), "fixed_exact_replay": replays,
            "test_source_hashes": hashes, "locked_weights_sha256": lock_hash}


def run(root):
    run_validation(root, smoke=False)
    validation = read(root / "result.json")
    if validation != read(PREVIOUS_VAL):
        raise ValueError("Archived Validation crossfit result did not reproduce")
    lock_test_weights(root, validation)
    test = evaluate_locked_test(root)
    verify_hashes(read(root / "protocol.json")["source_hashes"])
    write(root / "test.json", test)
    write(root / "completion.json", {
        "status": "complete", "seed": 1, "validation_crossfit_exact_replay": True,
        "validation_eligible": read(root / "test_weights.json")["validation_eligible"],
        "baseline_exact_replays": len(validation["fixed_exact_replay"]) + len(test["fixed_exact_replay"]),
        "summaries": {
            split: {name: groups["all"]["summary"] for name, groups in records.items()}
            for split, records in (("val_oof", validation["pooled_oof"]), ("test", test["results"]))
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
