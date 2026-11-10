"""Image-group three-fold evaluation of Frozen9 fusion on archived Validation logits."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.fuse_face_endpoint_validation import load_face_reliability
from multi_lane.track_a.r1_crossvit_experiment import read, sha, write
from multi_lane.track_a.runner import TASK_SIZES, compute_metrics, git_metadata, summarize_tasks, task_indices
from scripts.emotic.r1_asymmetric_forward import verify_replay
from scripts.emotic.run_r1_frozen9_learned_fusion import (
    ARCHIVE, FIXED, OPTIONS, ROOT as TRAIN_FUSION_ROOT, Rows, archive_files,
    archived_rows, bce, simplex_fit, stage_logits, verify_hashes,
)
from scripts.emotic.run_r1_frozen_cross_view import input_paths, score_suffix


ROOT = Path("output/r1_frozen9_fusion_val_crossfit/seed1_20261110")
VAL_MANIFEST = Path("../multi-lane-main/output/emotic_face_manifest/face_manifest_audit_v1_20260908")
TRAIN_FUSION_PROTOCOL = (Path("../multi-lane-main-r1-frozen9-learned-fusion-20261110")
                         / TRAIN_FUSION_ROOT / "protocol.json")
FOLDS = range(3)
FOLD_SALT = "frozen9-val-fusion-image-group-v1"
METHODS = ("fixed", "independent", "bounded")
POPS = ("all", "reliable", "unreliable")
RHO = 0.10


def image_key(sample_id):
    image, separator, person = sample_id.rpartition("#person=")
    if not image.startswith("val:") or not separator or not person.isdecimal():
        raise ValueError(f"Invalid person ID: {sample_id}")
    return image


def image_fold(image):
    digest = hashlib.sha256(f"{FOLD_SALT}:{image}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % len(FOLDS)


def fit_fold(rows, fit_mask, classes):
    good = rows.reliable
    weights = {name: FIXED.copy() for name in METHODS}
    support = {}
    for group, name in ((0, "unreliable"), (1, "reliable")):
        mask = fit_mask & (good == bool(group))
        active = 3 if group else 2
        logits = rows.logits[mask, :active][:, :, classes].astype(np.float64)
        labels = rows.targets[mask][:, classes].astype(np.float64)
        support[name] = {
            "people": len(labels),
            "all_current_negative": int((labels.sum(axis=1) == 0).sum()),
            "positives_per_class": labels.sum(axis=0).astype(int).tolist(),
        }
        if not len(labels):
            continue
        prior = FIXED[group, :active].astype(np.float64)
        prior /= prior.sum()
        weights["independent"][group, :active] = simplex_fit(logits, labels, prior)
        anchor = np.einsum("nvc,v->nc", logits, prior)
        bounded_logits = (1 - RHO) * anchor[:, None, :] + RHO * logits
        u = simplex_fit(bounded_logits, labels, prior)
        weights["bounded"][group, :active] = (1 - RHO) * prior + RHO * u
    for name, matrix in weights.items():
        if (not np.isfinite(matrix).all() or (matrix < 0).any() or matrix[0, 2] != 0
                or not np.allclose(matrix.sum(axis=1), 1, rtol=0, atol=1e-7)):
            raise ValueError(f"Invalid weights for {name}")
    if (weights["bounded"] < (1 - RHO) * FIXED - 1e-7).any():
        raise AssertionError("Bounded fusion exceeds its allowed deviation")
    return weights, support


def make_records():
    return {name: {pop: [] for pop in POPS} for name in METHODS}


def record(records, task, logits, targets, reliable, classes):
    for method, z in logits.items():
        for pop, mask in (("all", np.ones_like(reliable)), ("reliable", reliable),
                          ("unreliable", ~reliable)):
            if not mask.any():
                raise ValueError(f"Empty holdout population {task}/{pop}")
            values = compute_metrics(task, torch.from_numpy(z[mask]).sigmoid(),
                                     torch.from_numpy(targets[mask]), 0.5)
            current_bce = bce(z[mask][:, classes].astype(np.float64), targets[mask][:, classes])
            records[method][pop].append((values, current_bce))


def summarize(records, smoke):
    return {
        method: {
            pop: {
                "metrics": [dict(asdict(row), current_class_bce=loss) for row, loss in entries],
                "summary": {
                    **(summarize_tasks([row for row, _ in entries]) if not smoke else {}),
                    "mean_current_class_bce": float(np.mean([loss for _, loss in entries])),
                },
            }
            for pop, entries in populations.items()
        }
        for method, populations in records.items()
    }


def compare_folds(fold_reports):
    results = {}
    for method in METHODS[1:]:
        rows = []
        for fold in FOLDS:
            source = fold_reports[fold]
            base = source["fixed"]["all"]["summary"]
            fitted = source[method]["all"]["summary"]
            rows.append({
                "fold": fold,
                **{key: fitted[key] - base[key]
                   for key in ("final_mAP", "average_mAP", "mean_current_class_bce")},
            })
        mean = {key: float(np.mean([row[key] for row in rows]))
                for key in ("final_mAP", "average_mAP", "mean_current_class_bce")}
        both = sum(row["final_mAP"] > 0 and row["average_mAP"] > 0 for row in rows)
        results[method] = {
            "paired_deltas": rows, "mean_delta": mean, "folds_both_positive": both,
            "passes": both >= 2 and mean["final_mAP"] > 0 and mean["average_mAP"] > 0,
        }
    return results


def run(root, smoke):
    if root.exists() and any(smoke or path.name != "smoke" for path in root.iterdir()):
        raise FileExistsError(f"Output directory already contains a run: {root}")
    torch.set_num_threads(2)
    tasks = (0, 1) if smoke else tuple(range(len(TASK_SIZES)))
    val_manifest = VAL_MANIFEST / "manifests/val.jsonl"
    manifest_audit = VAL_MANIFEST / "audit_summary.json"
    if "test" in read(manifest_audit)["splits"]:
        raise ValueError("Crossfit requires the Validation-only Face audit")
    prior = read(TRAIN_FUSION_PROTOCOL)
    old_manifest = input_paths().face_manifest_root / "manifests/val.jsonl"
    if sha(val_manifest) != prior["hashes"][str(old_manifest)]:
        raise ValueError("Validation-only Face mask does not match Frozen9")
    files = [*archive_files("val", tasks), val_manifest, manifest_audit,
             VAL_MANIFEST / "manifests/train.jsonl", VAL_MANIFEST / "detector_config.json",
             TRAIN_FUSION_PROTOCOL]
    hashes = {str(file): sha(file) for file in files}
    write(root / "protocol.json", {
        "split": "val", "seed": 1, "tasks": tasks, "folds": list(FOLDS), "fold_salt": FOLD_SALT,
        "methods": METHODS, "rho": RHO, "optimizer": "CPU float64 SLSQP BCE", "options": OPTIONS,
        "decision": "positive mean final/average fold deltas and both positive in >=2/3 folds",
        "git": git_metadata(Path.cwd()), "source_hashes": hashes,
    })
    reliable_lookup = load_face_reliability(VAL_MANIFEST, "val")
    banks = {fold: {} for fold in FOLDS}
    fold_records = {fold: make_records() for fold in FOLDS}
    pooled_records = make_records()
    fit_stats = []
    replays = []
    previous_people = set()

    for task in tasks:
        rows = archived_rows("val", task, reliable_lookup)
        if not previous_people.issubset(set(rows.ids)):
            raise ValueError("Validation cohorts are not cumulative")
        previous_people = set(rows.ids)
        images = np.asarray([image_key(person) for person in rows.ids])
        folds = np.asarray([image_fold(image) for image in images])
        classes = list(task_indices(task))
        pooled = {name: np.full_like(rows.targets, np.nan) for name in METHODS}
        for fold in FOLDS:
            holdout = folds == fold
            if set(images[holdout]).intersection(images[~holdout]):
                raise AssertionError("An image crosses the fitting and holdout groups")
            banks[fold][task], support = fit_fold(rows, ~holdout, classes)
            held = Rows(rows.ids[holdout], rows.logits[holdout], rows.targets[holdout],
                        rows.reliable[holdout])
            fused = {name: stage_logits(held, banks[fold], task, name) for name in METHODS}
            for name in METHODS:
                pooled[name][holdout] = fused[name]
            record(fold_records[fold], task, fused, held.targets, held.reliable, classes)
            fit_stats.append({
                "task": task, "fold": fold, "fit": support,
                "held_people": int(holdout.sum()), "held_images": len(set(images[holdout])),
                "held_reliable": int(held.reliable.sum()),
            })
        if any(not np.isfinite(z).all() for z in pooled.values()):
            raise AssertionError("Out-of-fold coverage is incomplete")
        ref = ARCHIVE / "fusion_scores/D_both_edges" / score_suffix("val", task)
        replays.append(verify_replay(ref, rows.ids, torch.from_numpy(rows.targets),
                                     torch.from_numpy(pooled["fixed"])))
        record(pooled_records, task, pooled, rows.targets, rows.reliable, classes)
        print("VAL_CROSSFIT", task, len(rows.ids), np.bincount(folds, minlength=len(FOLDS)).tolist(), flush=True)
    verify_hashes(hashes)
    fold_reports = {fold: summarize(records, smoke) for fold, records in fold_records.items()}
    write(root / "weights.json", {
        "weights": {fold: {task: {name: matrix.tolist() for name, matrix in by_method.items()}
                            for task, by_method in bank.items()} for fold, bank in banks.items()},
        "fit_support": fit_stats,
    })
    write(root / "result.json", {
        "status": "complete", "smoke": smoke, "fixed_exact_replay": replays, "folds": fold_reports,
        "comparison": compare_folds(fold_reports) if not smoke else None,
        "pooled_oof": summarize(pooled_records, smoke),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.root / "smoke" if args.smoke else args.root, args.smoke)
