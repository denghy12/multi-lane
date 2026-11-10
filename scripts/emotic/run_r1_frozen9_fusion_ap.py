"""Fit task-local conditional weights by joint calibration mAP with no R1 prior."""
import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.r1_crossvit_experiment import read, sha, write
from scripts.emotic.run_r1_frozen9_fusion_val_crossfit import (
    METHODS as BASELINES, fit_fold, run as run_validation,
)
from scripts.emotic.run_r1_frozen9_fusion_val_test import (
    ROOT as PREVIOUS_ROOT, evaluate_locked_test, lock_test_weights,
)
from scripts.emotic.run_r1_frozen9_learned_fusion import bce, verify_hashes


ROOT = Path("output/r1_frozen9_fusion_ap/seed1_20261110")
PREVIOUS = Path("../multi-lane-main-r1-frozen9-fusion-val-test-20261110") / PREVIOUS_ROOT
METHOD = "direct_ap"
METHODS = (*BASELINES, METHOD)
UNITS = 100
COARSE_STEP = 5
FINE_RADIUS = 5
CANDIDATE_BATCH = 64
AP_TIE = 1e-12


@dataclass(frozen=True)
class Candidate:
    coordinates: tuple
    mean_ap: float
    mean_bce: float


def candidate_weights(coordinates):
    """Coordinates are reliable Person/Face and unreliable Person percentages."""
    a, b, c = np.asarray(coordinates, dtype=np.int64).T
    if (a < 0).any() or (b < 0).any() or (a + b > UNITS).any() or ((c < 0) | (c > UNITS)).any():
        raise ValueError("Candidate is outside the conditional simplices")
    unreliable = np.stack((UNITS - c, c, np.zeros_like(c)), axis=1)
    reliable = np.stack((UNITS - a - b, a, b), axis=1)
    return np.stack((unreliable, reliable), axis=1).astype(np.float32) / np.float32(UNITS)


def candidate_logits(logits, reliability, coordinates):
    weights = candidate_weights(coordinates)[:, reliability.astype(np.int8), :]
    return (weights[:, :, 0, None] * logits[None, :, 0, :]
            + weights[:, :, 1, None] * logits[None, :, 1, :]
            + weights[:, :, 2, None] * logits[None, :, 2, :])


def candidate_map(fused, targets):
    # Match the evaluation sigmoid and descending argsort, then compute AP per
    # class over all people, including comparisons across reliability groups.
    scores = np.ascontiguousarray(torch.from_numpy(fused).sigmoid().numpy().transpose(0, 2, 1))
    order = np.argsort(scores, axis=2)[:, :, ::-1]
    positives = np.take_along_axis(targets.T[None, :, :] == 1, order, axis=2)
    hits = np.cumsum(positives, axis=2)
    hits[~positives] = 0
    rank = np.arange(1, scores.shape[2] + 1)
    ap = (hits / rank).sum(axis=2) / (targets.sum(axis=0, dtype=np.float64) + 1e-8)
    return 100 * ap.mean(axis=1)


def search_grid(logits, targets, reliable, coordinates):
    # Keep both grids' tie-breaking independent of the original fixed weights.
    best = None
    for offset in range(0, len(coordinates), CANDIDATE_BATCH):
        batch = coordinates[offset:offset+CANDIDATE_BATCH]
        fused = candidate_logits(logits, reliable, batch)
        aps = candidate_map(fused, targets)
        for i in np.flatnonzero(aps >= aps.max() - AP_TIE):
            score = float(aps[i])
            if best is not None and score < best.mean_ap - AP_TIE:
                continue
            candidate = Candidate(batch[i], score, bce(fused[i].astype(np.float64), targets))
            if (best is None or candidate.mean_ap > best.mean_ap + AP_TIE
                    or (abs(candidate.mean_ap - best.mean_ap) <= AP_TIE
                        and (candidate.mean_bce, candidate.coordinates)
                        < (best.mean_bce, best.coordinates))):
                best = candidate
    if best is None:
        raise ValueError("Empty AP grid")
    return best


def coarse_coordinates():
    values = range(0, UNITS + 1, COARSE_STEP)
    return [(a, b, c) for a, b, c in product(values, repeat=3) if a + b <= UNITS]


def fine_coordinates(center):
    spans = [range(max(0, value - FINE_RADIUS), min(UNITS, value + FINE_RADIUS) + 1)
             for value in center]
    return [(a, b, c) for a, b, c in product(*spans) if a + b <= UNITS]


def fit_ap_fold(rows, fit_mask, classes):
    weights, support = fit_fold(rows, fit_mask, classes)
    logits = rows.logits[fit_mask][:, :, classes]
    targets = rows.targets[fit_mask][:, classes]
    reliable = rows.reliable[fit_mask]
    if not len(targets) or not reliable.any() or reliable.all():
        raise ValueError("AP fitting requires both Face reliability groups")
    coarse = coarse_coordinates()
    seed = search_grid(logits, targets, reliable, coarse)
    fine = fine_coordinates(seed.coordinates)
    winner = search_grid(logits, targets, reliable, fine)
    if winner.mean_ap < seed.mean_ap - AP_TIE:
        raise AssertionError("Local search lost the best coarse candidate")
    weights[METHOD] = candidate_weights([winner.coordinates])[0]
    support[METHOD] = {
        "coarse_candidates": len(coarse), "fine_candidates": len(fine),
        "coarse_coordinates": seed.coordinates, "coarse_mAP": seed.mean_ap,
        "selected_coordinates": winner.coordinates, "fit_mAP": winner.mean_ap,
        "fit_bce": winner.mean_bce,
    }
    print("FIT_AP", classes[0], len(targets), winner.coordinates,
          f"{winner.mean_ap:.4f}", flush=True)
    return weights, support


def verify_baselines(current, previous):
    for name in BASELINES:
        if current[name] != previous[name]:
            raise ValueError(f"Archived {name} baseline did not reproduce")


def run(root, smoke):
    run_validation(root, smoke, methods=METHODS, fitter=fit_ap_fold, fit_config={
        "ap_search": {
            "scope": "joint mixed-cohort current-class macro AP, no R1 prior",
            "coarse_step": COARSE_STEP / UNITS, "fine_step": 1 / UNITS,
            "fine_radius": FINE_RADIUS / UNITS, "candidate_batch": CANDIDATE_BATCH,
            "tie_break": "lower fitting BCE then lexicographic coordinates",
        },
    })
    if smoke:
        return
    validation = read(root / "result.json")
    previous_val = read(PREVIOUS / "result.json")
    verify_baselines(validation["pooled_oof"], previous_val["pooled_oof"])
    for fold, groups in validation["folds"].items():
        verify_baselines(groups, previous_val["folds"][fold])
    lock_test_weights(root, validation, fitter=fit_ap_fold, methods=METHODS)
    test = evaluate_locked_test(root)
    verify_baselines(test["results"], read(PREVIOUS / "test.json")["results"])
    verify_hashes(read(root / "protocol.json")["source_hashes"])
    write(root / "test.json", test)
    write(root / "completion.json", {
        "status": "complete", "seed": 1, "exact_baseline_method_replays": list(BASELINES),
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
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.root / "smoke" if args.smoke else args.root, args.smoke)
