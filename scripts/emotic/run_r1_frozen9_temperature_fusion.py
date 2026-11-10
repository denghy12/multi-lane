"""Ablate Smooth-AP temperature in the task0-only Frozen9 calibration."""
import argparse
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.r1_crossvit_experiment import read, sha, write
from multi_lane.track_a.runner import task_indices
from scripts.emotic.run_r1_frozen9_fusion_ap import candidate_map
from scripts.emotic.run_r1_frozen9_fusion_backprop import CHUNK, LR, STEPS, TEMPERATURE, rank_support
from scripts.emotic.run_r1_frozen9_fusion_val_crossfit import fit_fold as fit_controls, run as run_validation
from scripts.emotic.run_r1_frozen9_fusion_val_test import evaluate_locked_test, lock_test_weights
from scripts.emotic.run_r1_frozen9_learned_fusion import blend, verify_hashes
from scripts.emotic.run_r1_frozen9_shared_fusion import (
    CONTROLS, EQUAL, ROOT as SHARED_ROOT, SMOKE_STEPS, calibration_fold, fit_smooth_ap, grid_utility,
    select_grid, weight_matrices,
)


ROOT = Path("output/r1_frozen9_temperature_fusion/seed1_20261110")
REFERENCE = Path("../multi-lane-main-r1-frozen9-shared-fusion-20261110") / SHARED_ROOT
LOW_TEMPERATURE = .15
WARM_STEPS = 100
COOL_STEPS = 150
HARD = "frozen_hard_ap"
ORIGINAL = "frozen_smooth_ap"
LOW = "frozen_smooth_ap_low"
ANNEALED = "frozen_smooth_ap_annealed"
REPLAYS = (*CONTROLS, HARD, ORIGINAL)
METHODS = (*REPLAYS, LOW, ANNEALED)


def annealed_temperature(step):
    if step < WARM_STEPS:
        return TEMPERATURE
    if step >= WARM_STEPS + COOL_STEPS - 1:
        return LOW_TEMPERATURE
    progress = (step - WARM_STEPS + 1) / COOL_STEPS
    return TEMPERATURE + progress * (LOW_TEMPERATURE - TEMPERATURE)


SCHEDULES = {
    ORIGINAL: None,
    LOW: lambda step: LOW_TEMPERATURE,
    ANNEALED: annealed_temperature,
}


def fit_task0(rows, mask, classes, device, steps):
    logits = rows.logits[mask][:, :, classes]
    labels = rows.targets[mask][:, classes]
    reliable = rows.reliable[mask]
    if not reliable.any() or reliable.all():
        raise ValueError("Both Face reliability groups are required")

    targets = torch.as_tensor(labels, dtype=torch.float32, device=device)
    cohort = (torch.as_tensor(logits, dtype=torch.float32, device=device),
              torch.as_tensor(reliable, device=device), rank_support(targets))
    initial = torch.as_tensor(EQUAL, dtype=torch.float32, device=device).logit()
    hard, grid_report = select_grid(grid_utility(logits, labels, reliable))
    weights = {HARD: weight_matrices(hard)}
    report = {HARD: {"shares": hard.tolist(), **grid_report}}
    for method, schedule in SCHEDULES.items():
        raw, trace = fit_smooth_ap([cohort], initial, steps, temperature_at=schedule)
        shares = raw.sigmoid().cpu().numpy()
        weights[method] = weight_matrices(shares)
        fitted = blend(logits, reliable, weights[method])
        report[method] = {
            "shares": shares.tolist(),
            "optimization": trace,
            "hard_fit_mAP": float(candidate_map(fitted[None], labels)[0]),
        }
    return weights, report


class Task0TemperatureFit:
    def __init__(self, device, *, full_val=False, steps=STEPS):
        self.device = device
        self.full_val = full_val
        self.steps = steps
        self.locked = {}

    def __call__(self, rows, mask, classes):
        fold = calibration_fold(rows, mask, full_val=self.full_val)
        controls, support = fit_controls(rows, mask, classes)
        if tuple(classes) == task_indices(0):
            if fold in self.locked:
                raise AssertionError("A task0 fusion was already fitted for this fold")
            self.locked[fold], report = fit_task0(rows, mask, classes, self.device, self.steps)
            support["temperature_fusion"] = report
            print("FIT_TEMPERATURE", fold, {m: values["shares"] for m, values in report.items()}, flush=True)
        elif fold not in self.locked:
            raise AssertionError("The task0 fusion must be fitted before subsequent tasks")
        support["temperature_fit_task"] = 0
        return {**{m: controls[m] for m in CONTROLS}, **self.locked[fold]}, support


def verify_controls(current, previous):
    for name in REPLAYS:
        if current[name] != previous[name]:
            raise AssertionError(f"Historical {name} did not reproduce")


def run(root, smoke, device):
    steps = SMOKE_STEPS if smoke else STEPS
    reference_val = REFERENCE / "result.json"
    ref_hash = sha(reference_val)
    run_validation(root, smoke, methods=METHODS, fitter=Task0TemperatureFit(device, steps=steps), fit_config={
        "optimizer": "FP32 Adam, task0 only, shared reliability-conditioned two-parameter fusion",
        "temperature_fusion": {
            "steps": steps, "learning_rate": LR, "positive_chunk": CHUNK, "initialization": EQUAL.tolist(),
            "fixed_temperatures": [TEMPERATURE, LOW_TEMPERATURE],
            "annealed": {"warm_steps": WARM_STEPS, "cool_steps": COOL_STEPS,
                         "hold_steps": STEPS - WARM_STEPS - COOL_STEPS, "interpolation": "linear"},
            "fit_task": 0, "held_out_unit": "original image", "device": str(device),
            "validation_reference_sha256": ref_hash,
        },
    })
    if smoke:
        return
    val = read(root / "result.json")
    previous = read(reference_val)
    verify_controls(val["pooled_oof"], previous["pooled_oof"])
    for fold, report in val["folds"].items():
        verify_controls(report, previous["folds"][fold])

    lock_test_weights(root, val, fitter=Task0TemperatureFit(device, full_val=True), methods=METHODS)
    test = evaluate_locked_test(root)
    verify_controls(test["results"], read(REFERENCE / "test.json")["results"])
    verify_hashes(read(root / "protocol.json")["source_hashes"])
    verify_hashes({str(reference_val): ref_hash})
    write(root / "test.json", test)
    write(root / "completion.json", {
        "status": "complete", "seed": 1, "exact_baseline_method_replays": REPLAYS,
        "baseline_exact_replays": len(val["fixed_exact_replay"]) + len(test["fixed_exact_replay"]),
        "validation_eligible": read(root / "test_weights.json")["validation_eligible"],
        "summaries": {
            split: {name: groups["all"]["summary"] for name, groups in results.items()}
            for split, results in (("val_oof", val["pooled_oof"]), ("test", test["results"]))
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    run(args.root / "smoke" if args.smoke else args.root, args.smoke, torch.device(args.device))
