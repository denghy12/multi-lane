"""Compare two-parameter Frozen9 fusion under frozen and chronological fitting."""
import argparse
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.r1_crossvit_experiment import read, sha, write
from multi_lane.track_a.runner import task_indices
from scripts.emotic.run_r1_frozen9_fusion_ap import AP_TIE, CANDIDATE_BATCH, candidate_map
from scripts.emotic.run_r1_frozen9_fusion_backprop import (
    CHUNK, LR, ROOT as BACKPROP_ROOT, STEPS, TEMPERATURE, rank_support, smooth_ap_backward,
)
from scripts.emotic.run_r1_frozen9_fusion_val_crossfit import (
    FOLDS, fit_fold as fit_controls, image_fold, image_key, run as run_validation,
)
from scripts.emotic.run_r1_frozen9_fusion_val_test import evaluate_locked_test, lock_test_weights
from scripts.emotic.run_r1_frozen9_learned_fusion import verify_hashes


ROOT = Path("output/r1_frozen9_shared_fusion/seed1_20261110")
REFERENCE = Path("../multi-lane-main-r1-frozen9-fusion-backprop-20261110") / BACKPROP_ROOT
CONTROLS = ("fixed", "bounded")
NEW = ("frozen_smooth_ap", "frozen_hard_ap", "prefix_smooth_ap", "prefix_hard_ap")
METHODS = (*CONTROLS, *NEW)
GRID_DIVISIONS = 40
GRID = np.array(list(product(range(GRID_DIVISIONS + 1), repeat=2))) / GRID_DIVISIONS
EQUAL = np.array([.5, 1 / 3])
SMOKE_STEPS = 10


def weight_matrices(shares):
    full, face = np.asarray(shares, dtype=np.float64).T
    fallback = np.stack((full, 1 - full, np.zeros_like(full)), axis=-1)
    reliable = np.stack(((1 - face) * full, (1 - face) * (1 - full), face), axis=-1)
    return np.stack((fallback, reliable), axis=-2).astype(np.float32)


def grid_utility(logits, targets, reliable):
    utilities = []
    matrices = weight_matrices(GRID)
    groups = reliable.astype(np.int8)
    for start in range(0, len(matrices), CANDIDATE_BATCH):
        w = matrices[start:start + CANDIDATE_BATCH, groups, :]
        fused = (w[:, :, 0, None] * logits[None, :, 0]
                 + w[:, :, 1, None] * logits[None, :, 1]
                 + w[:, :, 2, None] * logits[None, :, 2])
        utilities.append(candidate_map(fused, targets))
    return np.concatenate(utilities)


def select_grid(utility):
    best = utility.max()
    ties = np.flatnonzero(utility >= best - AP_TIE)
    index = ties[np.argmin(((GRID[ties] - EQUAL) ** 2).sum(axis=1))]
    return GRID[index], {
        "macro_ap": float(utility[index]),
        "points_within_0p05": int((utility >= best - .05).sum()),
    }


def fused_scores(shares, logits, reliable):
    full, face = shares.unbind()
    fallback = full * logits[:, 0] + (1 - full) * logits[:, 1]
    with_face = ((1 - face) * full * logits[:, 0]
                 + (1 - face) * (1 - full) * logits[:, 1] + face * logits[:, 2])
    return torch.where(reliable[:, None], with_face, fallback)


def fit_smooth_ap(tasks, initial, steps):
    raw = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([raw], lr=LR)
    diagnostics = []
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        objectives = [
            smooth_ap_backward(fused_scores(raw.sigmoid(), logits, reliable), support,
                               gradient_scale=1 / len(tasks))
            for logits, reliable, support in tasks
        ]
        if step in (0, 29, 99, steps):
            objective = float(torch.stack(objectives).mean())
            norm = float(raw.grad.norm())
            if not np.isfinite(objective) or not np.isfinite(norm):
                raise ValueError("Nonfinite shared fusion objective or gradient")
            diagnostics.append({"step": step, "objective": objective, "gradient_norm": norm})
        if step < steps:
            optimizer.step()
    return raw.detach(), diagnostics


@dataclass
class History:
    cohorts: list = field(default_factory=list)
    utility_sum: np.ndarray = field(default_factory=lambda: np.zeros(len(GRID), dtype=np.float64))
    raw: torch.Tensor = None
    frozen: dict = field(default_factory=dict)


class ChronologicalFit:
    def __init__(self, device, *, full_val=False, steps=STEPS):
        self.device = device
        self.full_val = full_val
        self.steps = steps
        self.histories = {fold: History() for fold in ((None,) if full_val else FOLDS)}

    def __call__(self, rows, mask, classes):
        if self.full_val:
            if not mask.all():
                raise AssertionError("Full Validation fit received a held-out image")
            fold = None
        else:
            excluded = {image_fold(image_key(person)) for person in rows.ids[~mask]}
            if len(excluded) != 1:
                raise AssertionError("Exactly one image fold must be excluded")
            fold = next(iter(excluded))
            if any(image_fold(image_key(person)) == fold for person in rows.ids[mask]):
                raise AssertionError("A held-out image entered the fitting cohort")

        history = self.histories[fold]
        task = len(history.cohorts)
        if tuple(classes) != task_indices(task):
            raise AssertionError("Fitting order must follow class introduction")
        logits = rows.logits[mask][:, :, classes]
        labels = rows.targets[mask][:, classes]
        reliable = rows.reliable[mask]
        if not reliable.any() or reliable.all():
            raise ValueError("Both Face reliability groups are required")
        initial = history.raw
        if initial is None:
            initial = torch.as_tensor(EQUAL, dtype=torch.float32, device=self.device).logit()

        target_tensor = torch.as_tensor(labels, dtype=torch.float32, device=self.device)
        history.cohorts.append((
            torch.as_tensor(logits, dtype=torch.float32, device=self.device),
            torch.as_tensor(reliable, device=self.device), rank_support(target_tensor),
        ))
        # Each introduction cohort contributes one equally weighted class-macro objective.
        history.utility_sum += grid_utility(logits, labels, reliable)
        history.raw, diagnostics = fit_smooth_ap(history.cohorts, initial, self.steps)
        smooth = history.raw.sigmoid().cpu().numpy()
        hard, grid_report = select_grid(history.utility_sum / (task + 1))
        prefix = {"prefix_smooth_ap": smooth, "prefix_hard_ap": hard}
        if task == 0:
            history.frozen = {"frozen_smooth_ap": smooth.copy(), "frozen_hard_ap": hard.copy()}
        shares = {**history.frozen, **prefix}

        base, support = fit_controls(rows, mask, classes)
        weights = {**{name: base[name] for name in CONTROLS},
                   **{name: weight_matrices(pair) for name, pair in shares.items()}}
        support["shared"] = {
            "task": task, "fold": fold, "prefix_tasks": list(range(task + 1)),
            "frozen_fit_task": 0, "shares": {name: pair.tolist() for name, pair in shares.items()},
            "prefix_smooth_ap": diagnostics, "prefix_hard_ap": grid_report,
        }
        print("FIT_SHARED", fold, task, "smooth", np.round(smooth, 5).tolist(),
              "hard", hard.tolist(), flush=True)
        return weights, support


def verify_controls(current, previous):
    for method in CONTROLS:
        if current[method] != previous[method]:
            raise AssertionError(f"Historical {method} did not reproduce")


def run(root, smoke, device):
    references = {str(REFERENCE / f): sha(REFERENCE / f) for f in ("result.json", "test.json")}
    steps = SMOKE_STEPS if smoke else STEPS
    run_validation(root, smoke, methods=METHODS, fitter=ChronologicalFit(device, steps=steps), fit_config={
        "optimizer": "FP32 Adam; per-task class macro Smooth-AP then equal mean of available tasks",
        "shared_fusion": {
            "device": str(device), "parameters": ["Full share within Full/Person", "reliable Face share"],
            "initialization": EQUAL.tolist(), "prefix_warm_start": True, "old_weights_locked": True,
            "steps": steps, "learning_rate": LR, "temperature": TEMPERATURE, "positive_chunk": CHUNK,
            "hard_ap_step": 1 / GRID_DIVISIONS, "hard_ap_candidates": len(GRID),
            "hard_ap_tie": "nearest equal-view initialization, then grid order", "reference_hashes": references,
        },
    })
    if smoke:
        return
    val = read(root / "result.json")
    old_val = read(REFERENCE / "result.json")
    verify_controls(val["pooled_oof"], old_val["pooled_oof"])
    for fold, report in val["folds"].items():
        verify_controls(report, old_val["folds"][fold])

    lock_test_weights(root, val, fitter=ChronologicalFit(device, full_val=True), methods=METHODS)
    test = evaluate_locked_test(root)
    verify_controls(test["results"], read(REFERENCE / "test.json")["results"])
    verify_hashes(read(root / "protocol.json")["source_hashes"])
    verify_hashes(references)
    write(root / "test.json", test)
    write(root / "completion.json", {
        "status": "complete", "seed": 1, "exact_baseline_method_replays": CONTROLS,
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
