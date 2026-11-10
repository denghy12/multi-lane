"""Learn task and person fusion from frozen Frozen9 logits by backpropagation."""
import argparse
from functools import partial
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from multi_lane.track_a.fuse_face_endpoint_validation import load_face_reliability
from multi_lane.track_a.r1_crossvit_experiment import read, sha, write
from multi_lane.track_a.runner import task_indices
from scripts.emotic.r1_asymmetric_forward import verify_replay
from scripts.emotic.run_r1_frozen9_fusion_ap import (
    METHODS as CONTROLS, ROOT as AP_ROOT, fit_ap_fold, verify_baselines,
)
from scripts.emotic.run_r1_frozen9_fusion_val_crossfit import (
    VAL_MANIFEST, image_fold, image_key, run as run_validation,
)
from scripts.emotic.run_r1_frozen9_fusion_val_test import evaluate_locked_test, lock_test_weights
from scripts.emotic.run_r1_frozen9_learned_fusion import ARCHIVE, FIXED, archived_rows, blend, verify_hashes
from scripts.emotic.run_r1_frozen_cross_view import score_suffix


ROOT = Path("output/r1_frozen9_fusion_backprop/seed1_20261110")
REFERENCE = Path("../multi-lane-main-r1-frozen9-fusion-ap-20261110") / AP_ROOT
STEPS, LR, TEMPERATURE, SLOPE_L2, CHUNK = 300, .02, .5, .001, 128
TRAINED = ("task_bce", "sample_bce", "task_smooth_ap", "sample_smooth_ap")
ABLATIONS = ("sample_bce_mean", "sample_smooth_ap_mean")
METHODS = (*CONTROLS, *TRAINED, *ABLATIONS)


def logit_features(logits):
    distances = [(logits[:, a] - logits[:, b]).abs().mean(dim=-1)
                 for a, b in combinations(range(logits.shape[1]), 2)]
    return torch.cat((logits.abs().mean(dim=-1), torch.stack(distances, dim=-1)), dim=-1)


class ConditionalWeights(nn.Module):
    def __init__(self, logits, reliable, sample):
        super().__init__()
        self.biases = nn.ParameterList()
        self.slopes = nn.ParameterList()
        self.indices, self.inputs, self.means, self.scales = [], [], [], []
        for group, active in enumerate((2, 3)):
            idx = torch.where(reliable == bool(group))[0]
            if not len(idx):
                raise ValueError("Expected both reliability groups in the calibration set")
            features = logit_features(logits[idx, :active]) if sample else logits.new_empty((len(idx), 0))
            mean = features.mean(dim=0)
            scale = features.std(dim=0, unbiased=False).clamp_min(1e-6)
            normalized = (features - mean) / scale
            self.biases.append(nn.Parameter(logits.new_zeros(active - 1)))
            if sample:
                self.slopes.append(nn.Parameter(logits.new_zeros((active - 1, features.shape[1]))))
            self.indices.append(idx)
            self.inputs.append(normalized)
            self.means.append(mean)
            self.scales.append(scale)

    def forward(self, logits):
        z = logits.new_zeros((len(logits), logits.shape[-1]))
        for group, (idx, features, bias) in enumerate(zip(self.indices, self.inputs, self.biases)):
            scores = F.linear(features, self.slopes[group], bias) if self.slopes else bias.expand(len(idx), -1)
            weights = available_weights(scores)
            z = z.index_copy(0, idx, fuse_group(logits[idx, :weights.shape[1]], weights))
        return z

    def export(self):
        return [
            {"bias": bias.detach().cpu().tolist(),
             "slopes": self.slopes[group].detach().cpu().tolist() if self.slopes else [],
             "mean": mean.cpu().tolist(), "scale": scale.cpu().tolist()}
            for group, (bias, mean, scale) in enumerate(zip(self.biases, self.means, self.scales))
        ]


def available_weights(scores):
    return F.pad(scores, (1, 0)).softmax(dim=-1)


def fuse_group(logits, weights):
    return (logits * weights[:, :, None]).sum(dim=1)


def predict_weights(logits, reliable, model):
    weights = logits.new_zeros((len(logits), 3))
    for group, params in enumerate(model):
        mask = reliable == bool(group)
        active = group + 2
        bias = logits.new_tensor(params["bias"])
        if params["mean"]:
            features = logit_features(logits[mask, :active])
            features = (features - logits.new_tensor(params["mean"])) / logits.new_tensor(params["scale"])
            bias = F.linear(features, logits.new_tensor(params["slopes"]), bias)
        else:
            bias = bias.expand(int(mask.sum()), -1)
        weights[mask, :active] = available_weights(bias)
    return weights


def rank_support(labels):
    return [(c, torch.where(y.bool())[0], y) for c, y in enumerate(labels.T)
            if bool(y.any()) and not bool(y.all())]


def smooth_ap_backward(logits, support, chunk=CHUNK, *, gradient_scale=1.0):
    """Accumulate full-list AP gradients in chunks and backpropagate once through the gate."""
    if not support:
        raise ValueError("Smooth-AP needs a class with positive and negative people")
    scores = logits.detach().requires_grad_()
    people = torch.arange(len(scores), device=scores.device)
    loss = scores.new_ones(())
    for col, positives, targets in support:
        values = scores[:, col]
        for anchors in positives.split(chunk):
            ahead = ((values[None] - values[anchors, None]) / TEMPERATURE).sigmoid()
            ahead = ahead.masked_fill(anchors[:, None] == people, 0)
            precision = (1 + (ahead * targets).sum(dim=1)) / (1 + ahead.sum(dim=1))
            contribution = -precision.sum() / (len(positives) * len(support))
            contribution.backward()
            loss += contribution.detach()
    logits.backward(scores.grad * gradient_scale)
    return loss


def fit_gate(logits, labels, reliable, *, sample, ranking, steps):
    gate = ConditionalWeights(logits, reliable, sample)
    opt = torch.optim.Adam(gate.parameters(), lr=LR)
    support = rank_support(labels) if ranking else []
    checkpoints = []
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        z = gate(logits)
        if ranking:
            objective = smooth_ap_backward(z, support)
        else:
            objective = F.binary_cross_entropy_with_logits(z, labels)
            objective.backward()
        if sample:
            penalty = SLOPE_L2 * sum(slope.square().sum() for slope in gate.slopes)
            penalty.backward()
        if step in (0, 1, 29, 99, steps):
            gradient = torch.cat([p.grad.flatten() for p in gate.parameters()]).norm()
            diagnostics = {"step": step, "objective": float(objective.detach()),
                           "gradient_norm": float(gradient)}
            if not all(np.isfinite(value) for value in diagnostics.values()):
                raise ValueError("Nonfinite gate fit")
            checkpoints.append(diagnostics)
        if step < steps:
            opt.step()
    model = gate.export()
    with torch.no_grad():
        w = predict_weights(logits, reliable, model)
        means, distribution = [], []
        for group in (False, True):
            values = w[reliable == group]
            means.append(values.mean(dim=0).cpu().tolist())
            distribution.append({"mean": means[-1], "std": values.std(dim=0, unbiased=False).cpu().tolist(),
                                 "min": values.amin(dim=0).cpu().tolist(),
                                 "max": values.amax(dim=0).cpu().tolist()})
        fused = (w[:, :, None] * logits).sum(dim=1)
        torch.testing.assert_close(fused, z.detach(), atol=2e-6, rtol=1e-6)
        fit_bce = float(F.binary_cross_entropy_with_logits(fused, labels))
    report = {"optimization": checkpoints, "fit_bce": fit_bce, "weights": distribution,
              "ranking_classes": [col for col, _, _ in support]}
    return model, np.asarray(means, dtype=np.float32), report


def fit_learned(rows, mask, classes, device, steps=STEPS):
    logits = torch.as_tensor(rows.logits[mask][:, :, classes], dtype=torch.float32, device=device)
    labels = torch.as_tensor(rows.targets[mask][:, classes], dtype=torch.float32, device=device)
    reliable = torch.as_tensor(rows.reliable[mask], device=device)
    models, reports = {}, {}
    for method in TRAINED:
        sample = method.startswith("sample")
        models[method], mean, reports[method] = fit_gate(
            logits, labels, reliable, sample=sample, ranking=method.endswith("smooth_ap"), steps=steps,
        )
        if sample:
            models[method + "_mean"] = mean
    return models, reports


def fit_fold(rows, mask, classes, *, device):
    controls, support = fit_ap_fold(rows, mask, classes)
    models, reports = fit_learned(rows, mask, classes, device)
    print("FIT_BACKPROP", classes[0], int(mask.sum()),
          {method: round(r["optimization"][-1]["objective"], 6) for method, r in reports.items()}, flush=True)
    return {**controls, **models}, {**support, **reports}


def predict(rows, bank, stage, method):
    parts = []
    reliable = torch.from_numpy(rows.reliable)
    for task in range(stage + 1):
        logits = rows.logits[:, :, list(task_indices(task))]
        model = bank[task][method]
        if method in TRAINED:
            w = predict_weights(torch.from_numpy(logits), reliable, model).numpy()
            fused = (w[:, 0, None] * logits[:, 0] + w[:, 1, None] * logits[:, 1]
                     + w[:, 2, None] * logits[:, 2])
        else:
            fused = blend(logits, rows.reliable, model)
        parts.append(fused)
    return np.concatenate(parts, axis=1)


def encode(model):
    return model.tolist() if isinstance(model, np.ndarray) else model


def decode(model):
    return model if isinstance(model[0], dict) else np.asarray(model, dtype=np.float32)


def smoke(root, device):
    reliability = load_face_reliability(VAL_MANIFEST, "val")
    report = {}
    for task in (0, 6, 7):
        rows = archived_rows("val", task, reliability)
        ref = ARCHIVE / "fusion_scores/D_both_edges" / score_suffix("val", task)
        verify_replay(ref, rows.ids, torch.from_numpy(rows.targets),
                      torch.from_numpy(blend(rows.logits, rows.reliable, FIXED)))
        mask = np.asarray([image_fold(image_key(person)) != 0 for person in rows.ids])
        models, stats = fit_learned(rows, mask, list(task_indices(task)), device, steps=STEPS)
        report[task] = stats
        for method in TRAINED:
            optim = stats[method]["optimization"]
            if optim[0]["gradient_norm"] <= 0 or optim[-1]["objective"] >= optim[0]["objective"]:
                raise AssertionError(f"Gate did not learn on task {task}: {method}")
            logits = torch.from_numpy(rows.logits[:, :, list(task_indices(task))])
            reliable = torch.from_numpy(rows.reliable)
            weights = predict_weights(logits, reliable, models[method])
            logits[~reliable, 2] += 100
            torch.testing.assert_close(weights, predict_weights(logits, reliable, models[method]),
                                       atol=0, rtol=0)
            if weights[~reliable, 2].count_nonzero():
                raise AssertionError("Unavailable Face received nonzero weight")
        print("SMOKE_BACKPROP", task, {method: r["optimization"][-1] for method, r in stats.items()}, flush=True)
    write(root / "smoke.json", report)


def compare_controls(current, previous):
    verify_baselines(current, previous)
    if current["direct_ap"] != previous["direct_ap"]:
        raise ValueError("Archived direct AP did not reproduce")


def run(root, device):
    references = {str(REFERENCE / name): sha(REFERENCE / name) for name in ("result.json", "test.json")}
    fitter = partial(fit_fold, device=device)
    run_validation(root, False, methods=METHODS, fitter=fitter, predict=predict, encode=encode, fit_config={
        "optimizer": "Adam FP32, full calibration cohort; static historical controls unchanged",
        "backprop": {"device": str(device), "steps": STEPS, "learning_rate": LR, "temperature": TEMPERATURE,
                     "slope_l2": SLOPE_L2, "positive_chunk": CHUNK, "initialization": "uniform available views",
                     "ranking": "joint reliability groups, current-class Smooth-AP, skip single-label classes",
                     "features": "per-view mean absolute logit, pairwise mean absolute logit difference",
                     "baseline_reference_hashes": references},
    })
    validation = read(root / "result.json")
    previous = read(REFERENCE / "result.json")
    compare_controls(validation["pooled_oof"], previous["pooled_oof"])
    for fold, groups in validation["folds"].items():
        compare_controls(groups, previous["folds"][fold])
    lock_test_weights(root, validation, fitter=fitter, methods=METHODS, encode=encode)
    test = evaluate_locked_test(root, predict=predict, decode=decode)
    compare_controls(test["results"], read(REFERENCE / "test.json")["results"])
    verify_hashes(read(root / "protocol.json")["source_hashes"])
    verify_hashes(references)
    write(root / "test.json", test)
    write(root / "completion.json", {
        "status": "complete", "seed": 1, "exact_baseline_method_replays": CONTROLS,
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if args.smoke:
        smoke(args.root / "smoke", device)
    else:
        run(args.root, device)
