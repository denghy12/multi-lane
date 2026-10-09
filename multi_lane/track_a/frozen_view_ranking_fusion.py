"""Calibrate bounded view weights on image-group-held-out TRAIN predictions.

No visual parameters or feature vectors are updated. Each task owns a gate,
shared across its classes, fitted only at introduction and then frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

VIEWS = ("full", "person", "face")
METHODS = ("static_bce", "dynamic_bce", "dynamic_ranking")


def image_group(sample_id):
    return str(sample_id).rsplit("#person=", 1)[0]


def check_provenance(records):
    """Check cross-task exclusion, not just the current task's partition."""
    fit, held = set(), set()
    for record in records:
        if record.get("source_split") != "train" or record.get("training_exclusion") is not True:
            raise ValueError("Fusion fitting requires excluded train image groups")
        if record.get("split_salt") != "emotic-reliability-calibration-v1":
            raise ValueError("Unexpected calibration split")
        ids = record["fit_sample_ids"] + record["calibration_sample_ids"]
        if not ids or any(not str(value).startswith("train:") for value in ids):
            raise ValueError("Calibration must use train IDs only")
        fit.update(image_group(value) for value in record["fit_sample_ids"])
        held.update(image_group(value) for value in record["calibration_sample_ids"])
    if fit & held:
        raise ValueError("Held-out images overlap base training across tasks")
    return {"fit_image_groups": len(fit), "heldout_image_groups": len(held), "overlap": 0}


def read_scores(path):
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key].copy() for key in source.files}
    ids = data["sample_ids"].tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate score IDs")
    shape = data["targets"].shape
    for key in ("fused_logits", *(f"{view}_logits" for view in VIEWS)):
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError("Invalid view score shape/values")
    return data


def aligned_baseline(path, data):
    """Use ordinary saved logits/probabilities for exact baseline equivalence."""
    with np.load(path, allow_pickle=False) as source:
        if not np.array_equal(source["sample_ids"], data["sample_ids"]) or not np.array_equal(source["targets"], data["targets"]):
            raise ValueError("Baseline/view IDs or labels are not aligned")
        data["fused_logits"] = source["logits"].copy()
        data["fused_probabilities"] = source["probabilities"].copy()
    return data


def tensors(data, columns):
    views = torch.from_numpy(np.stack([data[f"{name}_logits"][:, columns] for name in VIEWS], axis=-1)).float()
    return (views, torch.from_numpy(data["fused_logits"][:, columns]).float(),
            torch.from_numpy(data["targets"][:, columns]).float(),
            torch.from_numpy(data["face_reliable"]).bool())


class BoundedFusionGate(nn.Module):
    def __init__(self, train_views, train_fixed, dynamic=True, radius=0.05):
        super().__init__()
        if not 0 < radius <= 0.05:
            raise ValueError("Radius must keep all prior weights positive")
        self.radius = float(radius)
        self.register_buffer("mean", train_views.mean(dim=0))
        self.register_buffer("std", train_views.std(dim=0, unbiased=False).clamp_min(1.0))
        self.register_buffer("temperature", train_fixed.std(dim=0, unbiased=False).clamp_min(1.0))
        if dynamic:
            self.router = nn.Sequential(nn.Linear(7, 8), nn.Tanh(), nn.Linear(8, 2))
            nn.init.zeros_(self.router[-1].weight)
            nn.init.zeros_(self.router[-1].bias)
        else:
            self.raw_delta = nn.Parameter(torch.zeros(2))

    def forward(self, views, fixed, reliable):
        if hasattr(self, "router"):
            x = (views - self.mean) / self.std
            # Invalid Face must not influence even the Person gate indirectly.
            x = torch.stack((x[..., 0], x[..., 1],
                             x[..., 2] * reliable[:, None]), dim=-1)
            quality = reliable[:, None, None].expand(-1, views.shape[1], 1).float()
            x = torch.cat((x, x[..., 1:2] - x[..., :1], x[..., 2:3] - x[..., :1],
                           x[..., 2:3] - x[..., 1:2], quality), dim=-1)
            raw = self.router(x)
        else:
            raw = self.raw_delta.expand(*fixed.shape, 2)
        delta = self.radius * torch.tanh(raw)
        dp, df = delta[..., 0], delta[..., 1] * reliable[:, None]
        # The original saved fused logits are the identity reference. Do not
        # reconstruct them from rounded AMP view logits.
        output = fixed + dp * (views[..., 1] - views[..., 0]) + df * (views[..., 2] - views[..., 0])
        return output, torch.stack((dp, df), dim=-1)


def implied_weights(delta, reliable):
    prior = torch.where(reliable[:, None, None],
                        delta.new_tensor([0.64, 0.16, 0.20]), delta.new_tensor([0.80, 0.20, 0.0]))
    dp, df = delta.unbind(dim=-1)
    return prior + torch.stack((-dp - df, dp, df), dim=-1)


def ranking_loss(scores, labels, temperature):
    """Class-balanced positive/negative softplus ranking, pairs within batch."""
    losses = []
    for c in range(scores.shape[1]):
        pos = scores[labels[:, c] > 0.5, c]
        neg = scores[labels[:, c] <= 0.5, c]
        if pos.numel() and neg.numel():
            losses.append(F.softplus((neg[None, :] - pos[:, None]) / temperature[c]).mean())
    # No BCE fallback: retain a well-defined differentiable zero and log skips.
    return torch.stack(losses).mean() if losses else scores.sum() * 0.0


def fit_gate(views, fixed, labels, reliable, method, device, epochs=30, batch_size=256, seed=0):
    torch.manual_seed(seed)
    gate = BoundedFusionGate(views, fixed, dynamic=method != "static_bce").to(device)
    initial, _ = gate(views.to(device), fixed.to(device), reliable.to(device))
    if not torch.equal(initial.detach().cpu(), fixed):
        raise RuntimeError("Initial fusion is not exactly baseline")
    del initial
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.001, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    history = []
    for epoch in range(epochs):
        loss_sum, valid_pairs, steps, grad_sq = 0., 0, 0, 0.
        for ix in torch.randperm(len(labels)).split(batch_size):
            v, z, y, r = (value[ix].to(device) for value in (views, fixed, labels, reliable))
            output, delta = gate(v, z, r)
            if method == "dynamic_ranking":
                objective = ranking_loss(output, y, gate.temperature)
                valid_pairs += sum(bool((y[:, c] > .5).any() and (y[:, c] <= .5).any()) for c in range(y.shape[1]))
            else:
                objective = F.binary_cross_entropy_with_logits(output, y)
            loss = objective + 0.1 * F.mse_loss(output, z) + delta.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for parameter in gate.parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError("Invalid fusion gradient")
                    grad_sq += float(parameter.grad.square().sum())
            nn.utils.clip_grad_norm_(gate.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ix)
            steps += 1
        history.append({"epoch": epoch + 1, "loss": loss_sum / len(labels), "updates": steps,
                        "valid_ranking_class_batches": valid_pairs,
                        "gradient_rms_norm": (grad_sq / steps) ** .5})
        scheduler.step()
    gate.cpu().eval().requires_grad_(False)
    return gate, {"initial_logit_difference": 0., "samples": len(labels), "history": history,
                  "trainable_parameters": sum(p.numel() for p in gate.parameters()),
                  "positive_counts": labels.sum(dim=0).tolist(),
                  "negative_counts": (1 - labels).sum(dim=0).tolist()}


def digest_files(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def pair_changes(base, updated, labels):
    counts = []
    for c in range(labels.shape[1]):
        p, n = labels[:, c] > .5, labels[:, c] <= .5
        corrected, damaged = 0, 0
        for positive in torch.nonzero(p).flatten().split(256):
            old_order = base[positive, c, None] > base[n, c][None, :]
            new_order = updated[positive, c, None] > updated[n, c][None, :]
            corrected += int((~old_order & new_order).sum())
            damaged += int((old_order & ~new_order).sum())
        counts.append({"class": c, "corrected_pairs": corrected, "damaged_pairs": damaged})
    return counts


def run(args):
    from .runner import compute_metrics, summarize_tasks, write_evaluation_scores, git_metadata
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise ValueError("Refuse to overwrite experiment output")
    config = json.loads((source / "config.json").read_text())
    if (config["reporting_split"] != "val" or config.get("also_report_test")
        or not config.get("calibration_training_exclusion") or config.get("calibration_fraction") != .2
        or not config.get("export_calibration_view_scores") or config["parax_mode"] != "disabled"
        or config["adapter_mode"] != "disabled" or config["view_fusion"] != "fixed_three_view"):
        raise ValueError("Requires completed 20% heldout B0 validation source")
    rows = json.loads((source / "task_metrics.json").read_text())
    source_summary = json.loads((source / "seed_summary.json").read_text())
    if len(rows) != args.tasks or source_summary.get("status") != "complete":
        raise ValueError("Source training incomplete")
    records = [json.loads((source / "calibration_split_provenance" / f"task{t}.json").read_text()) for t in range(args.tasks)]
    provenance = check_provenance(records)
    tracked = [path for path in source.rglob("*") if path.is_file()]
    before = digest_files(tracked)
    output.mkdir(parents=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sizes = config["task_sizes"]
    gates = {method: [] for method in METHODS}
    history = {method: {} for method in METHODS}
    for task in range(args.tasks):
        columns = slice(sum(sizes[:task]), sum(sizes[:task + 1]))
        data = aligned_baseline(source / "calibration_scores" / f"task{task}.npz",
                                read_scores(source / "calibration_view_scores" / f"task{task}.npz"))
        if data["sample_ids"].tolist() != records[task]["calibration_sample_ids"]:
            raise ValueError("Calibration IDs do not match exclusion provenance")
        for method in METHODS:
            gate, fit = fit_gate(*tensors(data, columns), method, torch.device(args.device),
                                 args.epochs, args.batch_size, config["seed"] + task)
            gates[method].append(gate)
            history[method][str(task)] = fit
            gate_path = output / method / "gates" / f"task{task}.pt"
            gate_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": gate.state_dict(), "task_id": task, "method": method,
                        "class_slice": [columns.start, columns.stop], "radius": .05,
                        "source_split": "excluded_train_groups"}, gate_path)
            print(f"fit task={task} method={method} samples={fit['samples']} loss={fit['history'][-1]['loss']:.6f}", flush=True)
    metrics = {method: [] for method in ("fixed", *METHODS)}
    diagnostics = {method: {} for method in METHODS}
    predictions = {method: [] for method in ("fixed", *METHODS)}
    # All fitting finished before labels from validation are read.
    for task in range(args.tasks):
        data = aligned_baseline(source / "val_scores" / f"task{task}.npz",
                                read_scores(source / "view_val_scores" / f"task{task}.npz"))
        fixed = torch.from_numpy(data["fused_logits"])
        targets = torch.from_numpy(data["targets"])
        baseline_prob = torch.from_numpy(data["fused_probabilities"])
        baseline_row = compute_metrics(task, baseline_prob, targets, .5)
        if abs(baseline_row.mAP - rows[task]["mAP"]) > 1e-8:
            raise RuntimeError("Saved baseline metrics differ")
        metrics["fixed"].append(baseline_row)
        predictions["fixed"].append((data["sample_ids"], fixed))
        for method in METHODS:
            updated = fixed.clone()
            deltas = []
            for t in range(task + 1):
                columns = slice(sum(sizes[:t]), sum(sizes[:t + 1]))
                v, z, _, r = tensors(data, columns)
                with torch.no_grad():
                    updated[:, columns], delta = gates[method][t](v, z, r)
                weights = implied_weights(delta, r)
                if (weights < 0).any() or (weights[~r, :, 2] != 0).any():
                    raise RuntimeError("Invalid fusion weights")
                deltas.append(delta)
            delta = torch.cat(deltas, dim=1)
            # Match saved CPU sigmoid batching; preserve exact zero-change cells.
            probs = torch.cat([torch.sigmoid(chunk) for chunk in updated.split(64)])
            probs = torch.where(updated == fixed, baseline_prob, probs)
            metrics[method].append(compute_metrics(task, probs, targets, .5))
            diagnostics[method][str(task)] = {
                "delta_mean": delta.mean((0, 1)).tolist(), "delta_std": delta.std((0, 1), unbiased=False).tolist(),
                "delta_max_abs": delta.abs().amax((0, 1)).tolist(),
                "pair_changes": pair_changes(fixed, updated, targets),
            }
            predictions[method].append((data["sample_ids"], updated))
            write_evaluation_scores(output / method / "val_scores" / f"task{task}.npz", task,
                                    data["sample_ids"].tolist(), updated, targets,
                                    probs, data["batch_lengths"].tolist())
    # Compare old task scores on the SAME samples, independent of changing pools.
    for method in ("fixed", *METHODS):
        ids0, scores0 = predictions[method][0]
        ids1, scores1 = predictions[method][-1]
        index = {value: i for i, value in enumerate(ids1)}
        scores1 = scores1[[index[value] for value in ids0], :sizes[0]]
        drift = float((scores1 - scores0).abs().mean())
        root = output / method
        root.mkdir(exist_ok=True)
        (root / "task_metrics.json").write_text(json.dumps([asdict(row) for row in metrics[method]], indent=2) + "\n")
        summary = summarize_tasks(metrics[method])
        summary["same_task0_sample_logit_drift"] = drift
        (root / "seed_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if method != "fixed":
            (root / "fit_history.json").write_text(json.dumps(history[method], indent=2) + "\n")
            (root / "diagnostics.json").write_text(json.dumps(diagnostics[method], indent=2) + "\n")
        print(f"result {method}: {summary}", flush=True)
    after = digest_files(tracked)
    if before != after:
        raise RuntimeError("Fusion fitting changed source artifacts")
    manifest = {"source_git": config.get("git"), "fusion_git": git_metadata(Path(__file__).resolve().parents[2]), "source": str(source), "source_sha256_before": before, "source_sha256_after": after,
                "provenance": provenance, "tasks": args.tasks, "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": .001, "optimizer": "Adam", "scheduler": "cosine", "radius": .05,
                "consistency_weight": .1, "delta_penalty": 1., "router_hidden": 8,
                "fit_split": "20% train image groups excluded from all base tasks", "validation_used_only_for_evaluation": True,
                "test_forbidden": True, "no_bce_fallback_in_pairless_ranking_batches": True}
    (output / "experiment.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "complete.txt").write_text("FROZEN_VIEW_FUSION_COMPLETE\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.tasks <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        parser.error("Positive tasks/epochs/batch-size required")
    run(args)


if __name__ == "__main__":
    main()
