"""Evaluate a jointly trained D9 model and compare readouts on identical branch logits."""
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit_experiment import single_thread_worker
from multi_lane.track_a.runner import (
    TASK_SIZES, TaskMetrics, compute_metrics, dataset_view, seen_indices, set_seed, summarize_tasks,
)
from scripts.emotic.r1_asymmetric_forward import fuse_logits as fixed_r1_logits


VIEWS = ("full", "person", "face")
SHUFFLE_SEED = 20261116


def shuffle_sources(ids, reliable, tasks):
    """Pick a different person's weights within the same availability group, per head."""
    if len(set(ids)) != len(ids) or len(ids) != len(reliable):
        raise ValueError("Evaluation requires one distinct ID and one mask per person")
    # Sort by stable ID so the pairing does not depend on loader order or batch size.
    ordered = np.argsort(np.asarray(ids), kind="stable")
    mask = np.asarray(reliable, dtype=bool)
    donors = np.repeat(np.arange(len(ids))[:, None], tasks, axis=1)
    for task in range(tasks):
        for flag in (False, True):
            members = ordered[mask[ordered] == flag]
            rng = np.random.default_rng(np.random.SeedSequence([SHUFFLE_SEED, task, int(flag)]))
            cycle = rng.permutation(members)
            # Singletons retain their own weights. All other members have a different donor.
            donors[cycle, task] = np.roll(cycle, 1)
    return torch.from_numpy(donors)


def weighted_logits(branches, weights, sizes):
    """Apply each person's task weights to all classes introduced by that task."""
    if branches.ndim != 3 or weights.ndim != 3 or branches.shape[2] != 3:
        raise ValueError("Expected person x class/task x view tensors")
    if weights.shape != (branches.shape[0], len(sizes), 3) or branches.shape[1] != sum(sizes):
        raise ValueError("Gate weights and seen task heads have inconsistent dimensions")
    per_class = weights.repeat_interleave(torch.as_tensor(sizes), dim=1)
    return (branches[..., 0] * per_class[..., 0]
            + branches[..., 1] * per_class[..., 1]
            + branches[..., 2] * per_class[..., 2])


def readouts(task, ids, reliable, branches, weights, actual):
    sizes = TASK_SIZES[:task + 1]
    if not all(torch.isfinite(x).all() for x in (branches, weights, actual)):
        raise FloatingPointError("Non-finite evaluation logits or weights")
    if (weights < 0).any() or not torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]), atol=2e-6):
        raise ValueError("Invalid fusion weights")
    if torch.count_nonzero(weights[~reliable, :, 2]):
        raise ValueError("Unreliable face contributed to fusion")
    replay = weighted_logits(branches, weights, sizes)
    torch.testing.assert_close(replay, actual, rtol=2e-6, atol=5e-6)
    donors = shuffle_sources(ids, reliable, len(sizes))
    donor_weights = weights[donors, torch.arange(len(sizes))]
    shuffled = weighted_logits(branches, donor_weights, sizes)
    same = (weights == donor_weights).all(-1).repeat_interleave(torch.as_tensor(sizes), dim=1)
    shuffled = torch.where(same, actual, shuffled)
    fixed = fixed_r1_logits(*(branches[..., i] for i in range(3)), reliable)
    predictions = {"model": actual, "shuffled_within_reliability": shuffled, "fixed_same_branches": fixed}
    predictions.update({view: branches[..., i] for i, view in enumerate(VIEWS)})
    return predictions, donors


def metrics_for_stage(task, targets, reliable, predictions):
    result = {}
    for method, logits in predictions.items():
        result[method] = {}
        for name, mask in (("all", torch.ones_like(reliable)), ("reliable", reliable),
                           ("unreliable", ~reliable)):
            if mask.any():
                row = compute_metrics(task, logits[mask].sigmoid(), targets[mask], .5)
                result[method][name] = asdict(row)
    return result


def weight_statistics(weights, reliable):
    stats = {}
    for name, mask in (("reliable", reliable), ("unreliable", ~reliable)):
        subset = weights[mask]
        if len(subset):
            stats[name] = {"people": len(subset), "mean": subset.mean(0).tolist(),
                           "std": subset.std(0, unbiased=False).tolist()}
    return stats


def collect(model, loader, device):
    ids, labels, validity, branches, weights, actual = [], [], [], [], [], []
    model.requires_grad_(False).eval()
    with torch.inference_mode():
        for images, target, batch_ids in loader:
            inputs = move_model_inputs(images, device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(inputs, all_seen=True)
            ids.extend(str(value) for value in batch_ids)
            labels.append(target.float().cpu())
            validity.append(images["evidence_face_reliable"].bool().reshape(-1))
            branches.append(torch.stack([output["branches"][v].float().cpu() for v in VIEWS], dim=-1))
            weights.append(output["weights"].float().cpu())
            actual.append(output["fused"].float().cpu())
    return (ids, torch.cat(labels), torch.cat(validity), torch.cat(branches),
            torch.cat(weights), torch.cat(actual))


def write_json(path, payload):
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def evaluate(root, split, batch_size, workers, smoke, device):
    from multi_lane.track_a.d9_joint_data import build_joint_source
    from scripts.emotic.run_d9_relation_training import load_evaluation_model
    from scripts.emotic.run_r1_frozen_cross_view import input_paths

    torch.set_num_threads(2)
    set_seed(1, True)
    directory = root / ("smoke_evaluation" if smoke else "evaluation") / split
    directory.mkdir(parents=True, exist_ok=True)
    config_hash = sha256_file(root / "config.json")
    source = build_joint_source(input_paths(), split)
    stages = []
    task_count = 2 if smoke else len(TASK_SIZES)
    for task in range(task_count):
        checkpoint = root / "compact_checkpoints" / f"task{task}.pth"
        checkpoint_hash = sha256_file(checkpoint)
        protocol = dict(schema=1, split=split, task=task, smoke=smoke, shuffle_seed=SHUFFLE_SEED,
                        batch_size=batch_size, config_sha256=config_hash, checkpoint_sha256=checkpoint_hash)
        report = directory / f"task{task}.json"
        scores = directory / f"task{task}.npz"
        if report.exists() and scores.exists():
            cache = json.loads(report.read_text())
            if cache["protocol"] == protocol and cache["scores_sha256"] == sha256_file(scores):
                stages.append(cache)
                continue
        data = dataset_view(source, seen_indices(task), include_sample_id=True)
        if smoke:
            data = Subset(data, range(min(32, len(data))))
        loader = DataLoader(data, batch_size=batch_size, num_workers=workers,
                            worker_init_fn=single_thread_worker, shuffle=False,
                            generator=torch.Generator().manual_seed(77191 + task),
                            **({"prefetch_factor": 1} if workers else {}))
        model = load_evaluation_model(root, task, device=str(device))
        ids, targets, reliable, branch_logits, weights, actual = collect(model, loader, device)
        del model
        predictions, donors = readouts(task, ids, reliable, branch_logits, weights, actual)
        if sha256_file(checkpoint) != checkpoint_hash:
            raise RuntimeError(f"Checkpoint changed during evaluation of task {task}")
        tmp = scores.with_suffix(f".{os.getpid()}.tmp")
        with tmp.open("wb") as buffer:
            np.savez_compressed(buffer, sample_ids=np.asarray(ids), targets=targets.numpy(),
                                reliable=reliable.numpy(), branch_logits=branch_logits.numpy(),
                                view_order=np.asarray(VIEWS), task_sizes=np.asarray(TASK_SIZES[:task+1]),
                                weights=weights.numpy(), shuffled_source_rows=donors.numpy(),
                                fused_logits=actual.numpy(), fixed_logits=predictions["fixed_same_branches"].numpy(),
                                shuffled_logits=predictions["shuffled_within_reliability"].numpy())
        tmp.replace(scores)
        row = dict(protocol=protocol, scores_sha256=sha256_file(scores),
                   metrics=metrics_for_stage(task, targets, reliable, predictions),
                   weights=weight_statistics(weights, reliable))
        write_json(report, row)
        stages.append(row)
        print(f"{split} task{task} people={len(ids)} mAP={row['metrics']['model']['all']['mAP']:.4f}", flush=True)
    summary = {}
    for method in stages[-1]["metrics"]:
        summary[method] = {}
        for population in ("all", "reliable", "unreliable"):
            rows = [s["metrics"][method].get(population) for s in stages]
            if all(row is not None for row in rows):
                summary[method][population] = summarize_tasks([TaskMetrics(**row) for row in rows])
    write_json(directory / "result.json", dict(split=split, smoke=smoke, summary=summary, stages=stages))
    print(f"Saved {directory / 'result.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive and workers nonnegative")
    evaluate(args.root, args.split, args.batch_size, args.workers, args.smoke, torch.device(args.device))


if __name__ == "__main__":
    main()
