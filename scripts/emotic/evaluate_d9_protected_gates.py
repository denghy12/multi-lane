"""Four protected readouts on identical subjects, with exact historical branch replay."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.d9_relation_fusion import READOUTS, DIRECTION_READOUTS, SAMPLE_MODES, sample_readouts
from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES, TaskMetrics, set_seed, summarize_tasks
from scripts.emotic.evaluate_d9_relation import (
    VIEWS, SHUFFLE_SEED, collect_readouts, metrics_for_stage, readouts, stage_loader,
    weight_statistics, weighted_logits, write_json,
)

GROUPS = ("unreliable", "reliable")


def verify_reference(path, task, ids, targets, reliable, branches, fixed_weights, fixed_logits):
    with np.load(path, allow_pickle=False) as old:
        old_ids = old["sample_ids"].tolist()
        if len(set(ids)) != len(ids) or len(set(old_ids)) != len(old_ids) or set(ids) != set(old_ids):
            raise ValueError("Historical fixed replay: unique person IDs differ")
        if tuple(old["view_order"]) != VIEWS or list(old["task_sizes"]) != list(TASK_SIZES[:task+1]):
            raise ValueError("Historical fixed replay: view or class/task order differs")
        by_id = {key: row for row, key in enumerate(old_ids)}
        order = [by_id[key] for key in ids]
        for name, tensor in (("targets", targets), ("reliable", reliable), ("branch_logits", branches),
                             ("weights", fixed_weights), ("fused_logits", fixed_logits)):
            ref = old[name][order]
            current = tensor.numpy()
            if ref.shape != current.shape or ref.dtype != current.dtype:
                raise ValueError(f"Historical fixed replay: {name} shape or dtype differs")
            if not np.array_equal(ref, current):
                pos = tuple(np.argwhere(ref != current)[0])
                delta = float(np.max(np.abs(ref.astype(float) - current)))
                raise ValueError(f"Historical fixed replay: {name} differs for {ids[pos[0]]}, "
                                 f"coordinates={pos[1:]}, ref={ref[pos]}, current={current[pos]}, max_abs={delta}")


def cohort_means(weights, reliable):
    """One unlabelled Validation mean per task head and availability cohort."""
    means = {}
    for name, mask in zip(GROUPS, (~reliable, reliable)):
        group = weights[mask]
        if len(group):
            # Preserve a task-static weight exactly, including on large populations.
            mean = group.double().mean(0).to(group.dtype)
            mean = torch.where((group == group[0]).all(0), group[0], mean)
            means[name] = mean.tolist()
    return means


def mean_readout(task, branches, weights, scores, reliable, means):
    assigned = torch.full_like(weights, torch.nan)
    valid = torch.zeros_like(reliable)
    for name, mask in zip(GROUPS, (~reliable, reliable)):
        if name in means:
            assigned[mask] = torch.as_tensor(means[name], dtype=weights.dtype)
            valid |= mask
    sizes = TASK_SIZES[:task + 1]
    fused = weighted_logits(branches, assigned, sizes)
    same = (weights == assigned).all(-1).repeat_interleave(torch.as_tensor(sizes), dim=1)
    return torch.where(same, scores, fused), valid


def verify_sample_controls(result, reliable):
    """A newly trained 90% branch has no historical replay peer; check both controls independently."""
    for name, available in (("fixed", (.64, .16, .20)), ("equal", (1/3, 1/3, 1/3))):
        weights = result[name][0]
        unavailable = (.8, .2, 0.) if name == "fixed" else (.5, .5, 0.)
        expected = torch.where(reliable[:, None], weights.new_tensor(available),
                               weights.new_tensor(unavailable))
        torch.testing.assert_close(weights, expected[:, None, :].expand_as(weights),
                                   rtol=0., atol=1e-7)


def assess(task, batch, reference_file, validation_means, names=READOUTS):
    ids, targets, reliable, branches, result = batch
    if set(result) != set(names):
        raise ValueError(f"Missing or unexpected readouts: {list(result)}")
    if reference_file is not None:
        verify_reference(reference_file, task, ids, targets, reliable, branches, *result["fixed"])
    if "equal" in names:
        verify_sample_controls(result, reliable)
    all_predictions = {v: branches[..., i] for i, v in enumerate(VIEWS)}
    mean_predictions, means, stats = {}, {}, {}
    arrays = dict(sample_ids=np.asarray(ids), targets=targets.numpy(), reliable=reliable.numpy(),
                  branch_logits=branches.numpy(), view_order=np.asarray(VIEWS),
                  task_sizes=np.asarray(TASK_SIZES[:task+1]))
    for name in names:
        weights, scores = result[name]
        # Legacy recomputation verifies normalization, finite values, masking and score reconstruction.
        checked, donors = readouts(task, ids, reliable, branches, weights, scores)
        all_predictions[name] = scores
        all_predictions[name + "_shuffled"] = checked["shuffled_within_reliability"]
        arrays["shuffled_source_rows"] = donors.numpy()
        arrays[name + "_weights"] = weights.numpy()
        arrays[name + "_logits"] = scores.numpy()
        arrays[name + "_shuffled_logits"] = all_predictions[name + "_shuffled"].numpy()
        stats[name] = weight_statistics(weights, reliable)
        means[name] = (cohort_means(weights, reliable) if validation_means is None
                       else validation_means[name])
        mean_scores, ok = mean_readout(task, branches, weights, scores, reliable, means[name])
        mean_predictions[name + "_val_mean"] = (mean_scores, ok)
        arrays[name + "_val_mean_logits"] = mean_scores.numpy()
    metrics = metrics_for_stage(task, targets, reliable, all_predictions)
    for name, (scores, ok) in mean_predictions.items():
        # Only the optional group-mean diagnostic can be unavailable when Val lacked a cohort.
        groups = metrics_for_stage(task, targets[ok], reliable[ok], {name: scores[ok]})[name] if ok.any() else {}
        if not ok.all():
            groups.pop("all", None)
        metrics[name] = groups
    return dict(metrics=metrics, weights=stats, validation_cohort_means=means, exact_replay=reference_file is not None), arrays


def evaluate(root, reference_root, split, batch_size, workers, smoke):
    from multi_lane.track_a.d9_joint_data import build_joint_source
    from scripts.emotic.run_d9_relation_training import load_evaluation_model
    from scripts.emotic.run_r1_frozen_cross_view import input_paths

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    set_seed(1, True)
    device = torch.device("cuda")
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    mode = config["mode"]
    sample_mode = mode in SAMPLE_MODES
    if mode not in ("multi", "direction") and not sample_mode:
        raise ValueError(f"Unsupported protected evaluation mode: {mode}")
    if mode == "sample_holdout" and reference_root is not None:
        raise ValueError("The 90% branches have a different TRAIN cohort from the historical fixed run")
    if (not sample_mode or mode == "sample_full_asl") and reference_root is None:
        raise ValueError("Historical reference required to verify this protected run")
    if reference_root is not None:
        reference_config = json.loads((reference_root / "config.json").read_text())
        if (reference_config["mode"] != "fixed" or reference_config["smoke"] != smoke
                or not reference_config.get("protect_branches")):
            raise ValueError("Expected matching protected fixed reference")
    if config["smoke"] != smoke or not config.get("protect_branches"):
        raise ValueError("Protected evaluation needs matching smoke/formal configurations")
    direction_mode = config["mode"] == "direction"
    names = DIRECTION_READOUTS if direction_mode else READOUTS
    if sample_mode:
        names = sample_readouts(mode)
    title = "smoke_evaluation" if smoke else "evaluation"
    directory = root / title / split
    directory.mkdir(parents=True, exist_ok=True)
    source = build_joint_source(input_paths(), split)
    stages = []
    val_file = root / title / "val/result.json"
    validation = json.loads(val_file.read_text()) if split == "test" else None
    config_hash = sha256_file(config_path)
    val_hash = sha256_file(val_file) if validation is not None else None
    task_count = 2 if smoke else len(TASK_SIZES)
    if validation is not None and (validation["split"] != "val" or len(validation["stages"]) != task_count):
        raise ValueError("Complete Validation before Test")
    for task in range(task_count):
        checkpoint = root / "compact_checkpoints" / f"task{task}.pth"
        ref = ref_hash = None
        if reference_root is not None:
            ref = reference_root / title / split / f"task{task}.npz"
            ref_hash = sha256_file(ref)
            ref_report = json.loads(ref.with_suffix(".json").read_text())
            if ref_report["scores_sha256"] != ref_hash or ref_report["protocol"]["batch_size"] != batch_size:
                raise ValueError(f"Reference artifact hash or eval batch size mismatch for task {task}")
        protocol = dict(schema={"direction": 2, "multi": 1}.get(mode, 3), task=task, split=split, smoke=smoke, batch_size=batch_size,
                        checkpoint_sha256=sha256_file(checkpoint), config_sha256=config_hash,
                        reference_scores_sha256=ref_hash, validation_result_sha256=val_hash,
                        shuffle_seed=SHUFFLE_SEED)
        means = None
        if validation is not None:
            val = validation["stages"][task]
            if any(val["protocol"][key] != protocol[key] for key in ("task", "checkpoint_sha256", "config_sha256")):
                raise ValueError(f"Validation and Test checkpoint mismatch at task {task}")
            means = val["validation_cohort_means"]
        report = directory / f"task{task}.json"
        scores = report.with_suffix(".npz")
        if report.exists() and scores.exists():
            previous = json.loads(report.read_text())
            if previous["protocol"] == protocol and sha256_file(scores) == previous["scores_sha256"]:
                stages.append(previous)
                continue
        model = load_evaluation_model(root, task, device=str(device))
        data = stage_loader(source, task, smoke, batch_size, workers)
        batch = collect_readouts(model, data, device, multi=True, include_direction=direction_mode)
        del model
        directions = batch[-1] if direction_mode else None
        if direction_mode:
            batch = batch[:-1]
        row, arrays = assess(task, batch, ref, means, names=names)
        if direction_mode:
            from scripts.emotic.d9_direction_metrics import direction_metrics
            row["direction_selection"], selection_arrays = direction_metrics(
                batch[1], batch[2], batch[3], directions, TASK_SIZES[:task + 1],
                center_weights=batch[4]["center"][0])
            arrays.update(selection_arrays)
        if sha256_file(checkpoint) != protocol["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint changed during evaluation of task {task}")
        temp = scores.with_suffix(f".{os.getpid()}.tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temp.replace(scores)
        row.update(protocol=protocol, scores_sha256=sha256_file(scores))
        write_json(report, row)
        stages.append(row)
        maps = {key: round(row["metrics"][key]["all"]["mAP"], 4) for key in names}
        print(f"{split} task{task} people={len(batch[0])} exact_replay={row['exact_replay']} mAP={maps}", flush=True)
    summary = {}
    for method in stages[-1]["metrics"]:
        summary[method] = {}
        for group in ("all", *GROUPS):
            rows = [s["metrics"][method].get(group) for s in stages]
            if all(r is not None for r in rows):
                summary[method][group] = summarize_tasks([TaskMetrics(**r) for r in rows])
    write_json(directory / "result.json", dict(split=split, smoke=smoke, summary=summary, stages=stages,
               mean_protocol="Per-stage unlabelled Validation group means, locked for Test"))
    print(f"Saved {directory / 'result.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch size must be positive; workers must be nonnegative")
    evaluate(args.root, args.reference_root, args.split, args.batch_size, args.workers, args.smoke)


if __name__ == "__main__":
    main()
