"""Fit task-local, reliability-conditioned logit fusion on Frozen9 TRAIN outputs."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch.utils.data import DataLoader, Subset

from multi_lane.track_a.evaluation_scores import align_evaluation_scores, load_evaluation_scores
from multi_lane.track_a.fuse_face_endpoint_validation import load_face_reliability
from multi_lane.track_a.fuse_fixed_three_view_test import (
    LOCKED_FALLBACK_WEIGHTS, LOCKED_RELIABLE_WEIGHTS,
)
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit_experiment import (
    model_from_config, read, sha, single_thread_worker, write,
)
from multi_lane.track_a.runner import (
    TASK_SIZES, TaskMetrics, compute_metrics, dataset_view, git_metadata,
    seen_indices, set_seed, summarize_tasks, task_indices,
)
from scripts.emotic.r1_asymmetric_forward import AsymmetricReceivers, verify_replay
from scripts.emotic.r1_frozen9_edge_forward import ROOT as EDGE_ROOT, branch_paths
from scripts.emotic.run_r1_frozen_cross_view import input_paths, load_source, restore, score_suffix


ROOT = Path("output/r1_frozen9_learned_fusion/seed1_20261110")
ARCHIVE = Path("../multi-lane-main-r1-frozen9-edges-20261108") / EDGE_ROOT / "evaluation/readonly"
BRANCHES = ("full", "person_all", "face_all")
FIXED = np.stack((LOCKED_FALLBACK_WEIGHTS, LOCKED_RELIABLE_WEIGHTS))
METHODS = ("fixed", "reliable_only", "unreliable_only", "independent", "shared_fp")
BATCH_SIZE = 64
OPTIONS = {"ftol": 1e-12, "maxiter": 200}


@dataclass
class Rows:
    ids: np.ndarray
    logits: np.ndarray  # [person, view, class], in Full/Person/Face order
    targets: np.ndarray
    reliable: np.ndarray


def bce(z, targets):
    return float(np.mean(np.logaddexp(0, z) - targets * z))


def simplex_fit(logits, targets, prior):
    """Fit the auxiliary shares on a simplex, with Full as the remainder."""
    if not len(targets):
        return prior.copy()
    full = logits[:, 0]
    differences = logits[:, 1:] - logits[:, :1]

    def objective(shares):
        z = full + np.sum(differences * shares[None, :, None], axis=1)
        error = expit(z) - targets
        gradient = (differences * error[:, None, :]).mean(axis=(0, 2))
        return bce(z, targets), gradient

    x0 = prior[1:].copy()
    result = minimize(
        objective, x0, jac=True, method="SLSQP",
        bounds=[(0.0, 1.0)] * len(x0),
        constraints=[{"type": "ineq", "fun": lambda x: 1 - x.sum(),
                      "jac": lambda x: -np.ones_like(x)}],
        options=OPTIONS,
    )
    if not result.success or not np.isfinite(result.x).all() or result.fun > objective(x0)[0] + 1e-10:
        raise RuntimeError(f"Simplex fitting failed: {result.message}")
    if result.x.sum() > 1 + 1e-9 or np.min(result.x) < -1e-9:
        raise ValueError("Simplex solution is outside its constraints")
    w = np.maximum(np.r_[1 - result.x.sum(), result.x], 0)
    return w / w.sum()


def shared_fp_fit(logits, targets, reliable, independent):
    """Tie the Person share within Full/Person across the two conditions."""
    full, person, face = np.moveaxis(logits, 1, 0)
    available = reliable[:, None].astype(np.float64)

    def objective(params):
        person_share, face_share = params
        fp = full + person_share * (person - full)
        face_weight = available * face_share
        z = fp + face_weight * (face - fp)
        error = expit(z) - targets
        gradient = np.array([
            (error * (1 - face_weight) * (person - full)).mean(),
            (error * available * (face - fp)).mean(),
        ])
        return bce(z, targets), gradient

    if not reliable.any():
        person_share, face_share = independent[0, 1], 0.20
    elif reliable.all():
        face_share = independent[1, 2]
        fp_mass = independent[1, :2].sum()
        person_share = independent[1, 1] / fp_mass if fp_mass > 1e-12 else 0.20
    else:
        starts = ((0.20, 0.20), (0.5, 1 / 3), (independent[0, 1], independent[1, 2]))
        candidates = [
            minimize(objective, start, jac=True, method="SLSQP",
                     bounds=[(0, 1), (0, 1)], options=OPTIONS)
            for start in starts
        ]
        successful = [res for res in candidates if res.success and np.isfinite(res.fun)]
        if not successful:
            raise RuntimeError("Shared Full/Person fitting failed")
        best = min(successful, key=lambda res: res.fun)
        if best.fun > objective(starts[0])[0] + 1e-10:
            raise RuntimeError("Shared Full/Person fit is worse than its initialization")
        person_share, face_share = best.x
    fallback = np.array([1 - person_share, person_share, 0.0])
    both = (1 - face_share) * fallback
    both[2] = face_share
    return np.stack((fallback, both))


def fit_task(rows):
    logits, targets = rows.logits.astype(np.float64), rows.targets.astype(np.float64)
    prior = FIXED.astype(np.float64)
    prior /= prior.sum(axis=1, keepdims=True)
    good = rows.reliable
    bad = ~good
    fallback = simplex_fit(logits[bad, :2], targets[bad], prior[0, :2])
    independent = np.stack((np.r_[fallback, 0.0], simplex_fit(logits[good], targets[good], prior[1])))
    independent = independent.astype(np.float32)
    weights = {
        "fixed": FIXED,
        "reliable_only": np.stack((FIXED[0], independent[1])),
        "unreliable_only": np.stack((independent[0], FIXED[1])),
        "independent": independent,
        "shared_fp": shared_fp_fit(logits, targets, good, independent.astype(np.float64)).astype(np.float32),
    }
    # Preserve the archived arithmetic for branches that fall back to the prior.
    for name in METHODS[1:]:
        for group, mask in enumerate((bad, good)):
            if not mask.any():
                weights[name][group] = FIXED[group]
    for matrix in weights.values():
        if (not np.isfinite(matrix).all() or (matrix < 0).any() or matrix[0, 2] != 0
                or not np.allclose(matrix.sum(axis=1), 1, rtol=0, atol=1e-7)):
            raise ValueError("Invalid fusion weights")
    fit_report = {}
    for group, mask in (("reliable", good), ("unreliable", bad)):
        fit_report[group] = {
            "samples": int(mask.sum()),
            "positives_per_class": targets[mask].sum(axis=0).astype(int).tolist(),
            "fallback_empty_group": not bool(mask.any()),
            "train_bce": {
                name: bce(blend(rows.logits, good, matrix)[mask], targets[mask])
                if mask.any() else None for name, matrix in weights.items()
            },
        }
    return weights, fit_report


def blend(logits, reliable, matrix):
    weights = matrix[reliable.astype(np.int8)]
    # Keep float32 evaluation and the operation order of the existing R1 readout.
    return (weights[:, 0, None] * logits[:, 0]
            + weights[:, 1, None] * logits[:, 1]
            + weights[:, 2, None] * logits[:, 2])


def stage_logits(rows, bank, stage, method):
    parts = []
    for task in range(stage + 1):
        columns = list(task_indices(task))
        parts.append(blend(rows.logits[:, :, columns], rows.reliable, bank[task][method]))
    return np.concatenate(parts, axis=1)


def source_files(paths, tasks, manifest):
    files = [manifest / "manifests/train.jsonl", manifest / "manifests/val.jsonl"]
    for path in paths.values():
        files.append(path / "config.json")
        files.extend(path / "compact_checkpoints" / f"task{task}.pth" for task in tasks)
    files.extend(archive_files("val", tasks))
    return {str(path): sha(path) for path in files}


def archive_files(split, tasks):
    files = []
    for task in tasks:
        suffix = score_suffix(split, task)
        files.extend(ARCHIVE / "branches" / name / suffix for name in BRANCHES)
        files.append(ARCHIVE / "fusion_scores/D_both_edges" / suffix)
    return files


def verify_hashes(hashes):
    if any(sha(Path(path)) != value for path, value in hashes.items()):
        raise AssertionError("An input artifact changed during the run")


@torch.no_grad()
def export_rows(models, paths, source, split, task, smoke=False):
    for name, model in models.items():
        restore(model, paths[name], task)
        model.requires_grad_(False).eval()
    classes = task_indices(task) if split == "train" else seen_indices(task)
    data = dataset_view(source, classes, include_sample_id=True)
    if smoke:
        data = Subset(data, range(min(BATCH_SIZE, len(data))))
    loader = DataLoader(
        data, batch_size=BATCH_SIZE, num_workers=2, prefetch_factor=1,
        worker_init_fn=single_thread_worker, generator=torch.Generator().manual_seed(77191 + task),
    )
    logits, targets, reliable, ids = [], [], [], []
    for images, target, names in loader:
        images = move_model_inputs(images, torch.device("cuda"))
        branch_logits = []
        for name, model in models.items():
            with torch.autocast("cuda"):
                z = model.seen_logits(AsymmetricReceivers.receiver_inputs(images, name))
            branch_logits.append(z[:, list(classes)].float().cpu().numpy())
        logits.append(np.stack(branch_logits, axis=1))
        targets.append(target.float().cpu().numpy())
        reliable.append(images["evidence_face_reliable"].bool().cpu().numpy())
        ids.extend(names)
    rows = Rows(np.asarray(ids, dtype=str), np.concatenate(logits), np.concatenate(targets),
                np.concatenate(reliable))
    validate_rows(rows, classes, split)
    return rows


def validate_rows(rows, classes, split):
    if (rows.logits.shape != (len(rows.ids), 3, len(classes))
            or rows.targets.shape != (len(rows.ids), len(classes))
            or rows.reliable.shape != (len(rows.ids),)
            or rows.reliable.dtype != np.bool_
            or len(set(rows.ids)) != len(rows.ids) or not len(rows.ids)
            or any(not value.startswith(split + ":") for value in rows.ids)
            or not np.isfinite(rows.logits).all()
            or not np.isin(rows.targets, (0, 1)).all()):
        raise ValueError(f"Invalid {split} prediction export")


def cached_train(root, models, paths, source, task, smoke):
    path = root / "train_logits" / f"task{task}.npz"
    classes = task_indices(task)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["class_indices"], classes):
                raise ValueError(f"Cached class indices do not match task {task}")
            rows = Rows(data["ids"], data["logits"], data["targets"], data["reliable"])
        validate_rows(rows, classes, "train")
        return rows
    rows = export_rows(models, paths, source, "train", task, smoke)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, ids=rows.ids, logits=rows.logits, targets=rows.targets,
                        reliable=rows.reliable, class_indices=np.asarray(classes))
    temporary.replace(path)
    return rows


def archived_rows(split, task, reliability):
    suffix = score_suffix(split, task)
    files = [ARCHIVE / "branches" / name / suffix for name in BRANCHES]
    full, person, face = [load_evaluation_scores(file, BATCH_SIZE) for file in files]
    if full.task_id != task or not np.array_equal(full.class_indices, seen_indices(task)):
        raise ValueError(f"Unexpected archived task or class indices for {split}/{task}")
    other = [align_evaluation_scores(full, item)[2] for item in (person, face)]
    mask = np.asarray([reliability[value] for value in full.sample_ids], dtype=np.bool_)
    rows = Rows(full.sample_ids, np.stack((full.logits, *other), axis=1), full.targets, mask)
    validate_rows(rows, seen_indices(task), split)
    return rows


def evaluate(split, bank, manifest):
    reliable_lookup = load_face_reliability(manifest, split)
    results = {name: {pop: [] for pop in ("all", "reliable", "unreliable")} for name in METHODS}
    replays = []
    for stage in range(len(TASK_SIZES)):
        rows = archived_rows(split, stage, reliable_lookup)
        fused = {name: stage_logits(rows, bank, stage, name) for name in METHODS}
        reference = ARCHIVE / "fusion_scores/D_both_edges" / score_suffix(split, stage)
        replays.append(verify_replay(reference, rows.ids, torch.from_numpy(rows.targets),
                                     torch.from_numpy(fused["fixed"])))
        good = rows.reliable
        if (not np.array_equal(fused["reliable_only"][~good], fused["fixed"][~good])
                or not np.array_equal(fused["unreliable_only"][good], fused["fixed"][good])):
            raise AssertionError("Single-condition learning changed the untouched population")
        for name, logits in fused.items():
            for pop, mask in (("all", np.ones_like(good)), ("reliable", good), ("unreliable", ~good)):
                metric = compute_metrics(stage, torch.from_numpy(logits[mask]).sigmoid(),
                                         torch.from_numpy(rows.targets[mask]), 0.5)
                results[name][pop].append(metric)
    summaries = {
        name: {pop: {"summary": summarize_tasks(metrics),
                     "metrics": [asdict(metric) for metric in metrics]}
               for pop, metrics in populations.items()}
        for name, populations in results.items()
    }
    return {"results": summaries, "fixed_replay": replays}


def select(validation):
    metrics = {name: validation["results"][name]["all"]["summary"] for name in METHODS}
    fixed = metrics["fixed"]
    eligible = [
        name for name in METHODS[1:]
        if all(metrics[name][key] > fixed[key] for key in ("final_mAP", "average_mAP"))
    ]
    degrees = {"reliable_only": 2, "unreliable_only": 1, "independent": 3, "shared_fp": 2}
    chosen = max(eligible, key=lambda name: (metrics[name]["final_mAP"], metrics[name]["average_mAP"],
                                             -degrees[name])) if eligible else "fixed"
    return {"selected": chosen, "eligible": eligible, "criterion": "both Val mAPs above fixed; rank by final, average, then fewer parameters"}


def run(root, smoke):
    started = time.monotonic()
    inputs = input_paths()
    paths = branch_paths(EDGE_ROOT, "readonly")
    tasks = (0, 7) if smoke else tuple(range(len(TASK_SIZES)))
    protocol = {
        "seed": 1, "dataset": str(inputs.data_root), "clip_checkpoint": str(inputs.clip_checkpoint),
        "face_manifest": str(inputs.face_manifest_root), "archive": str(ARCHIVE),
        "batch_size": BATCH_SIZE, "workers": 2, "tasks": tasks, "smoke": smoke,
        "optimizer": "CPU float64 constrained SLSQP, unregularized BCEWithLogits",
        "optimizer_options": OPTIONS,
        "fit_split": "TRAIN", "source": git_metadata(Path.cwd()), "methods": METHODS,
        "hashes": source_files(paths, tasks, inputs.face_manifest_root),
    }
    protocol_path = root / "protocol.json"
    if protocol_path.exists():
        if read(protocol_path) != json.loads(json.dumps(protocol)):
            raise ValueError("Existing output protocol differs")
        if (root / "completion.json").exists():
            raise FileExistsError(f"Run already complete: {root}")
    elif root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {root}")
    write(protocol_path, protocol)
    set_seed(1, True)
    torch.set_num_threads(2)
    models = {
        name: model_from_config(read(path / "config.json"), inputs.clip_checkpoint)
        for name, path in paths.items()
    }
    source = load_source(paths["person_all"], inputs, "train")
    val_source = load_source(paths["person_all"], inputs, "val") if smoke else None
    bank, report, replay = {}, {}, []
    for task in tasks:
        write(root / "progress.json", {"state": "train_export", "task": task, "done": list(bank)})
        rows = cached_train(root, models, paths, source, task, smoke)
        bank[task], report[task] = fit_task(rows)
        if smoke:
            val = export_rows(models, paths, val_source, "val", task, True)
            for i, name in enumerate(BRANCHES):
                replay.append(verify_replay(
                    ARCHIVE / "branches" / name / score_suffix("val", task), val.ids,
                    torch.from_numpy(val.targets), torch.from_numpy(val.logits[:, i]),
                ))
        print("FIT", task, "samples", len(rows.ids), "reliable", int(rows.reliable.sum()),
              "independent", bank[task]["independent"].tolist(),
              "elapsed", round(time.monotonic() - started, 1), flush=True)
    del models, source, val_source
    verify_hashes(protocol["hashes"])
    write(root / "weights.json", {"weights": {
        task: {name: values.tolist() for name, values in weights.items()}
        for task, weights in bank.items()}, "fit_report": report})
    if smoke:
        write(root / "completion.json", {"status": "complete", "smoke": True, "val_source_replay": replay,
                                         "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2})
        return
    write(root / "progress.json", {"state": "validation"})
    validation = evaluate("val", bank, inputs.face_manifest_root)
    write(root / "validation.json", validation)
    selection = select(validation)
    write(root / "selection.json", selection)
    print("VALIDATION_SELECTION", selection, flush=True)
    write(root / "progress.json", {"state": "test", **selection})
    test_files = [*archive_files("test", tasks), inputs.face_manifest_root / "manifests/test.jsonl"]
    test_hashes = {str(path): sha(path) for path in test_files}
    test = evaluate("test", bank, inputs.face_manifest_root)
    verify_hashes({**protocol["hashes"], **test_hashes})
    write(root / "test.json", test)
    write(root / "completion.json", {
        "status": "complete", "smoke": False, "selection": selection,
        "elapsed_seconds": time.monotonic() - started,
        "test_source_hashes": test_hashes,
        "baseline_exact_replays": len(validation["fixed_replay"]) + len(test["fixed_replay"]),
        "summaries": {
            split: {name: values["all"]["summary"] for name, values in record["results"].items()}
            for split, record in (("val", validation), ("test", test))
        },
    })
    write(root / "progress.json", {"state": "complete", **selection})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.root / "smoke" if args.smoke else args.root, args.smoke)
