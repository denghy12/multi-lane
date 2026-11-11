"""Predeclared TRAIN 80/20 branch/fusion learning, with automatic locked Val/Test."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from multi_lane.track_a.r1_crossvit_experiment import flags, model_from_config, read, sha, write
from multi_lane.track_a.runner import TASK_SIZES, git_metadata, set_seed, task_indices
from multi_lane.track_a.task_logit_fusion import TaskLogitFusion
from scripts.emotic.run_r1_frozen9_fusion_backprop import rank_support, smooth_ap_backward
from scripts.emotic.run_r1_frozen9_fusion_val_crossfit import make_records, record, summarize
from scripts.emotic.run_r1_frozen9_learned_fusion import FIXED, Rows, blend, export_rows, verify_hashes
from scripts.emotic.run_r1_frozen_cross_view import input_paths, load_source, queue, preflight
from scripts.emotic.run_r1_train_only_fusion import LR, STEPS, TEMPERATURE, save_checkpoint, load_checkpoint


ROOT = Path("output/r1_train_holdout_fusion/seed1_20261111")
LOGS = Path("logs/r1_train_holdout_fusion/seed1_20261111")
VIEWS = ("full", "person", "face")
METHODS = ("fixed", "equal", "seen_smooth_ap", "heldout_bce", "heldout_smooth_ap")
LEARNED = METHODS[2:]
SALT = "emotic-reliability-calibration-v1"


def image_key(sample_id):
    image, sep, person = str(sample_id).rpartition("#person=")
    if not image.startswith("train:") or not sep or not person.isdecimal():
        raise ValueError("Expected original TRAIN image and person ID")
    return image


def heldout_id(sample_id):
    digest = hashlib.sha256(f"{SALT}:{image_key(sample_id)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") < int(.2*(1 << 64))


def fit_masks(ids):
    heldout = np.array([heldout_id(value) for value in ids], dtype=np.bool_)
    if not heldout.any() or heldout.all():
        raise ValueError("TRAIN holdout partition is empty")
    images = np.array([image_key(value) for value in ids])
    candidates = sorted(set(images[~heldout]), key=lambda s: hashlib.sha256(("seen-control:"+s).encode()).digest())
    # Deterministic original-image grouping; approximate matching of person count.
    chosen, count, target = set(), 0, int(heldout.sum())
    for image in candidates:
        size = int((images == image).sum())
        if count and abs(count-target) <= abs(count+size-target):
            break
        chosen.add(image)
        count += size
    seen = np.array([value in chosen for value in images])
    if np.any(seen & heldout) or set(images[seen]).intersection(images[heldout]):
        raise AssertionError("A fusion control image crosses the TRAIN partitions")
    return seen, heldout


def training_paths(root):
    return {view: root / "branches" / ("native" if view == "full" else "attention") / view / "train"
            for view in VIEWS}


def training_command(root, view, smoke):
    inputs = input_paths()
    kind = "native" if view == "full" else "attention"
    args = flags(SimpleNamespace(output=root / "branches", **vars(inputs)), kind, view, smoke)
    # The runner's historical Face endpoint guard is bypassed only by the new,
    # explicit 80/20 study flag. Validation remains read-only.
    args[args.index("--reporting-split")+1] = "val"
    args[args.index("--evaluation-score-purpose")+1] = "validation_search"
    args += ["--calibration-fraction", ".2", "--save-calibration-scores", "--train-only-fusion-holdout"]
    if view != "full":
        args += ["--r1-frozen-mode", "patch", "--r1-exchange-layer", "8",
                 "--r1-frozen-evidence", "all", "--r1-fixed-loss-scale", "4096",
                 "--r1-fusion-readout", "logit"]
    if smoke:
        args += ["--r1-smoke-eval-samples", "64"]
    return [sys.executable, "-u", "-m", "multi_lane.track_a.runner", *args]


def check_training(root, smoke):
    results = {}
    for view, path in training_paths(root).items():
        config, counts, history = (read(path/name) for name in
                                  ("config.json", "calibration_counts.json", "training_history.json"))
        if not config["train_only_fusion_holdout"] or config["calibration_fraction"] != .2:
            raise ValueError("Branches did not exclude the declared TRAIN images")
        summary = read(path / "seed_summary.json")
        if summary["status"] != "complete":
            raise ValueError("Branch training incomplete")
        rows = []
        for task in range(2 if smoke else 8):
            epochs = history[str(task)]
            if any(row["skipped_optimizer_steps"] for row in epochs):
                raise ValueError("An AMP optimizer update was skipped")
            actual = sum(row["optimizer_steps"] for row in epochs)
            expected = 2 if smoke else 30*math.ceil(counts[str(task)]["fit"]/64)
            if actual != expected or (not smoke and len(epochs) != 30):
                raise ValueError(f"80% branch budget mismatch: {view}/{task}, {actual}!={expected}")
            rows.append({"task": task, "updates": actual, **counts[str(task)]})
        results[view] = rows
    write(root / "training_checks.json", results)


def cached_rows(root, models, paths, source, split, task, smoke):
    path = root / "branch_logits" / split / f"task{task}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            return Rows(data["ids"], data["logits"], data["targets"], data["reliable"])
    rows = export_rows(models, paths, source, split, task, smoke=smoke)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=rows.ids, logits=rows.logits, targets=rows.targets, reliable=rows.reliable)
    return rows


def fit_gate(rows, mask, task, bank, bce_loss, steps):
    if any(not str(value).startswith("train:") for value in rows.ids):
        raise ValueError("Only TRAIN may train the fusion parameters")
    logits = torch.as_tensor(rows.logits[mask], device="cuda").detach()
    labels = torch.as_tensor(rows.targets[mask], device="cuda")
    reliable = torch.as_tensor(rows.reliable[mask], device="cuda")
    if not bool(reliable.any()) or bool(reliable.all()):
        raise ValueError("Both reliability groups must occur in this fusion training subset")
    before = {k: v.clone() for k, v in bank.state_dict().items()}
    gate = bank.add_task(task).cuda()
    optimizer = torch.optim.Adam(gate.parameters(), lr=LR, weight_decay=0)
    support = rank_support(labels)
    if not bce_loss and not support:
        raise ValueError("No sortable current class")
    trace = []
    for step in range(steps+1):
        optimizer.zero_grad(set_to_none=True)
        z = gate(logits, reliable)
        if bce_loss:
            loss = F.binary_cross_entropy_with_logits(z, labels)
            loss.backward()
        else:
            loss = smooth_ap_backward(z, support, temperature=TEMPERATURE)
        if not torch.isfinite(loss) or not torch.isfinite(gate.raw_shares.grad).all():
            raise FloatingPointError("Invalid TRAIN fusion gradients")
        if step in (0, 29, 99, steps):
            trace.append({"step": step, "objective": float(loss),
                          "gradient_norm": float(gate.raw_shares.grad.norm()),
                          "shares": gate.raw_shares.sigmoid().detach().cpu().tolist()})
        if step < steps:
            optimizer.step()
    if any(not torch.equal(v, bank.state_dict()[k]) for k, v in before.items()):
        raise AssertionError("Old fusion parameters changed")
    gate.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    return {"people": int(mask.sum()), "images": len({image_key(s) for s in rows.ids[mask]}),
            "reliable": int(rows.reliable[mask].sum()),
            "positives_per_class": rows.targets[mask].sum(0).astype(int).tolist(),
            "negatives_per_class": (1-rows.targets[mask]).sum(0).astype(int).tolist(),
            "trace": trace}, optimizer.state_dict()


@torch.no_grad()
def evaluate_cached(root, split, banks, tasks, smoke):
    records = make_records(METHODS)
    state = {name: {k: v.clone() for k, v in m.state_dict().items()} for name, m in banks.items()}
    for task in tasks:
        path = root / "branch_logits" / split / f"task{task}.npz"
        with np.load(path, allow_pickle=False) as data:
            rows = Rows(data["ids"], data["logits"], data["targets"], data["reliable"])
        z = torch.as_tensor(rows.logits, device="cuda")
        reliable = torch.as_tensor(rows.reliable, device="cuda")
        fused = {"fixed": blend(rows.logits, rows.reliable, FIXED),
                 "equal": blend(rows.logits, rows.reliable, np.array([[.5,.5,0],[1/3,1/3,1/3]], dtype=np.float32))}
        fused.update({name: model(z, reliable, task).cpu().numpy() for name, model in banks.items()})
        record(records, task, fused, rows.targets, rows.reliable, list(task_indices(task)))
        out = root / "predictions" / split / f"task{task}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, ids=rows.ids, targets=rows.targets, reliable=rows.reliable, **fused)
    if any(not torch.equal(v, banks[name].state_dict()[k])
           for name, values in state.items() for k, v in values.items()):
        raise AssertionError("Evaluation changed fusion parameters")
    return summarize(records, smoke)


def finish(root, smoke):
    inputs, paths = input_paths(), training_paths(root)
    set_seed(1, True)
    models = {view: model_from_config(read(path/"config.json"), inputs.clip_checkpoint)
              for view, path in paths.items()}
    # export_rows follows Full/Person/Face insertion order and receives aligned views.
    tasks = tuple(range(2 if smoke else 8))
    source_hashes = {str(p): sha(p) for path in paths.values() for p in
                     [path/"config.json", *[path/"compact_checkpoints"/f"task{t}.pth" for t in tasks]]}
    source_hashes.update({str(p): sha(p) for p in
                         [inputs.clip_checkpoint, *[inputs.face_manifest_root/"manifests"/f"{s}.jsonl"
                                                   for s in ("train", "val", "test")]]})
    write(root/"branch_source_hashes.json", source_hashes)
    for split in ("train", "val"):
        source = load_source(paths["person"], inputs, split)
        for task in tasks:
            cached_rows(root, models, paths, source, split, task, smoke)
            print("HOLDOUT_EXPORT", split, task, flush=True)
    banks = {name: TaskLogitFusion(TASK_SIZES) for name in LEARNED}
    optimizer_states = {name: {} for name in LEARNED}
    reports = {}
    for task in tasks:
        path = root / "branch_logits/train" / f"task{task}.npz"
        with np.load(path, allow_pickle=False) as data:
            rows = Rows(data["ids"], data["logits"], data["targets"], data["reliable"])
        seen, heldout = fit_masks(rows.ids)
        reports[task] = {}
        for name, bank in banks.items():
            mask = seen if name == "seen_smooth_ap" else heldout
            reports[task][name], optimizer_states[name][task] = fit_gate(
                rows, mask, task, bank, name == "heldout_bce", 10 if smoke else STEPS)
            save_checkpoint(root / "fusion_checkpoints" / name / f"task{task}.pth", bank,
                            optimizer_states[name], {"fit_split": "train", "subset": name})
        write(root / "fit_report.json", reports)
    checkpoint_hashes = {}
    for name in banks:
        path = root / "fusion_checkpoints" / name / f"task{tasks[-1]}.pth"
        checkpoint_hashes[str(path)] = sha(path)
        banks[name] = load_checkpoint(path, "cuda")
    write(root / "locked_train_weights.json", {"fit_split": "train", "checkpoint_hashes": checkpoint_hashes})
    checkpoint_hashes[str(root/"locked_train_weights.json")] = sha(root/"locked_train_weights.json")
    val = evaluate_cached(root, "val", banks, tasks, smoke)
    write(root / "validation.json", val)
    selection = {"criterion": "both Val final/average above matched 80% fixed; no Val updates"}
    if not smoke:
        base = val["fixed"]["all"]["summary"]
        selection["eligible"] = [m for m in LEARNED if all(
            val[m]["all"]["summary"][k] > base[k] for k in ("final_mAP", "average_mAP"))]
    write(root / "selection.json", selection)
    # TEST is exported and evaluated only after fitting/locking and Val decision.
    source = load_source(paths["person"], inputs, "test")
    for task in tasks:
        cached_rows(root, models, paths, source, "test", task, smoke)
        print("HOLDOUT_EXPORT", "test", task, flush=True)
    test = evaluate_cached(root, "test", banks, tasks, smoke)
    write(root / "test.json", test)
    verify_hashes(checkpoint_hashes)
    verify_hashes(source_hashes)
    write(root / "completion.json", {"status": "complete", "smoke": smoke,
        "selection": selection, "fusion_fit_split": "train", "validation_parameter_updates": False,
        "branch_source_hashes_unchanged": True,
        "summaries": {split: {m: groups["all"]["summary"] for m, groups in data.items()}
                      for split, data in (("val", val), ("test", test))}})


def run(root, logs, smoke):
    if root.exists():
        raise FileExistsError(f"Fresh output required: {root}")
    inputs = input_paths()
    if git_metadata(Path.cwd())["dirty"]:
        raise ValueError("Committed clean source required")
    logs.mkdir(parents=True, exist_ok=True)
    write(root/"protocol.json", {"seed": 1, "tasks": 2 if smoke else 8, "epochs": 30,
        "batch_size": 64, "train_fraction": .8, "fusion_fraction": .2, "image_hash_salt": SALT,
        "smoke": smoke, "methods": METHODS, "git": git_metadata(Path.cwd()),
        "fusion_fit_split": "train", "fusion_steps": 10 if smoke else STEPS,
        "lr": LR, "temperature": TEMPERATURE, "initial_shares": [.5,1/3],
        "validation_role": "evaluation and selection only; zero updates"})
    env = dict(os.environ, OMP_NUM_THREADS="72", MKL_NUM_THREADS="72", OPENBLAS_NUM_THREADS="1")
    jobs = []
    smoke_root = root.with_name(root.name+"_smoke")
    for view in VIEWS:
        command = training_command(root, view, smoke)
        if smoke:
            memory = 8500
        else:
            peak = read(smoke_root/"preflight"/f"{view}.json")["admission_memory_mib"]
            memory = max(5000, peak)
        # Queue's subprocesses inherit the historical training thread environment.
        jobs.append({"name": view+"_train", "command": command, "memory_mib": memory})
    previous_env = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    try:
        os.environ.update({key: env[key] for key in previous_env})
        queue(jobs, logs, root, "train", 1 if smoke else 3)
    finally:
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    check_training(root, smoke)
    if smoke:
        for view, path in training_paths(root).items():
            if view == "full":
                history = read(path/"training_history.json")
                peak = max(row["cuda_peak_reserved_mib"] for rows in history.values() for row in rows)
                write(root/"preflight/full.json", {"admission_memory_mib": int(peak*1.2+1024)})
            else:
                preflight(path, inputs, root/"preflight"/f"{view}.json")
    torch.set_num_threads(2)
    finish(root, smoke)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--logs", type=Path, default=LOGS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.root.with_name(args.root.name+"_smoke") if args.smoke else args.root,
        args.logs / "smoke" if args.smoke else args.logs, args.smoke)
