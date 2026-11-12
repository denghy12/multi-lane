"""One full-TRAIN branch trajectory with concurrent, predeclared fusion schedules.

Three branch workers keep their historical independent RNG/Adam/ASL streams.
At each epoch a barrier records the complete model and fusion optimizer state.
No trained parameter is imported from another run. Fixed/C1/C2 are concurrent
readout controls over the SAME live branch parameters (fusion is detached).
"""
import argparse
import copy
import itertools
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from multi_lane.track_a import runner
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit import atomic_save
from multi_lane.track_a.r1_crossvit_experiment import (
    flags, model_from_config, read, sha, single_thread_worker, write,
)
from multi_lane.track_a.single_trajectory_fusion import (
    METHODS, OnlineFusionSchedules, isolated_observation,
)
from scripts.emotic.r1_asymmetric_forward import AsymmetricReceivers
from scripts.emotic.run_r1_frozen_cross_view import input_paths, load_source
from scripts.emotic.run_r1_frozen9_fusion_val_crossfit import make_records, record, summarize
from scripts.emotic.run_r1_frozen9_learned_fusion import FIXED, blend

ROOT = Path("output/r1_single_trajectory_fusion/seed1_20261112")
LOGS = Path("logs/r1_single_trajectory_fusion/seed1_20261112")
VIEWS = ("full", "person", "face")


def branch_paths(root):
    return {v: root / "branches" / ("native" if v == "full" else "attention") / v / "train" for v in VIEWS}


def branch_flags(root, view, smoke):
    inputs = input_paths()
    args = flags(SimpleNamespace(output=root/"branches", **vars(inputs)),
                 "native" if view == "full" else "attention", view, smoke)
    if smoke:
        i = args.index("--optimizer-updates-per-task")
        args[i+1] = "6"
        args[args.index("--epochs")+1] = "3"
        args += ["--r1-smoke-eval-samples", "64"]
    if view != "full":
        args += ["--r1-frozen-mode", "patch", "--r1-exchange-layer", "8",
                 "--r1-frozen-evidence", "all", "--r1-fixed-loss-scale", "4096",
                 "--r1-fusion-readout", "logit"]
    # No TRAIN exclusion, calibration predictions, external recovery or teacher.
    return args


class FirstBatches:
    def __init__(self, loader):
        self.loader = loader
    def __iter__(self):
        return itertools.islice(self.loader, 2)


def wait_for(path, root, timeout=5400):
    start = time.monotonic()
    while not path.exists():
        if (root/"abort.json").exists():
            raise RuntimeError(f"Controller aborted: {read(root/'abort.json')}")
        if time.monotonic()-start > timeout:
            raise TimeoutError(f"Barrier timed out: {path}")
        time.sleep(1)


@torch.no_grad()
def export_live(model, source, view, task, split, smoke=False):
    classes = list(runner.task_indices(task) if split == "train" else runner.seen_indices(task))
    data = runner.dataset_view(source, classes, include_sample_id=True)
    # Smoke TRAIN still uses the complete cohort; it must include both mask groups.
    if smoke and split != "train":
        data = Subset(data, range(min(64, len(data))))
    loader = DataLoader(data, batch_size=64, num_workers=2, prefetch_factor=1,
        worker_init_fn=single_thread_worker, generator=torch.Generator().manual_seed(77191+task))
    zs, ys, masks, ids = [], [], [], []
    model.eval()
    for images, targets, names in loader:
        images = move_model_inputs(images, torch.device("cuda"))
        with torch.autocast("cuda"):
            z = model.seen_logits(AsymmetricReceivers.receiver_inputs(images, view))[:, classes]
        zs.append(z.float().cpu().numpy())
        ys.append(targets.numpy().astype(np.float32))
        masks.append(images["evidence_face_reliable"].cpu().numpy())
        ids.extend(names)
    result = dict(ids=np.asarray(ids, dtype=str), logits=np.concatenate(zs),
                  targets=np.concatenate(ys), reliable=np.concatenate(masks))
    if (len(set(ids)) != len(ids) or any(not s.startswith(split+":") for s in ids)
            or not np.isfinite(result["logits"]).all()):
        raise AssertionError("Invalid live observation")
    return result


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def worker(root, view, smoke, control=False):
    """Use the unchanged historical runner; wrap its existing epoch-save hook."""
    train = branch_paths(root)[view]
    original = runner.train_task
    source_cache = {}

    def train_with_barrier(model, loader, *args, **kwargs):
        epoch_save = model.save_training_epoch
        if smoke:
            # The runner guard requires an explicit capped budget. Execute the
            # same six updates as 3 epochs x 2 batches to exercise both schedules.
            kwargs["optimizer_updates_per_task"] = None
        if not control:
            def observe(task, epoch, history, optimizer, scheduler, scaler, updates):
                # Save optimizer/RNG BEFORE any observation; every accepted epoch is resumable.
                epoch_save(task, epoch, history, optimizer, scheduler, scaler, updates)
                start = 2 if smoke else 6
                if epoch >= start:
                    wait_for(branch_paths(root)["person"]/"config.json", root)
                    versions = [(p, p._version) for p in model.parameters()]
                    with isolated_observation(model):
                        if "train" not in source_cache:
                            source_cache["train"] = load_source(branch_paths(root)["person"], input_paths(), "train")
                        rows = export_live(model, source_cache["train"], view, task, "train", smoke)
                        save_npz(root/"observations"/f"task{task}"/f"epoch{epoch:02d}_{view}.npz", **rows)
                    if any(p._version != version for p, version in versions):
                        raise AssertionError("Observation modified a branch parameter")
                write(root/"barriers"/f"task{task}"/f"epoch{epoch:02d}_{view}.json",
                      {"task": task, "epoch": epoch, "view": view, "updates": updates,
                       "checkpoint": str(train/"epoch_recovery"/f"task{task}"/f"epoch{epoch:02d}.pth"),
                       "extra_forward_parameter_updates": 0})
                wait_for(root/"barriers"/f"task{task}"/f"epoch{epoch:02d}_accepted.json", root)
            model.save_training_epoch = observe
        try:
            return original(model, FirstBatches(loader) if smoke else loader, *args, **kwargs)
        finally:
            model.save_training_epoch = epoch_save

    runner.train_task = train_with_barrier
    sys.argv = [sys.argv[0], *branch_flags(root, view, smoke)]
    runner.main()


def aligned_observations(root, task, epoch):
    rows = []
    for view in VIEWS:
        with np.load(root/"observations"/f"task{task}"/f"epoch{epoch:02d}_{view}.npz", allow_pickle=False) as d:
            rows.append({k: d[k] for k in d.files})
    first = rows[0]
    for row in rows[1:]:
        for key in ("ids", "targets", "reliable"):
            if not np.array_equal(first[key], row[key]):
                raise AssertionError(f"Cross-view observation alignment failed: {key}")
    return first["ids"], np.stack([r["logits"] for r in rows], axis=1), first["targets"], first["reliable"]


def complete_checkpoint(root, task, epoch, schedules, run_id):
    branches = {}
    for view, path in branch_paths(root).items():
        value = torch.load(path/"epoch_recovery"/f"task{task}"/f"epoch{epoch:02d}.pth", map_location="cpu")
        if value["task"] != task or value["training"]["next_epoch"] != epoch:
            raise AssertionError("Branch epochs differ")
        branches[view] = value
    state = {"format": "single_trajectory_multilane_v1", "run_id": run_id,
             "task": task, "epoch": epoch, "branches": branches,
             "fusion": schedules.state_dict(), "fit_split": "train",
             "source_git": runner.git_metadata(Path.cwd()), "pretrained": str(input_paths().clip_checkpoint)}
    recovery = root/"recovery"/f"task{task}"/f"epoch{epoch:02d}.pth"
    atomic_save(state, recovery)
    write(root/"checkpoints/latest.json", {"path": str(recovery), "task": task, "epoch": epoch})
    if epoch == (3 if schedules.smoke else 30):
        atomic_save(state, root/"checkpoints"/f"task{task}.pth")
    return {v: int(p["training"]["completed_task_updates"]) for v, p in branches.items()}


def assert_same(a, b, name):
    if isinstance(a, torch.Tensor):
        equal = isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        equal = isinstance(b, np.ndarray) and np.array_equal(a, b)
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"Replay keys differ: {name}")
        for k in a:
            assert_same(a[k], b[k], f"{name}/{k}")
        return
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"Replay length differs: {name}")
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{name}/{i}")
        return
    else:
        equal = a == b
    if not equal:
        raise AssertionError(f"Historical trajectory changed: {name}")


def smoke_replay(root, logs):
    control = root/"unobserved_control"
    jobs = []
    for view in VIEWS:
        handle = (logs/f"control_{view}.log").open("w")
        command = [sys.executable, "-u", "-m", "scripts.emotic.run_r1_single_trajectory_fusion",
                   "worker", "--root", str(control), "--view", view, "--smoke", "--control"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="72", MKL_NUM_THREADS="72")
        jobs.append((view, subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT), handle))
    try:
        for view, process, handle in jobs:
            if process.wait(timeout=5400):
                raise RuntimeError(f"Historical control failed: {view}")
        reports = []
        for view in VIEWS:
            for task in (0, 1):
                actual = torch.load(branch_paths(root)[view]/"task_recovery"/f"task{task}.pth", map_location="cpu")
                expected = torch.load(branch_paths(control)[view]/"task_recovery"/f"task{task}.pth", map_location="cpu")
                assert_same(actual["model"], expected["model"], f"{view}/{task}/model")
                assert_same(actual["rng"], expected["rng"], f"{view}/{task}/rng")
                for epoch in (1, 2, 3):
                    a = torch.load(branch_paths(root)[view]/"epoch_recovery"/f"task{task}"/f"epoch{epoch:02d}.pth", map_location="cpu")
                    b = torch.load(branch_paths(control)[view]/"epoch_recovery"/f"task{task}"/f"epoch{epoch:02d}.pth", map_location="cpu")
                    for key in ("optimizer", "scheduler", "scaler", "completed_task_updates"):
                        assert_same(a["training"][key], b["training"][key], f"{view}/{task}/{epoch}/{key}")
                reports.append({"view": view, "task": task, "model_optimizer_rng_exact": True})
        write(root/"smoke_replay.json", {"status": "passed", "comparisons": reports})
    finally:
        for _, process, handle in jobs:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)
            handle.close()


def finish(root, smoke):
    """Real image forwards load each WHOLE same-run task checkpoint."""
    paths, inputs = branch_paths(root), input_paths()
    tasks = range(2 if smoke else 8)
    budget_report = {}
    for view, path in paths.items():
        config, summary = read(path/"config.json"), read(path/"seed_summary.json")
        if config["calibration_fraction"] != 0 or summary["status"] != "complete":
            raise AssertionError("Training excluded data or did not complete")
        history = read(path/"training_history.json")
        budget_report[view] = []
        for task in tasks:
            rows, counts = history[str(task)], summary["calibration_counts"][str(task)]
            expected = 6 if smoke else 30*math.ceil(counts["fit"]/64)
            actual = int(sum(r["optimizer_steps"] for r in rows))
            if (len(rows) != (3 if smoke else 30) or actual != expected
                    or any(r["skipped_optimizer_steps"] for r in rows)
                    or counts["calibration"] != 0
                    or counts["fit"]+counts["face_filter_removed"] != counts["eligible"]):
                raise AssertionError(f"TRAIN/update budget mismatch {view}/{task}")
            budget_report[view].append({"task": task, "updates": actual, **counts})
    write(root/"training_budget_checks.json", budget_report)
    models = {v: model_from_config(read(p/"config.json"), inputs.clip_checkpoint) for v, p in paths.items()}
    records = {s: make_records(METHODS) for s in ("val", "test")}
    hashes = {str(root/"checkpoints"/f"task{t}.pth"): sha(root/"checkpoints"/f"task{t}.pth") for t in tasks}
    write(root/"locked_evaluation.json", {"whole_checkpoints": hashes, "gradient_updates": False,
        "methods": METHODS, "epoch": 3 if smoke else 30, "fit_split": "train"})
    for split in ("val", "test"):
        source = load_source(paths["person"], inputs, split)
        for task in tasks:
            state = torch.load(root/"checkpoints"/f"task{task}.pth", map_location="cpu")
            observed = []
            for view, model in models.items():
                missing = model.load_state_dict(state["branches"][view]["model"], strict=False)
                if missing.unexpected_keys or any(not n.startswith("visual_encoder.") for n in missing.missing_keys):
                    raise ValueError("Whole checkpoint is incomplete")
                model.restore_task(task)
                observed.append(export_live(model, source, view, task, split, smoke))
            first = observed[0]
            for row in observed[1:]:
                if any(not np.array_equal(first[k], row[k]) for k in ("ids", "targets", "reliable")):
                    raise AssertionError("Evaluation views differ")
            z = np.stack([r["logits"] for r in observed], axis=1)
            schedules = OnlineFusionSchedules(runner.TASK_SIZES, "cuda", smoke)
            schedules.load_state_dict(state["fusion"])
            with torch.no_grad():
                fused = {"fixed": blend(z, first["reliable"], FIXED)}
                fused.update({n: bank(torch.as_tensor(z, device="cuda"),
                    torch.as_tensor(first["reliable"], device="cuda"), task).cpu().numpy()
                    for n, bank in schedules.banks.items()})
            record(records[split], task, fused, first["targets"], first["reliable"], list(runner.task_indices(task)))
            save_npz(root/"predictions"/split/f"task{task}.npz", **{**first, "branch_logits": z, **fused})
            write(root/"evaluation_progress.json", {"split": split, "task": task})
            print("WHOLE_MODEL_FORWARD", split, task, len(first["ids"]), flush=True)
    if any(sha(Path(p)) != digest for p, digest in hashes.items()):
        raise AssertionError("Evaluation modified a trained checkpoint")
    result = {s: summarize(rows, smoke) for s, rows in records.items()}
    write(root/"result.json", result)
    write(root/"completion.json", {"status": "complete", "smoke": smoke,
        "training_sources": 1, "all_legal_train_used": True, "validation_parameter_updates": 0,
        "whole_checkpoint_forward": True, "results": result})


def run(root, logs, smoke):
    if root.exists():
        raise FileExistsError("Use a fresh run root; never overwrite another trajectory")
    root.mkdir(parents=True)
    logs.mkdir(parents=True, exist_ok=True)
    inputs = input_paths()
    write(root/"protocol.json", {"run_id": str(root), "seed": 1, "smoke": smoke,
        "dataset": str(inputs.data_root), "pretrained": str(inputs.clip_checkpoint),
        "pretrained_sha256": sha(inputs.clip_checkpoint), "git": runner.git_metadata(Path.cwd()),
        "methods": METHODS, "branch_epochs": 3 if smoke else 30, "train_batch": 64,
        "branch_threads": 72, "observation_threads": 2, "fusion_lr": .02,
        "fusion_steps_per_task": 2 if smoke else 100,
        "single_branch_trajectory_shared_by_detached_readouts": True,
        "branch_commands": {v: branch_flags(root, v, smoke) for v in VIEWS},
        "evaluation": "final complete checkpoints, both Val and Test, no fitting"})
    processes, handles = {}, []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="72", MKL_NUM_THREADS="72")
    try:
        for view in VIEWS:
            handle = (logs/f"{view}.log").open("w")
            handles.append(handle)
            cmd = [sys.executable, "-u", "-m", "scripts.emotic.run_r1_single_trajectory_fusion",
                   "worker", "--root", str(root), "--view", view, *(["--smoke"] if smoke else [])]
            processes[view] = subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT)
        write(root/"workers.json", {v: p.pid for v, p in processes.items()})
        torch.set_num_threads(2)
        runner.set_seed(1, True)
        schedules = OnlineFusionSchedules(runner.TASK_SIZES, "cuda", smoke)
        report = {}
        for task in range(2 if smoke else 8):
            schedules.begin_task(task)
            report[str(task)] = {}
            for epoch in range(1, (3 if smoke else 30)+1):
                ready = [root/"barriers"/f"task{task}"/f"epoch{epoch:02d}_{v}.json" for v in VIEWS]
                start = time.monotonic()
                while not all(p.exists() for p in ready):
                    failed = {v: p.poll() for v, p in processes.items() if p.poll() is not None}
                    if failed or time.monotonic()-start > 5400:
                        raise RuntimeError(f"Branch barrier failed {task}/{epoch}: {failed}")
                    time.sleep(2)
                if epoch >= (2 if smoke else 6):
                    args = aligned_observations(root, task, epoch)
                    fit = schedules.observe(epoch, *args)
                    fit["people"] = len(args[0])
                    fit["reliable"] = int(args[3].sum())
                else:
                    fit = schedules.warmup(epoch)
                updates = complete_checkpoint(root, task, epoch, schedules, str(root))
                report[str(task)][str(epoch)] = fit
                write(root/"fusion_history.json", report)
                write(root/"progress.json", {"task": task, "epoch": epoch, "branch_updates": updates,
                      "fusion_updates": schedules.updates, "status": "training"})
                write(root/"barriers"/f"task{task}"/f"epoch{epoch:02d}_accepted.json", {"checkpoint_saved": True})
                print("ACCEPTED_EPOCH", task, epoch, updates, schedules.updates, flush=True)
            expected = 2 if smoke else 100
            if any(v != expected for v in schedules.updates.values()):
                raise AssertionError("Fusion update budget differs")
        for view, p in processes.items():
            if p.wait(timeout=5400) != 0:
                raise RuntimeError(f"Branch failed after last epoch: {view}")
        torch.set_num_threads(2)
        write(root/"training_complete.json", {"status": "complete", "whole_checkpoints": len(report)})
        finish(root, smoke)
        if smoke:
            # Release evaluation models before the three independent replay controls.
            torch.cuda.empty_cache()
            smoke_replay(root, logs)
    except BaseException as exc:
        write(root/"abort.json", {"error": repr(exc)})
        for p in processes.values():
            if p.poll() is None:
                p.terminate()
        for p in processes.values():
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "worker", "evaluate"))
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--logs", type=Path, default=LOGS)
    p.add_argument("--view", choices=VIEWS)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--control", action="store_true", help="Unobserved historical smoke branch for exact replay")
    a = p.parse_args()
    if a.action == "run":
        run(a.root, a.logs, a.smoke)
    elif a.action == "worker":
        worker(a.root, a.view, a.smoke, a.control)
    else:
        torch.set_num_threads(2)
        runner.set_seed(1, True)
        finish(a.root, a.smoke)
