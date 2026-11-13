"""One trajectory: temporary TRAIN roles, fusion fit, then full-TRAIN return.

Full-extend is a separate matched extension control. Holdout-return has two
predeclared readouts over its single branch trajectory, never imported weights.
"""
import argparse
import copy
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from multi_lane.track_a import runner
from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit import atomic_save
from multi_lane.track_a.r1_crossvit_experiment import model_from_config, read, sha, single_thread_worker, write
from multi_lane.track_a.single_trajectory_fusion import isolated_observation
from multi_lane.track_a.task_logit_fusion import TaskLogitFusion
from scripts.emotic.r1_asymmetric_forward import AsymmetricReceivers
from scripts.emotic.run_r1_frozen_cross_view import input_paths, load_source
from scripts.emotic.run_r1_frozen9_fusion_val_crossfit import make_records, record, summarize
from scripts.emotic.run_r1_frozen9_learned_fusion import FIXED, blend
from scripts.emotic.run_r1_single_trajectory_fusion import (
    FirstBatches, VIEWS, branch_paths, export_live, save_npz, wait_for,
    branch_flags as original_branch_flags, assert_same,
)

ROOT = Path("output/r1_fusion_return_train/seed1_20261113")
LOGS = Path("logs/r1_fusion_return_train/seed1_20261113")
CONDITIONS = ("holdout_return", "full_extend")


def phase_sizes(smoke=False):
    return (2, 1) if smoke else (30, 5)


def branch_flags(root, view, smoke=False, condition="holdout_return"):
    if condition not in CONDITIONS:
        raise ValueError("Unknown trajectory")
    args = original_branch_flags(root, view, smoke)
    # Retain the explicit tiny budget for the runner's bounded-smoke guard;
    # the worker replaces it with the identical 3 epochs x 2 batches budget.
    args[args.index("--epochs")+1] = str(sum(phase_sizes(smoke)))
    args[args.index("--reporting-split")+1] = "val"
    args[args.index("--evaluation-score-purpose")+1] = "validation_search"
    if condition == "holdout_return":
        args += ["--calibration-fraction", ".2", "--save-calibration-scores", "--train-only-fusion-holdout"]
    return args


class ReturnFusion:
    """TRAIN-only finite fit, sealed before any held-out image is returned."""
    def __init__(self, task_sizes, device="cpu", steps=100):
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("Positive fusion update budget required")
        self.bank = TaskLogitFusion(task_sizes).to(device)
        self.device, self.steps = device, steps
        self.task = -1
        self.updates, self.optimizers, self.reports = {}, {}, {}

    def begin_task(self, task):
        if task != self.task+1:
            raise ValueError("Consecutive tasks required")
        self.bank.add_task(task).to(self.device)
        self.task = task
        self.updates[task] = 0

    def fit(self, ids, logits, targets, reliable):
        if self.task < 0 or self.updates[self.task]:
            raise ValueError("Fit once at the pre-return barrier")
        n, c = len(ids), self.bank.task_sizes[self.task]
        if (not n or len(set(ids)) != n or any(not str(i).startswith("train:") for i in ids)
                or tuple(logits.shape) != (n, 3, c) or tuple(targets.shape) != (n, c)
                or tuple(reliable.shape) != (n,)):
            raise ValueError("Only aligned current-task TRAIN observations may train fusion")
        z = torch.as_tensor(logits, dtype=torch.float32, device=self.device).detach()
        y = torch.as_tensor(targets, dtype=torch.float32, device=self.device).detach()
        raw_mask = torch.as_tensor(reliable)
        if raw_mask.dtype != torch.bool:
            raise ValueError("Reliability must be boolean")
        mask = raw_mask.to(self.device)
        if not torch.isfinite(z).all() or not torch.isfinite(y).all():
            raise FloatingPointError("Nonfinite fusion input")
        if not ((y == 0) | (y == 1)).all():
            raise ValueError("Expected binary current-task targets")
        old = {k: v.clone() for k, v in self.bank.state_dict().items()
               if not k.startswith(f"tasks.{self.task}.")}
        gate = self.bank.tasks[self.task]
        gate.requires_grad_(True)
        opt = torch.optim.Adam(gate.parameters(), lr=.02, weight_decay=0.)
        before = float(F.binary_cross_entropy_with_logits(gate(z, mask), y).detach())
        gradients, trajectory = [], []
        for step in range(self.steps):
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(gate(z, mask), y)
            loss.backward()
            grad = gate.raw_shares.grad
            if not torch.isfinite(loss) or grad is None or not torch.isfinite(grad).all():
                raise FloatingPointError("Nonfinite fusion update")
            gradients.append(float(grad.norm()))
            opt.step()
            trajectory.append({"step": step+1, "weights": gate.weight_matrix().detach().cpu().tolist(),
                               "loss_before_update": float(loss.detach())})
        opt.zero_grad(set_to_none=True)
        self.bank.seal()
        if any(not torch.equal(v, self.bank.state_dict()[k]) for k, v in old.items()):
            raise AssertionError("Old task fusion changed")
        self.updates[self.task] = self.steps
        self.optimizers[self.task] = copy.deepcopy(opt.state_dict())
        result = {"people": n, "reliable": int(mask.sum()), "steps": self.steps,
                  "bce_before": before, "bce_after": float(F.binary_cross_entropy_with_logits(gate(z, mask), y)),
                  "gradient_norms": gradients, "trajectory": trajectory,
                  "weights": gate.weight_matrix().detach().cpu().tolist(), "fit_split": "train"}
        self.reports[self.task] = result
        return result

    def snapshot(self):
        return {"task": self.task, "steps": self.steps, "fit_split": "train",
                "bank": {k: v.detach().cpu().clone() for k, v in self.bank.state_dict().items()},
                "updates": dict(self.updates), "optimizers": copy.deepcopy(self.optimizers),
                "reports": copy.deepcopy(self.reports)}

    def load_snapshot(self, state):
        if self.task != -1 or state["fit_split"] != "train" or state["steps"] != self.steps:
            raise ValueError("Restore a fresh matching TRAIN fusion bank")
        for task in range(state["task"]+1):
            self.begin_task(task)
        self.bank.load_state_dict(state["bank"], strict=True)
        self.bank.seal()
        self.updates, self.optimizers, self.reports = (copy.deepcopy(state[k]) for k in ("updates", "optimizers", "reports"))


def full_loader_like(loader, view, task, smoke):
    source = loader.dataset.source
    indices, _ = runner.fit_calibration_indices(source, runner.task_indices(task), 0.)
    if view == "face":
        indices = runner.filter_face_training_indices(source, indices)
    data = runner.LabelView(source, indices, runner.task_indices(task))
    result = DataLoader(data, batch_size=loader.batch_size, shuffle=True,
                        num_workers=loader.num_workers, pin_memory=loader.pin_memory, drop_last=False)
    return FirstBatches(result) if smoke else result


def exposure_record(root, loader, view, task, condition, smoke=False):
    source = loader.dataset.source
    eligible, _ = runner.fit_calibration_indices(source, runner.task_indices(task), 0.)
    a, h = runner.fit_calibration_indices(source, runner.task_indices(task), .2)
    actual_a = list(loader.dataset.indices)
    returned = runner.filter_face_training_indices(source, eligible) if view == "face" else eligible
    group = lambda i: str(source.sample_ids[i]).rsplit("#person=", 1)[0]
    prior = set()
    for previous in range(task):
        prior.update(read(root/"exposure"/view/f"task{previous}.json")["returned_groups"])
    result = {"task": task, "view": view, "condition": condition, "eligible": len(eligible),
              "fit": len(actual_a), "holdout": len(h) if condition == "holdout_return" else 0,
              "returned": len(returned), "returned_groups": sorted({group(i) for i in returned}),
              "heldout_ids": [str(source.sample_ids[i]) for i in h] if condition == "holdout_return" else [],
              "initial_ids": [str(source.sample_ids[i]) for i in actual_a],
              "heldout_prior_seen_people": sum(group(i) in prior for i in h) if condition == "holdout_return" else 0,
              "heldout_prior_seen_groups": len({group(i) for i in h}&prior) if condition == "holdout_return" else 0,
              "heldout_whole_trajectory_unseen_claim": task == 0 and condition == "holdout_return",
              "initial_return_exposures": list(phase_sizes(smoke)),
              "smoke_capped_batches": smoke}
    if condition == "holdout_return" and ({group(i) for i in actual_a}&{group(i) for i in h}):
        raise AssertionError("Image group leaked across initial/heldout roles")
    path = root/"exposure"/view/f"task{task}.json"
    if path.exists() and read(path) != result:
        raise AssertionError("Resume exposure manifest changed")
    write(path, result)
    return result


@torch.no_grad()
def export_heldout(model, source, view, task):
    classes = list(runner.task_indices(task))
    _, indices = runner.fit_calibration_indices(source, classes, .2)
    data = runner.LabelView(source, indices, classes, include_sample_id=True)
    loader = DataLoader(data, batch_size=64, num_workers=2, prefetch_factor=1,
                        worker_init_fn=single_thread_worker,
                        generator=torch.Generator().manual_seed(77191+task))
    rows, targets, masks, ids = [], [], [], []
    for images, y, names in loader:
        images = move_model_inputs(images, torch.device("cuda"))
        with torch.autocast("cuda"):
            z = model.seen_logits(AsymmetricReceivers.receiver_inputs(images, view))[:, classes]
        rows.append(z.float().cpu().numpy()); targets.append(y.numpy().astype(np.float32))
        masks.append(images["evidence_face_reliable"].bool().cpu().numpy()); ids.extend(names)
    if len(set(ids)) != len(ids) or any(not s.startswith("train:") for s in ids):
        raise AssertionError("Invalid TRAIN observation")
    return dict(ids=np.asarray(ids, dtype=str), logits=np.concatenate(rows),
                targets=np.concatenate(targets), reliable=np.concatenate(masks))


def worker(root, view, smoke, condition):
    original = runner.train_task
    initial, returned = phase_sizes(smoke)
    source_cache = {}
    write(branch_paths(root)[view]/"return_data_provenance.json", {
        "initial_calibration_exclusion_only": condition == "holdout_return",
        "all_legal_train_return_epoch": initial+1,
        "fusion_fit_source": "observations/task*/VIEW.npz at the initial-phase barrier only",
        "legacy_calibration_scores": "exported after return; already seen by branch; diagnostic only, never used to fit fusion",
        "official_validation_and_test_updates": 0})

    def train_with_return(model, loader, *args, **kwargs):
        # runner passes validation_loader, device, task_id, epochs positionally.
        task = args[2]
        exposure_record(root, loader, view, task, condition, smoke)
        if smoke:
            kwargs["optimizer_updates_per_task"] = None
        model.r1_return_training = {"initial_epochs": initial, "return_epochs": returned,
            "lr_factor": .1, "loader": full_loader_like(loader, view, task, smoke)}
        original_save = model.save_training_epoch

        def publish(task, epoch, updates):
            if epoch == initial and condition == "holdout_return":
                wait_for(branch_paths(root)["person"]/"config.json", root)
                with isolated_observation(model):
                    if "train" not in source_cache:
                        source_cache["train"] = load_source(branch_paths(root)["person"], input_paths(), "train")
                    observation = export_heldout(model, source_cache["train"], view, task)
                    expected = read(root/"exposure"/view/f"task{task}.json")["heldout_ids"]
                    if list(observation["ids"]) != expected:
                        raise AssertionError("Heldout prediction provenance mismatch")
                    save_npz(root/"observations"/f"task{task}"/f"{view}.npz", **observation)
            write(root/"barriers"/f"task{task}"/f"epoch{epoch:02d}_{view}.json",
                  {"task": task, "epoch": epoch, "view": view, "updates": updates})
            wait_for(root/"barriers"/f"task{task}"/f"epoch{epoch:02d}_accepted.json", root)

        def observe(task, epoch, history, optimizer, scheduler, scaler, updates):
            original_save(task, epoch, history, optimizer, scheduler, scaler, updates)
            if history[-1]["skipped_optimizer_steps"]:
                raise RuntimeError("Skipped optimizer update: stop trajectory for diagnosis")
            publish(task, epoch, updates)

        model.save_training_epoch = observe
        try:
            resume = getattr(model, "r1_resume", None)
            if resume and not (root/"barriers"/f"task{task}"/f"epoch{resume['next_epoch']:02d}_accepted.json").exists():
                publish(task, resume["next_epoch"], resume["completed_task_updates"])
            return original(model, FirstBatches(loader) if smoke else loader, *args, **kwargs)
        finally:
            model.save_training_epoch = original_save
            del model.r1_return_training

    runner.train_task = train_with_return
    sys.argv = [sys.argv[0], *branch_flags(root, view, smoke, condition)]
    runner.main()


def aligned(root, task):
    rows = []
    for view in VIEWS:
        with np.load(root/"observations"/f"task{task}"/f"{view}.npz", allow_pickle=False) as data:
            rows.append({k: data[k] for k in data.files})
    first = rows[0]
    for other in rows[1:]:
        if any(not np.array_equal(first[k], other[k]) for k in ("ids", "targets", "reliable")):
            raise AssertionError("Cross-view heldout observations differ")
    return first["ids"], np.stack([r["logits"] for r in rows], axis=1), first["targets"], first["reliable"]


def complete_checkpoint(root, task, epoch, fusion, smoke, condition):
    branches = {}
    for view, path in branch_paths(root).items():
        value = torch.load(path/"epoch_recovery"/f"task{task}"/f"epoch{epoch:02d}.pth", map_location="cpu")
        if value["task"] != task or value["training"]["next_epoch"] != epoch:
            raise AssertionError("Branch epochs differ")
        branches[view] = value
    state = {"format": "single_trajectory_train_return_v1", "run_id": str(root),
             "condition": condition, "task": task, "epoch": epoch, "branches": branches,
             "fusion": fusion.snapshot(), "fit_split": "train", "phase_sizes": phase_sizes(smoke),
             "source_git": runner.git_metadata(Path.cwd()), "pretrained": str(input_paths().clip_checkpoint)}
    recovery = root/"recovery"/f"task{task}"/f"epoch{epoch:02d}.pth"
    if recovery.exists():
        existing = torch.load(recovery, map_location="cpu")
        assert_same(existing["branches"], branches, "resume/branches")
        assert_same(existing["fusion"], state["fusion"], "resume/fusion")
    else:
        atomic_save(state, recovery)
    write(root/"checkpoints/latest.json", {"path": str(recovery), "task": task, "epoch": epoch})
    if epoch == sum(phase_sizes(smoke)):
        final = root/"checkpoints"/f"task{task}.pth"
        if not final.exists():
            atomic_save(state, final)
    return {v: int(p["training"]["completed_task_updates"]) for v, p in branches.items()}


def check_budget(root, smoke):
    initial, returned = phase_sizes(smoke)
    result = {}
    for view, path in branch_paths(root).items():
        if read(path/"seed_summary.json")["status"] != "complete":
            raise AssertionError("Incomplete branch training")
        history = read(path/"training_history.json")
        result[view] = []
        for task in range(2 if smoke else 8):
            counts = read(root/"exposure"/view/f"task{task}.json")
            rows = history[str(task)]
            expected_a = initial*(2 if smoke else math.ceil(counts["fit"]/64))
            expected_b = returned*(2 if smoke else math.ceil(counts["returned"]/64))
            a = int(sum(r["optimizer_steps"] for r in rows[:initial]))
            b = int(sum(r["optimizer_steps"] for r in rows[initial:]))
            if (len(rows) != initial+returned or (a, b) != (expected_a, expected_b)
                    or any(r["skipped_optimizer_steps"] for r in rows)):
                raise AssertionError(f"Phase update budget mismatch: {view}/{task}")
            if (any(r.get("data_phase") != "initial_train" for r in rows[:initial])
                    or any(r.get("data_phase") != "return_full_train" for r in rows[initial:])
                    or not math.isclose(rows[initial]["learning_rate"], .1*rows[0]["learning_rate"], abs_tol=1e-12)):
                raise AssertionError(f"Phase scheduler/loader metadata mismatch: {view}/{task}")
            before = torch.load(path/"epoch_recovery"/f"task{task}"/f"epoch{initial:02d}.pth", map_location="cpu")
            after = torch.load(path/"epoch_recovery"/f"task{task}"/f"epoch{initial+1:02d}.pth", map_location="cpu")
            left, right = before["training"]["optimizer"]["state"], after["training"]["optimizer"]["state"]
            increments = [float(right[k]["step"])-float(state["step"]) for k, state in left.items()]
            if not increments or any(n != rows[initial]["optimizer_steps"] for n in increments):
                raise AssertionError(f"Adam step counters reset at return: {view}/{task}")
            result[view].append(dict(task=task, initial_updates=a, return_updates=b,
                                     initial_people=counts["fit"], returned_people=counts["returned"],
                                     adam_boundary_step_counters_continued=True))
    write(root/"training_budget_checks.json", result)


def evaluate(root, smoke, condition):
    """Both splits use immutable complete same-trajectory checkpoints."""
    torch.set_num_threads(2)
    runner.set_seed(1, True)
    check_budget(root, smoke)
    initial, returned = phase_sizes(smoke)
    paths, inputs = branch_paths(root), input_paths()
    methods = ("fixed", "learned") if condition == "holdout_return" else ("fixed",)
    models = {v: model_from_config(read(p/"config.json"), inputs.clip_checkpoint) for v, p in paths.items()}
    records = {s: make_records(methods) for s in ("val", "test")}
    diagnostics = {s: make_records(methods) for s in ("val", "test")}
    tasks = range(2 if smoke else 8)
    todo = [(t, initial+returned, "final") for t in tasks]
    todo += [(t, initial, "pre_return") for t in tasks if smoke or t in (0, 2, 7)]
    hashes = {str(root/"recovery"/f"task{t}"/f"epoch{e:02d}.pth"):
              sha(root/"recovery"/f"task{t}"/f"epoch{e:02d}.pth") for t, e, _ in todo}
    write(root/"locked_evaluation.json", {"whole_checkpoints": hashes, "methods": methods,
          "gradient_updates": False, "official_validation_and_test_fitting": False})
    progress = []
    for split in ("val", "test"):
        source = load_source(paths["person"], inputs, split)
        for task, epoch, phase in todo:
            state = torch.load(root/"recovery"/f"task{task}"/f"epoch{epoch:02d}.pth", map_location="cpu")
            observed = []
            for view, model in models.items():
                missing = model.load_state_dict(state["branches"][view]["model"], strict=False)
                if missing.unexpected_keys or any(not n.startswith("visual_encoder.") for n in missing.missing_keys):
                    raise ValueError("Whole checkpoint incomplete")
                model.restore_task(task)
                observed.append(export_live(model, source, view, task, split, smoke))
            first = observed[0]
            for row in observed[1:]:
                if any(not np.array_equal(first[k], row[k]) for k in ("ids", "targets", "reliable")):
                    raise AssertionError("Evaluation views differ")
            z = np.stack([r["logits"] for r in observed], axis=1)
            with torch.no_grad():
                fused = {"fixed": blend(z, first["reliable"], FIXED)}
                if condition == "holdout_return":
                    fusion = ReturnFusion(runner.TASK_SIZES, "cuda", 2 if smoke else 100)
                    fusion.load_snapshot(state["fusion"])
                    fused["learned"] = fusion.bank(torch.as_tensor(z, device="cuda"),
                        torch.as_tensor(first["reliable"], device="cuda"), task).cpu().numpy()
            destination = records if phase == "final" else diagnostics
            record(destination[split], task, fused, first["targets"], first["reliable"], list(runner.task_indices(task)))
            save_npz(root/"predictions"/phase/split/f"task{task}.npz", **{**first, "branch_logits": z, **fused})
            progress.append({"split": split, "task": task, "epoch": epoch, "phase": phase})
            write(root/"evaluation_progress.json", progress)
            print("WHOLE_MODEL_FORWARD", condition, split, task, epoch, len(first["ids"]), flush=True)
    if any(sha(Path(path)) != digest for path, digest in hashes.items()):
        raise AssertionError("Evaluation modified checkpoint")
    result = {s: summarize(r, smoke) for s, r in records.items()}
    write(root/"result.json", result)
    # Sparse task0/2/7 diagnostics must not be advertised as an eight-stage mean.
    write(root/"pre_return_diagnostic.json", {s: summarize(r, True) for s, r in diagnostics.items()})
    write(root/"completion.json", {"status": "complete", "condition": condition, "smoke": smoke,
          "training_sources": 1, "all_legal_train_eventually_used": not smoke,
          "validation_parameter_updates": 0, "whole_checkpoint_forward": True, "results": result})


def run_one(root, logs, smoke, condition, resume=False):
    if root.exists() and not resume:
        raise FileExistsError("Fresh output required; use explicit --resume for this same trajectory")
    if (root/"completion.json").exists():
        return
    root.mkdir(parents=True, exist_ok=resume)
    logs.mkdir(parents=True, exist_ok=True)
    initial, returned = phase_sizes(smoke)
    inputs = input_paths()
    protocol = {"run_id": str(root), "condition": condition, "seed": 1, "smoke": smoke,
                "dataset": str(inputs.data_root), "pretrained": str(inputs.clip_checkpoint),
                "pretrained_sha256": sha(inputs.clip_checkpoint), "git": runner.git_metadata(Path.cwd()),
                "initial_epochs": initial, "return_epochs": returned, "return_lr_factor": .1,
                "train_batch": 64, "fusion_lr": .02, "fusion_steps": 2 if smoke else 100,
                "split_salt": "emotic-reliability-calibration-v1",
                "train_group_fraction": .2 if condition == "holdout_return" else 0.,
                "full_parameter_trajectory": True, "fusion_updates_after_return": 0,
                "calibration_exclusion": "initial phase only; all legal TRAIN returned before final checkpoint",
                "fusion_fit_source": "observations at initial phase barrier; never legacy post-return calibration_scores",
                "branch_commands": {v: branch_flags(root, v, smoke, condition) for v in VIEWS},
                "evaluation": "both Val/Test; final and task0/2/7 pre-return diagnostic; no fitting"}
    if (root/"protocol.json").exists() and read(root/"protocol.json") != protocol:
        raise AssertionError("Resume protocol changed")
    write(root/"protocol.json", protocol)
    if (root/"training_complete.json").exists():
        evaluate(root, smoke, condition)
        return
    if (root/"abort.json").exists():
        (root/"abort.json").rename(root/f"abort_previous_{time.time_ns()}.json")
    jobs, handles = {}, []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="72", MKL_NUM_THREADS="72")
    try:
        for view in VIEWS:
            handle = (logs/f"{view}.log").open("a" if resume else "w")
            handles.append(handle)
            command = [sys.executable, "-u", "-m", "scripts.emotic.run_r1_fusion_return_train", "worker",
                       "--root", str(root), "--view", view, "--condition", condition]
            if smoke:
                command.append("--smoke")
            jobs[view] = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        write(root/"workers.json", {v: p.pid for v, p in jobs.items()})
        torch.set_num_threads(2)
        runner.set_seed(1, True)
        fusion = ReturnFusion(runner.TASK_SIZES, "cuda", 2 if smoke else 100)
        for task in range(2 if smoke else 8):
            fusion.begin_task(task)
            sealed = None
            for epoch in range(1, initial+returned+1):
                accepted = root/"barriers"/f"task{task}"/f"epoch{epoch:02d}_accepted.json"
                if accepted.exists():
                    state = torch.load(root/"recovery"/f"task{task}"/f"epoch{epoch:02d}.pth", map_location="cpu")
                    fusion = ReturnFusion(runner.TASK_SIZES, "cuda", 2 if smoke else 100)
                    fusion.load_snapshot(state["fusion"])
                    if epoch == initial:
                        sealed = copy.deepcopy(fusion.snapshot()["bank"])
                    continue
                ready = [root/"barriers"/f"task{task}"/f"epoch{epoch:02d}_{v}.json" for v in VIEWS]
                start = time.monotonic()
                while not all(p.exists() for p in ready):
                    failed = {v: p.poll() for v, p in jobs.items() if p.poll() is not None}
                    if failed or time.monotonic()-start > 5400:
                        raise RuntimeError(f"Branch barrier failed {task}/{epoch}: {failed}")
                    time.sleep(2)
                if epoch == initial:
                    if condition == "holdout_return":
                        fusion.fit(*aligned(root, task))
                    sealed = copy.deepcopy(fusion.snapshot()["bank"])
                if epoch > initial:
                    assert_same(sealed, fusion.snapshot()["bank"], "fusion_sealed_during_return")
                updates = complete_checkpoint(root, task, epoch, fusion, smoke, condition)
                write(root/"fusion_history.json", fusion.reports)
                write(root/"progress.json", {"task": task, "epoch": epoch, "condition": condition,
                      "phase": "initial" if epoch <= initial else "return", "updates": updates,
                      "fusion_updates": fusion.updates, "status": "training"})
                write(accepted, {"complete_checkpoint_saved": True})
                print("ACCEPTED_EPOCH", condition, task, epoch, updates, fusion.updates, flush=True)
        for view, process in jobs.items():
            if process.wait(timeout=5400):
                raise RuntimeError(f"Branch exited unsuccessfully: {view}")
        write(root/"training_complete.json", {"status": "complete", "condition": condition})
        evaluate(root, smoke, condition)
    except BaseException as error:
        write(root/"abort.json", {"error": repr(error)})
        for process in jobs.values():
            if process.poll() is None:
                process.terminate()
        for process in jobs.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker", "evaluate"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--logs", type=Path, default=LOGS)
    parser.add_argument("--condition", choices=(*CONDITIONS, "both"), default="both")
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "worker":
        if args.condition == "both" or args.view is None:
            raise ValueError("Worker needs one condition and view")
        worker(args.root, args.view, args.smoke, args.condition)
        return
    if args.action == "evaluate":
        if args.condition == "both":
            raise ValueError("Evaluation needs one condition")
        evaluate(args.root, args.smoke, args.condition)
        return
    for condition in CONDITIONS if args.condition == "both" else (args.condition,):
        run_one(args.root/condition, args.logs/condition, args.smoke, condition, args.resume)
        torch.cuda.empty_cache()
    write(args.root/"all_conditions_complete.json", {"status": "complete", "conditions":
          list(CONDITIONS) if args.condition == "both" else [args.condition]})


if __name__ == "__main__":
    main()
