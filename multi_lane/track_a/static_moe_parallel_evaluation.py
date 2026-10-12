"""Parallel fixed-checkpoint evaluation, including a handoff from an active v1 run."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .compare_joint26 import require
from .evaluate_static_moe import METHODS, audit_pairing, read, source_hashes
from .export_compact_test_scores import _sha256
from .runner import git_metadata


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def create_source_lock(batch, metadata, smoke_updates=0):
    lock = dict(evaluation_mode="parallel_fixed_checkpoints", batch=str(batch.resolve()),
                evaluator_git=metadata, source_sha256={m: source_hashes(batch / m) for m in METHODS},
                checkpoint_selection="fixed_smoke_updates" if smoke_updates else "fixed_final_epoch30",
                smoke_updates=smoke_updates, optimizer_updates=0,
                created_at=datetime.now(timezone.utc).isoformat())
    path = batch / "parallel_evaluation_source_lock.json"
    require(not path.exists(), "Parallel source lock already exists")
    write(path, lock)
    return path, lock


def validate_parallel_results(batch, lock_path, lock):
    digest = _sha256(lock_path)
    for split, count in (("val", 2397), ("test", 5368)):
        result = read(batch / ("full_" + split) / "comparison.json")
        require(result["status"] == "complete" and result["split"] == split
                and result["samples"] == count and set(result["results"]) == set(METHODS)
                and not result["training_performed"] and result["optimizer_updates"] == 0
                and result["evaluation_mode"] == "parallel_fixed_checkpoints"
                and result["evaluation_source_lock_sha256"] == digest
                and result["evaluator_git"] == lock["evaluator_git"], "Incomplete parallel evaluation")
        for method in METHODS:
            require(result["results"][method]["source_sha256"] == lock["source_sha256"][method]
                    == source_hashes(batch / method), "Val/test did not use identical fixed sources")


def run_parallel_evaluations(batch, logs, data, clip, face, gpu, state, state_path):
    metadata = git_metadata(Path(__file__).resolve().parents[2])
    require(not metadata["dirty"], "Dirty parallel evaluator")
    audit_pairing(batch)
    require(all(not (batch / ("full_" + s)).exists() for s in ("val", "test")),
            "Parallel evaluation outputs already exist")
    lock_path, lock = create_source_lock(batch, metadata)
    state.update(status="waiting_for_evaluation_memory", evaluation_parallel=True, evaluator_git=metadata,
                 evaluation_source_lock=str(lock_path))
    write(state_path, state)
    from .static_moe_pipeline import wait_memory
    wait_memory(gpu, 14000, 43200)
    state["status"] = "evaluating_val_test_parallel"
    write(state_path, state)

    def evaluate(split):
        command = [sys.executable, "-u", "-m", "multi_lane.track_a.evaluate_static_moe",
                   "--batch", str(batch), "--output", str(batch / ("full_" + split)),
                   "--split", split, "--data-root", str(data), "--clip-checkpoint", str(clip),
                   "--face-manifest-root", str(face), "--parallel-splits", "--source-lock", str(lock_path)]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1")
        print("STATIC_MOE_PARALLEL_EVALUATION_BEGIN " + split, flush=True)
        with (logs / (split + "_evaluation.log")).open("w") as handle:
            result = subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        print(f"STATIC_MOE_PARALLEL_EVALUATION_END {split} exit={result.returncode}", flush=True)
        return split, result.returncode

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(evaluate, split) for split in ("val", "test")]
        for future in as_completed(futures):
            split, code = future.result()
            state["phases"][split] = code
            write(state_path, state)
    require(all(state["phases"][s] == 0 for s in ("val", "test")), "Parallel evaluation failed; logs retained")
    validate_parallel_results(batch, lock_path, lock)
    state["parallel_sources_verified"] = True
    write(state_path, state)


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19], fields[0]
    except FileNotFoundError:
        return None


def prepare_handoff(batch, controller_pid, metadata):
    original = read(batch / "pipeline_state.json")
    require(original["status"] == "training" and not original["smoke"], "Handoff requires active formal training")
    identity = process_identity(controller_pid)
    require(identity is not None and identity[1] != "Z", "Legacy controller is not running")
    command = Path(f"/proc/{controller_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    require("multi_lane.track_a.static_moe_pipeline" in command and batch.name in command
            and Path(os.readlink(f"/proc/{controller_pid}/cwd")) == batch.parents[2], "Wrong training controller")
    require(not (batch / "full_test").exists() and not (batch / "parallel_evaluation_state.json").exists(),
            "An evaluation or handoff already exists")
    reservation = batch / "full_val"
    reservation.mkdir(exist_ok=False)
    marker = dict(purpose="handoff_to_parallel_evaluation", controller_pid=controller_pid,
                  controller_start_ticks=identity[0], training_git=original["git"], evaluator_git=metadata,
                  training_interrupted=False, optimizer_updates=0,
                  explanation="Reserve the legacy evaluator destination so it exits before inference; preserve training and adopt its completed sources.")
    write(reservation / "PARALLEL_EVALUATION_RESERVED.json", marker)
    state = dict(status="waiting_for_training", batch=str(batch), evaluation_parallel=True,
                 training_git=original["git"], evaluator_git=metadata, phases={}, handoff=marker,
                 started_at=datetime.now(timezone.utc).isoformat())
    write(batch / "parallel_evaluation_state.json", state)
    return state


def wait_for_legacy_training(batch, state, timeout=86400):
    started = time.monotonic()
    marker = state["handoff"]
    while True:
        exits = {m: read(batch / (m + ".exit.json"))["exit_code"] for m in METHODS
                 if (batch / (m + ".exit.json")).exists()}
        require(not any(exits.values()), "A training group failed; parallel evaluation was not started")
        identity = process_identity(marker["controller_pid"])
        alive = identity is not None and identity[0] == marker["controller_start_ticks"] and identity[1] != "Z"
        if not alive:
            require(set(exits) == set(METHODS), "Training controller exited before completing all groups")
            legacy = read(batch / "pipeline_state.json")
            require(legacy["status"] == "failed" and legacy["phases"].get("val") == 1
                    and legacy["phases"]["training"] == exits, "Unexpected legacy exit; results retained")
            reservation = batch / "full_val"
            require(list(reservation.iterdir()) == [reservation / "PARALLEL_EVALUATION_RESERVED.json"],
                    "Legacy evaluation wrote unexpected outputs")
            logs = batch.parents[2] / "logs/emotic_static_moe" / batch.name
            legacy_log = logs / "val_evaluation.log"
            require("Dirty evaluator or existing output" in legacy_log.read_text(), "Unexpected legacy evaluation failure")
            write(batch / "pipeline_legacy_handoff.json", legacy)
            legacy_log.rename(logs / "legacy_val_reservation_exit.log")
            reservation.rename(batch / "legacy_evaluation_reservation")
            state["phases"].update(training=exits, legacy_evaluation="reserved_destination_handoff")
            state["legacy_training_completed"] = True
            return
        require(time.monotonic() - started < timeout, "Timed out waiting for training; no process was stopped")
        time.sleep(30)


def handoff(batch, controller_pid, gpu="0", wait_timeout=86400):
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    batch = batch.resolve()
    metadata = git_metadata(root)
    require(not metadata["dirty"], "Clean evaluator worktree required")
    state = prepare_handoff(batch, controller_pid, metadata)
    state_path = batch / "parallel_evaluation_state.json"
    print("STATIC_MOE_PARALLEL_HANDOFF_ARMED waiting_for_training", flush=True)
    try:
        wait_for_legacy_training(batch, state, wait_timeout)
        audit_pairing(batch)
        c = read(batch / "BASELINE/config.json")
        # Resolve recorded relative input paths from the original training checkout.
        training_root = batch.parents[2]
        resolve_input = lambda value: (training_root / value).resolve()
        clip = resolve_input(c["clip_checkpoint"])
        face = resolve_input(c["face_manifest"]["root"])
        data = resolve_input(c["data_root"])
        write(batch / "pipeline_state.json", state)
        run_parallel_evaluations(batch, batch.parents[2] / "logs/emotic_static_moe" / batch.name,
                                 data, clip, face, gpu, state, batch / "pipeline_state.json")
        state.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        write(batch / "pipeline_state.json", state)
        print("STATIC_MOE_PARALLEL_HANDOFF_COMPLETE", flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}",
                     finished_at=datetime.now(timezone.utc).isoformat())
        if state.get("legacy_training_completed"):
            write(batch / "pipeline_state.json", state)
        raise
    finally:
        write(state_path, state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--wait-timeout", type=int, default=86400)
    handoff(**vars(parser.parse_args()))
