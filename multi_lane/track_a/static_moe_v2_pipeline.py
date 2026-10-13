"""Wait for v2's locked training checkpoint set, then parallel-evaluate val/test."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .evaluate_static_moe_v2 import METHODS, source_hashes
from .evaluate_static_moe import audit_source
from .runner import git_metadata


def write(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def run(batch: Path, logs: Path, data_root: Path, clip_checkpoint: Path,
        face_manifest_root: Path, poll_seconds: int = 120,
        max_wait_hours: int = 36, gpu_val: str = "3", gpu_test: str = "4") -> dict:
    root = Path(__file__).resolve().parents[2]
    meta = git_metadata(root)
    if meta["dirty"]:
        raise RuntimeError("Evaluator worktree must be clean")
    batch, logs = batch.resolve(), logs.resolve()
    state_path = batch / "parallel_evaluation_state.json"
    if state_path.exists():
        raise RuntimeError(f"Evaluation controller state already exists: {state_path}")
    state = {"status": "waiting_for_training", "batch": str(batch),
             "evaluation_git": meta, "started_at": datetime.now(timezone.utc).isoformat(),
             "split_gpus": {"val": gpu_val, "test": gpu_test}}
    write(state_path, state)
    started = time.monotonic()
    while True:
        if all((batch / m / "seed_summary.json").is_file() for m in METHODS):
            break
        if time.monotonic() - started > max_wait_hours * 3600:
            state.update(status="failed", error="Timed out waiting for all three training summaries")
            write(state_path, state)
            raise TimeoutError(state["error"])
        time.sleep(poll_seconds)
    configs, audits = {}, {}
    for method in METHODS:
        configs[method], audits[method] = audit_source(batch / method)
        if configs[method]["adapter_residual_bound_ratio"] != 0.1:
            raise RuntimeError(f"{method} has an unexpected residual bound")
        if configs[method]["reporting_split"] != "val" or configs[method]["also_report_test"]:
            raise RuntimeError("Training source was configured to load test")
    hashes = {m: source_hashes(batch, m) for m in METHODS}
    lock = {"evaluation_mode": "parallel_fixed_checkpoints", "split_pair": ["val", "test"],
            "batch": str(batch), "evaluation_git": meta, "source_sha256": hashes}
    lock_path = batch / "parallel_evaluation_source_lock.json"
    write(lock_path, lock)
    lock_hash = __import__("hashlib").sha256(lock_path.read_bytes()).hexdigest()
    state.update(status="evaluating_parallel", source_lock_sha256=lock_hash,
                 source_sha256=hashes, training_audits={m: "passed" for m in METHODS})
    write(state_path, state)

    def evaluate(split: str, gpu: str) -> tuple[str, int]:
        destination = batch / ("full_val" if split == "val" else "full_test")
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite {destination}")
        command = [sys.executable, "-u", "-m", "multi_lane.track_a.evaluate_static_moe_v2",
                   "--batch", str(batch), "--output", str(destination), "--split", split,
                   "--data-root", str(data_root), "--clip-checkpoint", str(clip_checkpoint),
                   "--face-manifest-root", str(face_manifest_root), "--source-lock", str(lock_path)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1")
        with (logs / f"{split}_evaluation.log").open("w") as handle:
            result = subprocess.run(command, cwd=root, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT)
        return split, result.returncode

    state["evaluation_started_at"] = datetime.now(timezone.utc).isoformat()
    write(state_path, state)
    with ThreadPoolExecutor(max_workers=2) as pool:
        exits = dict(pool.map(lambda pair: evaluate(*pair), (("val", gpu_val), ("test", gpu_test))))
    if any(exits.values()):
        state.update(status="failed", split_exit_codes=exits)
        write(state_path, state)
        raise RuntimeError(f"Split evaluation failed: {exits}")
    if {m: source_hashes(batch, m) for m in METHODS} != hashes:
        state.update(status="failed", error="A source checkpoint/config changed during evaluation")
        write(state_path, state)
        raise RuntimeError(state["error"])
    state.update(status="complete", split_exit_codes=exits,
                 finished_at=datetime.now(timezone.utc).isoformat())
    write(state_path, state)
    return state


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("batch", "logs", "data-root", "clip-checkpoint", "face-manifest-root"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--poll-seconds", type=int, default=120)
    p.add_argument("--max-wait-hours", type=int, default=36)
    p.add_argument("--gpu-val", default="3")
    p.add_argument("--gpu-test", default="4")
    run(**vars(p.parse_args()))
