"""Run the locked v1 groups, then evaluate fixed checkpoints on val/test in parallel."""
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

from .evaluate_static_moe import METHODS, audit_pairing
from .runner import git_metadata


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def wait_memory(gpu, minimum, timeout):
    started = time.monotonic()
    while True:
        result = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
            "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
        if int(result.strip()) >= minimum:
            return
        if time.monotonic() - started > timeout:
            raise RuntimeError("Timed out waiting for free GPU memory; other processes were not modified")
        time.sleep(30)


def launch(batch_id, gpu="0", smoke=False, parallel=2):
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    metadata = git_metadata(root)
    if metadata["dirty"] or not batch_id or Path(batch_id).name != batch_id:
        raise ValueError("Clean worktree and a simple fresh batch ID required")
    if parallel not in (1, 2):
        raise ValueError("v1 supports at most two concurrent groups")
    output = root / "output/emotic_static_moe" / batch_id
    logs = root / "logs/emotic_static_moe" / batch_id
    output.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    state_path = output / "pipeline_state.json"
    state = dict(status="waiting_for_memory", smoke=smoke, gpu=gpu, seed=0, methods=list(METHODS),
                 epochs=30, tasks=8, smoke_updates=2 if smoke else 0, parallel=parallel,
                 git=metadata, started_at=datetime.now(timezone.utc).isoformat(), phases={})
    write(output / "manifest.json", state)
    write(state_path, state)
    data = os.environ.get("DATA_ROOT", "../multi-lane-main/datasets/EMOTIC")
    clip = os.environ.get("CLIP_CHECKPOINT", "../CODE_DDP-benchmark/pretrained/clip/ViT-B-16.pt")
    face = os.environ.get("FACE_MANIFEST_ROOT", "../emotic_benchmark_runs/multi_lane_face_test_manifest_v0.1/face_manifest_train_val_test_v1_20260909")
    try:
        wait_memory(gpu, 14000 if parallel == 2 else 8000, 43200)
        state["status"] = "training"
        write(state_path, state)

        def train(method):
            environment = dict(os.environ, METHOD=method, RUN_ROOT=str(output / method),
                               LOG_PATH=str(logs / (method + ".log")), GPU=gpu, PYTHON=sys.executable,
                               DATA_ROOT=data, CLIP_CHECKPOINT=clip, FACE_MANIFEST_ROOT=face)
            if smoke:
                environment["SMOKE_UPDATES"] = "2"
            else:
                environment.pop("SMOKE_UPDATES", None)
            print("STATIC_MOE_TRAIN_BEGIN " + method, flush=True)
            result = subprocess.run(["bash", "scripts/emotic/run_static_moe_v1.sh"], env=environment,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            print(f"STATIC_MOE_TRAIN_END {method} exit={result.returncode}", flush=True)
            write(output / (method + ".exit.json"), dict(exit_code=result.returncode))
            return method, result.returncode

        training_order = METHODS if smoke else ("BASELINE", "MOE_OVERLAP", "SHARED", "MOE_UNIFORM")
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            exits = dict(executor.map(train, training_order))
        state["phases"]["training"] = exits
        write(state_path, state)
        if any(exits.values()):
            raise RuntimeError(f"Training failed: {exits}; logs retained")
        audit_pairing(output, 2 if smoke else 0)
        state["phases"]["pairing"] = "passed"

        if not smoke:
            from .static_moe_parallel_evaluation import run_parallel_evaluations
            run_parallel_evaluations(output, logs, data, clip, face, gpu, state, state_path)
        splits = ("val",) if smoke else ()
        for split in splits:
            state["status"] = "preflight_evaluation" if smoke else "evaluating_" + split
            write(state_path, state)
            destination = output / ("preflight_val" if smoke else "full_" + split)
            command = [sys.executable, "-u", "-m", "multi_lane.track_a.evaluate_static_moe",
                       "--batch", str(output), "--output", str(destination), "--split", split,
                       "--data-root", data, "--clip-checkpoint", clip, "--face-manifest-root", face]
            if smoke:
                command += ["--preflight", "--smoke-updates", "2"]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1")
            with (logs / (split + "_evaluation.log")).open("w") as handle:
                result = subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            state["phases"][split] = result.returncode
            write(state_path, state)
            if result.returncode:
                raise RuntimeError(f"{split} evaluation failed; checkpoints/logs retained")
        state["status"] = "smoke_passed" if smoke else "complete"
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        write(state_path, state)
        print("STATIC_MOE_PIPELINE_COMPLETE " + state["status"], flush=True)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}",
                     finished_at=datetime.now(timezone.utc).isoformat())
        write(state_path, state)
        raise
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--parallel", type=int, default=2)
    launch(**vars(parser.parse_args()))

