"""Run checkpoint edge masks, then train two Full-only evidence receivers."""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from multi_lane.track_a.r1_crossvit_experiment import flags, read, sha, write
from multi_lane.track_a.runner import git_metadata
from scripts.emotic.r1_frozen9_edge_forward import ROOT, branch_paths
from scripts.emotic.run_r1_frozen_cross_view import check_training, input_paths, preflight, score_suffix
from scripts.emotic.run_r1_patch_receiver_depth import REFERENCE, dispatch


@dataclass(frozen=True)
class Job:
    view: str
    gpu: int

    @property
    def name(self):
        return "full_only_" + self.view

    def path(self, root):
        return root / "full_only/attention" / self.view


JOBS = (Job("person", 0), Job("face", 1))


def command_for(job, root, smoke):
    command = [
        sys.executable, "-u", "-m", "scripts.emotic.run_r1_frozen9_edges",
        "--root", str(root), "--worker", job.view,
    ]
    if smoke:
        command.append("--smoke")
    return command


def worker(job, root, smoke):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(job.gpu):
        raise ValueError("Worker must use its scheduled GPU")
    inputs = input_paths()
    path = job.path(root)
    command = [
        sys.executable, "-u", "-m", "multi_lane.track_a.runner",
        *flags(
            SimpleNamespace(output=root / "full_only", **vars(inputs)),
            "attention", job.view, smoke, seed=1,
        ),
        "--r1-frozen-mode", "patch", "--r1-exchange-layer", "8",
        "--r1-frozen-evidence", "full", "--r1-fixed-loss-scale", "4096",
        "--r1-fusion-readout", "logit",
    ]
    if smoke:
        command += ["--r1-smoke-eval-samples", "64"]
    print("TRAIN", " ".join(command), flush=True)
    subprocess.run(command, env=dict(
        os.environ, OMP_NUM_THREADS="72", MKL_NUM_THREADS="72",
        OPENBLAS_NUM_THREADS="1",
    ), check=True)
    counts = check_training(path / "train", REFERENCE, job.view, 2 if smoke else 8, smoke)
    if smoke:
        preflight(path / "train", inputs, path / "preflight.json")
    else:
        subprocess.run([
            sys.executable, "-u", "-m",
            "multi_lane.track_a.r1_crossvit_experiment", "val",
            "--training-root", str(path / "train"), "--output", str(path / "val"),
            "--data-root", str(inputs.data_root),
            "--clip-checkpoint", str(inputs.clip_checkpoint),
            "--face-manifest-root", str(inputs.face_manifest_root),
        ], check=True)
    write(path / "completion.json", {
        "status": "complete", "seed": 1, "view": job.view, "training": counts,
    })


def evaluate(root, logs, phase, smoke=False):
    name = phase + ("_smoke" if smoke else "_formal")
    command = [
        sys.executable, "-u", "-m", "scripts.emotic.r1_frozen9_edge_forward",
        "--root", str(root), "--phase", phase,
    ]
    if smoke:
        command.append("--smoke")
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
    )
    with (logs / (name + ".log")).open("x") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def run(root):
    git = git_metadata(Path.cwd())
    if git["dirty"]:
        raise ValueError("A clean experiment worktree is required")
    if shutil.disk_usage(".").free < 15 * 1024**3:
        raise RuntimeError("Insufficient disk for two recoverable receiver paths")
    sources = []
    for train in branch_paths(root, "readonly").values():
        sources.append(train / "config.json")
        for task in range(8):
            sources.append(train / "compact_checkpoints" / f"task{task}.pth")
            sources.extend(train.parent / score_suffix(split, task) for split in ("val", "test"))
    inputs = input_paths()
    sources.extend(
        inputs.face_manifest_root / "manifests" / f"{split}.jsonl"
        for split in ("train", "val", "test")
    )
    hashes = {str(path): sha(path) for path in sources}
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Experiment output already contains artifacts: {root}")
    root.mkdir(parents=True, exist_ok=True)
    logs = Path("logs/r1_frozen9_edges") / root.name
    logs.mkdir(parents=True, exist_ok=True)
    for phase in ("smoke", "formal"):
        (logs / phase).mkdir()
    write(root / "protocol.json", {
        "git": git, "seed": 1, "layer": 9, "tasks": 8, "epochs": 30, "batch": 64,
        "main_lr": .0125, "new_lr": .0004, "optimizer": "Adam",
        "scheduler": "cosine", "weight_decay": 0, "amp_scale": 4096, "cap": .05,
        "evidence_control": "mask_nonFull_sources_before_softmax",
        "training_paths": 2, "reuse_original_two_source_paths": True,
        "readout": "fixed_R1_weighted_logits_then_sigmoid",
        "source_hashes": hashes, "jobs": [vars(job) for job in JOBS],
    })
    write(root / "stage.json", {"stage": "readonly_smoke"})
    evaluate(root, logs, "readonly", smoke=True)
    write(root / "stage.json", {"stage": "readonly_full_evaluation"})
    evaluate(root, logs, "readonly")
    write(root / "stage.json", {"stage": "training_preflight"})
    smoke = dispatch(
        JOBS, root / "smoke", logs / "smoke", REFERENCE,
        {job.name: 6500 for job in JOBS}, True,
        reserve_peak=True, worker_command=command_for,
    )
    if set(smoke.values()) != {"complete"}:
        write(root / "completion.json", {"status": "preflight_failed", "smoke": smoke})
        raise AssertionError("Both Full-only preflights must pass")
    write(root / "smoke_queue.json", read(root / "smoke/queue.json"))
    budgets = {
        job.name: int(read(job.path(root / "smoke") / "preflight.json")[
            "smoke_train_peak_reserved_mib"
        ]) + 1024 for job in JOBS
    }
    write(root / "preflight_gate.json", {"states": smoke, "formal_budgets_mib": budgets})
    write(root / "stage.json", {"stage": "training_and_validation"})
    formal = dispatch(
        JOBS, root, logs / "formal", REFERENCE, budgets,
        False, reserve_peak=True, worker_command=command_for,
    )
    if set(formal.values()) != {"complete"}:
        write(root / "completion.json", {"status": "training_partial", "formal": formal})
        raise AssertionError("Both Full-only receiver paths are required")
    write(root / "stage.json", {"stage": "trained_four_group_forward"})
    evaluate(root, logs, "trained")
    if any(sha(Path(path)) != value for path, value in hashes.items()):
        raise AssertionError("Original source artifacts changed")
    write(root / "four_group_results.json", {
        "readonly": read(root / "evaluation/readonly/result.json"),
        "trained": read(root / "evaluation/trained/result.json"),
    })
    write(root / "stage.json", {"stage": "complete"})
    write(root / "completion.json", {
        "status": "complete", "seed": 1, "smoke": smoke, "formal": formal,
        "source_hashes_verified": True,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--worker", choices=("person", "face"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker is None:
        run(args.root)
    else:
        worker(next(job for job in JOBS if job.view == args.worker), args.root, args.smoke)
