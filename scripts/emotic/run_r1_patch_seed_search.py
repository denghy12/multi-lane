"""Compare single-block Frozen9 and Frozen10 patch receivers on four seeds."""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from types import SimpleNamespace

import torch

from multi_lane.track_a.r1_crossvit_experiment import flags, read, source_runs, write, sha
from multi_lane.track_a.runner import git_metadata
from scripts.emotic.run_r1_frozen_cross_view import check_training, input_paths, preflight
from scripts.emotic.run_r1_frozen_late_stable import summarize_family
from scripts.emotic.run_r1_frozen9_seed_sweep import native_preflight
from scripts.emotic.run_r1_patch_receiver_depth import REFERENCE, dispatch


ROOT = Path("output/r1_patch_seed_search/20261107")
SEEDS = (3047, 1024, 42, 123)
LAYERS = (9, 10)
MIN_FREE_DISK_GIB = 100


@dataclass(frozen=True)
class Job:
    seed: int
    layer: int
    view: str
    gpu: int

    @property
    def name(self):
        if self.view == "full":
            return f"seed{self.seed}_full"
        return f"seed{self.seed}_layer{self.layer}_{self.view}"

    @property
    def kind(self):
        return "native" if self.view == "full" else "attention"

    def arm_root(self, root):
        seed_root = root / f"seed{self.seed}"
        return seed_root if self.view == "full" else seed_root / f"layer{self.layer}"

    def path(self, root):
        return self.arm_root(root) / self.kind / self.view


def make_jobs():
    jobs = []
    for index, seed in enumerate(SEEDS):
        gpu = index % 2
        jobs.append(Job(seed, 0, "full", gpu))
        for layer in LAYERS:
            person_gpu = (gpu + layer - 9) % 2
            jobs.append(Job(seed, layer, "person", person_gpu))
            jobs.append(Job(seed, layer, "face", 1 - person_gpu))
    return tuple(jobs)


JOBS = make_jobs()


def command_for(job, root, smoke):
    command = [
        sys.executable, "-u", "-m", "scripts.emotic.run_r1_patch_seed_search",
        "--root", str(root), "--worker", job.name,
    ]
    if smoke:
        command.append("--smoke")
    return command


def worker(job, root, smoke):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(job.gpu):
        raise ValueError("Worker must use its scheduled GPU")
    inputs = input_paths()
    path = job.path(root)
    train = path / "train"
    command = [
        sys.executable, "-u", "-m", "multi_lane.track_a.runner",
        *flags(
            SimpleNamespace(output=job.arm_root(root), **vars(inputs)),
            job.kind, job.view, smoke, seed=job.seed,
        ),
        "--r1-fusion-readout", "logit",
    ]
    if job.view != "full":
        command += [
            "--r1-frozen-mode", "patch",
            "--r1-exchange-layer", str(job.layer - 1),
            "--r1-fixed-loss-scale", "4096",
        ]
    if smoke:
        command += ["--r1-smoke-eval-samples", "64"]
    # Preserve the archived initialization and training CPU random stream.
    training_env = dict(
        os.environ, OMP_NUM_THREADS="72", MKL_NUM_THREADS="72",
        OPENBLAS_NUM_THREADS="1",
    )
    print("TRAIN", " ".join(command), flush=True)
    subprocess.run(command, env=training_env, check=True)
    counts = check_training(train, REFERENCE, job.view, 2 if smoke else 8, smoke)
    if smoke:
        probe = native_preflight if job.view == "full" else preflight
        probe(train, inputs, path / "preflight.json")
    else:
        # Evaluation uses two CPU threads and bounded worker prefetch.
        subprocess.run([
            sys.executable, "-u", "-m",
            "multi_lane.track_a.r1_crossvit_experiment", "val",
            "--training-root", str(train), "--output", str(path / "val"),
            "--data-root", str(inputs.data_root),
            "--clip-checkpoint", str(inputs.clip_checkpoint),
            "--face-manifest-root", str(inputs.face_manifest_root),
        ], check=True)
    write(path / "completion.json", {
        "status": "complete", "seed": job.seed, "layer": job.layer,
        "view": job.view, "gpu": job.gpu, "training": counts,
    })


def result_tables(root, states, manifest):
    results = {}
    for seed in SEEDS:
        for layer in LAYERS:
            selected = [
                job for job in JOBS
                if job.seed == seed and job.layer in (0, layer)
            ]
            if not all(states.get(job.name) == "complete" for job in selected):
                continue
            seed_root = root / f"seed{seed}"
            destination = seed_root / f"layer{layer}" / "fusion.json"
            paths = {
                ("patch", job.view): job.path(root)
                for job in selected if job.view != "full"
            }
            summarize_family(
                root, seed_root, manifest, paths, destination,
                modes=("patch",), directions=("person_face",),
            )
            outcome = {
                split: row["patch_person_face"]
                for split, row in read(destination)["results"].items()
            }
            if any(row["status"] != "complete" for row in outcome.values()):
                raise ValueError(f"Incomplete metrics for seed{seed} layer{layer}")
            results[f"layer{layer}_seed{seed}"] = outcome
    values = {
        f"{split}_{metric}": {
            name: outcome[split]["populations"]["all"]["summary"][metric]
            for name, outcome in results.items()
        }
        for split in ("val", "test")
        for metric in ("final_mAP", "average_mAP")
    }
    rankings = {
        key: sorted(scores, key=lambda name: (-scores[name], name))
        for key, scores in values.items()
    }
    layer_statistics = {}
    for layer in LAYERS:
        layer_statistics[str(layer)] = {}
        for key, scores in values.items():
            selected = [
                value for name, value in scores.items()
                if name.startswith(f"layer{layer}_")
            ]
            layer_statistics[str(layer)][key] = {
                "count": len(selected),
                "mean": statistics.mean(selected) if selected else None,
                "sample_std": statistics.stdev(selected) if len(selected) > 1 else None,
            }
    return {
        "results": results, "values": values, "rankings": rankings,
        "layer_statistics": layer_statistics,
        "validation_preferred_group": (
            rankings["val_average_mAP"][0] if results else None
        ),
        "completed_groups": sorted(results),
        "shared_full_within_seed": True,
    }


def run(root):
    git = git_metadata(Path.cwd())
    if git["dirty"]:
        raise ValueError("A clean experiment worktree is required")
    if shutil.disk_usage(".").free < MIN_FREE_DISK_GIB * 1024**3:
        raise RuntimeError("Insufficient disk for all recoverable checkpoints")
    inputs = input_paths()
    sources = [path / "config.json" for path in source_runs(inputs.reference_root).values()]
    sources += [
        inputs.face_manifest_root / "manifests" / f"{split}.jsonl"
        for split in ("train", "val", "test")
    ]
    hashes = {str(path): sha(path) for path in sources}
    root.mkdir(parents=True, exist_ok=False)
    logs = Path("logs/r1_patch_seed_search") / root.name
    for phase in ("smoke", "formal"):
        (logs / phase).mkdir(parents=True, exist_ok=False)
    write(root / "protocol.json", {
        "git": git, "seeds": SEEDS, "layers": LAYERS,
        "interpretation": "two separate single-layer configurations",
        "groups": 8, "training_paths": 20, "tasks": 8, "epochs": 30,
        "batch": 64, "optimizer": "Adam", "scheduler": "cosine",
        "weight_decay": 0, "main_lr": .0125, "new_lr": .0004,
        "patch_cap": .05, "patch_amp_scale": 4096,
        "full_amp": "historical_dynamic", "shared_full_within_seed": True,
        "reliable_weights": [.64, .16, .20], "fallback_weights": [.8, .2, 0],
        "data": str(inputs.data_root), "pretrained": str(inputs.clip_checkpoint),
        "face_manifest": str(inputs.face_manifest_root),
        "jobs": [vars(job) for job in JOBS], "source_hashes": hashes,
        "validation_selection": "highest eight-stage average mAP",
    })
    smoke = dispatch(
        JOBS, root / "smoke", logs / "smoke", REFERENCE,
        {job.name: 4200 if job.view == "full" else 5700 for job in JOBS},
        True, reserve_peak=True, worker_command=command_for,
    )
    passed = [job for job in JOBS if smoke.get(job.name) == "complete"]
    budgets = {
        job.name: int(read(job.path(root / "smoke") / "preflight.json")[
            "smoke_train_peak_reserved_mib"
        ]) + 1024
        for job in passed
    }
    write(root / "smoke_queue.json", read(root / "smoke" / "queue.json"))
    write(root / "preflight_gate.json", {
        "states": smoke, "formal_budgets_mib": budgets,
    })
    formal = dispatch(
        passed, root, logs / "formal", REFERENCE, budgets, False,
        reserve_peak=True, worker_command=command_for,
    )
    torch.set_num_threads(2)
    if any(sha(Path(path)) != value for path, value in hashes.items()):
        raise ValueError("Reference recipes or Face manifests changed")
    result = result_tables(root, formal, inputs.face_manifest_root)
    write(root / "patch_seed_search_results.json", result)
    write(root / "completion.json", {
        "status": "complete" if len(result["completed_groups"]) == 8 else "partial",
        "smoke": smoke, "formal": formal, "source_hashes_verified": True,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--worker", choices=[job.name for job in JOBS])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.worker is None:
        run(args.root)
    else:
        worker(next(job for job in JOBS if job.name == args.worker), args.root, args.smoke)
