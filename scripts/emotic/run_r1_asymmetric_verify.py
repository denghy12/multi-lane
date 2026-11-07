"""Verify asymmetric receivers and train paired R1 and Face-only Pool controls."""
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
from torch.utils.data import DataLoader, Subset

from multi_lane.track_a.paired_transforms import move_model_inputs
from multi_lane.track_a.r1_crossvit_experiment import flags, model_from_config, read, sha, write
from multi_lane.track_a.runner import dataset_view, git_metadata, seen_indices, set_seed
from scripts.emotic.r1_asymmetric_forward import SOURCE, branch_paths
from scripts.emotic.run_r1_frozen_cross_view import check_training, input_paths, load_source, restore, score_suffix
from scripts.emotic.run_r1_frozen_late_stable import summarize_family
from scripts.emotic.run_r1_patch_receiver_depth import REFERENCE, dispatch


ROOT = Path("output/r1_asymmetric_verify/20261107")
SEEDS = (3047, 1024, 42, 123)


@dataclass(frozen=True)
class Job:
    seed: int
    arm: str
    view: str
    gpu: int

    @property
    def name(self):
        return f"seed{self.seed}_{self.arm}_{self.view}"

    def path(self, root):
        if self.arm == "verify":
            return root / "verify" / f"seed{self.seed}"
        return root / "controls" / f"seed{self.seed}" / self.arm / self.view


JOBS = tuple(
    [Job(seed, "verify", "all", index % 2) for index, seed in enumerate(SEEDS)]
    + [
        Job(seed, arm, view, (index + offset) % 2)
        for index, seed in enumerate(SEEDS)
        for offset, (arm, view) in enumerate(
            (("native", "person"), ("native", "face"), ("pool", "face"))
        )
    ]
)


def command_for(job, root, smoke):
    command = [
        sys.executable, "-u", "-m", "scripts.emotic.run_r1_asymmetric_verify",
        "--root", str(root), "--worker", job.name,
    ]
    if smoke:
        command.append("--smoke")
    return command


def baseline_preflight(train, destination):
    inputs = input_paths()
    config = read(train / "config.json")
    set_seed(config["seed"], True)
    torch.set_num_threads(2)
    model = model_from_config(config, inputs.clip_checkpoint)
    restore(model, train, 1)
    source = load_source(train, inputs, "val")
    data = dataset_view(source, seen_indices(1))
    images, _ = next(iter(DataLoader(Subset(data, range(64)), batch_size=64)))
    images = move_model_inputs(images, torch.device("cuda"))
    gradients = {}
    if config["r1_kind"] == "pool":
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            logits = model.current_all_logits(images).float()
        logits.square().mean().backward()
        gradients = {
            name: None if parameter.grad is None else float(parameter.grad.float().norm())
            for name, parameter in model.r1_bank[1].named_parameters()
        }
        if any(value is None or not torch.isfinite(torch.tensor(value)) for value in gradients.values()):
            raise AssertionError("Pool gradients are missing or nonfinite")
        if gradients["out.weight"] <= 0:
            raise AssertionError("Pool output projection has zero gradient")
    model.restore_task(7)
    model.requires_grad_(False).eval()
    with torch.no_grad(), torch.autocast("cuda"):
        logits = model.seen_logits(images).float()
    if logits.shape != (64, 26) or not torch.isfinite(logits).all():
        raise AssertionError("Invalid eight-lane control output")
    history = read(train / "training_history.json")
    peak = max(
        row["cuda_peak_reserved_mib"] for rows in history.values() for row in rows
    )
    write(destination, {
        "status": "passed", "seed": config["seed"],
        "gradient_norms": gradients, "eight_lane_shape": list(logits.shape),
        "smoke_train_peak_reserved_mib": max(
            peak, torch.cuda.max_memory_reserved() / 1024**2,
        ),
    })


def worker(job, root, smoke):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(job.gpu):
        raise ValueError("Worker GPU differs from the schedule")
    path = job.path(root)
    if job.arm == "verify":
        command = [
            sys.executable, "-u", "-m", "scripts.emotic.r1_asymmetric_forward",
            "--seed", str(job.seed), "--output", str(path),
        ]
        if smoke:
            command.append("--smoke")
        subprocess.run(command, check=True)
        if smoke:
            write(path / "preflight.json", read(path / "completion.json"))
        return
    inputs = input_paths()
    command = [
        sys.executable, "-u", "-m", "multi_lane.track_a.runner",
        *flags(
            SimpleNamespace(output=root / "controls" / f"seed{job.seed}", **vars(inputs)),
            job.arm, job.view, smoke, seed=job.seed,
        ),
        "--r1-fusion-readout", "logit",
    ]
    if smoke:
        command += ["--r1-smoke-eval-samples", "64"]
    env = dict(
        os.environ, OMP_NUM_THREADS="72", MKL_NUM_THREADS="72",
        OPENBLAS_NUM_THREADS="1",
    )
    print("TRAIN", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)
    counts = check_training(path / "train", REFERENCE, job.view, 2 if smoke else 8, smoke)
    if smoke:
        baseline_preflight(path / "train", path / "preflight.json")
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
        "status": "complete", "seed": job.seed, "training": counts, "gpu": job.gpu,
    })


def summarize(root, states, manifest):
    results = {}
    for seed in SEEDS:
        selected = [job for job in JOBS if job.seed == seed]
        if not all(states.get(job.name) == "complete" for job in selected):
            continue
        base = root / "controls" / f"seed{seed}"
        result = {
            "asymmetric": read(root / "verify" / f"seed{seed}" / "result.json"),
            "controls": {},
        }
        for name, face in (("R1", base / "native/face"), ("Face_only_Pool", base / "pool/face")):
            destination = base / "fusion" / f"{name}.json"
            summarize_family(
                root, SOURCE / f"seed{seed}", manifest,
                {("patch", "person"): base / "native/person", ("patch", "face"): face},
                destination, modes=("patch",), directions=("person_face",),
            )
            result["controls"][name] = {
                split: rows["patch_person_face"]
                for split, rows in read(destination)["results"].items()
            }
        results[str(seed)] = result
    deltas = {}
    for split in ("val", "test"):
        deltas[split] = {}
        for control in ("R1", "Face_only_Pool"):
            deltas[split][control] = {}
            for metric in ("final_mAP", "average_mAP"):
                values = {}
                for seed, result in results.items():
                    candidate = result["asymmetric"]["results"][split]["asymmetric"]["both_on"]["all"]["summary"]
                    baseline = result["controls"][control][split]["populations"]["all"]["summary"]
                    values[seed] = candidate[metric] - baseline[metric]
                deltas[split][control][metric] = {
                    "per_seed": values,
                    "mean": statistics.mean(values.values()) if values else None,
                    "sample_std": statistics.stdev(values.values()) if len(values) > 1 else None,
                }
    return {"results": results, "paired_deltas": deltas}


def run(root):
    git = git_metadata(Path.cwd())
    if git["dirty"]:
        raise ValueError("A clean experiment worktree is required")
    if shutil.disk_usage(".").free < 50 * 1024**3:
        raise RuntimeError("Insufficient disk for recoverable paired-control checkpoints")
    inputs = input_paths()
    sources = []
    for seed in SEEDS:
        for train in branch_paths(seed).values():
            sources.append(train / "config.json")
            for task in range(8):
                sources.append(train / "compact_checkpoints" / f"task{task}.pth")
                sources.extend(train.parent / score_suffix(split, task) for split in ("val", "test"))
    sources.extend(
        inputs.face_manifest_root / "manifests" / f"{split}.jsonl"
        for split in ("train", "val", "test")
    )
    hashes = {str(path): sha(path) for path in sources}
    root.mkdir(parents=True, exist_ok=False)
    logs = Path("logs/r1_asymmetric_verify") / root.name
    for phase in ("smoke", "formal"):
        (logs / phase).mkdir(parents=True, exist_ok=False)
    write(root / "protocol.json", {
        "git": git, "seeds": SEEDS, "source": str(SOURCE),
        "asymmetric": {"person": 10, "face": 9},
        "attribution_groups": ["layer9", "layer10", "asymmetric"],
        "attribution_tasks": [0, 2, 7],
        "attribution_states": ["both_on", "person_off", "face_off", "both_off"],
        "control_training_paths": 12, "reused_full_paths": 4,
        "tasks": 8, "epochs": 30, "batch": 64, "main_lr": .0125, "new_lr": .0004,
        "optimizer": "Adam", "scheduler": "cosine", "weight_decay": 0,
        "control_amp": "historical_dynamic", "pool_layer": 6, "pool_cap": None,
        "readout": "fixed_R1_weighted_logits_then_sigmoid",
        "jobs": [vars(job) for job in JOBS], "source_hashes": hashes,
    })
    smoke = dispatch(
        JOBS, root / "smoke", logs / "smoke", REFERENCE,
        {job.name: 9000 if job.arm == "verify" else 4200 for job in JOBS},
        True, reserve_peak=True, worker_command=command_for,
    )
    if set(smoke.values()) != {"complete"}:
        write(root / "completion.json", {"status": "preflight_failed", "smoke": smoke})
        raise AssertionError("Every replay and control preflight must pass")
    write(root / "smoke_queue.json", read(root / "smoke/queue.json"))
    budgets = {
        job.name: int(read(job.path(root / "smoke") / "preflight.json")[
            "smoke_train_peak_reserved_mib"
        ]) + 1024 for job in JOBS
    }
    write(root / "preflight_gate.json", {"states": smoke, "formal_budgets_mib": budgets})
    formal = dispatch(
        JOBS, root, logs / "formal", REFERENCE, budgets,
        False, reserve_peak=True, worker_command=command_for,
    )
    torch.set_num_threads(2)
    if any(sha(Path(path)) != value for path, value in hashes.items()):
        raise AssertionError("Source artifacts changed")
    results = summarize(root, formal, inputs.face_manifest_root)
    write(root / "asymmetric_paired_results.json", results)
    write(root / "completion.json", {
        "status": "complete" if len(results["results"]) == 4 else "partial",
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
