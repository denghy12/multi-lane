"""Read-only eight-task test completion; never train or select a checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .export_compact_test_scores import _read_json, _sha256, _torch_load, restore_compact_model_state
from .fixed_baseline_routes import build_model, loader, source_dataset, validate_source, write
from .evaluate_joint26_checkpoint import audit_face_manifest
from .openai_clip_loader import load_openai_clip_visual
from .post_task_calibration import parameter_hash
from .runner import (compute_metrics, evaluate, evaluate_view_diagnostics, git_metadata,
                     resolve_dataset_parent, seen_indices, set_seed, summarize_tasks,
                     validate_classes, write_evaluation_scores)


def require_eight(config, summary, cached=False):
    if (config.get("training_protocol") != "incremental" or config.get("max_tasks") != 8
            or config.get("task_sizes") != [5, 3, 3, 3, 3, 3, 3, 3]
            or summary.get("status") != "complete"):
        raise ValueError("A completed eight-task incremental source is required")
    count = summary.get("completed_tasks") if cached else len(summary.get("task_metrics", []))
    if count != 8:
        raise ValueError("Short or incomplete training cannot substitute for eight tasks")


def require_formal_budget(config):
    if (config.get("epochs_per_task") != 30 or config.get("training_budget_mode") != "epochs"
            or config.get("optimizer_updates_per_task") not in (None, 0)
            or config.get("optimizer_updates_by_task") not in (None, [])):
        raise ValueError("Formal completion requires the declared 30 epochs per task without a short update override")


def restore_cached(model, payload, base_payload, task, training_git, arm):
    expected_mode = {"dynamic": "image", "static": "constant"}[arm]
    if payload.get("router_input_mode") != expected_mode:
        raise ValueError("Checkpoint router mode does not match requested arm")
    original = {k: v for k, v in payload["model"].items() if not k.startswith("parax_bank.")}
    base = base_payload["model"]
    if (base_payload.get("task_id") != task or set(original) != set(base)
            or any(not torch.equal(original[k], base[k]) for k in original)):
        raise ValueError("Fitted checkpoint does not retain the exact source baseline")
    restore_compact_model_state(model, payload, task, training_git)
    model.parax_bank.router_input_mode = expected_mode  # Python option is not in state_dict.
    model.set_parax_runtime_enabled(True)
    model.eval().requires_grad_(False)


def model_digest(model):
    return parameter_hash(sorted(model.state_dict().items()))


def run_model(args):
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    cached = args.kind == "cached"
    fit_cfg = _read_json(source / "config.json") if cached else None
    baseline = Path(args.baseline_source) if cached else source
    config = _read_json(baseline / "config.json")
    summary = _read_json(baseline / "seed_summary.json")
    validate_source(config, summary, cached)
    require_eight(config, summary)
    if not (args.smoke or args.preflight):
        require_formal_budget(config)
    if cached:
        require_eight(config, _read_json(source / "summary.json"), True)
        if (fit_cfg.get("protocol") != "fixed_checkpoint_train_cache_v1"
                or fit_cfg.get("source_config") != config or fit_cfg.get("smoke")
                or fit_cfg.get("cache_split") != "train" or fit_cfg.get("max_tasks") != 8):
            raise ValueError("Cached fitting provenance mismatch")
    metadata = git_metadata(Path(__file__).resolve().parents[2])
    if metadata["dirty"]:
        raise RuntimeError("Clean evaluator worktree required")
    if _sha256(Path(args.clip_checkpoint)) != config["clip_checkpoint_sha256"]:
        raise ValueError("CLIP checkpoint changed")
    if (resolve_dataset_parent(Path(args.data_root)) / "EMOTIC").resolve() != Path(config["data_root"]).resolve():
        raise ValueError("Dataset root changed")
    face = audit_face_manifest(config, Path(args.face_manifest_root))
    dataset = source_dataset(config, args.data_root, args.face_manifest_root, "test")
    validate_classes(dataset)
    set_seed(config["seed"], config["tf32"])
    device = torch.device("cuda:0")
    model = build_model(config, load_openai_clip_visual(Path(args.clip_checkpoint)), frozen_pool=cached).float().to(device)
    tasks = [7] if args.smoke else list(range(8))
    ckpt_root = source / args.arm if cached else source
    hashes = {str(t): _sha256(ckpt_root / "compact_checkpoints" / f"task{t}.pth") for t in tasks}
    write(output / "config.json", {"kind": args.kind, "arm": args.arm if cached else None,
          "source": str(source), "baseline_source": str(baseline), "source_config": config,
          "git": metadata, "test_accessed": True, "training_updates": 0,
          "model_selection": False, "tasks": tasks, "smoke": args.smoke or args.preflight,
          "face_manifest_hashes": face})
    rows = []
    torch.cuda.reset_peak_memory_stats(device)
    for task in tasks:
        path = ckpt_root / "compact_checkpoints" / f"task{task}.pth"
        payload = _torch_load(path)
        if cached:
            base_path = baseline / "compact_checkpoints" / f"task{task}.pth"
            base_payload = _torch_load(base_path)
            if base_payload.get("source_git") != config["git"]:
                raise ValueError("Baseline checkpoint provenance changed")
            audit = _read_json(source / args.arm / f"task{task}_audit.json")
            if _sha256(base_path) != audit["source_checkpoint_sha256"]:
                raise ValueError("Baseline checkpoint differs from the fitted source")
            restore_cached(model, payload, base_payload, task, fit_cfg["git"], args.arm)
        else:
            restore_compact_model_state(model, payload, task, config["git"])
            model.eval().requires_grad_(False)
        model.assert_visual_frozen()
        before = model_digest(model)
        data = loader(dataset, seen_indices(task), config, args.smoke)
        row = evaluate(model, data, device, task, config["threshold"], config["amp"],
                       output / "test_scores" / f"task{task}.npz")
        views = evaluate_view_diagnostics(model, data, device, task, config["threshold"], config["amp"],
                                         output / "view_test_scores" / f"task{task}.npz")
        if model_digest(model) != before or _sha256(path) != hashes[str(task)]:
            raise RuntimeError("Test evaluation changed source parameters/checkpoint")
        rows.append(row)
        write(output / f"task{task}_audit.json", {"checkpoint_sha256": hashes[str(task)],
              "state_sha256_before": before, "state_sha256_after": before, "views": views})
        write(output / "summary.json", {"status": "running", "task_metrics": [asdict(r) for r in rows]})
        print(f"test task={task} mAP={row.mAP:.6f} source={args.kind} arm={args.arm}", flush=True)
    write(output / "summary.json", {"status": "smoke_complete" if (args.smoke or args.preflight) else "complete",
          "task_metrics": [asdict(r) for r in rows], **summarize_tasks(rows),
          "checkpoint_hashes": hashes, "training_updates": 0,
          "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20})


def apply_frozen_gates(data, gates, sizes, task):
    from .frozen_view_ranking_fusion import tensors, implied_weights
    fixed = torch.from_numpy(data["fused_logits"])
    updated = fixed.clone()
    for previous in range(task + 1):
        columns = slice(sum(sizes[:previous]), sum(sizes[:previous + 1]))
        v, z, _, r = tensors(data, columns)
        with torch.no_grad():
            updated[:, columns], delta = gates[previous](v, z, r)
        weights = implied_weights(delta, r)
        if (weights < 0).any() or (weights[~r, :, 2] != 0).any():
            raise RuntimeError("Invalid frozen fusion weights")
    probs = torch.cat([torch.sigmoid(chunk) for chunk in updated.split(64)])
    probs = torch.where(updated == fixed, torch.from_numpy(data["fused_probabilities"]), probs)
    return updated, probs


def run_fusion(args):
    from .frozen_view_ranking_fusion import (METHODS, BoundedFusionGate, aligned_baseline,
                                           read_scores, digest_files)
    source, output = Path(args.source), Path(args.output)
    test_root = Path(args.test_source)
    if output.exists():
        raise FileExistsError(output)
    if args.smoke:
        raise ValueError("Fusion completion uses full saved test scores only")
    manifest = _read_json(source / "experiment.json")
    config = _read_json(test_root / "config.json")
    if (manifest.get("tasks") != 8 or not (source / "complete.txt").exists()
            or config.get("kind") != "compact" or (config.get("smoke") and not args.preflight)
            or _read_json(test_root / "summary.json").get("status") != ("smoke_complete" if args.preflight else "complete")
            or manifest["source_git"] != config["source_config"]["git"]):
        raise ValueError("Completed eight-task frozen gates and same-source test scores required")
    sizes = config["source_config"]["task_sizes"]
    # Background calibration's source is augmented_source; its source_model points
    # to the exact compact model used for inference. Compare config and scores.
    gate_source = Path(manifest["source"])
    if _read_json(gate_source / "config.json")["git"] != config["source_config"]["git"]:
        raise ValueError("Fusion and test checkpoint provenance differ")
    for task in range(8):
        if _sha256(gate_source / "val_scores" / f"task{task}.npz") != _sha256(
                Path(config["source"]) / "val_scores" / f"task{task}.npz"):
            raise ValueError("Fusion calibration and test did not use the same baseline")
    paths = [p for p in source.rglob("*") if p.is_file()]
    paths += [p for p in test_root.rglob("*") if p.is_file()]
    before = digest_files(paths)
    all_gates = {}
    for method in METHODS:
        gates = []
        for task in range(8):
            payload = _torch_load(source / method / "gates" / f"task{task}.pt")
            columns = [sum(sizes[:task]), sum(sizes[:task + 1])]
            if (payload["task_id"] != task or payload["method"] != method
                    or payload["class_slice"] != columns or payload["source_split"] != "excluded_train_groups"):
                raise ValueError("Frozen gate provenance mismatch")
            n = sizes[task]
            gate = BoundedFusionGate(torch.zeros(2, n, 3), torch.zeros(2, n), method != "static_bce", payload["radius"])
            gate.load_state_dict(payload["state_dict"], strict=True)
            gates.append(gate.eval().requires_grad_(False))
        all_gates[method] = gates
    records = {method: [] for method in ("fixed", *METHODS)}
    for task in range(8):
        data = aligned_baseline(test_root / "test_scores" / f"task{task}.npz",
                                read_scores(test_root / "view_test_scores" / f"task{task}.npz"))
        targets = torch.from_numpy(data["targets"])
        for method in records:
            logits, probs = (torch.from_numpy(data["fused_logits"]), torch.from_numpy(data["fused_probabilities"])) if method == "fixed" else apply_frozen_gates(data, all_gates[method], sizes, task)
            records[method].append(compute_metrics(task, probs, targets, .5))
            write_evaluation_scores(output / method / "test_scores" / f"task{task}.npz", task,
                                    data["sample_ids"].tolist(), logits, targets, probs, data["batch_lengths"].tolist())
    if digest_files(paths) != before:
        raise RuntimeError("Read-only fusion evaluation changed source files")
    write(output / "summary.json", {"status": "smoke_complete" if args.preflight else "complete", "training_updates": 0,
          "source": str(source), "test_source": str(test_root), "source_sha256": before,
          "arms": {m: {"task_metrics": [asdict(r) for r in rows], **summarize_tasks(rows)} for m, rows in records.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("cached", "compact", "fusion"), required=True)
    for key in ("source", "output"):
        parser.add_argument("--" + key, required=True)
    for key in ("baseline-source", "test-source", "data-root", "clip-checkpoint", "face-manifest-root"):
        parser.add_argument("--" + key)
    parser.add_argument("--arm", choices=("dynamic", "static"), default="dynamic")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Full-cohort pipeline smoke; never a formal result")
    args = parser.parse_args()
    if args.kind != "fusion" and any(getattr(args, k) is None for k in ("data_root", "clip_checkpoint", "face_manifest_root")):
        parser.error("Model inference requires data, CLIP and Face manifest paths")
    if args.kind == "cached" and args.baseline_source is None:
        parser.error("Cached inference requires --baseline-source")
    if args.kind == "fusion" and args.test_source is None:
        parser.error("Frozen fusion requires --test-source")
    (run_fusion if args.kind == "fusion" else run_model)(args)


if __name__ == "__main__":
    main()
