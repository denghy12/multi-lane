"""Learn new/old parameter experts on an immutable incremental baseline.

This isolates expert reuse from original-model joint training. Test is opened
only after all requested tasks have been fitted and sealed, without selection.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path

import torch

from .complete_eight_task_test import require_eight, require_formal_budget, restore_cached, model_digest
from .evaluate_joint26_checkpoint import audit_face_manifest
from .export_compact_test_scores import _read_json, _sha256, _torch_load
from .fixed_baseline_routes import (build_model, cached_identity, compare_source_scores,
                                    loader, restore_base, source_dataset, validate_source, write)
from .openai_clip_loader import load_openai_clip_visual
from .post_task_calibration import base_hash, fit_cached_routes, parameter_hash
from .protected_parax_audit import ProtectedRouteAudit
from .runner import (compact_model_state_dict, dataset_view, evaluate, evaluate_view_diagnostics,
                     git_metadata, ids_digest, resolve_dataset_parent, seen_indices, set_seed,
                     summarize_tasks, task_indices, validate_classes)

POLICIES = ("fresh_only", "reuse_old")


def build_expanding_model(config, visual, policy):
    if policy not in POLICIES:
        raise ValueError("Only the equal-new-expert-budget policies are supported")
    options = dict(config)
    options.update(parax_mode="post_task_protected_fresh" if policy == "fresh_only" else "post_task_protected_reuse",
                   parax_rank=32, parax_num_experts=16, parax_layer_indices=[0], parax_router_hidden=16,
                   parax_residual_scale=1., parax_initialization="zero_output",
                   parax_trainable_components="all", parax_output_scale_mode="fixed",
                   parax_smooth_ratio_bound=.02, parax_level_conditioned=False,
                   parax_task_local_gate=False, parax_residual_ratio_cap=0.,
                   parax_projector_bottleneck_dim=0, adapter_mode="disabled")
    return build_model(options, visual)


def paired_banks(bank):
    if bank.policy != "fresh_only":
        raise ValueError("Pair construction requires a fresh-only bank")
    reuse = copy.deepcopy(bank)
    reuse.policy = "reuse_old"
    for task in range(reuse.num_tasks):
        reuse.expert_access[task].zero_()
        reuse.expert_access[task, :2 * task + 2] = True
    # Parameters, including unused future slots, must be paired exactly.
    if parameter_hash(bank.named_parameters()) != parameter_hash(reuse.named_parameters()):
        raise RuntimeError("Unpaired initial expert/router/projection tensors")
    return {"fresh_only": bank, "reuse_old": reuse}


def validate_cache(cache, task, checkpoint_sha, expected_ids, expected_targets, hidden_dim):
    if (cache.get("source_split") != "train" or cache.get("task_id") != task
            or cache.get("source_checkpoint_sha256") != checkpoint_sha):
        raise ValueError("Cache split/task/checkpoint provenance mismatch")
    ids = cache.get("sample_ids", [])
    if ids != expected_ids or len(set(ids)) != len(ids):
        raise ValueError("Cache sample IDs differ from full current-task TRAIN")
    targets = cache.get("targets")
    if not isinstance(targets, torch.Tensor) or not torch.equal(targets, expected_targets):
        raise ValueError("Cache labels differ from current-task TRAIN classes")
    n = len(ids)
    features = cache.get("features", {})
    if set(features) != {"full", "person", "face"}:
        raise ValueError("Three cached view features are required")
    for name, value in features.items():
        if value.shape != (n, 1, hidden_dim) or not bool(torch.isfinite(value).all()):
            raise ValueError("Invalid cache feature shape/finiteness: " + name)
    reliable = cache.get("face_reliable")
    if not isinstance(reliable, torch.Tensor) or reliable.shape != (n,) or reliable.dtype != torch.bool:
        raise ValueError("Invalid cached Face reliability mask")


def train_metadata(dataset, task):
    view = dataset_view(dataset, task_indices(task))
    ids = [str(dataset.sample_ids[i]) for i in view.indices]
    labels = torch.tensor([[float(c in dataset.targets[i]) for c in task_indices(task)]
                           for i in view.indices], dtype=torch.float32)
    return ids, labels


def smoke_subset(cache):
    n = min(64, len(cache["targets"]))
    return {**cache, "sample_ids": cache["sample_ids"][:n], "targets": cache["targets"][:n],
            "face_reliable": cache["face_reliable"][:n],
            "features": {k: v[:n] for k, v in cache["features"].items()}}


def checkpoint_payload(model, task, git, policy):
    return {"schema_version": 1, "task_id": task, "source_git": git,
            "expert_policy": policy, "router_input_mode": "image",
            "model": compact_model_state_dict(model)}


def restore_fitted(model, payload, original, task, git, policy):
    if payload.get("expert_policy") != policy or model.parax_bank.policy != policy:
        raise ValueError("Fitted expert policy mismatch")
    bank = model.parax_bank
    expected_access = torch.zeros_like(bank.expert_access, device="cpu")
    for index in range(bank.num_tasks):
        if policy == "fresh_only":
            expected_access[index, 2 * index:2 * index + 2] = True
        else:
            expected_access[index, :2 * index + 2] = True
    restore_cached(model, payload, original, task, git, "dynamic")
    if not torch.equal(model.parax_bank.expert_access.cpu(), expected_access):
        raise ValueError("Checkpoint changed the policy's permanent access masks")
    sealed = model.parax_bank.sealed_tasks.cpu()
    if not bool(sealed[:task + 1].all()) or bool(sealed[task + 1:].any()):
        raise ValueError("Checkpoint task sealing mismatch")


def assert_pair_budget(banks, task):
    active = {}
    for policy, bank in banks.items():
        bank.restore_task(task)
        active[policy] = sum(p.numel() for p in bank.active_parameters())
    if len(set(active.values())) != 1:
        raise RuntimeError("Unequal trainable parameter budget")
    fresh, reuse = (banks[p] for p in POLICIES)
    names = [f"expert_{side}.{i}" for side in ("a", "b") for i in fresh.new_expert_ids(task)]
    names += [k for k, _ in fresh.named_parameters()
              if k.startswith((f"routers.{task}_", f"task_projections.{task}."))]
    a, b = dict(fresh.named_parameters()), dict(reuse.named_parameters())
    if any(not torch.equal(a[k], b[k]) for k in names):
        raise RuntimeError("Current new expert/router/projection initialization differs")
    return {"trainable_parameters": active, "current_initialization_sha256": parameter_hash((k, a[k]) for k in names),
            "new_expert_ids": list(fresh.new_expert_ids(task)),
            "eligible_experts": {p: bank.expert_access[task].nonzero().flatten().tolist() for p, bank in banks.items()}}


def run(args):
    source, cached, output = map(Path, (args.source, args.cache_source, args.output))
    if output.exists():
        raise FileExistsError(output)
    config, source_summary = _read_json(source / "config.json"), _read_json(source / "seed_summary.json")
    validate_source(config, source_summary, True)
    require_eight(config, source_summary)
    require_formal_budget(config)
    if source_summary.get("test_metrics") is None or len(source_summary.get("test_task_metrics", [])) != 8:
        raise ValueError("Same-baseline eight-task test reference is required")
    if config["seed"] != 0:
        raise ValueError("This declared experiment is seed0 only")
    cache_config, cache_summary = _read_json(cached / "config.json"), _read_json(cached / "summary.json")
    if (cache_config.get("protocol") != "fixed_checkpoint_train_cache_v1"
            or cache_config.get("source_config") != config or cache_config.get("cache_split") != "train"
            or cache_config.get("smoke") or cache_summary.get("status") != "complete"
            or cache_summary.get("completed_tasks") != 8):
        raise ValueError("Complete same-baseline TRAIN cache source required")
    git = git_metadata(Path(__file__).resolve().parents[2])
    if git["dirty"]:
        raise RuntimeError("Clean experiment worktree required")
    clip = Path(args.clip_checkpoint)
    if _sha256(clip) != config["clip_checkpoint_sha256"]:
        raise ValueError("Pretrained CLIP changed")
    if (resolve_dataset_parent(Path(args.data_root)) / "EMOTIC").resolve() != Path(config["data_root"]).resolve():
        raise ValueError("Dataset root changed")
    faces = audit_face_manifest(config, Path(args.face_manifest_root))
    source_hashes = {str(t): _sha256(source / "compact_checkpoints" / f"task{t}.pth") for t in range(8)}
    cache_hashes = {str(t): _sha256(cached / "train_cache" / f"task{t}.pth") for t in range(8)}
    device = torch.device("cuda:0")
    set_seed(config["seed"], config["tf32"])
    model = build_expanding_model(config, load_openai_clip_visual(clip), "fresh_only").float().to(device)
    model.requires_grad_(False)
    banks = paired_banks(model.parax_bank)
    datasets = {s: source_dataset(config, args.data_root, args.face_manifest_root, s) for s in ("train", "val")}
    validate_classes(*datasets.values())
    audits = {p: ProtectedRouteAudit(output / p, device) for p in POLICIES}
    records = {p: [] for p in ("baseline", *POLICIES)}
    for p in POLICIES:
        (output / p).mkdir(parents=True)
    write(output / "config.json", {"protocol": "fixed_checkpoint_expanding_parameter_experts_v1",
          "source": str(source), "source_config": config, "cache_source": str(cached), "git": git,
          "seed": 0, "tasks": 8, "epochs": 5, "batch_size": 64, "optimizer": "Adam",
          "learning_rate": .0004, "weight_decay": 0., "scheduler": "cosine_reset_per_task",
          "gradient_clip_norm": 1., "auxiliary_weight": .1, "consistency_weight": 0.,
          "rank": 32, "new_experts_per_task": 2, "expert_slots": 16, "router_hidden": 16,
          "residual_bound": .02, "residual_scale": 1., "combination": "parameter_matrices",
          "old_paths_sealed": True, "baseline_retrained": False, "test_after_all_fitting": True,
          "model_selection": False, "smoke": args.smoke, "face_manifest_hashes": faces,
          "source_checkpoint_hashes": source_hashes, "train_cache_hashes": cache_hashes})
    previous_base = None
    for task in range(8):
        payload = _torch_load(source / "compact_checkpoints" / f"task{task}.pth")
        restored = restore_base(model, payload, task, config["git"])
        if previous_base is not None:
            current = audits["fresh_only"]._protected_state(model, task)
            if any(not torch.equal(current[k], v) for k, v in previous_base.items()):
                raise RuntimeError("Source baseline changed previously learned parameters")
        model.set_parax_runtime_enabled(False)
        val_loader = loader(datasets["val"], seen_indices(task), config, args.smoke)
        score = output / "baseline_scores" / f"task{task}.npz"
        records["baseline"].append(evaluate(model, val_loader, device, task, config["threshold"], config["amp"], score))
        recovery = {"full_cohort_recovery_skipped_for_smoke": True} if args.smoke else compare_source_scores(score, source / "val_scores" / f"task{task}.npz")
        cache_path = cached / "train_cache" / f"task{task}.pth"
        cache = _torch_load(cache_path)
        ids, labels = train_metadata(datasets["train"], task)
        validate_cache(cache, task, source_hashes[str(task)], ids, labels, model.output_dim)
        if args.smoke:
            cache = smoke_subset(cache)
        pair = assert_pair_budget(banks, task)
        write(output / f"task{task}_paired_protocol.json", {**pair, "train_ids_sha256": ids_digest(cache["sample_ids"]),
              "samples": len(cache["targets"]), "cache_sha256": cache_hashes[str(task)]})
        for policy, bank in banks.items():
            model.parax_bank = bank
            model.restore_task(task)
            model.requires_grad_(False)
            identity = cached_identity(model, cache)
            audits[policy].begin(model, val_loader, task)
            before = base_hash(model)
            set_seed(config["seed"] + 1009 * task, config["tf32"])
            fit = fit_cached_routes(model, cache, task_indices(task), device, epochs=5,
                                    learning_rate=.0004, batch_size=64, consistency_weight=0., auxiliary_weight=.1)
            if (base_hash(model) != before or any(p.grad is not None for n, p in model.named_parameters() if not n.startswith("parax_bank."))
                    or fit["initial_output_max_difference"] != 0. or not all(fit["new_experts_changed"].values())):
                raise RuntimeError("Base/identity/new-expert learning audit failed")
            model.set_parax_runtime_enabled(True)
            audits[policy].finish(model, task)
            arm_score = output / policy / "val_scores" / f"task{task}.npz"
            records[policy].append(evaluate(model, val_loader, device, task, config["threshold"], config["amp"], arm_score))
            views = evaluate_view_diagnostics(model, val_loader, device, task, config["threshold"], config["amp"],
                                             output / policy / "view_val_scores" / f"task{task}.npz")
            write(output / policy / f"task{task}_views.json", views)
            write(output / policy / f"task{task}_audit.json", {**fit, **identity, **recovery,
                  "source_base_tensor_sha256": restored, "source_checkpoint_sha256": source_hashes[str(task)],
                  "cache_sha256": cache_hashes[str(task)], "policy": bank.policy_summary(task)})
            target = output / policy / "compact_checkpoints" / f"task{task}.pth"
            target.parent.mkdir(exist_ok=True)
            torch.save(checkpoint_payload(model, task, git, policy), target)
        previous_base = {k: v for k, v in audits["fresh_only"]._protected_state(model, task + 1).items() if not k.startswith("parax.")}
        write(output / "summary.json", {"status": "fitting", "completed_tasks": task + 1,
              "arms": {k: {"task_metrics": [asdict(r) for r in rows], **summarize_tasks(rows)} for k, rows in records.items()}})
        print(f"paired frozen-baseline fit completed task={task}", flush=True)
    write(output / "fit_complete.json", {"tasks": 8, "test_accessed": False, "model_selection": False})
    # No test dataset exists until every training operation has finished.
    test_dataset = source_dataset(config, args.data_root, args.face_manifest_root, "test")
    validate_classes(test_dataset)
    test_rows = {p: [] for p in POLICIES}
    for policy in POLICIES:
        model = build_expanding_model(config, load_openai_clip_visual(clip), policy).float().to(device)
        for task in range(8):
            path = output / policy / "compact_checkpoints" / f"task{task}.pth"
            saved_sha = _sha256(path)
            payload = _torch_load(path)
            original = _torch_load(source / "compact_checkpoints" / f"task{task}.pth")
            restore_fitted(model, payload, original, task, git, policy)
            before = model_digest(model)
            test_loader = loader(test_dataset, seen_indices(task), config, args.smoke)
            test_rows[policy].append(evaluate(model, test_loader, device, task, config["threshold"], config["amp"],
                                             output / policy / "test_scores" / f"task{task}.npz"))
            views = evaluate_view_diagnostics(model, test_loader, device, task, config["threshold"], config["amp"],
                                             output / policy / "view_test_scores" / f"task{task}.npz")
            if model_digest(model) != before or _sha256(path) != saved_sha:
                raise RuntimeError("Read-only test changed model/checkpoint")
            write(output / policy / f"task{task}_test_audit.json", {"state_before": before, "state_after": before,
                  "checkpoint_sha256": saved_sha, "training_updates": 0, "views": views})
            print(f"read-only test completed policy={policy} task={task}", flush=True)
    if source_hashes != {str(t): _sha256(source / "compact_checkpoints" / f"task{t}.pth") for t in range(8)}:
        raise RuntimeError("Source checkpoints changed")
    if cache_hashes != {str(t): _sha256(cached / "train_cache" / f"task{t}.pth") for t in range(8)}:
        raise RuntimeError("Source TRAIN caches changed")
    write(output / "summary.json", {"status": "smoke_complete" if args.smoke else "complete", "completed_tasks": 8,
          "source_checkpoints_unchanged": True, "source_caches_unchanged": True,
          "validation": {k: {"task_metrics": [asdict(r) for r in rows], **summarize_tasks(rows)} for k, rows in records.items()},
          "test": {k: {"task_metrics": [asdict(r) for r in rows], **summarize_tasks(rows)} for k, rows in test_rows.items()},
          "baseline_test_from_source": source_summary["test_metrics"],
          "test_model_selection": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache-source", "output", "data-root", "clip-checkpoint", "face-manifest-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--smoke", action="store_true", help="All eight tasks, only first64 samples per split; not performance")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
