"""Read-only gradient geometry for paired frozen MULTI-LANE checkpoints.

No optimizer, .backward(), or parameter update is used. Gradient measurements
are conditional on endpoint checkpoints and are not Adam update directions.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_lane.continual_datasets.continual_datasets import EMOTIC
from .export_compact_test_scores import _sha256, _torch_load, restore_compact_model_state
from .model import MultiLaneModel
from .openai_clip_loader import load_openai_clip_visual
from .paired_transforms import ThreeViewTransform, move_model_inputs
from .runner import (TASK_SIZES, LabelView, _intersects, build_transforms,
                     compute_training_loss, compute_view_objective_losses,
                     git_metadata, ids_digest, resolve_dataset_parent, set_seed,
                     task_indices, tensor_state_digest)

MODES = ("baseline", "shared_fixed", "view_fixed", "dynamic")
PRECISIONS = ("deterministic_fp32", "training_amp")
BASE_GROUPS = ("selector", "prompt", "classifier", "base_representation", "base")
PATH_NAMES = ("fused_path_full", "fused_path_person", "fused_path_face")


def read(path):
    return json.loads(path.read_text())


def grouped_batches(sample_ids: Sequence[str], batch_size: int, count: int,
                    seed: int) -> list[list[int]]:
    """Shuffle source images, keep all eligible people of one image together."""
    if batch_size <= 0 or count <= 0:
        raise ValueError("Batch size and count must be positive")
    groups = {}
    for i, sample_id in enumerate(sample_ids):
        groups.setdefault(str(sample_id).rsplit("#person=", 1)[0], []).append(i)
    values = list(groups.values())
    order = np.random.default_rng(seed).permutation(len(values))
    result, current = [], []
    for i in order:
        group = values[int(i)]
        if len(group) > batch_size:
            raise ValueError("One image group exceeds the declared batch cap")
        if len(current) + len(group) > batch_size:
            result.append(current)
            if len(result) == count:
                return result
            current = []
        current.extend(group)
    if current:
        result.append(current)
    if len(result) < count:
        raise ValueError("Not enough image groups for the requested audit")
    return result[:count]


class ParameterView:
    """The active task coordinates, excluding old/future slices explicitly."""
    def __init__(self, model: MultiLaneModel, task_id: int):
        self.task = task_id
        self.current = task_indices(task_id)
        model.requires_grad_(False)
        model.selectors.requires_grad_(True)
        model.prompts.requires_grad_(True)
        model.head.requires_grad_(True)
        if model.adapter_bank is not None:
            model.adapter_bank.tasks[task_id].requires_grad_(True)
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.parameters = tuple(p for _, p in self.named)
        self.groups = {name: [] for name in BASE_GROUPS}
        for name in ("shared_expert", "specialists", "expert1", "expert2", "expert3", "router", "all_experts"):
            self.groups[name] = []
        for i, (name, _) in enumerate(self.named):
            if name == "selectors":
                self.groups["selector"].append(i)
            elif name.startswith("prompts."):
                self.groups["prompt"].append(i)
            elif name.startswith("head."):
                self.groups["classifier"].append(i)
            elif ".experts." in name:
                expert = int(name.split(".experts.")[1].split(".")[0])
                self.groups["shared_expert" if expert == 0 else "specialists"].append(i)
                if expert:
                    self.groups[f"expert{expert}"].append(i)
                self.groups["all_experts"].append(i)
            elif ".gates." in name:
                self.groups["router"].append(i)
        self.groups["base_representation"] = self.groups["selector"] + self.groups["prompt"]
        self.groups["base"] = self.groups["base_representation"] + self.groups["classifier"]

    def vectors(self, gradients, scale: float) -> dict[str, torch.Tensor]:
        pieces = []
        for (name, param), grad in zip(self.named, gradients):
            if name == "selectors":
                value = torch.zeros_like(param[self.task]) if grad is None else grad[self.task]
            elif name.startswith("prompts."):
                value = torch.zeros_like(param[:, self.task]) if grad is None else grad[:, self.task]
            elif name.startswith("head."):
                value = torch.zeros_like(param[list(self.current)]) if grad is None else grad[list(self.current)]
            else:
                value = torch.zeros_like(param) if grad is None else grad
            value = value.detach().float().flatten() / scale
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Non-finite gradient: {name}")
            pieces.append(value)
        reference = self.parameters[0]
        return {g: (torch.cat([pieces[i] for i in indices]) if indices
                    else reference.new_zeros(0, dtype=torch.float32))
                for g, indices in self.groups.items()}

    def grad(self, loss, scale=1.0):
        if not loss.requires_grad:
            grads = (None,) * len(self.parameters)
        else:
            grads = torch.autograd.grad(loss * scale, self.parameters,
                                        retain_graph=True, allow_unused=True)
        return self.vectors(grads, scale)


def configure_precision(model, precision: str) -> float:
    """Keep Router balance active; toggle only expert dropout and arithmetic."""
    model.train()
    model.visual_encoder.eval()
    use_amp = precision == "training_amp"
    torch.backends.cuda.matmul.allow_tf32 = use_amp
    torch.backends.cudnn.allow_tf32 = use_amp
    if model.adapter_bank is not None:
        for task in model.adapter_bank.tasks:
            for pool in task:
                for expert in pool.experts:
                    expert.train(use_amp and pool.training)
    return 1024.0 if use_amp else 1.0


def objective_losses(model, images, targets, current, config):
    logits, view_logits, features = model.current_all_logits_with_view_features(images)
    temp, mode = config["temperature"], config["training_loss_mode"]
    f = lambda z, y: compute_training_loss(z, y, current, temp, mode)
    views = compute_view_objective_losses(view_logits, images, targets, current, temp, mode, "bce")
    zero = logits.new_zeros(())
    fused = f(logits, targets)
    auxiliary = config["view_auxiliary_loss_weight"] * torch.stack(list(views.values())).mean()
    balance = config.get("moe_balance_weight") or 0.0
    balance_loss = balance * model.moe_balance_loss()
    objectives = {"fused": fused, "view_full": views["full"],
                  "view_person": views["person"]}
    if "face" in views:
        objectives["view_face_reliable"] = views["face"]
    reliable = images["face_reliable"].bool()
    for label, mask in (("reliable", reliable), ("unreliable", ~reliable)):
        if bool(mask.any()):
            objectives[f"fused_{label}"] = f(logits[mask], targets[mask])
            for view in ("full", "person"):
                objectives[f"view_{view}_{label}"] = f(view_logits[view][mask], targets[mask])
    if bool((~reliable).any()):
        objectives["face_unreliable_shadow"] = f(view_logits["face"][~reliable], targets[~reliable])
    objectives.update(auxiliary=auxiliary, supervised=fused + auxiliary,
                      balance=balance_loss, total=fused + auxiliary + balance_loss)
    return objectives, features, len(views), logits


def geometry(vectors, objective_names):
    """Cosines are null for absent/zero gradients, not false non-conflicts."""
    output = {}
    for group in next(iter(vectors.values())):
        matrix = torch.stack([vectors[name][group] for name in objective_names])
        if matrix.shape[1] == 0:
            output[group] = {"dimension": 0, "norms": [None] * len(objective_names),
                             "cosines": [[None] * len(objective_names) for _ in objective_names]}
            continue
        # Geometry must not inherit TF32 approximation from the AMP forward.
        with torch.autocast(device_type=matrix.device.type, enabled=False):
            prior = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            gram = matrix @ matrix.T
            torch.backends.cuda.matmul.allow_tf32 = prior
        norms = gram.diag().clamp_min(0).sqrt()
        cos = gram / (norms[:, None] * norms[None, :]).clamp_min(1e-30)
        norm_list = norms.cpu().tolist()
        cosine_list = cos.clamp(-1, 1).cpu().tolist()
        active = [n > 1e-12 and not (name in PATH_NAMES and group in ("classifier", "base"))
                  for name, n in zip(objective_names, norm_list)]
        for i, name in enumerate(objective_names):
            if not active[i]:
                norm_list[i] = None
            for j in range(len(objective_names)):
                if not active[i] or not active[j]:
                    cosine_list[i][j] = None
        output[group] = {"dimension": matrix.shape[1], "norms": norm_list,
                         "cosines": cosine_list}
    return output


def pair_metrics(left, right):
    result = {}
    for group, x in left.items():
        y = right[group]
        nx, ny = float(x.norm()), float(y.norm())
        dot = float(torch.dot(x, y)) if x.numel() else 0.0
        result[group] = {"left_norm": nx, "right_norm": ny,
                         "norm_ratio": nx / ny if ny > 1e-12 else None,
                         "cosine": max(-1., min(1., dot / (nx * ny))) if min(nx, ny) > 1e-12 else None,
                         "parallel_over_right": dot / ny ** 2 if ny > 1e-12 else None}
    return result


def measure(model, pack, images, targets, config, precision, seed):
    scale = configure_precision(model, precision)
    set_seed(seed, tf32=precision == "training_amp")
    with torch.cuda.amp.autocast(enabled=precision == "training_amp"):
        losses, features, valid_views, logits = objective_losses(
            model, images, targets, pack.current, config)
    vectors = {name: pack.grad(loss, scale) for name, loss in losses.items()}
    # Exact chain-rule branch attribution of the original fused loss. The head's
    # direct gradient is outside these feature paths and is deliberately N/A.
    path_tensors = torch.autograd.grad(losses["fused"] * scale,
                                     tuple(features[v] for v in ("full", "person", "face")),
                                     retain_graph=True)
    unreliable = ~images["face_reliable"].bool()
    face_invalid_grad_max = float(path_tensors[2][unreliable].abs().max()) if bool(unreliable.any()) else 0.0
    if face_invalid_grad_max != 0:
        raise AssertionError("Unreliable Face entered the actual fused gradient")
    for view, upstream in zip(("full", "person", "face"), path_tensors):
        gradients = torch.autograd.grad(features[view], pack.parameters,
                                        grad_outputs=upstream, retain_graph=True, allow_unused=True)
        vectors[f"fused_path_{view}"] = pack.vectors(gradients, scale)
    reconstruction = {}
    for group in vectors["fused"]:
        if group in ("base", "classifier") or vectors["fused"][group].numel() == 0:
            continue
        whole = vectors["fused"][group]
        summed = sum(vectors[name][group] for name in PATH_NAMES)
        reconstruction[group] = float((whole - summed).norm() / whole.norm().clamp_min(1e-12))
    if max(reconstruction.values(), default=0) > (0.06 if precision == "training_amp" else 1e-3):
        raise AssertionError(f"Fused feature-path reconstruction failed: {reconstruction}")
    names = list(vectors)
    record = {"precision": precision, "seed": seed,
              "samples": len(targets), "reliable_face_samples": int((~unreliable).sum()),
              "unreliable_face_samples": int(unreliable.sum()),
              "loss_values": {name: float(loss.detach()) for name, loss in losses.items()},
              "objective_names": names, "geometry": geometry(vectors, names),
              "auxiliary_each_view_coefficient": config["view_auxiliary_loss_weight"] / valid_views,
              "face_unreliable_actual_auxiliary_coefficient": 0,
              "face_unreliable_fused_feature_gradient_max": face_invalid_grad_max,
              "fused_path_reconstruction_relative_error": reconstruction,
              "auxiliary_vs_fused": pair_metrics(vectors["auxiliary"], vectors["fused"])}
    common = {name: {g: value[g].detach().cpu().clone() for g in BASE_GROUPS}
              for name, value in vectors.items() if name in ("fused", "auxiliary", "supervised", "total")}
    # Counterfactual anchor at IDENTICAL current base parameters. This differs
    # from the fresh B0 checkpoint, which lives at another learned parameter point.
    if model.adapter_bank is not None:
        model.set_adapter_runtime_enabled(False)
        try:
            with torch.cuda.amp.autocast(enabled=precision == "training_amp"):
                off, _, _, _ = objective_losses(model, images, targets, pack.current, config)
            anchor = pack.grad(off["supervised"], scale)
            for name in ("supervised", "total"):
                x = {g: common[name][g] for g in BASE_GROUPS}
                y = {g: anchor[g].cpu() for g in BASE_GROUPS}
                record[f"{name}_on_vs_same_parameter_moe_off"] = pair_metrics(x, y)
        finally:
            model.set_adapter_runtime_enabled(True)
    if any(p.grad is not None for p in model.parameters()):
        raise AssertionError("autograd.grad unexpectedly populated parameter .grad")
    return record, common


def make_train_source(config, data_root, face_root):
    train, _ = build_transforms(config["input_normalization"], config["train_crop_scale"])
    transform = ThreeViewTransform(
        train=True, normalization=config["input_normalization"],
        crop_scale=config["train_crop_scale"], margin=config["person_crop_margin"],
        jitter_strength=config["person_color_jitter_strength"],
        jitter_probability=config["person_color_jitter_probability"],
        full_crop_mode=config["full_crop_mode"])
    source = EMOTIC(str(resolve_dataset_parent(data_root)), train=True, transform=train,
                    multi_view_transform=transform, input_mode="full",
                    person_crop_margin=config["person_crop_margin"],
                    face_manifest_root=face_root)
    if source.splits != ["train"]:
        raise AssertionError("Gradient audit may only open the training split")
    return source


def build_model(config, clip):
    keys = ("num_selectors", "num_prompts", "num_prompt_layers", "selector_mode",
            "prompt_mode", "normalize", "adapter_mode", "adapter_bottleneck_dim",
            "adapter_layer_indices", "adapter_residual_scale", "view_fusion", "view_classifier_mode")
    kwargs = {key: config[key] for key in keys}
    if config["adapter_mode"] != "disabled":
        kwargs.update({key: config[key] for key in (
            "moe_num_experts", "moe_top_k", "moe_dropout", "moe_route_mode",
            "moe_shared_fraction", "moe_router_logit_scale", "moe_router_temperature", "moe_noise_std")})
    return MultiLaneModel(load_openai_clip_visual(clip), TASK_SIZES, **kwargs).float().cuda()


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU0 is required")
    if args.output.exists():
        raise FileExistsError(args.output)
    git = git_metadata(Path(__file__).resolve().parents[2])
    if git["dirty"]:
        raise RuntimeError("A clean diagnostic checkout is required")
    configs, runs, protocols = {}, {}, {}
    for mode in MODES:
        run_dir = args.source_root / f"formal_{mode}_seed0_20261017_01" / "train"
        runs[mode] = run_dir
        configs[mode] = read(run_dir / "config.json")
        protocols[mode] = read(run_dir / "paired_protocol_audit.json")
        summary = read(run_dir / "seed_summary.json")
        if summary["status"] != "complete" or summary["config"] != configs[mode]:
            raise ValueError("Incomplete source run")
        if configs[mode]["seed"] != 0 or configs[mode]["git"]["dirty"]:
            raise ValueError("Unexpected source provenance")
    baseline = configs["baseline"]
    for mode, c in configs.items():
        for key in ("training_loss_mode", "temperature", "view_auxiliary_loss_weight",
                    "view_fusion", "selector_mode", "prompt_mode", "parameter_group_loss_routing"):
            if c[key] != baseline[key]:
                raise ValueError(f"Unpaired objective: {mode}/{key}")
        if c["git"] != baseline["git"]:
            raise ValueError("Source methods use different training commits")
        if c["parameter_group_loss_routing"] != "joint_bce" or c["supervised_loss_scale"] != 1:
            raise ValueError("Audit is scoped to the v4 supervised objective")
    expected_face = baseline["face_manifest"]["artifact_sha256"]["train_manifest"]
    if _sha256(args.face_manifest_root / "manifests/train.jsonl") != expected_face:
        raise ValueError("Training Face manifest changed")
    source = make_train_source(baseline, args.data_root, args.face_manifest_root)
    cache, batch_metadata = {}, {}
    for task in args.tasks:
        eligible = [i for i, target in enumerate(source.targets) if _intersects(target, task_indices(task))]
        ids = [str(source.sample_ids[i]) for i in eligible]
        for mode in MODES:
            if ids_digest(ids) != protocols[mode]["tasks"][str(task)]["train_sample_ids_sha256"]:
                raise ValueError(f"Training population changed: {mode}/task{task}")
        seed = args.seed + 1009 * task
        positions = grouped_batches(ids, args.batch_size, args.batches, seed)
        view = LabelView(source, eligible, task_indices(task), include_sample_id=True)
        set_seed(seed, tf32=False)
        loader = DataLoader(view, batch_sampler=positions, num_workers=0)
        cache[task] = list(loader)
        batch_metadata[task] = []
        for batch_id, (images, y, sample_ids) in enumerate(cache[task]):
            batch_metadata[task].append({
                "batch": batch_id, "samples": len(y), "sample_ids": list(sample_ids),
                "source_images": len({s.rsplit("#person=", 1)[0] for s in sample_ids}),
                "input_tensor_sha256": tensor_state_digest(images),
                "target_sha256": tensor_state_digest({"targets": y}),
                "reliable_face_samples": int(images["face_reliable"].sum()),
            })
    args.output.mkdir(parents=True)
    header = {"diagnostic_git": git, "source_git": {m: c["git"] for m, c in configs.items()},
              "source_root": str(args.source_root), "tasks": args.tasks,
              "batch_cap": args.batch_size, "batches_per_task": args.batches,
              "precisions": PRECISIONS, "seed": args.seed, "training_updates": 0,
              "optimizer": None, "scheduler": None, "learning_rate": None,
              "split": "train", "validation_loaded": False, "test_loaded": False,
              "batch_metadata": batch_metadata,
              "notes": ["Endpoint local raw-gradient geometry; not Adam-conditioned updates",
                        "FP32/expert-dropout-off vs AMP/expert-dropout-on is sensitivity, not factor isolation",
                        "Unreliable Face shadow loss is never part of the training objective",
                        "Independent Router blocks are not shared-parameter conflict evidence"]}
    (args.output / "config.json").write_text(json.dumps(header, indent=2) + "\n")
    checkpoint_audits, baseline_gradients = [], {}
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    for mode in MODES:
        config = configs[mode]
        model = build_model(config, args.clip_checkpoint)
        model.assert_visual_frozen()
        if tensor_state_digest(model.visual_encoder.state_dict()) != protocols[mode]["frozen_visual_sha256_before"]:
            raise ValueError("Source CLIP tensor state changed")
        recorded_hashes = read(runs[mode].parent / "full_val/result.json")["checkpoint_sha256"]
        for task in args.tasks:
            checkpoint = runs[mode] / "compact_checkpoints" / f"task{task}.pth"
            checkpoint_sha = _sha256(checkpoint)
            if checkpoint_sha != recorded_hashes[str(task)]:
                raise ValueError("Source checkpoint differs from archived evaluation metadata")
            restore_compact_model_state(model, _torch_load(checkpoint), task, config["git"])
            state_before = tensor_state_digest(model.state_dict())
            pack = ParameterView(model, task)
            for precision in PRECISIONS:
                for batch_id, (cpu_images, cpu_y, _) in enumerate(cache[task]):
                    images = move_model_inputs(cpu_images, torch.device("cuda"))
                    targets = cpu_y.cuda().float()
                    record, common = measure(model, pack, images, targets, config, precision,
                                             args.seed + 1009 * task + 37 * batch_id)
                    record.update(method=mode, task=task, batch=batch_id,
                                  input_tensor_sha256=batch_metadata[task][batch_id]["input_tensor_sha256"])
                    key = (task, precision, batch_id)
                    if mode == "baseline":
                        baseline_gradients[key] = common
                    else:
                        record["fresh_B0_parameter_coordinate_alignment"] = {
                            name: pair_metrics(common[name], baseline_gradients[key][name])
                            for name in common}
                    path = args.output / f"{mode}_task{task}_{precision}_batch{batch_id}.json"
                    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
                    print(f"GRADIENT_BATCH method={mode} task={task} precision={precision} batch={batch_id} samples={len(targets)} fused={record['loss_values']['fused']:.7f}", flush=True)
                    del images, targets, record, common
                    gc.collect()
            unchanged = state_before == tensor_state_digest(model.state_dict())
            checkpoint_unchanged = checkpoint_sha == _sha256(checkpoint)
            if not unchanged or not checkpoint_unchanged:
                raise AssertionError("Read-only gradient audit modified model/checkpoint")
            checkpoint_audits.append({"method": mode, "task": task,
                                      "checkpoint_sha256": checkpoint_sha,
                                      "model_state_sha256_before": state_before,
                                      "model_state_unchanged": unchanged,
                                      "checkpoint_unchanged": checkpoint_unchanged,
                                      "all_parameter_grad_buffers_none": all(p.grad is None for p in model.parameters())})
        del model, pack
        gc.collect()
        torch.cuda.empty_cache()
    for task in args.tasks:
        for i, (images, targets, _) in enumerate(cache[task]):
            if tensor_state_digest(images) != batch_metadata[task][i]["input_tensor_sha256"]:
                raise AssertionError("Cached training pixels changed")
            if tensor_state_digest({"targets": targets}) != batch_metadata[task][i]["target_sha256"]:
                raise AssertionError("Cached training targets changed")
    result = {"status": "complete", "elapsed_seconds": time.time() - start,
              "snapshots": len(MODES) * len(args.tasks) * len(PRECISIONS) * args.batches,
              "training_updates": 0, "checkpoint_audits": checkpoint_audits,
              "input_tensors_unchanged": True, "test_loaded": False,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("GRADIENT_ATTRIBUTION_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--face-manifest-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 2, 7])
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if any(t not in (0, 2, 7) for t in args.tasks) or args.seed != 0:
        parser.error("Audit is locked to seed0 and task0/2/7")
    run(args)


if __name__ == "__main__":
    main()
