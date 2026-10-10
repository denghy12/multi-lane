"""Finite, task-boundary audits; no teacher and no live training monitor."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from .paired_transforms import move_model_inputs
from .post_task_calibration import isolated_rng, parameter_hash


@contextmanager
def audit_precision():
    """Use full FP32 for alignment audits, restoring training TF32 flags."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class ProtectedRouteAudit:
    def __init__(self, output: Path, device: torch.device):
        self.output, self.device = output, device
        self.images = None
        self.sample_ids = []
        self.references, self.rows = {}, {}
        self.protected_before = {}
        self.initial_difference = 0.0
        self.identity = {}

    def _identity_check(self, model, images, task):
        """Separate exact cached-input identity from independent CUDA forwards."""
        bank = model.parax_bank
        result = {"schema_version": 2, "passed": False, "views": {}}
        self.identity = result
        projection = bank.task_projections[task]
        for name, parameter in projection.named_parameters():
            if not bool(torch.isfinite(parameter).all()) or bool(torch.count_nonzero(parameter)):
                raise RuntimeError(f"New task output projection is not strictly zero: {name}")
        result["projection_strictly_zero"] = True
        model.set_parax_runtime_enabled(False)
        fused, features = model.encode_lanes_with_views(images, False)
        lane_ids = model._lane_ids(False)
        baseline, _ = model._lane_logits_with_views(fused, features, images, lane_ids)
        routed = {}
        for name, before in features.items():
            after, gate = bank(bank.layer_indices[0], before, view_name=name, task_id=task)
            if not bool(torch.isfinite(after).all()) or not torch.equal(before, after):
                raise RuntimeError(f"New task raw ParaX residual is not strictly zero: {name}")
            if not bool(torch.isfinite(gate).all()) or bool((gate[:, ~bank.expert_access[task]] != 0).any()):
                raise RuntimeError(f"Invalid initial expert gate/access: {name}")
            routed[name] = model.route_final_task_features(before, lane_ids, name, record=False)
            feature_delta = float((routed[name] - before).abs().max())
            second_norm_delta = float((F.normalize(after, dim=-1) - before).abs().max())
            result["views"][name] = {"raw_residual_max": 0.0,
                "identity_feature_max": feature_delta,
                "plain_second_normalize_feature_max": second_norm_delta}
            if not torch.equal(routed[name], before):
                raise RuntimeError(f"Cached routed features are not exact identity: {name}")
        routed_fused, _ = model.view_fusion_module(routed, lane_ids, images.get("face_reliable"))
        actual, _ = model._lane_logits_with_views(routed_fused, routed, images, lane_ids)
        result["cached_logit_max_difference"] = float((actual - baseline).abs().max())
        if not torch.equal(baseline, actual):
            raise RuntimeError("Cached-input zero residual changed logits")

        # Independent full forwards need a measured numerical floor. They do
        # not replace the exact raw-residual/feature/parameter checks above.
        disabled = [model.current_all_logits(images) for _ in range(3)]
        model.set_parax_runtime_enabled(True)
        enabled = [model.current_all_logits(images) for _ in range(3)]
        all_logits = disabled + enabled
        if any(not bool(torch.isfinite(value).all()) for value in all_logits):
            raise FloatingPointError("Non-finite initial full-forward logits")
        repeat = lambda rows: max(float((a - b).abs().max())
                                  for i, a in enumerate(rows) for b in rows[i + 1:])
        off_floor, on_floor = repeat(disabled), repeat(enabled)
        difference = max(float((on - off).abs().max()) for on, off in zip(enabled, disabled))
        magnitude = max(float(value.abs().max()) for value in all_logits)
        budget = 4 * max(off_floor, on_floor) + 8 * torch.finfo(torch.float32).eps * max(1., magnitude)
        result.update(disabled_repeat_logit_max=off_floor, enabled_repeat_logit_max=on_floor,
                      full_forward_logit_max_difference=difference,
                      full_forward_repeat_budget=budget,
                      repeat_budget_rule="4*max(off_repeat,on_repeat)+8*FP32_eps*max(1,logit_magnitude)")
        if difference > budget:
            raise RuntimeError(f"Initial full-forward mismatch exceeds measured repeat budget: {difference} > {budget}")
        result["passed"] = True
        return result

    def _capture(self, model):
        training = model.training
        model.eval()
        try:
            with isolated_rng(self.device), audit_precision(), torch.no_grad():
                images = move_model_inputs(self.images, self.device)
                fused, features = model.encode_lanes_with_views(images, True)
                lane_ids = model._lane_ids(True)
                logits, _ = model._lane_logits_with_views(fused, features, images, lane_ids)
                masks = model.task_class_mask[lane_ids].to(logits)
                logits = (logits * masks.unsqueeze(0)).sum(dim=1)[:, :model.seen_classes]
                return {
                    "logits": logits.float().cpu(),
                    "features": {k: v.float().cpu() for k, v in features.items()},
                    "gates": {f"{r['layer_id']}:{r['view']}": r['gates'].cpu().clone()
                              for r in model._parax_gate_records},
                    "diagnostics": model.parax_gate_diagnostics(),
                }
        finally:
            model.train(training)

    def _protected_state(self, model, task):
        result = ({f"parax.{k}": v for k, v in model.parax_bank.protected_state(task).items()}
                  if model.parax_bank is not None else {})
        old_classes = sum(model.task_sizes[:task])
        result["old_selectors"] = model.selectors[:task]
        for index, prompt in enumerate(model.prompts):
            result[f"old_prompts.{index}"] = prompt[:, :task]
        result["old_head.weight"] = model.head.weight[:old_classes]
        if model.head.bias is not None:
            result["old_head.bias"] = model.head.bias[:old_classes]
        result["old_task_class_mask"] = model.task_class_mask[:task]
        return {k: v.detach().cpu().clone() for k, v in result.items()}

    def begin(self, model, loader, task):
        if self.images is None:
            with isolated_rng(self.device):
                images, _, ids = next(iter(loader))
            self.images = {k: v[:16].detach().cpu().clone() for k, v in images.items()}
            self.sample_ids = [str(value) for value in ids[:16]]
            torch.save({"images": self.images, "sample_ids": self.sample_ids,
                        "source_split": "val", "optimization_use": False}, self.output / "fixed_anchor_inputs.pt")
        bank = model.parax_bank
        self.protected_before = self._protected_state(model, task)
        self.initial_difference = 0.0
        self.identity = {"schema_version": 2, "passed": True, "enabled": bank is not None}
        if bank is not None:
            training, runtime = model.training, model.parax_runtime_enabled
            model.eval()
            try:
                with isolated_rng(self.device), audit_precision(), torch.no_grad():
                    images = move_model_inputs(self.images, self.device)
                    self.identity = self._identity_check(model, images, task)
                    self.initial_difference = self.identity["full_forward_logit_max_difference"]
            finally:
                model.set_parax_runtime_enabled(runtime)
                model.train(training)
                (self.output / f"identity_task{task}.json").write_text(json.dumps(self.identity, indent=2) + '\n')

    def finish(self, model, task):
        bank = model.parax_bank
        after = self._protected_state(model, task)
        changes = {k: float((v.float() - after[k].float()).abs().max()) if v.numel() else 0.0
                   for k, v in self.protected_before.items()}
        unchanged = self.protected_before.keys() == after.keys() and all(
            torch.equal(v, after[k]) for k, v in self.protected_before.items())
        if not unchanged:
            raise RuntimeError(f"Protected old ParaX state changed: {changes}")
        policy = bank.policy_summary(task) if bank is not None else None
        if bank is not None:
            bank.seal_task(task)
        current = self._capture(model)
        drifts = {}
        for old_task, reference in self.references.items():
            offset = sum(model._task_sizes[:old_task])
            stop = offset + model._task_sizes[old_task]
            delta = current['logits'][:, offset:stop] - reference['logits'][:, offset:stop]
            view_drifts = {}
            for name, features in reference['features'].items():
                old = features[:, old_task]
                now = current['features'][name][:, old_task]
                view_drifts[name] = {"mean_cosine_distance": float((1 - F.cosine_similarity(old, now)).mean()),
                                     "max_absolute_difference": float((old - now).abs().max())}
            gate_drifts = {key: float((value - current['gates'][key]).abs().max())
                           for key, value in reference['gates'].items() if key.startswith(f"{old_task}:")}
            drifts[str(old_task)] = {"logit_mean_absolute_difference": float(delta.abs().mean()),
                                     "logit_max_absolute_difference": float(delta.abs().max()),
                                     "views": view_drifts, "gate_max_absolute_difference": gate_drifts}
        if bank is not None:
            for key, gate in current['gates'].items():
                route_task = int(key.split(':')[0])
                access = bank.expert_access[route_task].cpu()
                if bool((gate[:, ~access] != 0).any()):
                    raise RuntimeError("An old route accessed an ineligible or future expert")
        self.rows[str(task)] = {
            "initial_logit_max_difference": self.initial_difference,
            "identity_audit": self.identity,
            "protected_parameter_hash_before": parameter_hash(sorted(self.protected_before.items())),
            "protected_parameter_hash_after": parameter_hash(sorted(after.items())),
            "protected_tensors_unchanged": unchanged,
            "protected_tensor_max_absolute_changes": changes,
            "policy_before_sealing": policy, "old_task_anchor_drift": drifts,
            "diagnostics": current['diagnostics'],
        }
        self.references[task] = current
        torch.save(current, self.output / f"fixed_anchor_after_task{task}.pt")
        (self.output / "protected_route_audit.json").write_text(json.dumps({
            "anchor_split": "val", "anchor_sample_ids": self.sample_ids,
            "anchor_used_for_training": False, "FP32_evaluation": True, "TF32_evaluation": False, "tasks": self.rows,
        }, indent=2) + '\n')
