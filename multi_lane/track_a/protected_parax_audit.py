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
        if bank is not None:
            training, runtime = model.training, model.parax_runtime_enabled
            model.eval()
            try:
                with isolated_rng(self.device), audit_precision(), torch.no_grad():
                    images = move_model_inputs(self.images, self.device)
                    model.set_parax_runtime_enabled(False)
                    baseline = model.current_all_logits(images)
                    model.set_parax_runtime_enabled(True)
                    actual = model.current_all_logits(images)
                    self.initial_difference = float((actual - baseline).abs().max().cpu())
                    if self.initial_difference > 1e-5:
                        raise RuntimeError(f"New task ParaX path does not start at identity: max_difference={self.initial_difference}")
            finally:
                model.set_parax_runtime_enabled(runtime)
                model.train(training)

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
