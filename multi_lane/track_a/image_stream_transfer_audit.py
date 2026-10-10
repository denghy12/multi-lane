"""Matched transfer of the joint26 image-stream configuration; no protection of ParaX."""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F

from .paired_transforms import move_model_inputs
from .post_task_calibration import isolated_rng, parameter_hash
from .protected_parax_audit import audit_precision


def validate_transfer_config(args):
    required = dict(training_protocol="incremental", adapter_mode="disabled",
                    view_fusion="fixed_three_view", view_classifier_mode="shared_post_fusion",
                    selector_mode="shared", prompt_mode="shared", selector_conditioning="disabled",
                    view_gradient_routing="joint", loss_routing="joint_bce", reporting_split="val",
                    calibration_fraction=0, crossfit_folds=0, protected_parax_paired_audit=False,
                    fixed_view_paired_audit=False, view_auxiliary_loss_weight=.1,
                    supervised_loss_scale=1., skip_validation_eval=True)
    if any(getattr(args, key) != value for key, value in required.items()):
        raise ValueError("Image-stream transfer requires shared fixed three-view incremental paths and task-end validation")
    if args.parax_mode not in {"disabled", "image"}:
        raise ValueError("Transfer uses the original Frozen Forward image-stream ParaX only")
    if args.parax_mode == "image":
        expected = dict(parax_rank=32, parax_num_experts=3, parax_router_hidden=16,
                        parax_initialization="official", parax_residual_scale=.1,
                        parax_output_scale_mode="learnable", parax_trainable_components="all",
                        parax_freeze_center_after_task0=False, parax_task_local_gate=False,
                        parax_level_conditioned=False, parax_residual_ratio_cap=0.,
                        parax_smooth_ratio_bound=0., parax_projector_bottleneck_dim=0,
                        parax_distillation_weight=0., parax_residual_penalty_weight=0.)
        if tuple(args.parax_layer_indices) != (10,) or any(getattr(args, key) != value for key, value in expected.items()):
            raise ValueError("Transfer must preserve the locked joint26 image-stream ParaX configuration")


def old_lane_state(model, task):
    """Old task state excludes the intentionally live shared image transform."""
    count = sum(model.task_sizes[:task])
    state = {"selectors": model.selectors[:task], "head.weight": model.head.weight[:count],
             "class_mask": model.task_class_mask[:task]}
    if model.head.bias is not None:
        state["head.bias"] = model.head.bias[:count]
    state.update({f"prompt.{i}": prompt[:, :task] for i, prompt in enumerate(model.prompts)})
    return {key: value.detach().cpu().clone() for key, value in state.items()}


class ImageStreamTransferAudit:
    def __init__(self, output, device):
        self.output, self.device = output, device
        self.images = None
        self.references, self.rows = {}, {}

    def begin(self, model, loader, task):
        if self.images is None:
            with isolated_rng(self.device):
                images, _, ids = next(iter(loader))
            self.images = {key: value[:16].detach().cpu().clone() for key, value in images.items()}
            self.sample_ids = [str(value) for value in ids[:16]]
            torch.save(dict(images=self.images, sample_ids=self.sample_ids, source_split="val",
                            optimization_use=False), self.output/"fixed_anchor_inputs.pt")
        self.before = old_lane_state(model, task)
        self.bank_before = parameter_hash(model.parax_bank.named_parameters()) if model.parax_bank else None

    def finish(self, model, task):
        after = old_lane_state(model, task)
        if self.before.keys() != after.keys() or any(not torch.equal(value, after[key]) for key, value in self.before.items()):
            raise RuntimeError("Image-stream transfer modified old task parameters")
        training = model.training
        model.eval()
        try:
            with isolated_rng(self.device), audit_precision(), torch.no_grad():
                images = move_model_inputs(self.images, self.device)
                fused, features = model.encode_lanes_with_views(images, True)
                lanes = model._lane_ids(True)
                logits, _ = model._lane_logits_with_views(fused, features, images, lanes)
                logits = (logits * model.task_class_mask[lanes].to(logits).unsqueeze(0)).sum(dim=1)
                current = dict(logits=logits.float().cpu(), features={key: value.float().cpu() for key, value in features.items()})
        finally:
            model.train(training)
        drift = {}
        for old, reference in self.references.items():
            start, stop = sum(model.task_sizes[:old]), sum(model.task_sizes[:old+1])
            delta = current["logits"][:, start:stop]-reference["logits"][:, start:stop]
            drift[str(old)] = dict(logit_mean_absolute_difference=float(delta.abs().mean()),
                                   logit_max_absolute_difference=float(delta.abs().max()), views={})
            for view, value in reference["features"].items():
                previous, now = value[:, old], current["features"][view][:, old]
                drift[str(old)]["views"][view] = dict(mean_cosine_distance=float((1-F.cosine_similarity(previous, now)).mean()),
                                                     max_absolute_difference=float((previous-now).abs().max()))
        self.references[task] = current
        bank_after = parameter_hash(model.parax_bank.named_parameters()) if model.parax_bank else None
        self.rows[str(task)] = dict(old_lane_parameters_unchanged=True,
                                    old_lane_hash_before=parameter_hash(self.before.items()),
                                    old_lane_hash_after=parameter_hash(after.items()),
                                    shared_parax_hash_before=self.bank_before, shared_parax_hash_after=bank_after,
                                    shared_parax_changed=self.bank_before != bank_after,
                                    fixed_anchor_old_task_drift=drift)
        (self.output/"image_stream_transfer_audit.json").write_text(json.dumps(dict(
            sample_ids=self.sample_ids, source_split="val", precision="FP32_TF32_off",
            tasks=self.rows), indent=2)+"\n")
