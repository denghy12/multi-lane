"""Paired task-owned state, identity and fixed-cohort audits for static MoE v1."""
from __future__ import annotations

import json
import torch

from .image_stream_transfer_audit import ImageStreamTransferAudit
from .paired_transforms import move_model_inputs
from .post_task_calibration import isolated_rng, parameter_hash
from .protected_parax_audit import audit_precision
from .static_moe import StaticMoEImageTokenAdapterBank


def validate_static_config(args):
    required = dict(training_protocol="incremental", parax_mode="disabled",
                    view_fusion="fixed_three_view", view_classifier_mode="shared_post_fusion",
                    selector_mode="shared", prompt_mode="shared", selector_conditioning="disabled",
                    view_gradient_routing="joint", loss_routing="joint_bce", reporting_split="val",
                    calibration_fraction=0., crossfit_folds=0, view_auxiliary_loss_weight=.1,
                    supervised_loss_scale=1., skip_validation_eval=True, also_report_test=False,
                    image_stream_transfer_audit=False, fixed_view_paired_audit=False,
                    protected_parax_paired_audit=False, adapter_view_bottleneck_dim=0,
                    adapter_view_mode="shared", adapter_residual_gate_mode="fixed",
                    adapter_task_init="independent", adapter_regularization="none",
                    save_compact_checkpoints=True, save_evaluation_scores=True)
    required.update(seed=0, max_tasks=8, epochs=30, train_batch_size=64, eval_batch_size=64,
                    adapter_bottleneck_dim=48, adapter_residual_scale=.03, adapter_activation="relu",
                    adapter_learning_rate=.0004, adapter_weight_decay=0., source_learning_rate=.05,
                    source_reference_batch_size=256, weight_decay=0., scheduler_mode="cosine",
                    scheduler_min_lr_ratio=0., scheduler_warmup_ratio=0.,
                    amp_initial_scale=1024., amp_growth_interval=1000000000)
    if any(getattr(args, key) != value for key, value in required.items()):
        raise ValueError("Static MoE v1 requires paired shared three-view validation with task-owned experts")
    if args.adapter_mode not in ("disabled", "image_token"):
        raise ValueError("Static MoE audit supports the baseline or image-token adapters")
    if args.static_moe_mode != "disabled" and args.adapter_mode != "image_token":
        raise ValueError("Static experts require image-token adapters")
    if tuple(args.adapter_layer_indices) != (1,):
        raise ValueError("Static MoE v1 locks the Selector evidence adapter to layer index1")
    if args.static_moe_smoke_samples < 0 or (
        args.static_moe_smoke_samples and not args.optimizer_updates_per_task
    ):
        raise ValueError("Sample-limited smoke needs an explicit optimizer update budget")
    if args.optimizer_updates_by_task is not None:
        raise ValueError("Static MoE v1 uses full epochs or one common smoke update budget")


def adapter_old_state(model, task):
    if model.adapter_bank is None:
        return {}
    bank = model.adapter_bank
    state = {f"task{old}.{key}": value.detach().cpu().clone()
             for old in range(task)
             for key, value in bank.task_adapters[old].state_dict().items()}
    if isinstance(bank, StaticMoEImageTokenAdapterBank):
        state["route_weights"] = bank.route_weights.detach().cpu().clone()
    return state


class StaticMoEAudit(ImageStreamTransferAudit):
    def begin(self, model, loader, task):
        super().begin(model, loader, task)
        self.adapter_before = adapter_old_state(model, task)
        bank = model.adapter_bank
        if bank is not None:
            if any(p.requires_grad for old in range(task) for p in bank.task_adapters[old].parameters()):
                raise RuntimeError("An old adapter/expert is trainable")
            if isinstance(bank, StaticMoEImageTokenAdapterBank):
                bank.validate_routes()
            self.current_before = parameter_hash(bank.task_adapters[task].state_dict().items())
        else:
            self.current_before = None
        training, runtime = model.training, model.adapter_runtime_enabled
        self.identity = dict(raw_token_max_difference=0., full_forward_max_logit_difference=0.)
        model.eval()
        try:
            with isolated_rng(self.device), audit_precision(), torch.no_grad():
                images = move_model_inputs(self.images, self.device)
                if bank is not None:
                    tokens = torch.arange(6 * model.width, device=self.device, dtype=torch.float32).reshape(2, 3, model.width) / model.width
                    for layer in bank.layer_indices:
                        for view in ("full", "person", "face"):
                            adapted = bank.adapted_tokens_for_layer(layer, tokens, (task,), view)[0]
                            if not torch.equal(tokens, adapted):
                                raise RuntimeError("New task adapter is not exact raw-token identity")
                    model.set_adapter_runtime_enabled(False)
                    baseline = model.current_all_logits(images)
                    model.set_adapter_runtime_enabled(True)
                    actual = model.current_all_logits(images)
                    difference = float((actual - baseline).abs().max())
                    if not torch.isfinite(actual).all() or difference > 1e-4:
                        raise RuntimeError(f"Initial independent-forward identity exceeds FP32 tolerance: {difference}")
                    self.identity["full_forward_max_logit_difference"] = difference
        finally:
            model.set_adapter_runtime_enabled(runtime)
            model.train(training)

    def finish(self, model, task):
        after = adapter_old_state(model, task)
        if self.adapter_before.keys() != after.keys() or any(
            not torch.equal(value, after[key]) for key, value in self.adapter_before.items()
        ):
            raise RuntimeError("Training changed an old expert or static routing table")
        bank = model.adapter_bank
        if isinstance(bank, StaticMoEImageTokenAdapterBank):
            bank.validate_routes()
        if bank is not None:
            if any(p.requires_grad for old in range(task) for p in bank.task_adapters[old].parameters()):
                raise RuntimeError("Training unfroze an old expert")
            current_after = parameter_hash(bank.task_adapters[task].state_dict().items())
            if current_after == self.current_before:
                raise RuntimeError("Current adapter/expert pool did not learn")
        else:
            current_after = None
        super().finish(model, task)
        row = self.rows[str(task)]
        if any(r["logit_max_absolute_difference"] > 1e-4
               or any(abs(v["mean_cosine_distance"]) > 1e-5 for v in r["views"].values())
               for r in row["fixed_anchor_old_task_drift"].values()):
            raise RuntimeError("Old predictions drifted despite task-owned experts")
        row.update(adapter_old_state_unchanged=True,
                   adapter_old_sha256_before=parameter_hash(self.adapter_before.items()),
                   adapter_old_sha256_after=parameter_hash(after.items()),
                   current_adapter_sha256_before=self.current_before,
                   current_adapter_sha256_after=current_after,
                   current_adapter_learned=bank is not None,
                   current_adapter_trainable_parameters=sum(p.numel() for p in model.adapter_optimizer_parameters()),
                   identity=self.identity, static_moe_mode=model.static_moe_mode)
        (self.output / "static_moe_audit.json").write_text(json.dumps(dict(
            status="passed", source_split="val", optimization_use=False,
            precision="FP32_TF32_off", old_logit_tolerance=1e-4,
            sample_ids=self.sample_ids, tasks=self.rows), indent=2) + "\n")

