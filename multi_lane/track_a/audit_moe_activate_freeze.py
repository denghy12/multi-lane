"""Boundary audits for expert immutability and old-lane feature drift."""
from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader

from .paired_transforms import move_model_inputs
from .runner import LabelView, isolated_rng, tensor_state_digest


class ActivateFreezeAudit:
    def __init__(self, model, train_source, output, device):
        self.model, self.bank = model, model.adapter_bank
        self.output, self.device = output, device
        self.rows = {}
        indices = list(range(min(16, len(train_source))))
        with isolated_rng(device):
            loader = DataLoader(LabelView(train_source, indices, tuple(range(model.num_classes))),
                                batch_size=len(indices), shuffle=False, num_workers=0)
            images, _ = next(iter(loader))
        self.images = move_model_inputs(images, device)
        self.probe = dict(split="train", samples=len(indices),
                          sample_ids=[str(train_source.sample_ids[i]) for i in indices],
                          input_sha256=tensor_state_digest(images),
                          purpose="fixed_training_images_old_lane_drift_only_not_freeze_selection")

    def protected_state(self, task):
        state = {}
        for layer, experts in enumerate(self.bank.experts):
            for i, expert in enumerate(experts):
                if bool(self.bank.frozen_experts[layer, i]):
                    state.update({f"expert.{layer}.{i}.{n}": v for n, v in expert.state_dict().items()})
        for t in range(task):
            state.update({f"router.{t}.{n}": v for n, v in self.bank.tasks[t].state_dict().items()})
        return tensor_state_digest(state)

    @torch.no_grad()
    def features(self):
        training = self.model.training
        counts_before = self.bank.training_counts.clone()
        mass_before = self.bank.training_gate_mass.clone() if self.bank.freeze_metric == "gate_mass" else None
        self.model.eval()
        try:
            with isolated_rng(self.device):
                value = self.model.encode_lanes(self.images, all_seen_lanes=True).float().cpu()
        finally:
            self.model.train(training)
        if not torch.equal(counts_before, self.bank.training_counts):
            raise AssertionError("Audit inference contaminated freeze counts")
        if mass_before is not None and not torch.equal(mass_before, self.bank.training_gate_mass):
            raise AssertionError("Audit inference contaminated gate-mass statistics")
        return value

    def begin(self, task):
        self.before_hash = self.protected_state(task)
        self.before = self.features() if task else None
        self.frozen_before = self.bank.frozen_experts.clone()

    def finish(self, task):
        after_hash = self.protected_state(task)
        if after_hash != self.before_hash:
            raise AssertionError("Frozen expert or historical Router weights changed")
        drift = {}
        if task:
            after = self.features()[:, :task]
            for old in range(task):
                delta = after[:, old] - self.before[:, old]
                drift[str(old)] = dict(max_abs=float(delta.abs().max()),
                                      relative_l2=float(delta.norm()/self.before[:, old].norm().clamp_min(1e-12)))
        freeze = self.bank.finish_task(task)
        self.rows[str(task)] = dict(
            freeze=freeze, frozen_experts_and_old_routers_unchanged=True,
            protected_sha256_before=self.before_hash, protected_sha256_after=after_hash,
            frozen_before=self.frozen_before.tolist(),
            old_lane_feature_drift=drift,
            trainable_adapter_parameters_after_boundary=sum(p.numel() for p in self.bank.active_parameters()),
        )
        payload = dict(policy="cumulative_per_task_training_gate_mass" if self.bank.freeze_metric == "gate_mass" else f"cumulative_per_task_training_hard_top{self.bank.top_k}_frequency",
                       freeze_metric=self.bank.freeze_metric,
                       face_counting="reliable_only", count_views="sum_not_per_view_topk", router_top_k=self.bank.top_k,
                       freeze_per_layer=self.bank.freeze_count, can_reselect_already_frozen=True,
                       frozen_expert_dropout="follows_model_training_mode_weights_only_frozen",
                       probe=self.probe, tasks=self.rows)
        (self.output / "activate_freeze_audit.json").write_text(json.dumps(payload, indent=2)+"\n")
        print(f"ACTIVATE_FREEZE task={task} freeze={json.dumps(freeze)} protected_sha_unchanged=1 old_lane_drift={json.dumps(drift)}", flush=True)
