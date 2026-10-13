from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from multi_lane.track_a.adapter import TaskImageTokenAdapterBank, bound_residual_to_input
from multi_lane.track_a.static_moe import StaticMoEImageTokenAdapterBank, OutputExpertPool, ROUTES
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import compact_model_state_dict, parse_args, TASK_SIZES, CLASS_ORDER
from multi_lane.track_a.static_moe_audit import adapter_old_state, validate_static_config
from multi_lane.track_a.evaluate_static_moe import preceding_validation, build_static_model, audit_source, audit_pairing, METHODS
from multi_lane.track_a.export_compact_test_scores import restore_compact_model_state
from test_track_a_reproduction import FakeVisual


class StaticMoETest(unittest.TestCase):
    def test_smooth_residual_bound_is_exact_and_identity_grad_at_zero(self):
        x = torch.randn(2, 4, 8)
        d = torch.randn_like(x, requires_grad=True)
        ratio = 0.1
        bounded = bound_residual_to_input(d, x, ratio)
        self.assertTrue(torch.all(bounded.norm(dim=-1) <= ratio * x.norm(dim=-1) + 1e-6))
        zero = torch.zeros_like(x, requires_grad=True)
        out = bound_residual_to_input(zero, x, ratio)
        self.assertTrue(torch.equal(out, zero))
        out.sum().backward()
        self.assertTrue(torch.equal(zero.grad, torch.ones_like(zero)))
        self.assertTrue(torch.equal(bound_residual_to_input(d.detach(), x, 0), torch.zeros_like(x)))

    def bank(self, routing="overlap", width=8, tasks=2):
        return StaticMoEImageTokenAdapterBank(tasks, width, 6, (0,), routing=routing)

    def model(self, mode="overlap"):
        return MultiLaneModel(FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
                              num_prompt_layers=1, adapter_mode="image_token",
                              adapter_bottleneck_dim=6, adapter_layer_indices=(0,),
                              adapter_residual_scale=.03, static_moe_mode=mode,
                              view_fusion="fixed_three_view")

    def images(self):
        images = {v: torch.randn(3, 3, 4, 4) for v in ("full", "person", "face")}
        images["face_reliable"] = torch.tensor([True, False, True])
        return images

    def test_exact_budget_without_hidden_unused_bank(self):
        shared = TaskImageTokenAdapterBank(8, 768, 48, (1,), residual_scale=.03)
        bank = StaticMoEImageTokenAdapterBank(8, 768, 48, (1,))
        self.assertEqual(bank.per_task_parameter_counts(), (74544,) * 8)
        self.assertEqual(bank.total_parameter_count(), shared.total_parameter_count())
        self.assertEqual(bank.total_parameter_count(), sum(p.numel() for p in bank.parameters()))

    def test_zero_identity_and_routing_gradient_support(self):
        bank = self.bank()
        bank.activate_task(0)
        x = torch.ones(2, 3, 8)
        with torch.no_grad():
            for expert in bank.task_adapters[0]["0"].experts:
                expert.down.weight.fill_(.1)
        y = bank.adapted_tokens_for_layer(0, x, (0,), "full")[0]
        self.assertTrue(torch.equal(x, y))
        y.sum().backward()
        experts = bank.task_adapters[0]["0"].experts
        self.assertGreater(float(experts[0].up.weight.grad.norm()), 0)
        self.assertGreater(float(experts[1].up.weight.grad.norm()), 0)
        self.assertEqual(float(experts[2].up.weight.grad.norm()), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in bank.active_parameters() if p.grad is not None))

    def test_complete_expert_output_combination(self):
        pool = OutputExpertPool(3, 1, "relu")
        with torch.no_grad():
            for i, expert in enumerate(pool.experts):
                expert.down.weight.fill_(i + 1.)
                expert.down.bias.fill_(.2 * i)
                expert.up.weight.fill_(2. - i)
            pool.output_bias.fill_(.7)
        x = torch.tensor([[[1., -1., 2.]]])
        weights = torch.tensor([.2, .3, .5])
        expected = sum(weights[i] * expert(x) for i, expert in enumerate(pool.experts)) + pool.output_bias
        torch.testing.assert_close(pool(x, weights), expected)

    def test_uniform_and_overlap_reuse_same_pool_with_different_outputs(self):
        uniform = self.bank("uniform")
        overlap = self.bank("overlap")
        parameters = dict(uniform.named_parameters())
        with torch.no_grad():
            for name, parameter in overlap.named_parameters():
                parameter.copy_(parameters[name])
            for bank in (uniform, overlap):
                for i, expert in enumerate(bank.task_adapters[0]["0"].experts):
                    expert.up.weight.fill_(i + .1)
        for bank in (uniform, overlap):
            bank.activate_task(0)
        x = torch.randn(2, 4, 8)
        a = uniform.adapted_tokens_for_layer(0, x, (0,), "full")
        b = uniform.adapted_tokens_for_layer(0, x, (0,), "face")
        self.assertTrue(torch.equal(a, b))
        c = overlap.adapted_tokens_for_layer(0, x, (0,), "full")
        d = overlap.adapted_tokens_for_layer(0, x, (0,), "face")
        self.assertFalse(torch.equal(c, d))
        self.assertFalse(torch.equal(a, c))

    def test_old_experts_exact_under_adamw_and_new_task_learning(self):
        bank = self.bank()
        bank.activate_task(0)
        x = torch.randn(2, 3, 8)
        def update():
            opt = torch.optim.AdamW(bank.active_parameters(), lr=.1, weight_decay=.2)
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                bank.adapted_tokens_for_layer(0, x, (bank.current_task_id,), "full").square().sum().backward()
                opt.step()
        update()
        before = {k: v.clone() for k, v in bank.task_adapters[0].state_dict().items()}
        old_output = bank.adapted_tokens_for_layer(0, x, (0,), "face").detach().clone()
        bank.activate_task(1)
        update()
        self.assertTrue(all(torch.equal(value, bank.task_adapters[0].state_dict()[key]) for key, value in before.items()))
        self.assertTrue(torch.equal(old_output, bank.adapted_tokens_for_layer(0, x, (0,), "face")))
        self.assertFalse(any(p.requires_grad for p in bank.task_adapters[0].parameters()))
        self.assertGreater(float(bank.task_adapters[1]["0"].experts[0].up.weight.norm()), 0)

    def test_future_task_and_modified_routing_rejected(self):
        bank = self.bank()
        bank.activate_task(0)
        with self.assertRaises(ValueError):
            bank.adapted_tokens_for_layer(0, torch.ones(1, 2, 8), (1,), "full")
        with torch.no_grad():
            bank.route_weights[0, 0] = .2
        with self.assertRaises(RuntimeError):
            bank.validate_routes()

    def test_model_initialization_rng_matches_baseline(self):
        def construct(mode, enabled):
            torch.manual_seed(17)
            model = MultiLaneModel(FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
                num_prompt_layers=1, view_fusion="fixed_three_view",
                adapter_mode="image_token" if enabled else "disabled", adapter_bottleneck_dim=6,
                adapter_layer_indices=(0,), static_moe_mode=mode)
            return model, torch.get_rng_state()
        base, rng = construct("disabled", False)
        a, arng = construct("uniform", True)
        b, brng = construct("overlap", True)
        self.assertTrue(torch.equal(rng, arng) and torch.equal(rng, brng))
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, a.state_dict()[key]), key)
        for key, value in a.adapter_bank.named_parameters():
            self.assertTrue(torch.equal(value, dict(b.adapter_bank.named_parameters())[key]), key)

    def test_model_old_logits_face_mask_and_frozen_visual(self):
        model = self.model()
        images = self.images()
        model.activate_task(0)
        model.eval()
        old = model.seen_logits(images).detach().clone()
        model.activate_task(1)
        opt = torch.optim.Adam(model.adapter_optimizer_parameters(), lr=.1)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            model.current_logits(images).square().sum().backward()
            opt.step()
        actual = model.seen_logits(images).detach()
        torch.testing.assert_close(actual[:, :2], old, atol=1e-6, rtol=1e-6)
        self.assertEqual(tuple(actual.shape), (3, 3))
        self.assertEqual(float(model.last_fusion_weights()[1, :, 2].abs().max()), 0.)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.visual_encoder.parameters()))

    def test_compact_restore_preserves_all_seen_outputs_and_routes(self):
        model = self.model()
        model.activate_task(0)
        model.activate_task(1)
        with torch.no_grad():
            for task in model.adapter_bank.task_adapters:
                for expert in task["0"].experts:
                    expert.up.weight.normal_(std=.01)
        model.eval()
        images = self.images()
        expected = model.seen_logits(images).detach()
        git = dict(commit="test", branch="exp/test", dirty=False)
        payload = dict(model=compact_model_state_dict(model), task_id=1, source_git=git, schema_version=1)
        restored = self.model()
        restored.visual_encoder.load_state_dict(model.visual_encoder.state_dict())
        restore_compact_model_state(restored, payload, 1, git)
        restored.eval()
        torch.testing.assert_close(restored.seen_logits(images), expected, atol=0, rtol=0)
        restored.adapter_bank.validate_routes()
        before = adapter_old_state(restored, 1)
        self.assertTrue(any(key.startswith("task0.") for key in before))

    def test_parameter_names_match_optimizer_without_old_experts(self):
        model = self.model()
        model.activate_task(0)
        model.activate_task(1)
        by_name = dict(model.named_parameters())
        expected = {id(by_name[name]) for name in model.optimizer_parameter_names()}
        actual = {id(p) for p in model.optimizer_parameters()}
        self.assertEqual(expected, actual)
        self.assertFalse(any(id(p) in actual for p in model.adapter_bank.task_adapters[0].parameters()))

    def test_invalid_combination_rejected(self):
        with self.assertRaises(ValueError):
            MultiLaneModel(FakeVisual(), (2,), static_moe_mode="overlap")
        with self.assertRaises(ValueError):
            StaticMoEImageTokenAdapterBank(2, 8, 7, (0,))
        with self.assertRaises(ValueError):
            StaticMoEImageTokenAdapterBank(2, 8, 6, (0,), residual_gate_mode="learnable")

    def test_test_requires_complete_full_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                preceding_validation(root, "test", dict(commit="x"))
            (root / "full_val").mkdir()
            (root / "full_val/comparison.json").write_text(json.dumps(dict(status="preflight_passed")))
            with self.assertRaises(ValueError):
                preceding_validation(root, "test", dict(commit="x"))
            with self.assertRaises(ValueError):
                preceding_validation(root, "test", dict(commit="x"), preflight=True)

    def test_parallel_test_does_not_wait_for_validation_but_rejects_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(preceding_validation(root, "test", dict(commit="x"), parallel_splits=True))
            for options in (dict(preflight=True), dict(smoke_updates=2)):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    preceding_validation(root, "test", dict(commit="x"), parallel_splits=True, **options)

    def test_actual_update_budget_schema_and_formal_budget(self):
        for smoke in (0, 2):
            with self.subTest(smoke=smoke), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                c = dict(static_moe_paired_audit=True, training_protocol="incremental",
                         task_sizes=list(TASK_SIZES), class_order=list(CLASS_ORDER), max_tasks=8,
                         epochs_per_task=None if smoke else 30,
                         training_budget_mode="optimizer_updates" if smoke else "epochs",
                         optimizer_updates_per_task=smoke or None, static_moe_smoke_samples=64 if smoke else 0,
                         parax_mode="disabled", view_fusion="fixed_three_view", selector_mode="shared",
                         prompt_mode="shared", adapter_mode="disabled", reporting_split="val",
                         also_report_test=False, git=dict(dirty=False), save_compact_checkpoints=True,
                         train_batch_size=64)
                row = dict(skipped_optimizer_steps=0, amp_loss_scale_end=1024, current_loss=.6,
                           optimizer_steps=2 if smoke else 1)
                history = {str(t): [row] * (1 if smoke else 30) for t in range(8)}
                pa = dict(frozen_visual_unchanged=True, frozen_visual_sha256_before="a",
                          frozen_visual_sha256_after="a", tasks={str(t): dict(train_instances=64) for t in range(8)})
                audit = dict(status="passed", tasks={str(t): dict(old_lane_parameters_unchanged=True,
                    adapter_old_state_unchanged=True, adapter_old_sha256_before="b", adapter_old_sha256_after="b",
                    identity=dict(raw_token_max_difference=0.)) for t in range(8)})
                summary = dict(status="complete", config=c, completed_optimizer_updates=16 if smoke else 240,
                               completed_epochs=8 if smoke else 240)
                for name, value in (("config", c), ("seed_summary", summary), ("paired_protocol_audit", pa),
                                    ("static_moe_audit", audit), ("training_history", history)):
                    (root / (name + ".json")).write_text(json.dumps(value))
                audit_source(root, smoke)
                with self.assertRaises(ValueError):
                    audit_source(root, 0 if smoke else 2)

    def test_paired_runner_schema_and_negative_controls(self):
        common = dict(seed=0, class_order=list(CLASS_ORDER), task_sizes=list(TASK_SIZES), max_tasks=8,
            epochs_per_task=None, train_batch_size=64, eval_batch_size=64, adapter_layer_indices=[1],
            adapter_residual_scale=.03, view_fusion="fixed_three_view", view_auxiliary_loss_weight=.1,
            parameter_group_loss_routing="joint_bce", training_loss_mode="legacy_full_zero",
            learning_rate=.0125, optimizer="Adam_reset_per_task", scheduler_mode="cosine",
            scheduler_min_lr_ratio=0., scheduler_warmup_ratio=0., input_normalization="clip",
            full_crop_mode="legacy", train_crop_scale=[.05, 1.], person_crop_margin=.15,
            amp_initial_scale=1024., amp_growth_interval=1000000000, clip_checkpoint_sha256="clip",
            face_manifest=dict(root="face"), static_moe_smoke_samples=64, optimizer_updates_per_task=2,
            git=dict(commit="a", dirty=False), adapter_learning_rate=.0004, adapter_weight_decay=0.,
            adapter_parameter_counts_per_task=[74544] * 8, initial_adapter_parameters_sha256="experts")
        configs = {}
        for method, mode in zip(METHODS, ("disabled", "disabled", "uniform", "overlap")):
            configs[method] = dict(common, static_moe_mode=mode,
                adapter_mode="disabled" if method == "BASELINE" else "image_token",
                static_moe_route_weights=ROUTES.get(mode))
        tasks = {str(t): dict(train_sample_ids_sha256="ids", validation_sample_ids_sha256="val",
                 first_train_batch_ids_sha256="first", reporting_sample_ids_sha256="report",
                 train_instances=64, sampler_seed=t * 1009) for t in range(8)}
        audits = {m: dict(initial_selector_prompt_classifier_sha256="base", tasks=copy.deepcopy(tasks)) for m in METHODS}
        with patch("multi_lane.track_a.evaluate_static_moe.audit_source",
                   side_effect=lambda source, updates: (configs[source.name], audits[source.name])):
            audit_pairing(Path("unused"), 2)
            audits["MOE_OVERLAP"]["tasks"]["7"]["first_train_batch_ids_sha256"] = "wrong"
            with self.assertRaises(ValueError):
                audit_pairing(Path("unused"), 2)


if __name__ == "__main__":
    unittest.main()

