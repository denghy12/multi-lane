from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from multi_lane.track_a.adapter import TaskImageTokenAdapterBank
from multi_lane.track_a.static_moe import StaticMoEImageTokenAdapterBank, OutputExpertPool, ROUTES
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import compact_model_state_dict, parse_args
from multi_lane.track_a.static_moe_audit import adapter_old_state, validate_static_config
from multi_lane.track_a.evaluate_static_moe import preceding_validation, build_static_model
from multi_lane.track_a.export_compact_test_scores import restore_compact_model_state
from test_track_a_reproduction import FakeVisual


class StaticMoETest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

