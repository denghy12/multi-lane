from __future__ import annotations

import copy
import unittest

import torch

from multi_lane.track_a.adapter import ParaXImageAdapterBank
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import build_optimizer_groups
from test_track_a_reproduction import FakeVisual


class ParaXTest(unittest.TestCase):
    def test_post_task_router_keeps_old_lane_fixed_after_new_task_update(self) -> None:
        torch.manual_seed(61)
        visual = FakeVisual()
        common = dict(task_sizes=(2, 1), num_selectors=2, num_prompts=2,
                      num_prompt_layers=1)
        torch.manual_seed(62)
        baseline = MultiLaneModel(copy.deepcopy(visual), **common)
        torch.manual_seed(62)
        routed = MultiLaneModel(
            copy.deepcopy(visual), **common,
            parax_mode="post_task_router", parax_layer_indices=(0,),
            parax_rank=2, parax_num_experts=3, parax_initialization="zero_output",
            parax_residual_scale=0.001, parax_output_scale_mode="fixed",
            parax_trainable_components="router", parax_freeze_center_after_task0=True,
        )
        baseline.activate_task(0)
        routed.activate_task(0)
        images = torch.randn(2, 3, 4, 4)
        torch.testing.assert_close(
            routed.current_all_logits(images), baseline.current_all_logits(images),
            atol=1e-6, rtol=1e-6,
        )
        self.assertTrue(routed.parax_bank.expert_a.requires_grad)
        opt0 = torch.optim.Adam(routed.parax_optimizer_parameters(), lr=0.1)
        opt0.zero_grad()
        routed.current_all_logits(images).square().mean().backward()
        opt0.step()
        routed.activate_task(1)
        self.assertFalse(routed.parax_bank.expert_a.requires_grad)
        self.assertFalse(any(
            p.requires_grad for p in routed.parax_bank.routers["0_0"].parameters()
        ))
        self.assertTrue(all(
            p.requires_grad for p in routed.parax_bank.routers["1_0"].parameters()
        ))
        old_logits = routed.seen_logits(images)[:, :2].detach().clone()
        opt1 = torch.optim.Adam(routed.parax_optimizer_parameters(), lr=0.1)
        opt1.zero_grad()
        routed.current_all_logits(images).square().mean().backward()
        self.assertTrue(any(
            p.grad is not None and bool(torch.isfinite(p.grad).all())
            for p in routed.parax_bank.routers["1_0"].parameters()
        ))
        opt1.step()
        torch.testing.assert_close(
            routed.seen_logits(images)[:, :2], old_logits, atol=1e-6, rtol=1e-6,
        )
        routed.assert_visual_frozen()

    def test_projector_preserves_initial_p10_and_updates_with_expert_center(self) -> None:
        torch.manual_seed(41)
        visual = FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        common = dict(
            task_sizes=(2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1, parax_mode="image",
            parax_layer_indices=(0,), parax_rank=2, parax_num_experts=3,
            parax_initialization="official", parax_residual_scale=0.1,
            parax_trainable_components="all", parax_freeze_center_after_task0=False,
        )
        torch.manual_seed(42)
        plain = MultiLaneModel(copy.deepcopy(visual), **common)
        torch.manual_seed(42)
        projected = MultiLaneModel(
            copy.deepcopy(visual), **common, parax_projector_bottleneck_dim=4,
        )
        plain.activate_task(0)
        projected.activate_task(0)
        images = torch.randn(2, 3, 4, 4)
        torch.testing.assert_close(
            projected.current_all_logits(images), plain.current_all_logits(images),
            atol=1e-6, rtol=1e-6,
        )
        _, _, optimizer_groups = build_optimizer_groups(
            projected, weight_decay=0.0, adapter_learning_rate=4e-4,
        )
        optimizer_ids = [id(p) for group in optimizer_groups for p in group["params"]]
        self.assertEqual(len(optimizer_ids), len(set(optimizer_ids)))
        self.assertTrue({
            id(p) for p in projected.parax_bank.projector.parameters()
        } <= set(optimizer_ids))
        projected.zero_grad(set_to_none=True)
        logits = projected.current_all_logits(images)
        alignment = projected.parax_alignment_penalty()
        self.assertTrue(torch.isfinite(alignment))
        self.assertGreater(float(alignment.detach()), 0)
        (logits.square().mean() + 0.1 * alignment).backward()
        projector_gradients = [
            parameter.grad for parameter in projected.parax_bank.projector.parameters()
        ]
        self.assertTrue(any(
            grad is not None and torch.isfinite(grad).all() and torch.count_nonzero(grad)
            for grad in projector_gradients
        ))
        diagnostics = projected.parax_gate_diagnostics()
        self.assertIn("parax_full_layer0_raw_residual_ratio", diagnostics)
        self.assertIn("parax_full_layer0_projector_correction_ratio", diagnostics)
        projected.assert_visual_frozen()
        projected.activate_task(1)
        self.assertTrue(projected.parax_bank.expert_a.requires_grad)
        self.assertTrue(any(
            parameter.requires_grad for parameter in projected.parax_bank.routers.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad for parameter in projected.parax_bank.projector.parameters()
        ))

    def test_zero_parax_with_projector_is_exact_identity_initially(self) -> None:
        bank = ParaXImageAdapterBank(
            8, 2, 3, (0,), residual_scale=0.0,
            output_scale_mode="fixed", projector_bottleneck_dim=4,
        )
        bank.activate_task(0)
        tokens = torch.randn(2, 5, 8)
        output, _ = bank(0, tokens)
        self.assertTrue(torch.equal(output, tokens))
        self.assertAlmostEqual(float(bank.alignment_penalty()), 0.0, places=6)

    def test_image_token_adapter_and_p10_parax_run_together(self) -> None:
        torch.manual_seed(31)
        visual = FakeVisual()
        visual.transformer.resblocks.append(
            copy.deepcopy(visual.transformer.resblocks[0])
        )
        common = dict(
            task_sizes=(2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1, view_fusion="fixed_three_view",
            adapter_mode="image_token", adapter_bottleneck_dim=3,
            adapter_layer_indices=(0,), adapter_residual_scale=0.03,
        )
        torch.manual_seed(32)
        baseline = MultiLaneModel(copy.deepcopy(visual), **common)
        torch.manual_seed(32)
        combined = MultiLaneModel(
            copy.deepcopy(visual), **common,
            parax_mode="image", parax_layer_indices=(0,),
            parax_rank=2, parax_num_experts=3,
            parax_initialization="zero_output",
            parax_residual_scale=0.001,
            parax_output_scale_mode="fixed",
            parax_trainable_components="router",
            parax_freeze_center_after_task0=True,
        )
        baseline.activate_task(0)
        combined.activate_task(0)
        full = torch.randn(2, 3, 4, 4)
        images = {
            "full": full,
            "person": torch.flip(full, dims=(-1,)),
            "face": torch.flip(full, dims=(-2,)),
            "face_reliable": torch.tensor([True, False]),
        }
        baseline_logits = baseline.current_all_logits(images)
        combined_logits = combined.current_all_logits(images)
        torch.testing.assert_close(combined_logits, baseline_logits, atol=1e-6, rtol=1e-6)

        _, adapter_parameters, groups = build_optimizer_groups(
            combined, weight_decay=0.0, adapter_learning_rate=4e-4,
        )
        adapter_ids = {id(p) for p in combined.adapter_optimizer_parameters()}
        parax_ids = {id(p) for p in combined.parax_optimizer_parameters()}
        optimizer_ids = [id(p) for group in groups for p in group["params"]]
        self.assertTrue(adapter_ids)
        self.assertTrue(parax_ids)
        self.assertTrue(adapter_ids.isdisjoint(parax_ids))
        self.assertEqual(adapter_ids | parax_ids, {id(p) for p in adapter_parameters})
        self.assertEqual(len(optimizer_ids), len(set(optimizer_ids)))
        self.assertTrue(adapter_ids | parax_ids <= set(optimizer_ids))

        combined_logits.square().mean().backward()
        for parameters in (
            combined.adapter_optimizer_parameters(), combined.parax_optimizer_parameters()
        ):
            gradients = [p.grad for p in parameters]
            self.assertTrue(any(
                g is not None and torch.isfinite(g).all() and torch.count_nonzero(g) > 0
                for g in gradients
            ))
        combined.assert_visual_frozen()
        self.assertTrue(all(p.grad is None for p in combined.visual_encoder.parameters()))

        combined.activate_task(1)
        self.assertFalse(combined.parax_bank.expert_a.requires_grad)
        self.assertTrue(any(p.requires_grad for p in combined.parax_bank.routers.parameters()))
        self.assertTrue(list(combined.adapter_optimizer_parameters()))

    def test_bank_shape_static_gate_and_finite_gradient(self) -> None:
        torch.manual_seed(1)
        bank = ParaXImageAdapterBank(8, 2, 3, (0, 1), router_hidden=4)
        bank.activate_task(0)
        tokens = torch.randn(2, 5, 8, requires_grad=True)
        output, gates = bank(0, tokens, "full")
        self.assertEqual(output.shape, tokens.shape)
        self.assertEqual(gates.shape, (2, 3))
        self.assertTrue(torch.allclose(gates.sum(dim=-1), torch.ones(2)))
        output.square().mean().backward()
        active_names = {
            name for name, _ in bank.named_parameters()
            if not name.startswith("routers.1.")
        }
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for name, parameter in bank.named_parameters()
            if name in active_names
        ))
        static = ParaXImageAdapterBank(8, 2, 3, (0,), static=True)
        _, static_gates = static(0, tokens.detach())
        self.assertTrue(torch.allclose(static_gates, torch.full_like(static_gates, 1 / 3)))

    def test_level_conditioning_and_shared_expert_centers(self) -> None:
        bank = ParaXImageAdapterBank(8, 2, 3, (0, 1), level_conditioned=True)
        self.assertEqual(bank.expert_a.shape, (3, 2, 8))
        tokens = torch.randn(4, 5, 8)
        with torch.no_grad():
            bank.level_embeddings[0].fill_(1.0)
            bank.level_embeddings[1].fill_(-1.0)
        _, full_gates = bank(0, tokens, "full")
        _, face_gates = bank(0, tokens, "face")
        self.assertFalse(torch.equal(full_gates, face_gates))

    def test_model_keeps_visual_frozen_and_routes_gradient_to_parax(self) -> None:
        model = MultiLaneModel(
            FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1, parax_mode="post",
            parax_layer_indices=(0,), parax_rank=2, parax_num_experts=3,
        )
        model.activate_task(0)
        model.parax_bank.output_scale.data.fill_(0.1)
        logits = model.current_all_logits(torch.randn(3, 3, 4, 4))
        logits.square().mean().backward()
        model.assert_visual_frozen()
        self.assertTrue(any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parax_bank.parameters()
        ))
        self.assertTrue(model.parax_gate_diagnostics())

    def test_small_fixed_scale_and_component_freezing(self) -> None:
        bank = ParaXImageAdapterBank(
            8, 2, 3, (0,), residual_scale=1e-3,
            initialization="small", trainable_components="router",
            output_scale_mode="fixed", residual_ratio_cap=0.1,
        )
        bank.activate_task(0)
        self.assertFalse(bank.expert_a.requires_grad)
        self.assertFalse(bank.expert_b.requires_grad)
        self.assertTrue(next(bank.routers.parameters()).requires_grad)
        self.assertFalse(bank.output_scale.requires_grad)
        tokens = torch.randn(2, 5, 8)
        output, _ = bank(0, tokens)
        ratio = (output - tokens).norm() / tokens.norm()
        self.assertLessEqual(float(ratio), 0.1001)

    def test_identity_initialization_is_exact_and_learnable(self) -> None:
        torch.manual_seed(17)
        bank = ParaXImageAdapterBank(8, 2, 3, (0,), initialization="identity")
        bank.activate_task(0)
        tokens = torch.randn(2, 5, 8, requires_grad=True)
        output, _ = bank(0, tokens)
        self.assertTrue(torch.equal(output, tokens))
        output.square().mean().backward()
        self.assertIsNotNone(bank.output_scale.grad)
        self.assertTrue(torch.isfinite(bank.output_scale.grad).all())
        self.assertNotEqual(float(bank.output_scale.grad.abs()), 0.0)

    def test_zero_output_initialization_is_exact_and_fixed(self) -> None:
        bank = ParaXImageAdapterBank(
            8, 2, 3, (0,), initialization="zero_output",
            output_scale_mode="fixed",
        )
        bank.activate_task(0)
        tokens = torch.randn(2, 5, 8)
        output, _ = bank(0, tokens)
        self.assertTrue(torch.equal(output, tokens))
        self.assertFalse(bank.output_scale.requires_grad)

    def test_center_freeze_and_task_local_gate(self) -> None:
        bank = ParaXImageAdapterBank(
            8, 2, 3, (0,), initialization="zero_output",
            trainable_components="router", output_scale_mode="fixed",
            num_tasks=3, task_local_gate=True, freeze_center_after_task0=True,
        )
        bank.activate_task(0)
        self.assertTrue(bank.expert_a.requires_grad)
        bank.activate_task(1)
        self.assertFalse(bank.expert_a.requires_grad)
        self.assertTrue(bank.output_scale.requires_grad)

    def test_parax_initialization_does_not_shift_shared_rng(self) -> None:
        torch.manual_seed(21)
        visual = FakeVisual()
        visual_without = copy.deepcopy(visual)
        visual_with = copy.deepcopy(visual)
        torch.manual_seed(22)
        baseline = MultiLaneModel(visual_without, (2, 1), num_selectors=2, num_prompts=2, num_prompt_layers=1)
        baseline_state = torch.get_rng_state().clone()
        torch.manual_seed(22)
        candidate = MultiLaneModel(
            visual_with, (2, 1), num_selectors=2, num_prompts=2, num_prompt_layers=1,
            parax_mode="image", parax_layer_indices=(0,), parax_rank=2,
            parax_num_experts=3, parax_initialization="identity",
        )
        candidate_state = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(baseline_state, candidate_state))
        self.assertTrue(torch.equal(baseline.selectors, candidate.selectors))
        self.assertTrue(torch.equal(baseline.head.weight, candidate.head.weight))
        self.assertTrue(torch.equal(baseline.head.bias, candidate.head.bias))


if __name__ == "__main__":
    unittest.main()
