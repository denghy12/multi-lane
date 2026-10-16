from __future__ import annotations

import unittest

import torch

from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.moe_task_ffn_v2 import TaskForwardSharedMoEBank
from test_track_a_reproduction import FakeVisual


class TaskForwardSharedMoETest(unittest.TestCase):
    def bank(self, mode: str = "dynamic") -> TaskForwardSharedMoEBank:
        return TaskForwardSharedMoEBank(
            num_tasks=2, hidden_dim=8, bottleneck_dim=4,
            layer_indices=(0,), residual_scale=0.1,
            num_experts=7, dropout=0, route_mode=mode,
            noise_std=0,
        )

    def test_identity_shared_route_and_content_switch(self) -> None:
        torch.manual_seed(4)
        bank = self.bank()
        bank.activate_task(0)
        tokens = torch.randn(1, 2, 3, 8)
        evidence = torch.zeros(1, 2, 8)
        evidence[0, 0, 0] = 3
        evidence[0, 1, 1] = 3
        pool = bank.tasks[0][0]
        for gate in pool.gates:
            gate.data.zero_()
            for expert in range(6):
                gate.data[expert, expert] = 1
        self.assertTrue(torch.equal(
            bank.delta_for_layer(0, tokens, [0], "full", evidence),
            torch.zeros_like(tokens),
        ))
        for expert in pool.experts:
            expert.up.weight.data.fill_(0.01)
        bank.eval()
        result = bank.delta_for_layer(0, tokens, [0], "full", evidence)
        self.assertEqual(result.shape, tokens.shape)
        counts = bank.route_statistics()["0"]["0"]["full"]["expert_topk_count"]
        self.assertEqual(counts[0], 2)
        self.assertEqual(counts[1], 1)
        self.assertEqual(counts[2], 1)
        self.assertEqual(sum(counts), 4)

    def test_control_routes_and_old_pool_freeze(self) -> None:
        tokens = torch.randn(1, 2, 3, 8)
        evidence = torch.randn(1, 2, 8)
        for mode, view, expected in (
            ("shared_fixed", "face", {0}),
            ("view_fixed", "full", {0, 1}),
            ("view_fixed", "person", {0, 2}),
            ("view_fixed", "face", {0, 3}),
        ):
            bank = self.bank(mode)
            bank.activate_task(0)
            bank.eval()
            bank.delta_for_layer(0, tokens, [0], view, evidence)
            counts = bank.route_statistics()["0"]["0"][view]["expert_topk_count"]
            self.assertEqual({i for i, count in enumerate(counts) if count}, expected)
            old = bank.delta_for_layer(0, tokens, [0], view, evidence).detach().clone()
            bank.activate_task(1)
            bank.train()
            self.assertFalse(any(p.requires_grad for p in bank.tasks[0].parameters()))
            self.assertFalse(bank.tasks[0][0].training)
            self.assertTrue(torch.equal(
                old, bank.delta_for_layer(0, tokens, [0], view, evidence).detach()
            ))
            with self.assertRaises(ValueError):
                bank.delta_for_layer(0, tokens, [2], view, evidence)

    def test_dynamic_gate_gradient_balance_and_noise_schedule(self) -> None:
        bank = self.bank()
        bank.activate_task(0)
        for expert in bank.tasks[0][0].experts:
            expert.up.weight.data.normal_(std=0.01)
        tokens = torch.randn(1, 5, 3, 8, requires_grad=True)
        evidence = torch.randn(1, 5, 8, requires_grad=True)
        bank.reset_auxiliary_metrics()
        output = bank.delta_for_layer(0, tokens, [0], "person", evidence)
        loss = output.square().mean() + 0.01 * bank.balance_loss()
        loss.backward()
        gate_grad = bank.tasks[0][0].gates[1].grad
        self.assertIsNotNone(gate_grad)
        self.assertTrue(torch.isfinite(gate_grad).all())
        self.assertGreater(gate_grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(evidence.grad).all())
        self.assertGreaterEqual(bank.balance_loss().item(), 0)
        bank.set_training_progress(1.0)
        self.assertEqual(bank.noise_progress, 1.0)
        with self.assertRaises(ValueError):
            bank.set_training_progress(1.1)

    def test_model_passes_image_evidence_and_preserves_frozen_visual(self) -> None:
        torch.manual_seed(7)
        model = MultiLaneModel(
            FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1, normalize="pre-head",
            adapter_mode="moe_task_ffn_v2", adapter_layer_indices=(0,),
            adapter_bottleneck_dim=4, moe_num_experts=7,
            moe_noise_std=0, view_fusion="fixed_three_view",
        )
        model.activate_task(0)
        inputs = {
            name: torch.randn(3, 3, 4, 4)
            for name in ("full", "person", "face")
        }
        inputs["face_reliable"] = torch.tensor([True, False, True])
        logits, views, _ = model.current_all_logits_with_view_features(inputs)
        self.assertEqual(logits.shape, (3, 3))
        self.assertEqual(set(views), {"full", "person", "face"})
        self.assertTrue(torch.isfinite(model.moe_balance_loss()))
        self.assertEqual(len(model.adapter_bank._balance_terms), 3)
        self.assertTrue(all(not p.requires_grad for p in model.visual_encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
