from __future__ import annotations

import unittest

import torch

from multi_lane.track_a.adapter import ParaXImageAdapterBank
from multi_lane.track_a.model import MultiLaneModel
from test_track_a_reproduction import FakeVisual


class ParaXTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
