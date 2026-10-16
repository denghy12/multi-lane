from __future__ import annotations

import unittest

import torch

from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.moe_task_ffn_v3 import TaskForwardSoftMoEBank
from test_track_a_reproduction import FakeVisual


class TaskForwardSoftMoETest(unittest.TestCase):
    def bank(self, mode="dynamic"):
        bank = TaskForwardSoftMoEBank(2, 8, 4, (0,), 0.1, dropout=0,
                                      route_mode=mode)
        bank.activate_task(0)
        return bank

    def test_soft_top2_matches_train_eval_and_has_content_gradient(self):
        bank = self.bank()
        pool = bank.tasks[0][0]
        with torch.no_grad():
            for gate in pool.gates:
                gate.zero_()
                gate[:3].copy_(torch.eye(3))
        evidence = torch.zeros(2, 8, requires_grad=True)
        with torch.no_grad():
            evidence[0, 0] = 3
            evidence[0, 1] = 1
            evidence[1, 1] = 3
            evidence[1, 2] = 1
        gate = bank._specialist_gate(evidence, pool, 0)
        self.assertTrue(torch.equal((gate > 0).sum(1), torch.tensor([2, 2])))
        self.assertTrue(torch.allclose(gate.sum(1), torch.ones(2)))
        self.assertFalse(torch.equal(gate[0] > 0, gate[1] > 0))
        bank.eval()
        self.assertTrue(torch.equal(gate, bank._specialist_gate(evidence, pool, 0)))
        (gate * torch.tensor([1., 2., 4.])).sum().backward()
        self.assertGreater(evidence.grad.abs().sum().item(), 0)
        self.assertGreater(pool.gates[0].grad.abs().sum().item(), 0)

    def test_matched_controls_and_specialist_ablation(self):
        tokens = torch.randn(1, 3, 2, 8)
        evidence = torch.randn(1, 3, 8)
        for mode in ("shared_fixed", "view_fixed", "dynamic"):
            bank = self.bank(mode).eval()
            pool = bank.tasks[0][0]
            with torch.no_grad():
                for i, expert in enumerate(pool.experts):
                    expert.up.bias.fill_(i + 1)
            output = bank.delta_for_layer(0, tokens, [0], "full", evidence)
            gate = bank._specialist_gate(evidence[0], pool, 0)
            expected = 0.1 * (0.5 + 0.5 * (gate * torch.tensor([2., 3., 4.])).sum(1))
            self.assertTrue(torch.allclose(output[0], expected[:, None, None].expand_as(output[0])))
            bank.specialist_runtime_enabled = False
            off = bank.delta_for_layer(0, tokens, [0], "full", evidence)
            self.assertTrue(torch.allclose(off, torch.full_like(off, 0.05)))
            if mode == "shared_fixed":
                self.assertTrue(torch.equal(output, off))

    def test_identity_two_expert_gradients_dtype_and_old_freeze(self):
        bank = self.bank("view_fixed")
        tokens = torch.randn(1, 4, 3, 8)
        evidence = torch.randn(1, 4, 8)
        self.assertTrue(torch.equal(bank.delta_for_layer(0, tokens, [0], "full", evidence),
                                    torch.zeros_like(tokens)))
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = bank.delta_for_layer(0, tokens, [0], "full", evidence)
        self.assertEqual(result.dtype, tokens.dtype)
        result.sum().backward()
        for i in (0, 1, 2):
            self.assertGreater(bank.tasks[0][0].experts[i].up.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(bank.tasks[0][0].experts[3].up.weight.grad)
        bank.eval()
        old = bank.delta_for_layer(0, tokens, [0], "full", evidence).detach().clone()
        bank.activate_task(1)
        bank.train()
        self.assertFalse(any(p.requires_grad for p in bank.tasks[0].parameters()))
        self.assertFalse(bank.tasks[0][0].training)
        self.assertTrue(torch.equal(old, bank.delta_for_layer(0, tokens, [0], "full", evidence)))

    def test_model_diagnostics_counters_parameter_budget_and_base_rng(self):
        options = dict(task_sizes=(2, 1), num_selectors=2, num_prompts=2,
                       num_prompt_layers=1, adapter_layer_indices=(0,),
                       adapter_bottleneck_dim=4, view_fusion="fixed_three_view")
        torch.manual_seed(8)
        base = MultiLaneModel(FakeVisual(), **options)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(8)
        model = MultiLaneModel(FakeVisual(), adapter_mode="moe_task_ffn_v3",
                              moe_num_experts=4, moe_noise_std=0, **options)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        bank = TaskForwardSoftMoEBank(8, 768, 64, (1,), 0.1)
        self.assertEqual(bank.per_task_parameter_count(), 403456)
        model.activate_task(0)
        inputs = {v:torch.randn(3, 3, 4, 4) for v in ("full", "person", "face")}
        inputs["face_reliable"] = torch.tensor([True, False, True])
        model.eval()
        with torch.no_grad(): model.seen_logits(inputs)
        metrics = model.adapter_view_diagnostics()
        self.assertIn("moe_person_weighted_specialist_shared_ratio", metrics)
        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in metrics.values()))
        routes = model.adapter_bank.route_statistics()["0"]["0"]
        for view in routes.values():
            self.assertEqual(sum(view["expert_topk_count"]), 9)
            self.assertAlmostEqual(sum(view["expert_soft_weight_sum"]), 3, places=5)
        self.assertFalse(any("route_output" in key for key in model.state_dict()))
        model.assert_visual_frozen()


if __name__ == "__main__":
    unittest.main()
