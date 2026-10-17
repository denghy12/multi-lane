from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.moe_task_ffn_v4 import TaskForwardEvidenceMoEBank
from test_track_a_reproduction import FakeVisual


class TaskForwardEvidenceMoETest(unittest.TestCase):
    def bank(self, mode="dynamic"):
        bank = TaskForwardEvidenceMoEBank(2, 8, 4, (0,), 0.1,
                                          dropout=0, route_mode=mode)
        bank.activate_task(0)
        return bank

    def test_experts_consume_image_evidence_and_never_write_selectors(self):
        bank = self.bank("shared_fixed").eval()
        tokens = torch.randn(1, 4, 3, 8)
        evidence = torch.randn(1, 4, 8)
        pool = bank.tasks[0][0]
        with torch.no_grad():
            pool.experts[0].up.weight.normal_()
        captured = []
        hook = pool.experts[0].down.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        output = bank.delta_for_layer(0, tokens, [0], "full", evidence)
        hook.remove()
        self.assertTrue(torch.equal(captured[0][:, 0], F.layer_norm(evidence[0], (8,))))
        self.assertGreater(output[:, :, :1].abs().sum().item(), 0)
        self.assertEqual(output[:, :, 1:].count_nonzero().item(), 0)
        # With the same evidence, changing raw lane tokens cannot change delta.
        changed_lane = bank.delta_for_layer(0, tokens * 17, [0], "full", evidence)
        self.assertTrue(torch.equal(output, changed_lane))
        changed_image = bank.delta_for_layer(0, tokens, [0], "full", evidence.flip(1))
        self.assertFalse(torch.equal(output, changed_image))

    def test_identity_amp_evidence_and_selected_expert_gradients(self):
        bank = self.bank("view_fixed")
        tokens = torch.randn(1, 4, 3, 8)
        evidence = torch.randn(1, 4, 8, requires_grad=True)
        self.assertEqual(bank.delta_for_layer(0, tokens, [0], "full", evidence).count_nonzero(), 0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = bank.delta_for_layer(0, tokens, [0], "full", evidence)
        self.assertEqual(result.dtype, tokens.dtype)
        result.sum().backward()
        for i in (0, 1, 2):
            self.assertGreater(bank.tasks[0][0].experts[i].up.weight.grad.abs().sum(), 0)
        self.assertIsNone(bank.tasks[0][0].experts[3].up.weight.grad)
        # Evidence gradients begin once zero-initialized output weights move.
        with torch.no_grad():
            for expert in bank.tasks[0][0].experts:
                expert.up.weight.normal_()
        bank.zero_grad(set_to_none=True)
        evidence.grad = None
        bank.delta_for_layer(0, tokens, [0], "full", evidence).square().sum().backward()
        self.assertGreater(evidence.grad.abs().sum(), 0)

    def test_dynamic_gate_gradient_and_whole_old_pool_freeze(self):
        bank = self.bank().eval()
        tokens = torch.randn(1, 6, 3, 8)
        evidence = torch.randn(1, 6, 8)
        with torch.no_grad():
            for expert in bank.tasks[0][0].experts:
                expert.up.weight.normal_()
        bank.train()
        result = bank.delta_for_layer(0, tokens, [0], "person", evidence)
        result.square().sum().backward()
        self.assertGreater(bank.tasks[0][0].gates[1].grad.abs().sum(), 0)
        bank.eval()
        old = bank.delta_for_layer(0, tokens, [0], "person", evidence).detach().clone()
        bank.activate_task(1)
        bank.train()
        self.assertFalse(any(p.requires_grad for p in bank.tasks[0].parameters()))
        self.assertFalse(bank.tasks[0][0].training)
        with torch.no_grad():
            for parameter in bank.tasks[1].parameters():
                parameter.add_(1)
        self.assertTrue(torch.equal(old, bank.delta_for_layer(0, tokens, [0], "person", evidence)))
        with self.assertRaises(ValueError):
            bank.delta_for_layer(0, tokens, [2], "person", evidence)

    def test_matched_shared_fixed_dynamic_and_runtime_ablation(self):
        tokens, evidence = torch.randn(1, 4, 3, 8), torch.randn(1, 4, 8)
        for mode in ("shared_fixed", "view_fixed", "dynamic"):
            bank = self.bank(mode).eval()
            pool = bank.tasks[0][0]
            with torch.no_grad():
                for i, expert in enumerate(pool.experts):
                    expert.up.bias.fill_(i + 1)
            output = bank.delta_for_layer(0, tokens, [0], "full", evidence)
            gate = bank._specialist_gate(evidence[0], pool, 0)
            expected = 0.1 * (0.5 + 0.5 * (gate * torch.tensor([2., 3., 4.])).sum(1))
            self.assertTrue(torch.allclose(output[0, :, 0], expected[:, None].expand(-1, 8)))
            self.assertEqual(output[:, :, 1:].count_nonzero(), 0)
            bank.specialist_runtime_enabled = False
            off = bank.delta_for_layer(0, tokens, [0], "full", evidence)
            self.assertTrue(torch.allclose(off[:, :, 0], torch.full_like(off[:, :, 0], 0.05)))

    def test_model_identity_rng_visual_freeze_and_cls_diagnostics(self):
        options = dict(task_sizes=(2, 1), num_selectors=2, num_prompts=2,
                       num_prompt_layers=1, adapter_layer_indices=(0,),
                       adapter_bottleneck_dim=4, view_fusion="fixed_three_view")
        torch.manual_seed(8)
        base = MultiLaneModel(FakeVisual(), **options)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(8)
        model = MultiLaneModel(FakeVisual(), adapter_mode="moe_task_ffn_v4",
                              moe_num_experts=4, moe_noise_std=0, **options)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for key, value in base.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        base.activate_task(0)
        model.activate_task(0)
        inputs = {v: torch.randn(3, 3, 4, 4) for v in ("full", "person", "face")}
        inputs["face_reliable"] = torch.tensor([True, False, True])
        base.eval(); model.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(base.seen_logits(inputs), model.seen_logits(inputs)))
        metrics = model.adapter_view_diagnostics()
        self.assertEqual(metrics["moe_person_delta_cls_input_cls_ratio"], 0)
        self.assertGreater(metrics["moe_person_evidence_centered_l2"], 0)
        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in metrics.values()))
        routes = model.adapter_bank.route_statistics()["0"]["0"]
        for view in routes.values():
            self.assertEqual(sum(view["expert_topk_count"]), 9)
        model.assert_visual_frozen()
        bank = TaskForwardEvidenceMoEBank(8, 768, 64, (1,), 0.1)
        self.assertEqual(bank.per_task_parameter_count(), 403456)
        self.assertFalse(any("route_output" in key for key in model.state_dict()))


if __name__ == "__main__":
    unittest.main()
