from __future__ import annotations

import unittest

import torch

from multi_lane.track_a.moe_task_ffn import TaskForwardMoEBank


class TaskForwardMoETest(unittest.TestCase):
    def make_bank(self) -> TaskForwardMoEBank:
        return TaskForwardMoEBank(
            num_tasks=2, hidden_dim=8, bottleneck_dim=4,
            layer_indices=(1,), residual_scale=0.1,
            num_experts=4, top_k=2, dropout=0,
        )

    def test_zero_output_identity_and_frozen_old_task(self) -> None:
        torch.manual_seed(17)
        bank = self.make_bank()
        bank.activate_task(0)
        x = torch.randn(1, 3, 5, 8)
        self.assertTrue(torch.equal(bank.delta_for_layer(1, x, [0], "full"),
                                    torch.zeros_like(x)))
        bank.tasks[0][0].experts[0].up.weight.data.fill_(0.01)
        bank.eval()
        old = bank.delta_for_layer(1, x, [0], "full").detach().clone()
        bank.activate_task(1)
        bank.train()
        self.assertFalse(any(parameter.requires_grad
                             for parameter in bank.tasks[0].parameters()))
        self.assertFalse(bank.tasks[0][0].training)
        self.assertTrue(bank.tasks[1][0].training)
        self.assertTrue(all(parameter.requires_grad
                            for parameter in bank.tasks[1].parameters()))
        after = bank.delta_for_layer(1, x, [0], "full").detach()
        self.assertTrue(torch.equal(old, after))
        with self.assertRaises(ValueError):
            bank.delta_for_layer(1, x, [2], "full")

    def test_view_routers_share_experts_and_train_without_future_access(self) -> None:
        bank = self.make_bank()
        bank.activate_task(0)
        pool = bank.tasks[0][0]
        for expert in pool.experts:
            expert.up.weight.data.fill_(0.01)
        pool.gates[0].data[0] = torch.tensor([4., 3., -4., -3.])
        pool.gates[1].data[0] = torch.tensor([-4., -3., 4., 3.])
        x = torch.ones(1, 2, 3, 8, requires_grad=True)
        bank.eval()
        full = bank.delta_for_layer(1, x, [0], "full")
        person = bank.delta_for_layer(1, x, [0], "person")
        full_counts = bank.route_statistics()["0"]["1"]["full"]["expert_topk_count"]
        person_counts = bank.route_statistics()["0"]["1"]["person"]["expert_topk_count"]
        self.assertEqual(full_counts, [2., 2., 0., 0.])
        self.assertEqual(person_counts, [0., 0., 2., 2.])
        self.assertEqual(set(id(expert) for expert in pool.experts),
                         set(id(expert) for expert in bank.tasks[0][0].experts))
        (full + person).sum().backward()
        self.assertIsNotNone(pool.experts[0].up.weight.grad)
        self.assertIsNotNone(pool.experts[2].up.weight.grad)
        self.assertIsNone(bank.tasks[1][0].experts[0].up.weight.grad)

    def test_compact_state_excludes_route_counters(self) -> None:
        bank = self.make_bank()
        bank.activate_task(0)
        bank.eval()
        bank.delta_for_layer(1, torch.ones(1, 2, 3, 8), [0], "face")
        state = bank.state_dict()
        self.assertFalse(any("route_" in key for key in state))
        restored = self.make_bank()
        restored.load_state_dict(state)
        restored.restore_task(0)
        self.assertEqual(restored.route_statistics()["0"]["1"]["face"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
