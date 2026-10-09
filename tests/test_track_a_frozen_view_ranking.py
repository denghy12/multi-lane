import unittest
import torch

from multi_lane.track_a.frozen_view_ranking_fusion import (
    BoundedFusionGate, implied_weights, ranking_loss, check_provenance, fit_gate,
)


class FrozenViewRankingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.views = torch.randn(12, 2, 3)
        self.reliable = torch.arange(12) % 2 == 0
        self.fixed = self.views.mean(-1)
        self.labels = (torch.arange(24).reshape(12, 2) % 3 == 0).float()

    def test_identity_gradient_and_weight_mask(self):
        for dynamic in (False, True):
            gate = BoundedFusionGate(self.views, self.fixed, dynamic)
            out, delta = gate(self.views, self.fixed, self.reliable)
            self.assertTrue(torch.equal(out, self.fixed))
            torch.nn.functional.binary_cross_entropy_with_logits(out, self.labels).backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in gate.parameters()), 0)
            with torch.no_grad():
                for p in gate.parameters():
                    p.fill_(1e5)
            out, delta = gate(self.views, self.fixed, self.reliable)
            weights = implied_weights(delta, self.reliable)
            self.assertLessEqual(float(delta.abs().max()), .050001)
            self.assertTrue(bool((weights >= 0).all()))
            torch.testing.assert_close(weights.sum(-1), torch.ones_like(self.fixed))
            self.assertTrue(bool((weights[~self.reliable, :, 2] == 0).all()))
            self.assertTrue(bool(torch.isfinite(out).all()))

    def test_invalid_face_cannot_change_router_output(self):
        gate = BoundedFusionGate(self.views, self.fixed)
        with torch.no_grad():
            gate.router[-1].weight.fill_(.1)
        changed = self.views.clone()
        changed[~self.reliable, :, 2] += 1000
        a, _ = gate(self.views, self.fixed, self.reliable)
        b, _ = gate(changed, self.fixed, self.reliable)
        self.assertTrue(torch.equal(a, b))

    def test_ranking_rewards_order_and_handles_pairless_batch(self):
        y = torch.tensor([[1.], [0.]])
        temp = torch.ones(1)
        self.assertLess(float(ranking_loss(torch.tensor([[2.], [-2.]]), y, temp)),
                        float(ranking_loss(torch.tensor([[-2.], [2.]]), y, temp)))
        z = torch.randn(3, 1, requires_grad=True)
        value = ranking_loss(z, torch.ones_like(z), temp)
        value.backward()
        self.assertEqual(float(value), 0.)
        self.assertTrue(bool(torch.isfinite(z.grad).all()))

    def test_fitting_uses_only_train_statistics_and_freezes_output(self):
        for method in ("static_bce", "dynamic_bce", "dynamic_ranking"):
            views, fixed = self.views.clone(), self.fixed.clone()
            gate, audit = fit_gate(views, fixed, self.labels, self.reliable, method,
                                   torch.device("cpu"), epochs=2, batch_size=12)
            self.assertTrue(torch.equal(views, self.views))
            self.assertTrue(torch.equal(fixed, self.fixed))
            self.assertEqual(audit["initial_logit_difference"], 0)
            self.assertTrue(all(not p.requires_grad for p in gate.parameters()))
            torch.testing.assert_close(gate.mean, self.views.mean(0))
            state = {key: value.clone() for key, value in gate.state_dict().items()}
            gate(self.views * 100, self.fixed, self.reliable)
            self.assertTrue(all(torch.equal(state[k], v) for k, v in gate.state_dict().items()))

    def test_cross_task_image_groups_are_excluded(self):
        record = {"source_split": "train", "training_exclusion": True,
                  "split_salt": "emotic-reliability-calibration-v1",
                  "fit_sample_ids": ["train:a#person=0"],
                  "calibration_sample_ids": ["train:b#person=1"]}
        self.assertEqual(check_provenance([record])["overlap"], 0)
        leak = dict(record, fit_sample_ids=["train:b#person=0"])
        with self.assertRaises(ValueError):
            check_provenance([record, leak])
        with self.assertRaises(ValueError):
            check_provenance([dict(record, source_split="val")])


if __name__ == "__main__":
    unittest.main()
