import copy
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from multi_lane.track_a.d9_direction_objective import (
    deranged_targets, direction_benefits, gate_update_due)
from scripts.emotic.run_d9_relation_training import GateOptimizer


class DirectionTrainingTests(unittest.TestCase):
    def test_exact_budget_and_uniform_epoch_coverage(self):
        for batches in (84, 69, 14, 156, 68, 40, 10, 24):
            counts = [sum(gate_update_due(i, batches, e, 30, 300) for i in range(batches))
                      for e in range(1, 31)]
            self.assertEqual(counts, [10] * 30)
        self.assertEqual([gate_update_due(i, 2, 1, 1, 2) for i in range(2)], [True, True])
        with self.assertRaises(ValueError):
            gate_update_due(0, 9, 1, 30, 300)

    def test_finite_transfer_labels_match_explicit_fusion(self):
        branches = torch.tensor([[[0., 2., -2.]], [[0., -2., 2.]], [[1., 1., float('nan')]]],
                                requires_grad=True)
        labels = torch.tensor([[1.], [1.], [0.]])
        reliable = torch.tensor([True, True, False])
        benefit, valid = direction_benefits(branches, labels, reliable)
        self.assertTrue(torch.equal(benefit > 0, torch.tensor([[True, False], [False, True], [False, False]])))
        self.assertTrue(torch.equal(valid, torch.tensor([[True, True], [True, True], [True, False]])))
        self.assertFalse(benefit.requires_grad)
        self.assertEqual(benefit[2].tolist(), [0., 0.])
        prior = torch.tensor([0., torch.log(torch.tensor(.25)), torch.log(torch.tensor(.3125))]).softmax(0)
        base = (branches[:2, 0].detach() * prior).sum(1)
        after = base + .05 * (branches[:2, 0, 1].detach() - branches[:2, 0, 0].detach())
        expected = F.binary_cross_entropy_with_logits(base, labels[:2, 0], reduction='none')
        expected -= F.binary_cross_entropy_with_logits(after, labels[:2, 0], reduction='none')
        torch.testing.assert_close(benefit[:2, 0], expected)

    def test_learned_center_transfer_clamped_and_detached(self):
        branches = torch.tensor([[[0., 4., -3.]]], requires_grad=True)
        labels = torch.ones(1, 1)
        center = torch.tensor([[.01, .49, .5]], requires_grad=True)
        benefit, valid = direction_benefits(branches, labels, torch.tensor([True]), base_weights=center)
        original = (branches.detach()[0, 0] * center.detach()[0]).sum()
        desired = torch.stack([
            F.softplus(-original) - F.softplus(-(original + .01 * (branches.detach()[0, 0, v])) )
            for v in (1, 2)])
        torch.testing.assert_close(benefit[0], desired)
        self.assertTrue(valid.all())
        self.assertFalse(benefit.requires_grad)
        self.assertIsNone(branches.grad)
        self.assertIsNone(center.grad)

    def test_derangement_preserves_counts_without_rng_changes(self):
        reliable = torch.tensor([True, False, True, False, True, False])
        targets = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 0.], [1., 1.], [1., 0.]])
        valid = torch.stack((torch.ones_like(reliable), reliable), 1)
        before = torch.get_rng_state().clone()
        shuffled, usable, donors = deranged_targets(targets, valid, reliable, 53)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue((donors != torch.arange(6)).all())
        self.assertTrue(torch.equal(reliable[donors], reliable))
        self.assertTrue(torch.equal((shuffled * usable).sum(0), (targets * usable).sum(0)))
        self.assertTrue(torch.equal(shuffled, targets[donors]))
        self.assertTrue(torch.equal(usable, valid))

    def test_singleton_mask_is_shared(self):
        reliable = torch.tensor([True, True, True, False])
        targets = torch.ones(4, 2)
        valid = torch.stack((torch.ones_like(reliable), reliable), 1)
        _, usable, donors = deranged_targets(targets, valid, reliable, 54)
        self.assertFalse(usable[-1].any())
        self.assertTrue(usable[:3].all())
        self.assertEqual(donors[-1], 3)
        with self.assertRaises(ValueError):
            deranged_targets(targets[:2], valid[:2], torch.tensor([True, False]), 54)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA determinism regression')
    def test_singleton_mask_on_deterministic_cuda(self):
        with torch.random.fork_rng():
            original = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(True)
            try:
                for bits in ([True, True, True, False], [False] * 4):
                    reliable = torch.tensor(bits, device='cuda')
                    valid = torch.stack((torch.ones_like(reliable), reliable), 1)
                    labels = valid.float()
                    _, used, _ = deranged_targets(labels, valid, reliable, 12)
                    self.assertEqual(used.sum().item(), 6 if bits[0] else 4)
            finally:
                torch.use_deterministic_algorithms(original)

    def test_direction_adam_clock_resume_and_microbatch(self):
        torch.manual_seed(23)
        model = nn.Linear(4, 2, bias=False)
        twin = copy.deepcopy(model)
        cfg = {'mode': 'direction', 'epochs': 30, 'gate_lr': .0004, 'gate_updates': 300}
        opts = [GateOptimizer(net, cfg) for net in (model, twin)]
        feature = torch.randn(7, 4, requires_grad=True)
        label = torch.randint(2, (7, 2)).float()
        reliable = torch.tensor([True, False, True, True, False, True, False])
        valid = torch.stack((torch.ones_like(reliable), reliable), 1)
        weights = torch.ones(7, 1, 3) / 3

        def step(net, opt, split):
            opt.reset_batch('cpu')
            for rows in torch.arange(7).split(split):
                output = {'direction_scores': net(feature[rows].detach())[:, None], 'weights': weights[rows]}
                opt.accumulate_direction(output, reliable[rows])
            opt.backward_direction(label, valid, 1.)
            opt.check_gradients(1.)
            opt.step()

        step(model, opts[0], 2)
        step(twin, opts[1], 7)
        self.assertIsNone(feature.grad)
        torch.testing.assert_close(model.weight, twin.weight)
        torch.testing.assert_close(opts[0].loss, opts[1].loss)
        self.assertEqual(opts[0].updates, 1)
        self.assertEqual(opts[0].scheduler.last_epoch, 1)
        resume = copy.deepcopy(model)
        resumed = GateOptimizer(resume, cfg)
        resumed.load_state_dict(copy.deepcopy(opts[0].state_dict()))
        step(model, opts[0], 2)
        step(resume, resumed, 2)
        torch.testing.assert_close(model.weight, resume.weight, rtol=0, atol=0)
        self.assertEqual(resumed.updates, 2)
        self.assertEqual(resumed.scheduler.get_last_lr(), opts[0].scheduler.get_last_lr())


if __name__ == '__main__':
    unittest.main()
