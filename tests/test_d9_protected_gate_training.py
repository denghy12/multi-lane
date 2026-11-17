import copy
import unittest

import torch
from torch import nn

from scripts.emotic.run_d9_relation_training import GateOptimizer


class GateTrainingTests(unittest.TestCase):
    def test_microbatch_gradient_mask_and_adam_resume(self):
        torch.manual_seed(17)
        net = nn.Linear(2, 3)
        config = {'gate_lr': .0004, 'epochs': 30}
        features = torch.randn(7, 2)
        logits = torch.randn(7, 3, 2, requires_grad=True)
        labels = torch.randint(2, (7, 2)).float()
        reliable = torch.tensor([True, False, True, False, True, True, False])

        def step(gate, opt, split):
            opt.reset_batch('cpu')
            for ids in torch.arange(len(labels)).split(split):
                rel = reliable[ids]
                scores = gate(features[ids])
                mask = torch.stack((torch.ones_like(rel), torch.ones_like(rel), rel), 1)
                weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
                fused = (weights[:, :, None] * logits[ids].detach()).sum(1)
                opt.accumulate({'fused': fused, 'weights': weights[:, None]}, labels[ids],
                               rel, len(labels), 1.)
            opt.check_gradients(1.)
            opt.optimizer.step()
            opt.scheduler.step()

        whole = copy.deepcopy(net)
        opt = GateOptimizer(net, config)
        whole_opt = GateOptimizer(whole, config)
        step(net, opt, 3)
        step(whole, whole_opt, 7)
        self.assertIsNone(logits.grad)
        self.assertEqual(opt.weight_sums[0, 2], 0)
        torch.testing.assert_close(opt.weight_sums, whole_opt.weight_sums)
        torch.testing.assert_close(opt.loss, whole_opt.loss)
        for p, q in zip(net.parameters(), whole.parameters()):
            torch.testing.assert_close(p.grad, q.grad)
            torch.testing.assert_close(p, q)

        resumed = copy.deepcopy(net)
        resume_opt = GateOptimizer(resumed, config)
        resume_opt.load_state_dict(copy.deepcopy(opt.state_dict()))
        step(net, opt, 3)
        step(resumed, resume_opt, 3)
        self.assertEqual(opt.scheduler.get_last_lr(), resume_opt.scheduler.get_last_lr())
        for p, q in zip(net.parameters(), resumed.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
