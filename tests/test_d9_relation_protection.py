import unittest

import torch

from scripts.emotic.run_d9_relation_training import losses


class ProtectedFixedLossTests(unittest.TestCase):
    def test_only_the_fused_gradient_is_removed_from_both_parameter_groups(self):
        labels = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        valid = torch.tensor([True, True, False])
        branch_params = torch.randn(3, 3, 2, generator=torch.Generator().manual_seed(11),
                                    requires_grad=True)
        adapter_params = torch.zeros_like(branch_params, requires_grad=True)
        scores = branch_params + adapter_params
        full, person, face = scores.unbind()
        fused = torch.where(valid[:, None], .64 * full + .16 * person + .20 * face,
                            .80 * full + .20 * person)
        outputs = dict(fused=fused, all_columns=dict(full=full, person=person, face=face))
        args = (outputs, labels, valid, 3, 2, [0, 1])
        protected, bce, asl = losses(*args, protect_branches=True)
        joint, joint_bce, joint_asl = losses(*args)
        self.assertFalse(protected.requires_grad)
        self.assertTrue(joint.requires_grad)
        torch.testing.assert_close(torch.stack([protected, bce, asl]), torch.stack([joint, joint_bce, joint_asl]),
                                   rtol=0, atol=0)
        for params, branch_loss in ((branch_params, bce), (adapter_params, asl)):
            def gradient(loss):
                return torch.autograd.grad(loss, params, retain_graph=True)[0]
            expected = gradient(branch_loss)
            torch.testing.assert_close(gradient(branch_loss + protected), expected, rtol=0, atol=0)
            self.assertEqual(expected[2, 2].count_nonzero().item(), 0)
            joint_grad = gradient(branch_loss + joint)
            for view in range(3):
                self.assertGreater(expected[view].abs().sum().item(), 0.)
                self.assertFalse(torch.equal(joint_grad[view], expected[view]))


if __name__ == '__main__':
    unittest.main()
