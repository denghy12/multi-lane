import unittest

import numpy as np
import torch
import torch.nn.functional as F

from scripts.emotic.d9_matched_run import fit_average
from scripts.emotic.d9_natural_rank_fit import class_pairs, fit_arm, natural_rank


class NaturalRankTest(unittest.TestCase):
    def test_all_pairs_macro_and_cross_group_gradient(self):
        reliable = torch.tensor([True, False, False, True])
        labels = torch.tensor([[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 0, 1]])
        z = torch.tensor([[2., -3., 0.], [1., -2., 0.], [0., -1., 0.], [-1., 0., 0.]], requires_grad=True)
        pairs, counts = class_pairs(labels, reliable)
        expected = (F.softplus(torch.tensor([-1., -2., 2., 1.])).mean()
                    + F.softplus(torch.tensor([-1., 1., 2.])).mean())/2
        loss = natural_rank(z, pairs)
        torch.testing.assert_close(loss, expected)
        self.assertEqual([c['pairs'] for c in counts], [4, 3, 0])
        self.assertEqual([c['cross_group_pairs'] for c in counts], [4, 2, 0])
        loss.backward()
        self.assertTrue((z.grad[labels[:, 0] == 1, 0] < 0).all())
        self.assertTrue((z.grad[labels[:, 0] == 0, 0] > 0).all())
        self.assertTrue((z.grad[:, 2] == 0).all())

    def test_control_replays_original_and_empty_pairs_leave_bce_intact(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(31)
        z = rng.standard_normal((12, 2, 3)).astype(np.float32)
        data = dict(branch_logits=z, person_branch_logits=z * 0.5, face_branch_logits=z * 1.5,
                    reliable=np.arange(12) % 2 == 0, targets=np.ones((12, 2), np.float32))
        variants = {v: dict(branch_logits=data[v+'_branch_logits']) for v in ('person', 'face')}
        reference = fit_average(data, variants)
        paired, _ = fit_arm(data, False)
        candidate, _ = fit_arm(data, True)
        self.assertEqual(paired['center'], reference['params']['bias'])
        self.assertEqual(paired['weights'], candidate['weights'])
        self.assertEqual(candidate['fit']['final']['natural_rank'], 0.)
        self.assertEqual(candidate['weights'][0][2], 0.)
