import json
import unittest

import numpy as np
import torch

from scripts.emotic.d9_aux_budget_fit import apply_task, fit_task


def toy_train():
    rng = np.random.default_rng(5)
    n = 24
    r = np.arange(n) % 3 != 0
    positive = np.arange(n) % 2 == 0
    y = np.column_stack((positive, ~positive)).astype(np.float32)
    want_person = (np.arange(n) // 3) % 2 == 0
    signed = (2. * y - 1.) * np.where(want_person[:, None], 4., -4.)
    z = np.zeros((n, 2, 3), np.float32)
    z[:, :, 1], z[:, :, 2] = signed, -signed
    z[~r, :, 1] = (2. * y[~r] - 1.) * 4.
    relation = np.zeros((n, 8))
    relation[:, 0] = 2. * want_person - 1.
    relation[~r] = 1e4
    return dict(reliable=r, targets=y, branch_logits=z, relation=relation,
                delta=rng.normal(size=(n, 2, 4)), content=np.zeros((n, 3, 4)))


class AuxBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.data = toy_train()
        cls.model = json.loads(json.dumps(fit_task(cls.data), allow_nan=False))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_exact_shared_budget_and_face_mask(self):
        w = apply_task(self.data, self.model)
        r = self.data['reliable']
        self.assertTrue(np.all(w['person'][:, 0] >= .5))
        np.testing.assert_array_equal(w['person'][:, 0], w['group'][:, 0])
        np.testing.assert_array_equal(w['person'][~r], w['group'][~r])
        self.assertTrue(np.all(w['person'][~r, 2] == 0))
        for array in w.values():
            np.testing.assert_allclose(array.sum(1), 1., atol=1e-6)
            self.assertTrue(np.all(array >= 0))
        np.testing.assert_allclose(w['fixed'][r][0], [.64, .16, .20])

    def test_matched_allocation_learns_without_group_scale_drift(self):
        w = apply_task(self.data, self.model)
        z = self.data['branch_logits']
        r = self.data['reliable']
        want_p = self.data['relation'][:, 0] > 0
        self.assertTrue(np.all(w['person'][r & want_p, 1] > w['group'][r & want_p, 1]))
        self.assertTrue(np.all(w['person'][r & ~want_p, 1] < w['group'][r & ~want_p, 1]))
        bce = self.model['fit']
        self.assertLess(bce['person_train_bce'], bce['group_train_bce'])
        residual = (z[:, :, 0] + w['person'][:, 1:2] * (z[:, :, 1] - z[:, :, 0])
                    + w['person'][:, 2:3] * (z[:, :, 2] - z[:, :, 0]))
        np.testing.assert_allclose(residual, (w['person'][:, None] * z).sum(-1), atol=1e-6)
        self.assertEqual(self.model['feature_stats']['mean'][12], 0.)

    def test_inference_has_no_access_to_logits_or_labels_and_is_batch_invariant(self):
        keys = ('reliable', 'relation', 'content', 'delta')
        small = {k:self.data[k][:3] for k in keys}
        full = apply_task(self.data, self.model)
        for name, weights in apply_task(small, self.model).items():
            np.testing.assert_allclose(weights, full[name][:3], atol=1e-6)

    def test_nonfinite_usable_feature_fails(self):
        bad = toy_train()
        bad['relation'][1, 3] = np.inf
        with self.assertRaises(ValueError):
            fit_task(bad)
