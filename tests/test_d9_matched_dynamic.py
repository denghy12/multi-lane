import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.emotic.d9_matched_dynamic import _personal_weights, apply_dynamic, fit_dynamic


class SameMeanPersonalCorrectionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(61)
        n = 40
        valid = np.arange(n) % 3 != 0
        cls.a = np.array([[.75, .25, 0.], [.63, .2, .17]], dtype='float32')
        cls.data = dict(reliable=valid, content_proj=rng.normal(size=(n, 12)), relation=rng.normal(size=(n, 8)),
                        targets=rng.integers(0, 2, size=(n, 2)), branch_logits=rng.normal(size=(n, 2, 3)))
        variants = {}
        for view, idx in [('person', 1), ('face', 2)]:
            d = {k: v.copy() for k, v in cls.data.items()}
            rr = np.ones(n, dtype=bool) if idx == 1 else valid
            d['content_proj'][rr, idx*4] += 1
            d['branch_logits'][rr, :, idx] = 0.
            variants[view] = d
        with patch('scripts.emotic.d9_matched_dynamic.STEPS', 12):
            cls.model = json.loads(json.dumps(fit_dynamic(cls.data, variants, cls.a), allow_nan=False))

    def test_shifted_cohort_mean_equal_and_masked(self):
        mask = np.arange(40) < 12
        only = {k: self.data[k][mask].copy() for k in ('reliable', 'content_proj', 'relation')}
        only['content_proj'] += 7.
        for w in apply_dynamic(only, self.model, 'cohort').values():
            for r in (False, True):
                np.testing.assert_allclose(w[only['reliable'] == r].mean(0), self.a[int(r)], rtol=0, atol=2e-7)
            self.assertTrue((w >= 0).all())
            np.testing.assert_allclose(w.sum(-1), 1., atol=2e-7)
            self.assertTrue((w[~only['reliable'], 2] == 0.).all())

    def test_single_person_train_center_inference(self):
        only = {k: self.data[k] for k in ('reliable', 'content_proj', 'relation')}
        few = {k: v[:1] for k, v in only.items()}
        all_w = apply_dynamic(only, self.model, 'train')
        for m, w in apply_dynamic(few, self.model, 'train').items():
            np.testing.assert_allclose(w[0], all_w[m][0], atol=2e-7)
            self.assertGreater(self.model['heads'][m]['last_gradient_norm'], 0.)

    def test_zero_init_and_bounded_extreme_perturbations(self):
        x = torch.randn(24, 3, 20)
        r = torch.arange(24) % 3 != 0
        a = torch.from_numpy(self.a)
        w, _ = _personal_weights(x, r, torch.zeros((3, 20)), a)
        base = a[r.long(), None, :].expand_as(w)
        torch.testing.assert_close(w, base)
        matrix = torch.randn(3, 20) * 100
        w, _ = _personal_weights(x, r, matrix, a)
        self.assertTrue(torch.all((w >= 0) & (w <= 2 * base)))
        torch.testing.assert_close(w.sum(-1), torch.ones_like(w[..., 0]))


if __name__ == '__main__':
    unittest.main()
