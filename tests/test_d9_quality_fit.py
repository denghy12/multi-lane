import json
import unittest

import numpy as np
import torch

from scripts.emotic.d9_quality_fit import (_balanced_bce, _odds_change, _paired_loss,
                                          _utility_targets, apply_task, fit_task)


class QualityFitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        rng = np.random.default_rng(17)
        n = 12
        valid = np.arange(n) % 3 != 0
        data = dict(reliable=valid, sample_ids=np.array([str(i) for i in range(n)]),
                    targets=(np.arange(n) % 2)[:, None].astype('float32'),
                    branch_logits=rng.normal(size=(n, 1, 3)).astype('float32'),
                    content_proj=rng.normal(size=(n, 12)), relation=rng.normal(size=(n, 8)))
        data['branch_logits'][~valid, :, 2] = 0.
        cls.natural = data
        cls.variants = {}
        for v, name in enumerate(('person', 'face'), 1):
            variant = {k: val.copy() for k, val in data.items()}
            rows = np.ones(n, dtype=bool) if v == 1 else valid
            variant['branch_logits'][rows, :, v] = 0.
            variant['content_proj'][rows, v * 4] += 3.
            cls.variants[name] = variant
        cls.model = json.loads(json.dumps(fit_task(data, cls.variants), allow_nan=False))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_equal_person_training_mass(self):
        rows = torch.tensor([[0., 4., 100.], [0., 0., 6.]])
        usable = torch.tensor([[True, True, False], [True, True, True]])
        self.assertEqual(float(_balanced_bce(rows, usable)), 2.)

    def test_leave_one_out_relative_utility(self):
        z = torch.zeros(2, 3, 1, 3)
        z[:, 0, 0, 1] = 4.
        z[:, 1, 0, 1] = -2.
        z[:, 0, 0, 2] = 1.
        z[:, 2, 0, 2] = 0.
        labels = torch.ones(2, 1, 1)
        valid = torch.tensor([False, True])
        bias = torch.zeros(2, 3, 3)
        bias[0, :, 2] = -torch.inf
        pairs = _utility_targets(z, labels, bias, valid)
        self.assertEqual(pairs[0][1].shape, (2,))
        self.assertEqual(pairs[1][1].shape, (1,))
        # Unreliable: removing Person leaves just Full, unchanged between versions.
        expected = torch.nn.functional.softplus(torch.tensor(1.)) - torch.nn.functional.softplus(torch.tensor(-2.))
        self.assertAlmostEqual(float(pairs[0][1][0]), float(expected), places=6)
        self.assertTrue(torch.all(pairs[0][1] > 0))
        zero = [(mask, torch.zeros_like(delta)) for mask, delta in pairs]
        self.assertEqual(float(_paired_loss(bias, zero)), 0.)
        # Signs are based on labels, so the same change for opposite labels reverses the target.
        negative = _utility_targets(z, 1.-labels, bias, valid)
        self.assertTrue(torch.all(negative[0][1] < 0))
        bias[1, 0] = torch.tensor([1., 2., 3.])
        delta = _odds_change(bias, valid, 2)
        expected = 3. - torch.logsumexp(torch.tensor([1., 2.]), dim=0) + np.log(2.)
        self.assertAlmostEqual(float(delta), float(expected), places=6)

    def test_predictions_do_not_need_labels_or_logits(self):
        only_features = {k: self.natural[k] for k in ('content_proj', 'relation', 'reliable')}
        weights = apply_task(only_features, self.model)
        self.assertEqual(set(weights), {'A', 'B', 'C', 'fixed'})
        for w in weights.values():
            np.testing.assert_allclose(w.sum(1), 1., atol=2e-7)
            self.assertTrue((w[~only_features['reliable'], 2] == 0.).all())
        smaller = apply_task({k: v[:2] for k, v in only_features.items()}, self.model)
        for k, w in smaller.items():
            np.testing.assert_allclose(w, weights[k][:2], atol=1e-6)
        self.assertEqual(self.model['pair_targets']['face']['count'], int(only_features['reliable'].sum()))

    def test_invalid_alignment_fails(self):
        variants = dict(self.variants)
        variants['face'] = dict(variants['face'], sample_ids=self.natural['sample_ids'][::-1])
        with self.assertRaisesRegex(ValueError, 'rows differ'):
            fit_task(self.natural, variants)
