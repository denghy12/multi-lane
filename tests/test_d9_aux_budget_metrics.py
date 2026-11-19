import json
import unittest

import numpy as np
import torch

from multi_lane.track_a.runner import average_precision
from scripts.emotic.d9_aux_budget_metrics import evaluate
from scripts.emotic.fit_d9_task10_probe import mix


class AuxBudgetMetricsTest(unittest.TestCase):
    def setUp(self):
        valid = np.array([True, False, True, False, True, True])
        labels = np.array([[1, 1, 1], [0, 0, 0], [0, 1, 0],
                           [1, 0, 1], [0, 1, 0], [1, 0, 1]])
        z = np.random.default_rng(7).normal(size=(6, 3, 3)).astype('float32')
        fixed = np.where(valid[:, None], [[.64, .16, .2]], [[.8, .2, 0]]).astype('float32')
        group = np.where(valid[:, None], [[.6, .2, .2]], [[.7, .3, 0]]).astype('float32')
        person = group.copy()
        person[valid, 1:] = [[.4, 0], [0, .4], [.1, .3], [.3, .1]]
        self.data = dict(targets=labels, sample_ids=np.arange(6), reliable=valid, branch_logits=z)
        self.weights = dict(fixed=fixed, group=group, person=person)

    def test_exact_archived_ap_and_controls(self):
        archived = mix(torch.tensor(self.weights['fixed']), torch.tensor(self.data['branch_logits']))
        # Two near-tied raw values satisfy replay tolerance. The archived order governs fixed AP.
        self.data['branch_logits'][:, 0] = 0.
        archived[:, 0] = torch.tensor([1e-6, -1e-6, -1e-6, 1e-6, -1e-6, 1e-6])
        self.data['fixed_logits'] = archived.numpy()
        r = evaluate(self.data, self.weights, 6, permutations=5)
        json.dumps(r, allow_nan=False)
        actual = r['classes']['Sadness']['fixed']
        self.assertEqual(actual, 100 * average_precision(archived.sigmoid().numpy()[:, 0], self.data['targets'][:, 0]))
        self.assertAlmostEqual(actual, 100., places=5)
        self.assertEqual(r['permutation_seeds'], [1, 2, 3, 4, 5])
        self.assertEqual(r['slices']['unreliable']['macro']['group'], r['slices']['unreliable']['macro']['person'])
        self.assertEqual(r['macro']['group'], r['macro']['mean_person'])
        self.assertEqual(r['weight_stats']['reliable']['person']['std'][0], 0.)
        self.assertEqual(r['macro'], evaluate(self.data, self.weights, 6, permutations=5)['macro'])

    def test_constant_allocation_matches_shuffle(self):
        self.weights['person'] = self.weights['group'].copy()
        report = evaluate(self.data, self.weights, 6, permutations=3)
        for method in ('person', 'mean_person', 'shuffle'):
            self.assertAlmostEqual(report['macro']['group'], report['macro'][method], places=12)
        self.assertIsNone(report['fixed_max_abs_logit_error'])

    def test_reject_person_specific_full_share(self):
        self.weights['person'][0] += [-.01, .01, 0.]
        with self.assertRaisesRegex(ValueError, 'budget'):
            evaluate(self.data, self.weights, 6, permutations=1)


if __name__ == '__main__':
    unittest.main()
