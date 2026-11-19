import json
import unittest

import numpy as np
import torch

from multi_lane.track_a.runner import average_precision
from scripts.emotic.d9_quality_eval import evaluate
from scripts.emotic.fit_d9_task10_probe import mix
from scripts.emotic.d9_direction_permutation import permutation_sources


class QualityEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.valid = np.array([True, False, True, False, True, True])
        y = np.array([[1, 1, 1], [0, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]])
        z = np.random.default_rng(8).normal(size=(6, 3, 3)).astype('float32')
        fixed = np.where(self.valid[:, None], [[.64, .16, .2]], [[.8, .2, 0]]).astype('float32')
        a = np.where(self.valid[:, None], [[.4, .3, .3]], [[.5, .5, 0]]).astype('float32')
        c = a.copy()
        c[self.valid] = [[.3,.2,.5], [.5,.4,.1], [.7,.1,.2], [.1,.6,.3]]
        c[~self.valid] = [[.8,.2,0],[.2,.8,0]]
        self.data = dict(sample_ids=np.arange(20, 26), reliable=self.valid, targets=y, branch_logits=z)
        self.weights = dict(fixed=fixed, A=a, B=a.copy(), C=c)

    def test_subgroups_and_paired_shuffle_reconstructed(self):
        report = evaluate(self.data, self.weights, 6, 3)
        json.dumps(report, allow_nan=False)
        self.assertEqual(set(report['macro']), {'fixed','A','B','C','shuffle','mean_C'})
        self.assertEqual(report['slices']['reliable']['people'], 4)
        self.assertEqual(report['slices']['unreliable']['positives'], [1,0,1])
        for pos, seed in enumerate(report['permutation_seeds']):
            donors = permutation_sources(self.data['sample_ids'], self.valid, 7, seed)[:, 6]
            scores = mix(torch.from_numpy(self.weights['C'][donors]),
                         torch.from_numpy(self.data['branch_logits'])).sigmoid().numpy()
            aps = [100*average_precision(p,y) for p,y in zip(scores.T,self.data['targets'].T)]
            self.assertAlmostEqual(report['shuffled_task_maps'][pos], float(np.mean(aps)))
        self.assertAlmostEqual(report['macro']['shuffle'], np.mean(report['shuffled_task_maps']))

    def test_archived_fixed_order_and_constant_allocation(self):
        self.weights['C'] = self.weights['A'].copy()
        self.data['branch_logits'][:, 0, :] = 0
        frozen = mix(torch.from_numpy(self.weights['fixed']), torch.from_numpy(self.data['branch_logits'])).numpy()
        frozen[:, 0] = np.where(self.data['targets'][:, 0], 1e-6, -1e-6)
        self.data['fixed_logits'] = frozen
        report = evaluate(self.data, self.weights, 6, 4)
        self.assertAlmostEqual(report['classes']['Sadness']['fixed'],100,places=5)
        self.assertAlmostEqual(report['macro']['C'], report['macro']['mean_C'])
        self.assertAlmostEqual(report['macro']['C'], report['macro']['shuffle'])

    def test_invalid_face_and_nonconstant_A_fail(self):
        self.weights['C'][1] = [.5, .4, .1]
        with self.assertRaisesRegex(ValueError,'allocation'):
            evaluate(self.data, self.weights, 6, 1)
        self.weights['C'][1] = [.6,.4,0]
        self.weights['A'][0] = [.5, .3, .2]
        with self.assertRaisesRegex(ValueError,'constant'):
            evaluate(self.data, self.weights, 6, 1)
