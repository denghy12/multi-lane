import json
import unittest

import numpy as np
import torch

from multi_lane.track_a.runner import average_precision
from scripts.emotic.d9_disagreement_metrics import _shuffle, evaluate


class DisagreementMetricsTest(unittest.TestCase):
    def setUp(self):
        n, k = 54, 3
        target = np.zeros((n, k), dtype=np.int64)
        target[50] = 1
        fixed = np.tile(np.linspace(1, 0, n, dtype=np.float32)[:, None], (1, k))
        candidate = np.zeros((n, k), dtype=bool)
        candidate[50:52] = True
        mid = target.astype(np.float32)
        self.data = dict(targets=target, reliable=np.zeros(n, dtype=bool), sample_ids=np.arange(n))
        self.pred = dict(fixed_logits=fixed, gap=np.full((n, k), 4.), candidate=candidate,
                         middle_probability=mid, score_probability=1 - mid,
                         receiver=np.ones((n, k), dtype=int))

    def test_exact_ap_group_auc_and_top50_entries(self):
        report = evaluate(self.data, self.pred, 6, permutations=3)
        json.dumps(report, allow_nan=False)
        row = report['classes']['Sadness']
        prob = torch.from_numpy(self.pred['fixed_logits']).sigmoid().numpy()[:, 0]
        self.assertEqual(row['ap']['fixed'], 100 * average_precision(prob, self.data['targets'][:, 0]))
        cand = row['candidate']['all']
        self.assertEqual((cand['auc_middle'], cand['auc_score']), (1., 0.))
        self.assertIsNone(row['candidate']['face']['auc_middle'])
        self.assertEqual(cand['middle']['positive_accept_rate'], 1.)
        self.assertEqual(cand['middle']['negative_accept_rate'], 0.)
        self.assertGreater(row['ap']['middle'], row['ap']['fixed'])
        self.assertEqual(row['ap']['middle'], row['ap']['oracle_positive'])
        top = row['top50']
        self.assertEqual([top[k] for k in ('rescued_by_all', 'rescued_retained',
                                            'false_alarms_by_all', 'false_alarms_prevented')], [1, 1, 1, 1])

    def test_shuffle_keeps_decision_counts_within_each_group(self):
        selected = np.array([True, False, False, True, False])
        groups = [np.array([0, 1]), np.array([2, 3])]
        swapped = _shuffle(selected, groups, 0, 2, 1)
        np.testing.assert_equal(swapped, [False, True, True, False, False])
        for idx in groups:
            self.assertEqual(int(swapped[idx].sum()), int(selected[idx].sum()))

    def test_no_candidates_falls_back_exactly_and_has_undefined_auc(self):
        self.pred['candidate'][:] = False
        row = evaluate(self.data, self.pred, 6, permutations=2)['classes']['Sadness']
        self.assertIsNone(row['candidate']['all']['auc_middle'])
        self.assertEqual(len(set(row['ap'].values())), 1)


if __name__ == '__main__':
    unittest.main()
