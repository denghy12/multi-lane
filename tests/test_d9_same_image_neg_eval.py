import unittest

import torch

from scripts.emotic.d9_same_image_neg_eval import top_k, val_decision


class SameImageEvaluationTest(unittest.TestCase):
    def test_decision_compares_expansion_and_objective(self):
        def score(final, avg):
            return {'all': dict(final_mAP=final, average_mAP=avg)}
        summary = dict(paired=score(40, 50), rank=score(99, 99), expanded_bce=score(40.1, 50),
                       expanded_rank=score(40, 51))
        self.assertEqual(val_decision(summary)['winner'], 'expanded_bce')
        summary['expanded_rank'] = score(40.1, 50)
        self.assertEqual(val_decision(summary)['winner'], 'expanded_bce')
        summary['expanded_rank'] = score(40.1, 50.1)
        self.assertEqual(val_decision(summary)['winner'], 'expanded_rank')
        summary.update(expanded_bce=score(40.1, 49), expanded_rank=score(40, 49.5))
        self.assertEqual(val_decision(summary)['winner'], 'paired')

    def test_top_errors_separate_target_emotion_from_whole_task(self):
        logits = torch.tensor([[0., 1., 0., 0., 0.], [4., 0., 0., 0., 0.],
                               [5., 1., 0., 0., 0.], [6., 0., 0., 0., 0.]])
        targets = torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]])
        result = top_k(0, logits, targets, [10., 50., 0., 0., 0.])
        first = result['by_class']['Affection']
        self.assertEqual((first['k'], first['fp'], first['fp_head_all_negative']), (2, 2, 1))
        self.assertIsNone(result['by_class']['Aversion']['precision'])
        self.assertEqual(result['current']['mean_ap'], 12.)
