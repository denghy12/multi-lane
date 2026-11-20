import unittest

import torch

from scripts.emotic.d9_natural_rank_eval import group_attributions, val_decision


class NaturalRankEvaluationTest(unittest.TestCase):
    def test_only_preregistered_arm_can_advance_and_tradeoff_rejected(self):
        def score(final, avg):
            return {'all': dict(final_mAP=final, average_mAP=avg)}
        summary = dict(paired=score(41, 50), rank=score(41, 50), rank_reliable_only=score(99, 99))
        self.assertEqual(val_decision(summary)['winner'], 'paired')
        summary['rank'] = score(42, 49.999)
        self.assertEqual(val_decision(summary)['winner'], 'paired')
        summary['rank'] = score(41, 50.01)
        self.assertEqual(val_decision(summary)['winner'], 'rank')

    def test_group_readouts_use_whole_heads_of_corresponding_people(self):
        paired = torch.tensor([[1, 2], [3, 4]])
        rank = torch.tensor([[5, 6], [7, 8]])
        result = group_attributions(paired, rank, torch.tensor([False, True]))
        self.assertEqual(result['rank_reliable_only'].tolist(), [[1, 2], [7, 8]])
        self.assertEqual(result['rank_unreliable_only'].tolist(), [[5, 6], [3, 4]])
