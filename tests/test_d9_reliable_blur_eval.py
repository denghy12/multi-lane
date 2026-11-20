"""Validation alone selects a replacement for the historical paired readout."""
import unittest

from scripts.emotic.d9_reliable_blur_eval import CANDIDATES, val_decision


def summary(**methods):
    values = {name: (50., 55.) for name in (*CANDIDATES, 'paired')}
    values.update(methods)
    return {name: {'all': dict(final_mAP=final, average_mAP=mean)} for name,(final, mean) in values.items()}


class ReliableBlurDecisionTest(unittest.TestCase):
    def test_tradeoff_and_identical_control_do_not_advance(self):
        scores = summary(person=(51.,54.9), face=(49.9,56.))
        self.assertEqual(val_decision(scores)['winner'], 'paired')

    def test_eligible_candidates_rank_by_final_then_average(self):
        scores = summary(natural=(50.1,55.), person=(50.1,55.2), face=(50.,55.9))
        decision = val_decision(scores)
        self.assertEqual(decision['eligible'], ['natural','person','face'])
        self.assertEqual(decision['winner'], 'person')
