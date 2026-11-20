import unittest

from scripts.emotic.d9_face_budget_eval import validation_decision


class FaceBudgetDecisionTest(unittest.TestCase):
    def test_validation_requires_both_fixed_and_one_current_metric(self):
        fixed = dict(final_mAP=43.5, average_mAP=51.1)
        current = dict(final_mAP=43.8, average_mAP=51.3)
        for pair, expected in [((43.8, 51.1), True), ((43.5, 51.3), True),
                               ((43.79, 51.29), False), ((43.49, 51.4), False),
                               ((44.0, 51.09), False)]:
            with self.subTest(pair=pair):
                candidate = dict(zip(('final_mAP', 'average_mAP'), pair))
                self.assertEqual(validation_decision(candidate, fixed, current), expected)
