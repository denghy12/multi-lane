import unittest

from scripts.emotic.d9_blur_dose_eval import CANDIDATES, val_decision


def summary(**overrides):
    pairs = {name: (43., 51.) for name in ('paired_112', *CANDIDATES)}
    pairs.update(overrides)
    return {name: {'all': dict(final_mAP=final, average_mAP=average)}
            for name, (final, average) in pairs.items()}


class BlurDoseSelectionTest(unittest.TestCase):
    def test_final_average_tradeoffs_and_exact_replay_do_not_replace_control(self):
        result = val_decision(summary(gentle_168=(43.1, 50.9), clean_112=(42.9, 51.1)))
        self.assertEqual(result['winner'], 'paired_112')
        self.assertEqual(result['eligible'], [])

    def test_eligible_methods_use_prespecified_final_average_then_arm_order(self):
        scores = summary(gentle_168=(43.1, 51.), clean_112=(43.1, 51.1), clean_168=(43.0, 51.5))
        self.assertEqual(val_decision(scores)['winner'], 'clean_112')
        scores = summary(gentle_168=(43.1, 51.1), clean_112=(43.1, 51.1), clean_168=(43.1, 51.1))
        self.assertEqual(val_decision(scores)['winner'], 'gentle_168')
