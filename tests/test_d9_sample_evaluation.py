import unittest

import torch

from scripts.emotic.evaluate_d9_protected_gates import SAMPLE_READOUTS, assess
from scripts.emotic.evaluate_d9_relation import weighted_logits
from scripts.emotic.evaluate_d9_sample_fusion import comparisons, PROTECTED_BASELINE


class SampleFusionEvaluationTests(unittest.TestCase):
    def test_no_historical_reference_still_enforces_fixed_and_equal_for_holdout(self):
        ids = ["a", "b", "c", "d"]
        reliable = torch.tensor([True, False, False, True])
        branches = torch.randn(4, 5, 3, generator=torch.Generator().manual_seed(22))
        fixed = torch.tensor([[.64, .16, .2], [.8, .2, 0.]])[(~reliable).long(), None]
        equal = torch.tensor([[1/3, 1/3, 1/3], [.5, .5, 0.]])[(~reliable).long(), None]
        names = SAMPLE_READOUTS["sample_holdout"]
        data = {}
        for name in names:
            w = fixed if name == "fixed" else equal
            data[name] = (w.clone(), weighted_logits(branches, w, [5]))
        batch = (ids, torch.eye(5)[:4], reliable, branches, data)
        row, arrays = assess(0, batch, None, None, names)
        self.assertFalse(row["exact_replay"])
        self.assertIn("mid_asl_shuffled_logits", arrays)
        # A normalized but wrong equal rule would otherwise pass simple masking/reconstruction checks.
        data["equal"] = data["fixed"]
        with self.assertRaises(AssertionError):
            assess(0, batch, None, None, names)

    def test_relative_success_does_not_substitute_for_absolute_target(self):
        baseline = PROTECTED_BASELINE["test"]
        def metric(a, b):
            return {"all": {"final_mAP": baseline["final_mAP"] + a,
                            "average_mAP": baseline["average_mAP"] + b}}
        table = {"fixed": metric(-.2, -.5), "equal": metric(-1., -1.),
                 "mid_bce": metric(-.1, .1), "mid_asl": metric(0., .1)}
        row = comparisons(table, ("mid_asl", "mid_bce"), "test", False)
        self.assertTrue(row["mid_bce"]["same_branches_fixed"]["both_at_least_reference"])
        self.assertFalse(row["mid_bce"]["historical_protected_full_train"]["both_at_least_reference"])
        self.assertTrue(row["mid_asl"]["historical_protected_full_train"]["both_at_least_reference"])
        self.assertNotIn("historical_protected_full_train", comparisons(table, ("mid_asl", "mid_bce"), "test", True)["mid_asl"])


if __name__ == "__main__":
    unittest.main()
