import copy
import unittest

from multi_lane.track_a.summarize_dual_view_seeds import (
    LOCKED_RULE, LOCKED_VALIDATION_SUMMARY_SHA256, METRICS, aggregate_seed_rows,
)


class DualViewSeedSummaryTest(unittest.TestCase):
    @staticmethod
    def rows():
        rows = []
        for seed in range(3):
            rows.append((seed, {
                "evaluation_split": "test", "search_performed_on_test": False,
                "locked_rule": dict(LOCKED_RULE),
                "validation_selection": {"sha256": LOCKED_VALIDATION_SUMMARY_SHA256},
                "anchors": {
                    "full": {"metrics": {key: 30.0 + seed for key in METRICS}},
                    "person": {"metrics": {key: 25.0 + seed for key in METRICS}},
                },
                "fixed_fusion": {"metrics": {key: 31.0 + seed for key in METRICS}},
            }))
        return rows

    def test_paired_mean_and_sample_standard_deviation(self):
        result = aggregate_seed_rows(self.rows())
        self.assertEqual(result["groups"]["fusion"]["final_mAP"]["mean"], 32.0)
        self.assertEqual(result["groups"]["fusion"]["final_mAP"]["sample_std"], 1.0)
        self.assertEqual(result["paired_fusion_minus_full"]["final_mAP"]["values"], [1.0] * 3)
        self.assertEqual(result["positive_final_mAP_seeds"], 3)

    def test_rejects_duplicate_seed(self):
        rows = self.rows()
        rows[2] = (1, rows[2][1])
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            aggregate_seed_rows(rows)

    def test_rejects_changed_fusion_weight(self):
        rows = copy.deepcopy(self.rows())
        rows[1][1]["locked_rule"]["person_weight"] = 0.25
        with self.assertRaisesRegex(ValueError, "Fusion rule"):
            aggregate_seed_rows(rows)


if __name__ == "__main__":
    unittest.main()
