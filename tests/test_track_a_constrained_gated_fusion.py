import unittest

import numpy as np

from multi_lane.track_a.search_constrained_gated_fusion import (
    Geometry,
    geometry_cell,
    search_constrained_gate,
)


class ConstrainedGatedFusionTest(unittest.TestCase):
    def test_geometry_cell_uses_fixed_boundaries(self):
        self.assertEqual(
            geometry_cell(0.05, 1.5, 1),
            "area=small_lt_0.10|aspect=regular_le_2.0|people=single",
        )
        self.assertEqual(
            geometry_cell(0.10, 2.01, 2),
            "area=medium_0.10_0.30|aspect=extreme_gt_2.0|people=multi",
        )
        self.assertEqual(
            geometry_cell(0.30, 2.0, 3),
            "area=large_ge_0.30|aspect=regular_le_2.0|people=multi",
        )

    def test_gate_search_requires_every_seed_to_match_global_winner(self):
        # Exercise the hard eligibility logic with a tiny deterministic stub.
        import multi_lane.track_a.search_constrained_gated_fusion as module

        original = module._record_rule
        calls = []

        def fake_record(*args, **kwargs):
            class_delta = float(args[-2])
            quality_delta = float(args[-1])
            calls.append((class_delta, quality_delta))
            values = [10.0 + class_delta, 10.0 + quality_delta, 9.9]
            seeds = [
                {"seed": seed, "metrics": {"final_mAP": value, "average_mAP": value}}
                for seed, value in enumerate(values)
            ]
            return {
                "seeds": seeds,
                "aggregate": {
                    "final_mAP": {"mean": float(np.mean(values))},
                    "average_mAP": {"mean": float(np.mean(values))},
                },
            }

        global_winner = {
            "seeds": [
                {"seed": seed, "metrics": {"final_mAP": 10.0}}
                for seed in range(3)
            ],
            "aggregate": {"final_mAP": {"mean": 10.0}},
        }
        try:
            module._record_rule = fake_record
            candidates, winner = search_constrained_gate(
                {0: object(), 1: object(), 2: object()},
                {},
                [0] * 26,
                {},
                global_winner,
            )
        finally:
            module._record_rule = original
        self.assertEqual(len(candidates), 9)
        self.assertIsNone(winner)
        self.assertTrue(calls)

    def test_geometry_record_is_immutable_value(self):
        record = Geometry(0.2, 1.5, 2, geometry_cell(0.2, 1.5, 2))
        self.assertEqual(record.people_in_image, 2)
        with self.assertRaises(Exception):
            record.people_in_image = 1


if __name__ == "__main__":
    unittest.main()
