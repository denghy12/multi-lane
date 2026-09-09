from __future__ import annotations

import unittest

import numpy as np

from multi_lane.track_a.fuse_fixed_three_view_test import (
    LOCKED_FALLBACK_WEIGHTS,
    LOCKED_RELIABLE_WEIGHTS,
)


class FixedThreeViewTest(unittest.TestCase):
    def test_locked_weights_are_normalized_and_face_is_masked(self) -> None:
        self.assertAlmostEqual(float(LOCKED_RELIABLE_WEIGHTS.sum()), 1.0)
        self.assertAlmostEqual(float(LOCKED_FALLBACK_WEIGHTS.sum()), 1.0)
        np.testing.assert_array_equal(
            LOCKED_RELIABLE_WEIGHTS,
            np.asarray([0.64, 0.16, 0.20], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            LOCKED_FALLBACK_WEIGHTS,
            np.asarray([0.80, 0.20, 0.00], dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
