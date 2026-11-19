import unittest

from scripts.emotic.d9_equal_updates_compare import changes


class MatchedUpdatesComparisonTest(unittest.TestCase):
    def test_remaining_gap_and_recovery_use_same_direction(self):
        self.assertEqual(changes(20.,21.,24.), dict(original_gap=4., gain_from_updates=1.,
                                                  full_minus_matched=3., recovered_fraction=.25))
        self.assertIsNone(changes(21.,22.,20.)['recovered_fraction'])
        self.assertIsNone(changes(21.,21.,21.)['recovered_fraction'])


if __name__ == '__main__':
    unittest.main()
