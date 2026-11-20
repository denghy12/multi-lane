import unittest

from scripts.emotic.d9_split90_compare import verify_roles


class SplitComparisonTest(unittest.TestCase):
    def test_good_global_count_cannot_hide_changed_image_universe(self):
        old = {str(i): 'fusion' if i == 0 else 'branch' for i in range(10)}
        new = {str(i): 'fusion' if i == 1 else 'branch' for i in range(10)}
        self.assertEqual(verify_roles(new, old)['changed_roles'], 2)
        new['other'] = new.pop('9')
        with self.assertRaises(ValueError):
            verify_roles(new, old)
