import unittest

import numpy as np

from scripts.emotic.d9_image_gate_extract import group_key, pack_images, select_images, summarize_people


class ImageFeaturesTest(unittest.TestCase):
    def test_sorted_whole_images_and_reverse_person_order(self):
        rosters = {'val:b': [('p3', None, True, True)], 'val:a': [('p1', None, True, True),
                                                                  ('p2', None, False, False)]}
        self.assertEqual(select_images(rosters, {'val:b', 'val:a'}, True), ['val:a', 'val:b'])
        self.assertEqual(group_key('val:a#person=1', 'val'), 'val:a')
        with self.assertRaises(ValueError):
            group_key('test:a#person=1', 'val')
        self.assertEqual(pack_images([2, 1, 6, 1], 3), [[0, 1], [2], [3]])
        rel = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 0, 0, 0, 0]], dtype='float32')
        f = np.array([True, False])
        summary = summarize_people(rel, f)
        np.testing.assert_array_equal(summary, [2, 3, 4, 5, 5, 6, 7, 8])
        np.testing.assert_array_equal(summary, summarize_people(rel[::-1], f[::-1]))
        np.testing.assert_array_equal(summarize_people(rel, np.zeros_like(f))[4:], [0]*4)


if __name__ == '__main__':
    unittest.main()
