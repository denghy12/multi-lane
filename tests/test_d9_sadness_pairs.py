import unittest

import numpy as np

from scripts.emotic.d9_sadness_pairs import attribute


class SadnessPairTest(unittest.TestCase):
    def test_flip_directions_and_face_availability(self):
        # Full ranks positive0 above negative0 and positive1 below negative1.
        # Person pushes positive0 down; Face pushes positive1 up.
        z = np.array([[3, -1, 3], [0, 0, 2], [2, 2, 2], [1, 1, 1.]])
        wa = np.tile([1., 0., 0.], (4, 1))
        wc = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.], [1., 0., 0.]])
        out = attribute(np.arange(4), [1, 1, 0, 0], z, wa, wc, [False, True, False, True],
                        [1, 4, 2, 3], [4, 2, 1, 3])
        total = out['pairs']['all']
        self.assertEqual(total['total'], 4)
        self.assertEqual(total['lost']['count'], 2)
        self.assertEqual(total['lost']['person_larger_in_flip_direction'], 2)
        self.assertEqual(total['gained']['count'], 1)
        self.assertEqual(total['gained']['face_larger_in_flip_direction'], 1)
        self.assertEqual(total['lost']['person_mean_margin_change'], -4.)
        self.assertEqual(out['people_groups']['positive_reliable']['face_term']['mean'], 2.)
        self.assertEqual(out['people_groups']['positive_unreliable']['face_term']['mean'], 0.)

    def test_invalid_or_tied(self):
        wa = np.array([[1., 0., 0.], [1., 0., 0.]])
        data = (['p', 'n'], [1, 0], np.zeros((2, 3)), wa, wa, [True, False])
        with self.assertRaises(ValueError):
            attribute(*data, [0, 1], [1, 2])
        result = attribute(*data, [1, 2], [2, 1])
        self.assertEqual(result['pairs']['all']['lost']['contributions_tied'], 1)
        self.assertEqual(result['pairs']['all']['lost']['no_directional_raw_change'], 1)
        self.assertIsNone(result['pairs']['all']['gained']['face_mean_margin_change'])
