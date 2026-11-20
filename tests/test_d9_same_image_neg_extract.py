import unittest

from scripts.emotic.d9_same_image_neg_extract import extra_ids


class TrainRosterTest(unittest.TestCase):
    def test_same_image_unique_old_people_excluding_current(self):
        a, b = 'train:A#person=', 'train:B#person='
        self.assertEqual(extra_ids([a+'1'], [b+'0', a+'1', a+'0', a+'0']), [a+'0'])
        self.assertEqual(extra_ids([a+'1'], []), [])
