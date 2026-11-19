import unittest

import numpy as np

from scripts.emotic.d9_paired_a_extract import verify_pairs


class PairedHeldCacheTest(unittest.TestCase):
    def setUp(self):
        z = np.array([[[1., 2., 0.]], [[2., 3., 4.]]])
        self.reference = dict(sample_ids=np.array(['person1', 'person2']), targets=np.ones((2, 1)),
                              reliable=np.array([False, True]), branch_logits=z)
        self.data = {k: v.copy() for k, v in self.reference.items()}
        self.data.update(person_branch_logits=z.copy(), face_branch_logits=z.copy())

    def test_allowed_recipients_and_mask(self):
        self.data['person_branch_logits'][:, :, 1] += 1
        self.data['face_branch_logits'][1, :, 1:] += 1
        verify_pairs(self.data, self.reference)
        self.data['face_branch_logits'][0, :, 1] += 1
        with self.assertRaises(AssertionError):
            verify_pairs(self.data, self.reference)

    def test_rejects_wrong_natural_order_or_corrupted_full(self):
        self.data['sample_ids'] = self.data['sample_ids'][::-1]
        with self.assertRaises(AssertionError):
            verify_pairs(self.data, self.reference)
        self.data['sample_ids'] = self.reference['sample_ids']
        self.data['person_branch_logits'][0, :, 0] += 1
        with self.assertRaises(AssertionError):
            verify_pairs(self.data, self.reference)
