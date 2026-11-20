import unittest

import numpy as np

from scripts.emotic.d9_face_budget_fit import aligned_mix


class MatchedFaceBudgetTests(unittest.TestCase):
    def pair(self, offset):
        logits = np.arange(12).reshape(2, 2, 3) + offset
        data = dict(sample_ids=np.array(['a','b']), reliable=np.array([True, False]),
                    targets=np.ones((2,2)), branch_logits=logits)
        return data, {k: {'branch_logits': logits + i} for k,i in [('person', 100), ('face', 200)]}

    def test_each_view_uses_same_variant(self):
        old, new = self.pair(0), self.pair(1000)
        natural, variants = aligned_mix(old, new)
        for a,b,c in [(old[0],new[0],natural)] + [(old[1][v],new[1][v],variants[v]) for v in variants]:
            np.testing.assert_array_equal(c['branch_logits'][..., :2], b['branch_logits'][..., :2])
            np.testing.assert_array_equal(c['branch_logits'][..., 2], a['branch_logits'][..., 2])

    def test_rejects_unpaired_people(self):
        old, new = self.pair(0), self.pair(1000)
        new[0]['sample_ids'] = new[0]['sample_ids'][::-1]
        with self.assertRaises(AssertionError):
            aligned_mix(old, new)
