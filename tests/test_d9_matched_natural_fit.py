import unittest

import numpy as np
import torch

from scripts.emotic.d9_matched_natural_fit import fit_natural
from scripts.emotic.d9_matched_run import fit_average


class MatchedNaturalATest(unittest.TestCase):
    def test_duplicate_versions_keep_each_person_equally_weighted(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(19)
        logits = rng.normal(size=(20, 3, 3)).astype(np.float32)
        data = dict(branch_logits=logits, reliable=np.arange(20) % 3 == 0,
                    targets=rng.integers(0, 2, (20, 3)).astype(np.float32))
        variants = {v: {'branch_logits': logits} for v in ('person', 'face')}
        natural = fit_natural(data)['weights']
        paired = fit_average(data, variants)['weights']
        np.testing.assert_allclose(natural, paired, rtol=0, atol=1e-5)
        self.assertEqual(natural[0][2], 0.0)
