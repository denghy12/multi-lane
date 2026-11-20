import unittest

import numpy as np
import torch

from scripts.emotic.d9_matched_natural_fit import fit_natural
from scripts.emotic.d9_matched_run import fit_average
from scripts.emotic.d9_reliable_blur_fit import ARMS, fit_arm, variants_for


class ReliableBlurFitTest(unittest.TestCase):
    @staticmethod
    def example():
        rng = np.random.default_rng(17)
        shape = (26, 3, 3)
        natural = rng.standard_normal(shape).astype('float32')
        reliable = np.arange(26) % 3 != 0
        person = natural.copy()
        person[..., 1] -= rng.standard_normal(shape[:2]).astype('float32')
        face = natural.copy()
        face[reliable, :, 2] += rng.standard_normal((reliable.sum(), 3)).astype('float32')
        return dict(branch_logits=natural, person_branch_logits=person, face_branch_logits=face,
                    reliable=reliable, targets=rng.integers(0, 2, shape[:2]).astype('float32'))

    def test_slots_keep_missing_group_and_natural_mass(self):
        data = self.example()
        mask = data['reliable']
        for arm in ARMS:
            for view, slot in variants_for(data, arm).items():
                original = data[f'{view}_branch_logits']
                np.testing.assert_array_equal(slot['branch_logits'][~mask], original[~mask])
                expected = original if arm in ('paired', view) else data['branch_logits']
                np.testing.assert_array_equal(slot['branch_logits'][mask], expected[mask])

    def test_endpoints_replay_and_missing_face_row_is_shared(self):
        torch.set_num_threads(1)
        data = self.example()
        both = {v:dict(branch_logits=data[v+'_branch_logits']) for v in ('person', 'face')}
        paired = fit_average(data, both)
        sealed = dict(center=paired['params']['bias'], weights=paired['weights'])
        natural = fit_natural(data)
        fits = {m:fit_arm(data, m, sealed, natural) for m in ARMS}
        self.assertEqual(fits['paired']['center'], sealed['center'])
        self.assertEqual(fits['natural']['center'][1], natural['center'][1])
        for row in fits.values():
            self.assertEqual(row['center'][0], sealed['center'][0])
            self.assertEqual(row['weights'][0], sealed['weights'][0])
            self.assertEqual(row['weights'][0][2], 0.)


if __name__ == '__main__':
    unittest.main()
