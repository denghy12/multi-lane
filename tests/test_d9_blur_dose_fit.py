import unittest

import numpy as np
import torch

from scripts.emotic.d9_blur_dose_fit import METHODS, arm_data, fit_arm, fit_weighted, weighted_bce
from scripts.emotic.d9_matched_run import fit_average


class BlurDoseFitTest(unittest.TestCase):
    def test_version_mass_preserves_people_and_face_unavailable_rows(self):
        reliable = torch.tensor([True, False])
        row = torch.tensor([[2., 4., 6.], [8., 10., 500.]], requires_grad=True)
        loss = weighted_bce(row, reliable, True)
        self.assertEqual(float(loss), (3.5 + 9.) / 2)
        loss.backward()
        torch.testing.assert_close(row.grad, torch.tensor([[.25, .125, .125], [.25, .25, 0.]]))
        self.assertEqual(float(weighted_bce(row, reliable, False)), (4. + 9.) / 2)

    def test_original_fit_and_unavailable_weights_replay(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(17)
        natural = rng.standard_normal((20, 3, 3)).astype(np.float32)
        data = dict(branch_logits=natural, person_branch_logits=natural + .5,
                    face_branch_logits=natural - .3, reliable=np.arange(20) % 3 != 0,
                    targets=rng.integers(0, 2, (20, 3)).astype(np.float32))
        mild = dict(data)
        for view in ('person', 'face'):
            key = view + '_branch_logits'
            mild[key] = (data[key] + data['branch_logits']) / 2
        base = fit_average(data, {v:dict(branch_logits=data[v+'_branch_logits']) for v in ('person', 'face')})
        reference = dict(center=base['params']['bias'], weights=base['weights'])
        self.assertEqual(fit_weighted(data, False), reference)
        for method in METHODS:
            altered = arm_data(data, mild, method)
            for view in ('person', 'face'):
                key = view + '_branch_logits'
                np.testing.assert_array_equal(altered[key][~data['reliable']], data[key][~data['reliable']])
                if method.endswith('168'):
                    np.testing.assert_array_equal(altered[key][data['reliable']], mild[key][data['reliable']])
            fit = fit_arm(data, mild, method, reference)
            self.assertEqual(fit['center'][0], reference['center'][0])
            self.assertEqual(fit['weights'][0], reference['weights'][0])
            self.assertTrue(np.isfinite(fit['fit']['final']['objective']))


if __name__ == '__main__':
    unittest.main()
