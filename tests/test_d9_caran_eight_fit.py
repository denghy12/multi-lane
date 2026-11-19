import unittest

import numpy as np
import torch

from scripts.emotic.d9_caran_eight_fit import METHODS, initialize, readout_weights, train_head


class FusionHeadTests(unittest.TestCase):
    def setUp(self):
        self.raw = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2)
        self.valid = torch.tensor([False, True, False, True])
        self.classes = 3

    def test_equal_start_and_exact_unavailable_face_mask(self):
        face_swapped = self.raw.clone()
        face_swapped[~self.valid, 2] = 1e6
        expected = torch.tensor([[.5, .5, 0.], [1/3]*3])[self.valid.long()]
        for name in METHODS:
            model = initialize(name, self.classes, 6)
            w = readout_weights(model, self.raw, self.valid, self.classes)
            torch.testing.assert_close(w, expected[:, None].expand_as(w))
            if 'fc' in model:
                model['fc'].fill_(0.4)
            torch.testing.assert_close(readout_weights(model, self.raw, self.valid, self.classes),
                                       readout_weights(model, face_swapped, self.valid, self.classes), rtol=0, atol=0)

    def test_head_output_varies_along_intended_axes(self):
        for name in METHODS:
            model = initialize(name, self.classes, 6)
            model['bias'][1, -1, 0] = .3
            if 'fc' in model:
                model['fc'][0, 0] = .1
            w = readout_weights(model, self.raw, self.valid, self.classes)
            self.assertEqual(not torch.equal(w[1, 0], w[1, -1]), name in ('class_static', 'person_class'))
            self.assertEqual(not torch.equal(w[1, 0], w[3, 0]), name in ('person_task', 'person_class'))

    def test_fullbatch_training_is_finite_and_learns_synthetic_example(self):
        rng = np.random.RandomState(3)
        targets = rng.randint(2, size=(4, 3)).astype(np.float32)
        signs = 6 * (targets - 0.5)
        logits = np.stack([signs, -signs, signs / 2], axis=-1)
        d = dict(reliable=self.valid.numpy(), targets=targets, branch_logits=logits)
        for name in METHODS:
            model, summary = train_head(name, d, self.raw / 30., self.valid, torch.tensor(targets),
                                       torch.tensor(logits), steps=8)
            self.assertLess(summary['final_bce'], summary['initial_bce'])
            self.assertTrue(all(torch.isfinite(p).all() for p in model.values()))
