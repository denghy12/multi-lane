import unittest

import numpy as np
import torch

from scripts.emotic.d9_shrink_fit import GRID, cross_fit, image_folds, interpolate, select


class ShrinkOOFTest(unittest.TestCase):
    @staticmethod
    def data():
        rng = np.random.default_rng(21)
        n_images = 21
        ids = np.array([f'train:group/{i}.jpg#person={p}' for i in range(n_images) for p in (0, 1)])
        n = len(ids)
        return dict(sample_ids=ids, targets=rng.integers(0, 2, (n, 3)).astype('float32'),
                    reliable=np.tile([False, True], n_images), branch_logits=rng.normal(size=(n,3,3)).astype('float32'),
                    relation=rng.normal(size=(n, 8)).astype('float32'),
                    content=rng.normal(size=(n,3,9)).astype('float32'), delta=rng.normal(size=(n,2,9)).astype('float32'))

    def test_global_folds_ignore_row_and_task_order_and_preserve_images(self):
        d = self.data()
        ids = d['sample_ids']
        self.assertEqual(image_folds({0: ids, 3: ids[:8]}), image_folds({3: ids[:8][::-1], 0: ids[::-1]}))
        assigned = image_folds({0: ids})
        self.assertEqual(np.bincount(list(assigned.values())).tolist(), [7, 7, 7])
        with self.assertRaises(ValueError):
            image_folds({0: ['val:group/0.jpg#person=0']})

    def test_fold_held_out_labels_cannot_change_its_weights(self):
        torch.set_num_threads(1)
        d = self.data()
        lookup = image_folds({0: d['sample_ids']})
        folds, a, person, counts = cross_fit(d, lookup, 4)
        self.assertTrue(np.array_equal(folds[::2], folds[1::2]))
        self.assertEqual(sum(row['held']['people'] for row in counts), len(folds))
        changed = d.copy()
        changed['targets'] = d['targets'].copy()
        changed['targets'][folds==0] = 1 - changed['targets'][folds==0]
        _, a1, d1, _ = cross_fit(changed, lookup, 4)
        torch.testing.assert_close(a1[folds==0], a[folds==0], rtol=0, atol=0)
        torch.testing.assert_close(d1[folds==0], person[folds==0], rtol=0, atol=0)
        self.assertTrue(torch.equal(interpolate(a, person, 0), a))
        self.assertTrue(torch.equal(interpolate(a, person, 1), person))
        with self.assertRaises(ValueError):
            cross_fit(d, dict.fromkeys(lookup,0), 4)

    def test_selection_uses_equal_task_weight_and_smallest_exact_tie(self):
        scores = np.ones((3, len(GRID)))
        scores[0, 97] = 2
        scores[1:, 2:4] = 2
        best, curve = select(scores)
        self.assertEqual(best, 2)
        self.assertGreater(curve[best], curve[97])
        self.assertEqual(select(np.zeros_like(scores))[0], 0)
