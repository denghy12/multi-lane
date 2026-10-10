import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from multi_lane.track_a.compare_incremental_view_contribution import fixed_cohort_drift
from multi_lane.track_a.runner import fixed_view_sampler_seed


class IncrementalViewAuditTest(unittest.TestCase):
    def test_explicit_sampler_is_paired_despite_unrelated_rng_and_preserves_defaults(self):
        self.assertIsNone(fixed_view_sampler_seed(1, 2, "incremental", False))
        self.assertEqual(fixed_view_sampler_seed(1, 2, "joint26", False), 1)
        seed = fixed_view_sampler_seed(1, 2, "incremental", True)
        self.assertEqual(seed, 2019)
        def order():
            return torch.cat(list(DataLoader(list(range(100)), batch_size=13, shuffle=True,
                                            generator=torch.Generator().manual_seed(seed))))
        first = order()
        torch.rand(1000)
        self.assertTrue(torch.equal(first, order()))

    def test_old_cohort_ignores_new_samples_and_class_columns(self):
        first = dict(sample_ids=np.asarray(["a", "b"]), targets=np.asarray([[1.], [0.]]),
                     logits=np.asarray([[1.], [-1.]]), probabilities=np.asarray([[.73], [.27]]), class_indices=np.asarray([0]))
        later = dict(sample_ids=np.asarray(["new", "b", "a"]), targets=np.asarray([[0.,1.], [0.,1.], [1.,0.]]),
                     logits=np.asarray([[100.,5.], [-1.,3.], [1.,7.]]),
                     probabilities=np.asarray([[1.,.9], [.27,.8], [.73,.99]]), class_indices=np.asarray([0,1]))
        result = fixed_cohort_drift(first, later)
        self.assertEqual(result["mean_absolute_logit_drift"], 0)
        self.assertEqual(result["mAP_change"], 0)
        later["logits"][2,0] += 2
        self.assertEqual(fixed_cohort_drift(first, later)["mean_absolute_logit_drift"], 1)
        later["targets"][2,0] = 0
        with self.assertRaisesRegex(ValueError, "labels changed"):
            fixed_cohort_drift(first, later)
