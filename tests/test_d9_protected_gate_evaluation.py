import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.emotic.evaluate_d9_protected_gates import (
    READOUTS, VIEWS, assess, cohort_means, mean_readout,
)
from scripts.emotic.evaluate_d9_relation import weighted_logits


class ProtectedGateEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ref = Path(self.temp.name) / "task0.npz"
        self.ids = ["c", "a", "b", "d"]
        self.y = torch.eye(5)[:4]
        self.mask = torch.tensor([True, False, True, False])
        self.logits = torch.randn(4, 5, 3, generator=torch.Generator().manual_seed(95))
        fixed = torch.tensor([[.64, .16, .20], [.8, .2, 0.]]).repeat(2, 1).unsqueeze(1)
        varying = fixed.clone()
        varying[2, :, 0] -= .15
        varying[2, :, 2] += .15
        self.output = {}
        for name in READOUTS:
            w = varying if name in ("relation", "summary") else fixed
            self.output[name] = (w, weighted_logits(self.logits, w, [5]))
        self.batch = self.ids, self.y, self.mask, self.logits, self.output
        # Physical storage order may differ. The semantic check is per person and class.
        order = [3, 1, 2, 0]
        np.savez(self.ref, sample_ids=np.asarray(self.ids)[order], targets=self.y[order].numpy(),
                 reliable=self.mask[order].numpy(), branch_logits=self.logits[order].numpy(),
                 view_order=np.asarray(VIEWS), task_sizes=np.asarray([5]),
                 weights=fixed[order].numpy(), fused_logits=self.output["fixed"][1][order].numpy())

    def test_validation_mean_locked_for_test_and_fixed_controls_replay(self):
        val, arrays = assess(0, self.batch, self.ref, None)
        self.assertTrue(val["exact_replay"])
        for name in ("fixed", "static"):
            for suffix in ("_val_mean_logits", "_shuffled_logits"):
                np.testing.assert_array_equal(arrays[name + suffix], arrays[name + "_logits"])
        # A new split can have different learned weights, but the mean is still the locked Val constant.
        changed = dict(self.output, summary=self.output["fixed"])
        test, test_arrays = assess(0, (*self.batch[:-1], changed), self.ref, val["validation_cohort_means"])
        self.assertEqual(test["validation_cohort_means"], val["validation_cohort_means"])
        np.testing.assert_array_equal(test_arrays["summary_val_mean_logits"], arrays["summary_val_mean_logits"])
        self.assertFalse(np.array_equal(test_arrays["summary_val_mean_logits"], test_arrays["summary_logits"]))

    def test_direction_names_preserve_all_replay_controls(self):
        from multi_lane.track_a.d9_relation_fusion import DIRECTION_READOUTS
        outputs = {name: self.output['summary'] for name in DIRECTION_READOUTS}
        outputs['fixed'] = self.output['fixed']
        row, arrays = assess(0, (*self.batch[:-1], outputs), self.ref, None,
                             names=DIRECTION_READOUTS)
        self.assertTrue(row['exact_replay'])
        self.assertEqual(set(row['weights']), set(DIRECTION_READOUTS))
        for name in DIRECTION_READOUTS:
            self.assertIn(name + '_val_mean_logits', arrays)
            self.assertIn(name + '_shuffled_logits', arrays)

    def test_replay_rejects_one_subject_logit_change(self):
        self.logits[1, 3, 2] += .05
        with self.assertRaisesRegex(ValueError, "branch_logits differs for a"):
            assess(0, self.batch, self.ref, None)

    def test_absent_validation_cohort_has_no_test_fitted_replacement(self):
        weights, scores = self.output["summary"]
        absent = cohort_means(weights[:1], self.mask[:1])
        self.assertEqual(set(absent), {"reliable"})
        fused, ok = mean_readout(0, self.logits, weights, scores, self.mask, absent)
        self.assertTrue(torch.equal(ok, self.mask))
        self.assertTrue(torch.isnan(fused[~ok]).all())
        means = {name: absent for name in READOUTS}
        row, _ = assess(0, self.batch, self.ref, means)
        self.assertEqual(set(row["metrics"]["summary_val_mean"]), {"reliable"})


if __name__ == "__main__":
    unittest.main()
