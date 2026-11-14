"""Regression checks for source-image OOF, ranking, and decision direction."""
import unittest

import numpy as np
import torch

from scripts.emotic.evaluate_r1_early_fusion import class_pairs, gate_deltas, pair_loss, train_folds


class EarlyFusionAnalysisTests(unittest.TestCase):
    def test_train_people_from_one_image_remain_on_one_side(self):
        ids = ["train:clip/a.jpg#person=0", "train:clip/a.jpg#person=1", "train:clip/b.jpg#person=0"]
        folds = train_folds(ids)
        self.assertEqual(folds[0], folds[1])
        np.testing.assert_array_equal(train_folds(list(reversed(ids))), folds[::-1])
        with self.assertRaises(ValueError):
            train_folds(["val:clip/a.jpg#person=0"])

    def test_exact_class_balanced_pair_objective_and_shift_invariance(self):
        y = torch.tensor([[1.,1.],[0.,1.],[0.,0.],[0.,0.]])
        z = torch.tensor([[1.,2.],[0.,0.],[-1.,-1.],[2.,-2.]], requires_grad=True)
        pairs = class_pairs(y)
        result = pair_loss(z, pairs)
        oracle = 0.
        for c in range(2):
            differences = [np.logaddexp(0, float(z[j,c]-z[i,c])) for i in range(4)
                           for j in range(4) if y[i,c] == 1 and y[j,c] == 0]
            oracle += np.mean(differences) / 2
        self.assertAlmostEqual(float(result), oracle, places=6)
        self.assertAlmostEqual(float(result), float(pair_loss(z+10.,pairs)), places=6)
        result.backward()
        self.assertTrue(bool((z.grad[y.bool()] < 0).all()))
        self.assertTrue(bool((z.grad[~y.bool()] > 0).all()))
        with self.assertRaises(ValueError):
            class_pairs(torch.zeros(3,1))

    def test_gate_uses_epoch30_oof_and_both_validation_metrics(self):
        report = [{"pooled_oof": {15: {"learned":{"mAP":1.},"fixed":{"mAP":99.}},
                                  30: {"learned":{"mAP":4.},"fixed":{"mAP":3.}}}}]
        val = {k: {"all": {"summary":{"final_mAP":f, "average_mAP":a}}} for k,f,a in
               [("e15_bce", 42., 49.),("fixed", 41., 50.)]}
        d = gate_deltas(report,val)
        self.assertEqual(d["train_crossfit_e15_to_e30_taskmean_delta"], 1.)
        self.assertEqual(d["val_e35_final_delta"], 1.)
        self.assertEqual(d["val_e35_average_delta"], -1.)


if __name__ == "__main__":
    unittest.main()
