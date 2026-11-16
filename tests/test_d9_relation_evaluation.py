import unittest

import torch

from scripts.emotic.evaluate_d9_relation import (
    metrics_for_stage, readouts, shuffle_sources, weight_statistics, weighted_logits,
)


class RelationEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.ids = ["z", "a", "b", "h", "i", "n", "d"]
        self.mask = torch.tensor([True, False, True, False, True, False, True])

    def test_external_weights_move_only_within_availability_and_same_task(self):
        donors = shuffle_sources(self.ids, self.mask, 2)
        for column in donors.T:
            self.assertEqual(sorted(column.tolist()), list(range(7)))
            self.assertTrue(torch.equal(self.mask[column], self.mask))
            self.assertFalse((column == torch.arange(7)).any())
        order = torch.tensor([3, 2, 0, 4, 5, 6, 1])
        other = shuffle_sources([self.ids[i] for i in order], self.mask[order], 2)
        self.assertTrue(torch.equal(order[other], donors[order]))
        self.assertEqual(shuffle_sources(["a", "b"], torch.tensor([True, False]), 1).flatten().tolist(), [0, 1])

    def test_metrics_use_model_outputs_and_intervene_only_on_readout(self):
        values = torch.arange(7 * 8 * 3).reshape(7, 8, 3).float() / 64
        weights = torch.zeros(7, 2, 3)
        weights[..., 0] = torch.linspace(.1, .7, 7)[:, None]
        weights[..., 1] = 1 - weights[..., 0]
        weights[self.mask, 1, 1] -= .1
        weights[self.mask, 1, 2] += .1
        native = weighted_logits(values, weights, (5, 3))
        predictions, donors = readouts(1, self.ids, self.mask, values, weights, native)
        expected = weighted_logits(values, weights[donors, torch.arange(2)], (5, 3))
        torch.testing.assert_close(predictions["shuffled_within_reliability"], expected)
        self.assertTrue(torch.equal(predictions["model"], native))
        self.assertTrue(torch.equal(predictions["person"], values[..., 1]))
        self.assertFalse(torch.equal(native, expected))
        fixed = predictions["fixed_same_branches"]
        torch.testing.assert_close(fixed[self.mask], (values[self.mask] * torch.tensor([.64, .16, .20])).sum(-1))
        torch.testing.assert_close(fixed[~self.mask], (values[~self.mask] * torch.tensor([.8, .2, 0.])).sum(-1))
        scores = metrics_for_stage(1, (values[..., 1] > 1).float(), self.mask, predictions)
        self.assertEqual(scores["model"]["all"]["samples"], 7)
        self.assertEqual(scores["full"]["reliable"]["samples"], 4)
        self.assertEqual(weight_statistics(weights, self.mask)["unreliable"]["std"][1][2], 0.)

    def test_static_readout_replays_exactly_and_invalid_face_is_rejected(self):
        weights = torch.tensor([[.8, .2, 0]]).repeat(7, 1, 1)
        weights[self.mask] = torch.tensor([.64, .16, .2])
        logits = torch.randn(7, 5, 3)
        native = weighted_logits(logits, weights, (5,))
        scores, _ = readouts(0, self.ids, self.mask, logits, weights, native)
        self.assertTrue(torch.equal(scores["shuffled_within_reliability"], native))
        weights[1] = torch.tensor([.64, .16, .2])
        with self.assertRaisesRegex(ValueError, "Unreliable"):
            readouts(0, self.ids, self.mask, logits, weights, native)


if __name__ == "__main__":
    unittest.main()
