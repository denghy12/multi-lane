import unittest

import numpy as np
import torch

from multi_lane.track_a.runner import task_indices
from scripts.emotic.run_r1_frozen9_fusion_backprop import (
    TEMPERATURE, ConditionalWeights, predict, predict_weights, rank_support, smooth_ap_backward,
)
from scripts.emotic.run_r1_frozen9_learned_fusion import Rows


class FusionBackpropTests(unittest.TestCase):
    def test_chunked_gradient_matches_dense_reference(self):
        scores = torch.tensor([[.2, 1.], [-.1, .2], [.5, .4], [.6, .5]], requires_grad=True)
        labels = torch.tensor([[1., 0.], [0., 0.], [1., 0.], [0., 0.]])
        support = rank_support(labels)
        self.assertEqual(len(support), 1)
        result = smooth_ap_backward(scores, support, chunk=1)
        gradient = scores.grad.clone()
        scores.grad = None
        ahead = ((scores[:, 0][None] - scores[:, 0][:, None]) / TEMPERATURE).sigmoid()
        ahead = ahead.masked_fill(torch.eye(4, dtype=torch.bool), 0)
        precision = (1 + ahead @ labels[:, 0]) / (1 + ahead.sum(dim=1))
        expected = 1 - precision[labels[:, 0].bool()].mean()
        expected.backward()
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(gradient, scores.grad)
        self.assertTrue((gradient[labels[:, 0].bool(), 0] < 0).all())

    def test_gate_starts_uniform_and_ignores_unavailable_face(self):
        logits = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2) / 10
        reliable = torch.tensor([True, False, True, False])
        task = ConditionalWeights(logits, reliable, False)
        sample = ConditionalWeights(logits, reliable, True)
        self.assertEqual(sum(p.numel() for p in task.parameters()), 3)
        self.assertEqual(sum(p.numel() for p in sample.parameters()), 18)
        torch.testing.assert_close(task(logits), sample(logits), rtol=0, atol=0)
        sample(logits).sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in sample.parameters()))
        with torch.no_grad():
            for slope in sample.slopes:
                slope.fill_(.4)
        model = sample.export()
        weights = predict_weights(logits, reliable, model)
        changed = logits.clone()
        changed[~reliable, 2] = 100
        torch.testing.assert_close(weights, predict_weights(changed, reliable, model), atol=0, rtol=0)
        torch.testing.assert_close(weights[~reliable, 2], torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(4))

    def test_future_class_logits_do_not_change_old_gate(self):
        rng = np.random.default_rng(7)
        first, second = list(task_indices(0)), list(task_indices(1))
        shape = (4, 3, len(first) + len(second))
        logits = rng.normal(size=shape).astype(np.float32)
        reliable = np.array([True, False, True, False])
        models = []
        for columns in (first, second):
            gate = ConditionalWeights(torch.from_numpy(logits[:, :, columns]),
                                      torch.from_numpy(reliable), True)
            with torch.no_grad():
                for slope in gate.slopes:
                    slope.fill_(.4)
            models.append({"sample_bce": gate.export()})
        bank = dict(enumerate(models))
        rows = Rows(np.arange(4), logits, np.zeros((4, shape[-1]), dtype=np.float32), reliable)
        stage0 = predict(rows, bank, 0, "sample_bce")
        np.testing.assert_array_equal(predict(rows, bank, 1, "sample_bce")[:, :len(first)], stage0)
        logits[:, :, second] *= -30
        np.testing.assert_array_equal(predict(rows, bank, 1, "sample_bce")[:, :len(first)], stage0)
