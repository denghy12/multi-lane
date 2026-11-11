import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.task_logit_fusion import ReliabilityLogitFusion, TaskLogitFusion
from scripts.emotic.run_r1_frozen9_fusion_backprop import smooth_ap_backward, rank_support
from scripts.emotic.run_r1_frozen9_learned_fusion import Rows
from scripts.emotic.run_r1_train_only_fusion import fit_task, load_checkpoint, save_checkpoint


def examples(task=0):
    size = 5 if task == 0 else 3
    targets = np.repeat(np.array([[1.], [0.], [1.], [0.]], dtype=np.float32), size, axis=1)
    signal = 2*targets-1
    reliable = np.array([True, True, False, False])
    logits = np.stack((-signal, 2*signal, 3*signal), axis=1)
    return Rows(np.array([f"train:img{i}#person=0" for i in range(4)]), logits, targets, reliable)


class TrainOnlyFusionTests(unittest.TestCase):
    def test_equal_start_convex_weights_and_availability(self):
        gate = ReliabilityLogitFusion()
        torch.testing.assert_close(gate.weight_matrix(),
                                   torch.tensor([[.5, .5, 0], [1/3, 1/3, 1/3]]))
        rows = examples()
        z = torch.from_numpy(rows.logits)
        mask = torch.from_numpy(rows.reliable)
        original = gate(z, mask)
        z[~mask, 2] += 1000
        torch.testing.assert_close(gate(z, mask)[~mask], original[~mask], rtol=0, atol=0)
        for raw in (-100., 0., 100.):
            with torch.no_grad():
                gate.raw_shares.fill_(raw)
            weights = gate.weight_matrix()
            self.assertTrue(bool((weights >= 0).all()))
            torch.testing.assert_close(weights.sum(1), torch.ones(2))
            self.assertEqual(float(weights[0, 2]), 0.)

    def test_gradient_agrees_with_analytic_fusion_derivative(self):
        gate = ReliabilityLogitFusion().double()
        z = torch.tensor([[[3.], [1.], [8.]], [[2.], [-1.], [9.]]], dtype=torch.float64)
        mask = torch.tensor([True, False])
        gate(z, mask).sum().backward()
        a, b = gate.raw_shares.sigmoid().detach()
        expected_a = a*(1-a)*((1-b)*(3-1)+(2-(-1)))
        expected_b = b*(1-b)*(8-(a*3+(1-a)*1))
        torch.testing.assert_close(gate.raw_shares.grad, torch.stack((expected_a, expected_b)))

    def test_training_rejects_validation_and_test_people(self):
        for split in ("val", "test"):
            rows = examples()
            rows.ids = np.array([v.replace("train:", split+":") for v in rows.ids])
            model = TaskLogitFusion((5, 3))
            with self.assertRaisesRegex(ValueError, "Invalid train"):
                fit_task(rows, 0, model, steps=1)
            self.assertEqual(len(model.tasks), 0)

    def test_train_fit_has_gradient_and_preserves_previous_task_function(self):
        model = TaskLogitFusion((5, 3))
        report, _ = fit_task(examples(), 0, model, steps=20)
        self.assertGreater(report["trace"][0]["gradient_norm"], 0)
        self.assertLess(report["trace"][-1]["smooth_ap_loss"], report["trace"][0]["smooth_ap_loss"])
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        inputs = torch.randn(7, 3, 8)
        mask = torch.arange(7) % 2 == 0
        old = model(inputs[:, :, :5], mask, 0).detach().clone()
        fit_task(examples(1), 1, model, steps=5)
        for key, value in saved.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(model(inputs, mask, 1)[:, :5], old, rtol=0, atol=0)
        self.assertFalse(any(p.requires_grad for p in model.tasks[0].parameters()))

    def test_chunked_loss_and_gradient_equal_full_list_autograd(self):
        y = torch.tensor([[1.], [0.], [1.], [0.]])
        scores = torch.tensor([[.1], [.3], [-.2], [-.4]], requires_grad=True)
        z = scores.detach().clone().requires_grad_()
        soft = torch.sigmoid((z[:, 0][None] - z[:, 0][:, None])/.15)
        soft = soft.masked_fill(torch.eye(4, dtype=torch.bool), 0)
        precision = (1+(soft*y[:, 0]).sum(1))/(1+soft.sum(1))
        expected = 1-precision[y[:, 0].bool()].mean()
        expected.backward()
        actual = smooth_ap_backward(scores, rank_support(y), chunk=1, temperature=.15)
        torch.testing.assert_close(actual, expected.detach())
        torch.testing.assert_close(scores.grad, z.grad)

    def test_saved_model_and_optimizer_roundtrip(self):
        model = TaskLogitFusion((5, 3))
        _, optimizer = fit_task(examples(), 0, model, steps=3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"task0.pth"
            save_checkpoint(path, model, {0: optimizer}, {"fit_split": "train"})
            restored = load_checkpoint(path, "cpu")
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
            state = torch.load(path)
            self.assertEqual(state["fit_split"], "train")
            self.assertTrue(state["optimizer_states"][0]["state"])

    def test_order_and_seen_class_width_are_enforced(self):
        model = TaskLogitFusion((5, 3))
        with self.assertRaisesRegex(ValueError, "order"):
            model.add_task(1)
        model.add_task(0)
        with self.assertRaisesRegex(ValueError, "class columns"):
            model(torch.zeros(2, 3, 8), torch.ones(2, dtype=torch.bool), 0)
