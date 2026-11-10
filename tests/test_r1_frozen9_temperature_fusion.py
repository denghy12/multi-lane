import unittest
from unittest.mock import patch

import numpy as np
import torch

from multi_lane.track_a.runner import task_indices
from scripts.emotic.run_r1_frozen9_fusion_backprop import rank_support, smooth_ap_backward
from scripts.emotic.run_r1_frozen9_learned_fusion import FIXED, Rows
from scripts.emotic.run_r1_frozen9_temperature_fusion import (
    ANNEALED, LOW_TEMPERATURE, ORIGINAL, TEMPERATURE, Task0TemperatureFit, annealed_temperature,
)


class TemperatureFusionTests(unittest.TestCase):
    def test_annealing_schedule_updates(self):
        values = np.array([annealed_temperature(step) for step in range(301)])
        np.testing.assert_array_equal(values[:100], TEMPERATURE)
        np.testing.assert_allclose(np.diff(values[99:250]), (LOW_TEMPERATURE - TEMPERATURE) / 150)
        np.testing.assert_array_equal(values[250:], LOW_TEMPERATURE)
        self.assertEqual(values[249], LOW_TEMPERATURE)

    def test_temperature_changes_objective_and_gradient(self):
        support = rank_support(torch.tensor([[1.], [0.]]))
        gradients = []
        for temperature in (TEMPERATURE, LOW_TEMPERATURE):
            z = torch.tensor([[0.], [.4]], requires_grad=True)
            loss = smooth_ap_backward(z, support, temperature=temperature)
            ahead = torch.sigmoid(torch.tensor(.4 / temperature))
            torch.testing.assert_close(loss, 1 - 1 / (1 + ahead))
            self.assertTrue(torch.isfinite(z.grad).all())
            self.assertLess(z.grad[0, 0].item(), 0)
            gradients.append(z.grad[0, 0].item())
        self.assertNotEqual(*gradients)

    def test_only_task0_invokes_temperature_fit(self):
        ids = np.array(["val:test#person=0", "val:test#person=1"])
        rows = Rows(ids, None, None, np.array([False, True]))
        mask = np.ones(len(ids), dtype=bool)
        fitter = Task0TemperatureFit("cpu", full_val=True)
        with patch("scripts.emotic.run_r1_frozen9_temperature_fusion.fit_controls",
                   return_value=({"fixed": FIXED, "bounded": FIXED}, {})), patch(
            "scripts.emotic.run_r1_frozen9_temperature_fusion.fit_task0",
            return_value=({ORIGINAL: FIXED, ANNEALED: FIXED}, {}),
        ) as fit_task0:
            first, _ = fitter(rows, mask, list(task_indices(0)))
            next_task, support = fitter(rows, mask, list(task_indices(1)))
            self.assertEqual(support["temperature_fit_task"], 0)
            np.testing.assert_array_equal(first[ANNEALED], next_task[ANNEALED])
            fit_task0.assert_called_once()
