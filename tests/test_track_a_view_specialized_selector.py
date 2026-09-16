from __future__ import annotations

import unittest

import torch

from test_track_a_reproduction import FakeVisual
from multi_lane.track_a.model import MultiLaneModel


def make_model(selector_mode: str = "shared", num_selectors: int = 2) -> MultiLaneModel:
    return MultiLaneModel(
        FakeVisual(),
        (5, 3),
        num_selectors=num_selectors,
        num_prompts=2,
        num_prompt_layers=1,
        selector_mode=selector_mode,
        view_fusion="fixed_three_view",
    )


def view_batch() -> dict[str, torch.Tensor]:
    return {
        "full": torch.randn(3, 3, 4, 4),
        "person": torch.randn(3, 3, 4, 4),
        "face": torch.randn(3, 3, 4, 4),
        "face_reliable": torch.ones(3, dtype=torch.bool),
    }


class ViewSpecializedSelectorTest(unittest.TestCase):
    def test_view_specific_initialization_matches_shared_forward(self) -> None:
        torch.manual_seed(91)
        shared = make_model("shared")
        torch.manual_seed(91)
        specialized = make_model("view_specific")
        shared.activate_task(0)
        specialized.activate_task(0)
        inputs = view_batch()
        with torch.no_grad():
            expected = shared.current_all_logits(inputs)
            actual = specialized.current_all_logits(inputs)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(specialized.selectors.ndim, 4)
        self.assertTrue(torch.equal(
            specialized.selectors[:, 0], specialized.selectors[:, 1]
        ))
        self.assertTrue(torch.equal(
            specialized.selectors[:, 1], specialized.selectors[:, 2]
        ))

    def test_view_specific_banks_receive_independent_gradients(self) -> None:
        model = make_model("view_specific")
        model.activate_task(0)
        logits = model.current_all_logits(view_batch())
        logits.sum().backward()
        self.assertIsNotNone(model.selectors.grad)
        self.assertEqual(tuple(model.selectors.grad.shape), tuple(model.selectors.shape))
        for view_index in range(3):
            self.assertTrue(torch.isfinite(model.selectors.grad[:, view_index]).all())
            self.assertGreater(
                torch.count_nonzero(model.selectors.grad[:, view_index]).item(), 0
            )

    def test_task_activation_copies_each_view_and_freezes_previous_task(self) -> None:
        model = make_model("view_specific")
        model.activate_task(0)
        with torch.no_grad():
            model.selectors[0].normal_()
        previous = model.selectors[0].detach().clone()
        model.activate_task(1)
        self.assertTrue(torch.equal(model.selectors[1], previous))
        model.current_all_logits(view_batch()).sum().backward()
        self.assertIsNotNone(model.selectors.grad)
        self.assertEqual(torch.count_nonzero(model.selectors.grad[0]).item(), 0)
        self.assertGreater(torch.count_nonzero(model.selectors.grad[1]).item(), 0)

    def test_selector_capacity_control_matches_parameter_count(self) -> None:
        specialized = make_model("view_specific", num_selectors=2)
        shared_wide = make_model("shared", num_selectors=6)
        self.assertEqual(specialized.selectors.numel(), shared_wide.selectors.numel())


if __name__ == "__main__":
    unittest.main()
