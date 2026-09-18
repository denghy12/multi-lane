from __future__ import annotations

import unittest

import torch

from test_track_a_reproduction import FakeVisual
from multi_lane.track_a.model import MultiLaneModel


def make_model(
    selector_mode: str = "shared",
    num_selectors: int = 2,
    prompt_mode: str = "shared",
    adapter_mode: str = "disabled",
    adapter_view_mode: str = "shared",
    adapter_view_bottleneck_dim: int = 0,
    view_classifier_mode: str = "shared_post_fusion",
    num_prompt_layers: int = 1,
    prompt_private_layers: int = 1,
    selector_view_residual_scale: float = 0.1,
) -> MultiLaneModel:
    return MultiLaneModel(
        FakeVisual(),
        (5, 3),
        num_selectors=num_selectors,
        num_prompts=2,
        num_prompt_layers=num_prompt_layers,
        selector_mode=selector_mode,
        selector_view_residual_scale=selector_view_residual_scale,
        prompt_mode=prompt_mode,
        prompt_private_layers=prompt_private_layers,
        adapter_mode=adapter_mode,
        adapter_bottleneck_dim=3,
        adapter_layer_indices=(0,),
        adapter_view_mode=adapter_view_mode,
        adapter_view_bottleneck_dim=adapter_view_bottleneck_dim,
        view_fusion="fixed_three_view",
        view_classifier_mode=view_classifier_mode,
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

    def test_shared_selector_residual_starts_equal_and_routes_by_view(self) -> None:
        torch.manual_seed(94)
        shared = make_model("shared")
        torch.manual_seed(94)
        residual = make_model("shared_residual")
        shared.activate_task(0)
        residual.activate_task(0)
        inputs = view_batch()
        with torch.no_grad():
            _, expected, _ = shared.current_all_logits_with_view_features(inputs)
            _, actual, _ = residual.current_all_logits_with_view_features(inputs)
        for name in ("full", "person", "face"):
            self.assertTrue(torch.equal(expected[name], actual[name]))
        self.assertTrue(torch.equal(
            residual.selector_view_residuals,
            torch.zeros_like(residual.selector_view_residuals),
        ))
        with torch.no_grad():
            residual.selector_view_residuals[0, 1].add_(0.1)
            _, changed, _ = residual.current_all_logits_with_view_features(inputs)
        self.assertGreater(torch.max((changed["person"] - actual["person"]).abs()).item(), 0)
        self.assertTrue(torch.equal(changed["full"], actual["full"]))
        self.assertTrue(torch.equal(changed["face"], actual["face"]))

    def test_late_prompt_residual_keeps_early_layers_shared(self) -> None:
        model = make_model(
            prompt_mode="late_view_residual_full_face",
            num_prompt_layers=2,
            prompt_private_layers=1,
        )
        model.activate_task(0)
        inputs = view_batch()
        with torch.no_grad():
            _, before, _ = model.current_all_logits_with_view_features(inputs)
            model.prompts[0][0, 1].add_(0.1)
            _, after, _ = model.current_all_logits_with_view_features(inputs)
        self.assertTrue(torch.equal(after["person"], before["person"]))
        self.assertTrue(torch.equal(after["full"], before["full"]))
        self.assertTrue(torch.equal(after["face"], before["face"]))
        with torch.no_grad():
            model.prompts[2][0, 1].add_(0.1)
            _, final, _ = model.current_all_logits_with_view_features(inputs)
        self.assertGreater(torch.max((final["full"] - after["full"]).abs()).item(), 0)
        self.assertTrue(torch.equal(final["person"], after["person"]))

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

    def test_view_specific_prompts_initialize_like_shared_and_route_by_view(self) -> None:
        torch.manual_seed(92)
        shared = make_model(prompt_mode="shared")
        torch.manual_seed(92)
        specialized = make_model(prompt_mode="view_specific")
        shared.activate_task(0)
        specialized.activate_task(0)
        inputs = view_batch()
        with torch.no_grad():
            _, shared_views, _ = shared.current_all_logits_with_view_features(inputs)
            _, actual_views, _ = specialized.current_all_logits_with_view_features(inputs)
        for name in ("full", "person", "face"):
            self.assertTrue(torch.equal(shared_views[name], actual_views[name]))
        self.assertEqual(specialized.prompts[0].ndim, 6)
        with torch.no_grad():
            specialized.prompts[0][0, 1].add_(0.1)
            _, changed, _ = specialized.current_all_logits_with_view_features(inputs)
        self.assertTrue(torch.equal(changed["full"], actual_views["full"]))
        self.assertTrue(torch.equal(changed["face"], actual_views["face"]))
        self.assertGreater(
            torch.max((changed["person"] - actual_views["person"]).abs()).item(), 0
        )

    def test_selective_prompt_keeps_person_shared_and_private_full_face(self) -> None:
        torch.manual_seed(93)
        shared = make_model(prompt_mode="shared")
        torch.manual_seed(93)
        selective = make_model(prompt_mode="view_specific_full_face")
        shared.activate_task(0)
        selective.activate_task(0)
        inputs = view_batch()
        with torch.no_grad():
            _, expected, _ = shared.current_all_logits_with_view_features(inputs)
            _, actual, _ = selective.current_all_logits_with_view_features(inputs)
        for name in ("full", "person", "face"):
            self.assertTrue(torch.equal(expected[name], actual[name]))
        self.assertEqual(selective.prompts[0].ndim, 6)
        with torch.no_grad():
            # Selective layout is [Person-shared, Full-private, Face-private].
            selective.prompts[0][0, 0].add_(0.1)
            _, changed, _ = selective.current_all_logits_with_view_features(inputs)
        self.assertTrue(torch.equal(changed["full"], actual["full"]))
        self.assertTrue(torch.equal(changed["face"], actual["face"]))
        self.assertGreater(
            torch.max((changed["person"] - actual["person"]).abs()).item(), 0
        )

    def test_independent_image_adapters_route_by_view(self) -> None:
        model = make_model(
            adapter_mode="image_token",
            adapter_view_mode="independent",
            adapter_view_bottleneck_dim=3,
        )
        model.activate_task(0)
        inputs = view_batch()
        with torch.no_grad():
            _, before, _ = model.current_all_logits_with_view_features(inputs)
            model.adapter_bank.view_task_adapters[0]["person"]["0"].up.weight.fill_(0.1)
            _, after, _ = model.current_all_logits_with_view_features(inputs)
        self.assertTrue(torch.equal(after["full"], before["full"]))
        self.assertTrue(torch.equal(after["face"], before["face"]))
        self.assertGreater(torch.max((after["person"] - before["person"]).abs()).item(), 0)
        self.assertEqual(
            model.adapter_bank.per_task_parameter_count(0),
            sum(p.numel() for p in model.adapter_bank.view_task_adapters[0].parameters()),
        )

    def test_private_per_view_heads_are_copied_and_optimizer_unique(self) -> None:
        model = make_model(view_classifier_mode="private_per_view")
        model.activate_task(0)
        self.assertEqual(len(model.private_view_heads), 2)
        self.assertTrue(torch.equal(model.head.weight, model.private_view_heads[0].weight))
        self.assertTrue(torch.equal(model.head.weight, model.private_view_heads[1].weight))
        parameters = list(model.classifier_optimizer_parameters())
        self.assertEqual(len({id(parameter) for parameter in parameters}), len(parameters))
        logits, views, _ = model.current_all_logits_with_view_features(view_batch())
        # ``current_all_logits_with_view_features`` preserves the historical
        # all-class output contract; the training loss selects current-task
        # columns using its ``current`` index list.
        self.assertEqual(tuple(logits.shape), (3, 8))
        self.assertEqual(set(views), {"full", "person", "face"})


if __name__ == "__main__":
    unittest.main()
