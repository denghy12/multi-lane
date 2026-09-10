from __future__ import annotations

import copy
import unittest

import torch
from PIL import Image

from test_track_a_reproduction import FakeVisual
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.paired_transforms import ThreeViewTransform
from multi_lane.track_a.runner import (
    add_view_auxiliary_loss,
    build_optimizer_groups,
    compute_training_loss,
)
from multi_lane.track_a.view_fusion import TaskwiseViewFusion


def tiny_model(mode: str) -> MultiLaneModel:
    torch.manual_seed(19)
    model = MultiLaneModel(
        FakeVisual(),
        (5, 3),
        num_selectors=2,
        num_prompts=2,
        num_prompt_layers=1,
        adapter_mode="image_token",
        adapter_layer_indices=(0,),
        adapter_bottleneck_dim=3,
        view_fusion=mode,
        view_fusion_hidden_dim=4,
    )
    model.activate_task(0)
    return model


def three_view_batch(batch_size: int = 3):
    return {
        "full": torch.randn(batch_size, 3, 4, 4),
        "person": torch.randn(batch_size, 3, 4, 4),
        "face": torch.randn(batch_size, 3, 4, 4),
        "face_reliable": torch.tensor([1, 0, 1], dtype=torch.float32)[:batch_size],
    }


class TaskwiseViewFusionTest(unittest.TestCase):
    def test_three_view_transform_emits_face_masks_and_aligned_tensors(self) -> None:
        transform = ThreeViewTransform(
            train=False,
            normalization="clip",
            crop_scale=(0.05, 1.0),
            margin=0.15,
            jitter_strength=0,
            jitter_probability=0,
            full_crop_mode="legacy",
        )
        result = transform(
            Image.new("RGB", (80, 60), color=(180, 100, 40)),
            [10, 5, 70, 55],
            {"face_crop_bbox": [25, 10, 45, 30]},
            True,
            True,
        )
        self.assertEqual(tuple(result["full"].shape), (3, 224, 224))
        self.assertEqual(tuple(result["person"].shape), (3, 224, 224))
        self.assertEqual(tuple(result["face"].shape), (3, 224, 224))
        self.assertEqual(float(result["face_training_valid"]), 1)
        self.assertEqual(float(result["face_reliable"]), 1)
        unreliable = transform(
            Image.new("RGB", (80, 60), color=(180, 100, 40)),
            [10, 5, 70, 55],
            {"face_crop_bbox": [25, 10, 45, 30]},
            True,
            False,
        )
        self.assertEqual(float(unreliable["face_training_valid"]), 1)
        self.assertEqual(float(unreliable["face_reliable"]), 0)
        self.assertTrue(torch.isfinite(unreliable["face"]).all())

    def test_soft_three_view_starts_from_locked_fixed_priors(self) -> None:
        module = TaskwiseViewFusion(2, 4, "soft_three_view", hidden_dim=3)
        module.restore_task(0)
        features = {name: torch.randn(3, 1, 4) for name in module.view_names}
        reliable = torch.tensor([True, False, True])
        weights = module.weights(features, (0,), reliable)[:, 0]
        self.assertTrue(torch.allclose(
            weights[0], torch.tensor([0.64, 0.16, 0.20]), atol=1e-7
        ))
        self.assertTrue(torch.allclose(
            weights[1], torch.tensor([0.80, 0.20, 0.00]), atol=1e-7
        ))

    def test_later_router_cannot_change_an_old_task_lane(self) -> None:
        module = TaskwiseViewFusion(2, 4, "soft_three_view", hidden_dim=3)
        features = {name: torch.randn(3, 2, 4) for name in module.view_names}
        reliable = torch.tensor([True, False, True])
        module.restore_task(1)
        before, before_weights = module(features, (0, 1), reliable)
        with torch.no_grad():
            for parameter in module.task_routers[1].parameters():
                parameter.add_(torch.randn_like(parameter))
        after, after_weights = module(features, (0, 1), reliable)
        self.assertTrue(torch.equal(before[:, 0], after[:, 0]))
        self.assertTrue(torch.equal(before_weights[:, 0], after_weights[:, 0]))
        self.assertFalse(torch.equal(before_weights[:, 1], after_weights[:, 1]))
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in module.task_routers[0].parameters()
        ))

    def test_invalid_face_is_exactly_removed_from_fusion(self) -> None:
        module = TaskwiseViewFusion(1, 4, "fixed_three_view", hidden_dim=3)
        features = {name: torch.randn(2, 1, 4) for name in module.view_names}
        reliable = torch.tensor([False, True])
        first, _ = module(features, (0,), reliable)
        changed = copy.deepcopy(features)
        changed["face"][0].fill_(1e6)
        second, _ = module(changed, (0,), reliable)
        self.assertTrue(torch.equal(first[0], second[0]))

    def test_residual_fusion_starts_as_exact_full_and_is_bounded(self) -> None:
        module = TaskwiseViewFusion(
            1, 4, "residual_full_person", hidden_dim=3,
            residual_scale=0.1,
        )
        module.restore_task(0)
        full = torch.randn(2, 1, 4, requires_grad=True)
        person = torch.randn(2, 1, 4, requires_grad=True)
        fused, coefficients = module(
            {"full": full, "person": person}, (0,), None
        )
        self.assertTrue(torch.equal(fused, full))
        self.assertTrue(torch.equal(coefficients[..., 0], torch.ones(2, 1)))
        self.assertTrue(torch.all(coefficients[..., 1] >= 0))
        self.assertTrue(torch.all(coefficients[..., 1] <= 0.1))
        fused.sum().backward()
        self.assertIsNotNone(full.grad)
        self.assertIsNone(person.grad)
        projection = module.task_residuals[0]["projections"]["person"]
        self.assertIsNotNone(projection[3].weight.grad)

    def test_residual_face_is_masked_and_old_task_is_frozen(self) -> None:
        module = TaskwiseViewFusion(
            2, 4, "residual_three_view", hidden_dim=3,
            residual_scale=0.1,
        )
        module.restore_task(1)
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in module.task_residuals[0].parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in module.task_residuals[1].parameters()
        ))
        features = {
            name: torch.randn(2, 2, 4) for name in module.view_names
        }
        reliable = torch.tensor([False, True])
        before, coefficients = module(features, (0, 1), reliable)
        self.assertTrue(torch.equal(
            coefficients[0, :, 2], torch.zeros(2)
        ))
        changed = copy.deepcopy(features)
        changed["face"][0].fill_(1e6)
        after, _ = module(changed, (0, 1), reliable)
        self.assertTrue(torch.equal(before[0], after[0]))

    def test_fused_model_returns_branch_logits_and_preserves_frozen_clip(self) -> None:
        model = tiny_model("soft_three_view")
        inputs = three_view_batch()
        logits, branches = model.current_all_logits_with_views(inputs)
        self.assertEqual(tuple(logits.shape), (3, 8))
        self.assertEqual(set(branches), {"full", "person", "face"})
        self.assertEqual(tuple(model.seen_logits(inputs).shape), (3, 5))
        logits.sum().backward()
        self.assertTrue(all(
            parameter.grad is None for parameter in model.visual_encoder.parameters()
        ))

    def test_face_auxiliary_loss_masks_unreliable_samples(self) -> None:
        primary_logits = torch.randn(3, 8, requires_grad=True)
        face_logits = torch.randn(3, 8, requires_grad=True)
        targets = torch.tensor([
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
        ]).float()
        primary = compute_training_loss(
            primary_logits, targets, range(5), 1, "legacy_full_zero"
        )
        loss = add_view_auxiliary_loss(
            primary,
            {"face": face_logits},
            {"face_reliable": torch.tensor([1, 0, 0])},
            targets,
            range(5),
            1,
            "legacy_full_zero",
            0.1,
            "bce",
        )
        loss.backward()
        self.assertGreater(torch.count_nonzero(face_logits.grad[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(face_logits.grad[1:]).item(), 0)

    def test_fusion_optimizer_is_task_local_and_has_its_own_lr(self) -> None:
        model = tiny_model("soft_full_person")
        base, adapter, groups = build_optimizer_groups(
            model,
            0,
            adapter_learning_rate=4e-4,
            view_fusion_learning_rate=2e-4,
        )
        active = list(model.fusion_optimizer_parameters())
        self.assertTrue(active)
        self.assertEqual(groups[-2]["lr"], 2e-4)
        self.assertEqual(groups[-1]["lr"], 4e-4)
        identifiers = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(set(identifiers), {
            id(parameter) for parameter in [*base, *adapter]
        })

    def test_residual_optimizer_contains_only_current_task_module(self) -> None:
        model = tiny_model("residual_three_view")
        active = list(model.fusion_optimizer_parameters())
        self.assertTrue(active)
        self.assertEqual(
            {id(parameter) for parameter in active},
            {
                id(parameter)
                for parameter in model.view_fusion_module.task_residuals[0].parameters()
            },
        )
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in model.view_fusion_module.task_residuals[1].parameters()
        ))

    def test_residual_forward_restores_full_adapter_runtime(self) -> None:
        model = tiny_model("residual_full_person")
        inputs = three_view_batch()
        model.current_all_logits_with_views({
            "full": inputs["full"], "person": inputs["person"],
        })
        self.assertTrue(model.adapter_runtime_enabled)


if __name__ == "__main__":
    unittest.main()
