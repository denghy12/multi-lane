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
    backward_dgl_training_losses,
    build_optimizer_groups,
    compute_training_loss,
    view_gradient_audit,
    view_path_gradient_audit,
)
from multi_lane.track_a.view_fusion import TaskwiseViewFusion


def tiny_model(
    mode: str, view_classifier_mode: str = "shared_post_fusion"
) -> MultiLaneModel:
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
        view_classifier_mode=view_classifier_mode,
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
    def test_full_private_head_copies_initialization_without_rng_drift(self) -> None:
        torch.manual_seed(41)
        shared = tiny_model("fixed_three_view", "shared_per_view")
        shared_rng = torch.random.get_rng_state()
        torch.manual_seed(41)
        private = tiny_model("fixed_three_view", "full_private_per_view")
        private_rng = torch.random.get_rng_state()
        self.assertTrue(torch.equal(shared_rng, private_rng))
        self.assertIsNot(private.head.weight, private.full_view_head.weight)
        self.assertTrue(torch.equal(private.head.weight, private.full_view_head.weight))
        self.assertTrue(torch.equal(private.head.bias, private.full_view_head.bias))
        inputs = three_view_batch()
        shared_logits, shared_views = shared.current_all_logits_with_views(inputs)
        private_logits, private_views = private.current_all_logits_with_views(inputs)
        self.assertTrue(torch.equal(shared_logits, private_logits))
        for name in ("full", "person", "face"):
            self.assertTrue(torch.equal(shared_views[name], private_views[name]))

    def test_full_private_head_separates_full_from_person_face_gradients(self) -> None:
        model = tiny_model("fixed_three_view", "full_private_per_view")
        _, branches = model.current_all_logits_with_views(three_view_batch())
        branches["full"].sum().backward()
        self.assertIsNotNone(model.full_view_head.weight.grad)
        self.assertIsNone(model.head.weight.grad)
        model.zero_grad(set_to_none=True)
        _, branches = model.current_all_logits_with_views(three_view_batch())
        (branches["person"].sum() + branches["face"].sum()).backward()
        self.assertIsNone(model.full_view_head.weight.grad)
        self.assertIsNotNone(model.head.weight.grad)

    def test_full_private_head_is_in_classifier_optimizer_group(self) -> None:
        model = tiny_model("fixed_three_view", "full_private_per_view")
        classifier = list(model.classifier_optimizer_parameters())
        self.assertEqual(len(classifier), 4)
        _, _, groups = build_optimizer_groups(
            model, 0.0, adapter_learning_rate=4e-4
        )
        classifier_ids = {id(parameter) for parameter in classifier}
        self.assertEqual(
            classifier_ids,
            {id(parameter) for parameter in groups[1]["params"]},
        )

    def test_view_specialized_adapter_changes_only_selected_view(self) -> None:
        torch.manual_seed(29)
        model = MultiLaneModel(
            FakeVisual(),
            (5, 3),
            num_selectors=2,
            num_prompts=2,
            num_prompt_layers=1,
            adapter_mode="image_token",
            adapter_layer_indices=(0,),
            adapter_bottleneck_dim=3,
            adapter_view_bottleneck_dim=2,
            view_fusion="fixed_three_view",
            view_fusion_hidden_dim=4,
        )
        model.activate_task(0)
        image = torch.randn(2, 3, 4, 4)
        full_before = model._encode_single_lanes(image, False, "full")
        person_before = model._encode_single_lanes(image, False, "person")
        self.assertTrue(torch.equal(full_before, person_before))
        person_adapter = model.adapter_bank.view_task_adapters[0]["person"]["0"]
        with torch.no_grad():
            person_adapter.up.bias.fill_(1.0)
        full_after = model._encode_single_lanes(image, False, "full")
        person_after = model._encode_single_lanes(image, False, "person")
        self.assertTrue(torch.equal(full_before, full_after))
        self.assertFalse(torch.equal(person_before, person_after))
        self.assertEqual(model.adapter_bank.per_task_parameter_count(), 185)

    def test_path_audit_separates_view_and_parameter_group(self) -> None:
        model = tiny_model("fixed_three_view")
        logits, view_logits, features = (
            model.current_all_logits_with_view_features(three_view_batch())
        )
        fused_bce = logits.square().mean()
        fused_asl = (logits - 0.25).square().mean()
        view_bce = {
            name: value.square().mean() for name, value in view_logits.items()
        }
        view_asl = {
            name: (value - 0.25).square().mean()
            for name, value in view_logits.items()
        }
        audit = view_path_gradient_audit(
            fused_bce,
            fused_asl,
            view_bce,
            view_asl,
            features,
            tuple(model.representation_optimizer_parameters()),
            tuple(model.adapter_optimizer_parameters()),
        )
        for group in ("representation", "adapter"):
            for view in ("full", "person", "face"):
                self.assertIn(
                    f"path_gradient_{group}_{view}_fused_to_unimodal_ratio",
                    audit,
                )
                self.assertGreater(
                    audit[f"path_gradient_{group}_{view}_fused_norm"], 0
                )
                self.assertGreater(
                    audit[f"path_gradient_{group}_{view}_unimodal_norm"], 0
                )
        self.assertTrue(all(torch.isfinite(torch.tensor(list(audit.values())))))

    def test_fused_loss_updates_every_view_specific_adapter(self) -> None:
        torch.manual_seed(31)
        model = MultiLaneModel(
            FakeVisual(),
            (5, 3),
            num_selectors=2,
            num_prompts=2,
            num_prompt_layers=1,
            adapter_mode="image_token",
            adapter_layer_indices=(0,),
            adapter_bottleneck_dim=3,
            adapter_view_bottleneck_dim=2,
            view_fusion="fixed_three_view",
            view_fusion_hidden_dim=4,
        )
        model.activate_task(0)
        fused, _ = model.current_all_logits_with_views(three_view_batch())
        fused.square().mean().backward()
        for view in ("full", "person", "face"):
            gradients = [
                parameter.grad
                for parameter in model.adapter_bank.view_active_parameters(view)
            ]
            self.assertTrue(any(
                gradient is not None and bool(torch.count_nonzero(gradient))
                for gradient in gradients
            ))

    def test_detached_fixed_fusion_preserves_branch_gradients_only(self) -> None:
        torch.manual_seed(23)
        model = MultiLaneModel(
            FakeVisual(),
            (5, 3),
            num_selectors=2,
            num_prompts=2,
            num_prompt_layers=1,
            adapter_mode="image_token",
            adapter_layer_indices=(0,),
            adapter_bottleneck_dim=3,
            view_fusion="fixed_three_view",
            view_fusion_hidden_dim=4,
            detach_view_fusion_features=True,
        )
        model.activate_task(0)
        fused, branches = model.current_all_logits_with_views(three_view_batch())
        fused.sum().backward()
        self.assertIsNotNone(model.head.weight.grad)
        self.assertIsNone(model.selectors.grad)
        self.assertTrue(all(
            parameter.grad is None
            for parameter in model.adapter_optimizer_parameters()
        ))
        model.zero_grad(set_to_none=True)
        _, branches = model.current_all_logits_with_views(three_view_batch())
        branches["full"].sum().backward()
        self.assertIsNotNone(model.selectors.grad)

    def test_dgl_backward_keeps_prediction_gradient_fusion_only(self) -> None:
        representation = torch.nn.Parameter(torch.tensor(2.0))
        adapter = torch.nn.Parameter(torch.tensor(3.0))
        prediction = torch.nn.Parameter(torch.tensor(5.0))
        representation_loss = representation * prediction
        adapter_loss = adapter * prediction
        fusion_loss = prediction.square()
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        backward_dgl_training_losses(
            representation_loss,
            adapter_loss,
            fusion_loss,
            (representation,),
            (adapter,),
            (prediction,),
            scaler,
        )
        self.assertEqual(float(representation.grad), 5.0)
        self.assertEqual(float(adapter.grad), 5.0)
        self.assertEqual(float(prediction.grad), 10.0)

    def test_gradient_audit_reports_opposing_shared_gradients(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        fused = parameter.sum()
        opposite = -parameter.sum()
        aligned = 2 * parameter.sum()
        audit = view_gradient_audit(
            fused,
            fused,
            {"full": opposite, "person": aligned},
            {"full": opposite, "person": aligned},
            (parameter,),
            (),
        )
        self.assertAlmostEqual(audit["gradient_cosine_fused_full"], -1.0)
        self.assertAlmostEqual(audit["gradient_cosine_fused_person"], 1.0)

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
