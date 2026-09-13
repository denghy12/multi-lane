from __future__ import annotations

import unittest

import torch
from PIL import Image
from torch import nn

from test_track_a_reproduction import FakeVisual
from multi_lane.track_a.face_dual_transform import DualFaceTransform
from multi_lane.track_a.face_expression_residual import FaceExpressionResidualModel
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import build_optimizer_groups


class TinyExpressionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 6)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(2, 3)))


def hybrid_model(seed: int = 17) -> FaceExpressionResidualModel:
    expression = TinyExpressionEncoder()
    classifier = nn.Linear(6, 8)
    torch.manual_seed(seed)
    model = FaceExpressionResidualModel(
        expression_encoder=expression,
        expression_classifier=classifier,
        expression_feature_dim=6,
        visual_encoder=FakeVisual(),
        task_sizes=(2, 1),
        num_selectors=2,
        num_prompts=2,
        num_prompt_layers=1,
        adapter_mode="image_token",
        adapter_bottleneck_dim=3,
        adapter_layer_indices=(0,),
        residual_rank=3,
        residual_scale=0.1,
    )
    model.activate_task(0)
    return model


class FaceExpressionResidualTest(unittest.TestCase):
    def test_zero_residual_preserves_clip_face_logits(self) -> None:
        torch.manual_seed(17)
        baseline = MultiLaneModel(
            FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1, adapter_mode="image_token",
            adapter_bottleneck_dim=3, adapter_layer_indices=(0,),
        )
        baseline.activate_task(0)
        hybrid = hybrid_model(17)
        inputs = {
            "clip": torch.randn(4, 3, 4, 4),
            "expression": torch.randn(4, 3, 4, 4),
        }
        torch.testing.assert_close(
            baseline.current_all_logits(inputs["clip"]),
            hybrid.current_all_logits(inputs), rtol=1e-6, atol=1e-7,
        )

    def test_expression_towers_are_frozen_and_residual_is_task_local(self) -> None:
        model = hybrid_model()
        model.train()
        model.assert_expression_frozen()
        self.assertFalse(model.expression_encoder.training)
        self.assertFalse(model.expression_classifier.training)
        self.assertTrue(all(
            parameter.requires_grad
            for parameter in model.expression_residual_bank.task_modules[0].parameters()
        ))
        model.activate_task(1)
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in model.expression_residual_bank.task_modules[0].parameters()
        ))

    def test_residual_uses_bce_group_at_locked_small_lr(self) -> None:
        model = hybrid_model()
        model_parameters, adapter_parameters, groups = build_optimizer_groups(
            model, 0.0, adapter_learning_rate=4e-4,
            adapter_weight_decay=0.0, view_fusion_learning_rate=4e-4,
        )
        residual = set(map(id, model.expression_residual_bank.active_parameters()))
        self.assertTrue(residual)
        self.assertTrue(residual <= set(map(id, model_parameters)))
        self.assertFalse(residual.intersection(map(id, adapter_parameters)))
        self.assertEqual(groups[-2]["lr"], 4e-4)

    def test_dual_transform_keeps_two_views_and_invalid_placeholders(self) -> None:
        transform = DualFaceTransform(train=False)
        image = Image.new("RGB", (80, 60), (160, 100, 60))
        record = {
            "face_bbox": [20, 10, 50, 45],
            "face_keypoints": [
                [29, 22], [41, 22], [35, 29], [30, 36], [40, 36]
            ],
        }
        valid = transform(image, record, True)
        invalid = transform(image, record, False)
        for result in (valid, invalid):
            self.assertEqual(set(result), {"clip", "expression"})
            self.assertEqual(tuple(result["clip"].shape), (3, 224, 224))
            self.assertEqual(tuple(result["expression"].shape), (3, 224, 224))
            self.assertTrue(torch.isfinite(result["clip"]).all())
            self.assertTrue(torch.isfinite(result["expression"]).all())


if __name__ == "__main__":
    unittest.main()
