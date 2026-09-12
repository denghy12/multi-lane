from __future__ import annotations

import unittest
import sys

import numpy as np
import torch
from PIL import Image
from torch import nn

from multi_lane.track_a.face_alignment import (
    align_face_five_points,
    canonical_five_point_template,
    estimate_similarity_transform,
    valid_five_point_landmarks,
)
from multi_lane.track_a.face_expression_model import (
    FaceExpressionIncrementalModel,
    _install_timm_checkpoint_compatibility_aliases,
)


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 6)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(2, 3)))


class FivePointAlignmentTest(unittest.TestCase):
    def test_similarity_transform_maps_landmarks_to_template(self) -> None:
        source = canonical_five_point_template(224, 0.0) * 0.6 + np.array([9.0, 17.0])
        destination = canonical_five_point_template(224, 0.15)
        matrix = estimate_similarity_transform(source, destination)
        projected = source @ matrix[:, :2].T + matrix[:, 2]
        np.testing.assert_allclose(projected, destination, atol=1e-8)

    def test_margin_retains_more_context(self) -> None:
        no_margin = canonical_five_point_template(224, 0.0)
        margin = canonical_five_point_template(224, 0.15)
        centre = np.asarray([112.0, 112.0])
        self.assertLess(
            np.linalg.norm(margin - centre, axis=1).mean(),
            np.linalg.norm(no_margin - centre, axis=1).mean(),
        )

    def test_degenerate_semantic_pair_is_rejected(self) -> None:
        points = canonical_five_point_template().copy()
        points[1] = points[0]
        self.assertFalse(valid_five_point_landmarks(points, (224, 224)))

    def test_alignment_returns_a_finite_rgb_image(self) -> None:
        image = Image.new("RGB", (224, 224), color=(80, 100, 120))
        record = {"face_keypoints": canonical_five_point_template(224, 0.0).tolist()}
        aligned = align_face_five_points(image, record, margin=0.15)
        self.assertIsNotNone(aligned)
        self.assertEqual(aligned.size, (224, 224))
        self.assertEqual(aligned.mode, "RGB")


class FaceExpressionModelTest(unittest.TestCase):
    @staticmethod
    def _model(variant: str) -> FaceExpressionIncrementalModel:
        return FaceExpressionIncrementalModel(
            TinyEncoder(), 6, (2, 3), variant,
            bottleneck_dim=2, residual_scale=0.1, activation="relu",
        )

    def test_encoder_is_frozen_and_kept_in_eval_mode(self) -> None:
        model = self._model("projection")
        model.activate_task(0)
        model.train()
        self.assertFalse(model.encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.heads[0].parameters()))

    def test_checkpoint_module_aliases_support_locked_timm(self) -> None:
        _install_timm_checkpoint_compatibility_aliases()
        self.assertIn("timm.layers.adaptive_avgmax_pool", sys.modules)
        self.assertIn("timm.models._efficientnet_blocks", sys.modules)

    def test_new_task_does_not_change_old_head_logits(self) -> None:
        torch.manual_seed(5)
        model = self._model("projection")
        images = torch.randn(4, 3, 8, 8)
        model.activate_task(0)
        before = model.seen_logits(images).detach().clone()
        model.activate_task(1)
        after = model.seen_logits(images)[:, :2].detach()
        torch.testing.assert_close(before, after)

    def test_zero_initialized_adapter_preserves_initial_features(self) -> None:
        model = self._model("bottleneck_adapter")
        model.activate_task(0)
        images = torch.randn(4, 3, 8, 8)
        frozen = model.frozen_features(images)
        adapted = model.adapter_bank(frozen, 0)
        torch.testing.assert_close(frozen, adapted)
        self.assertTrue(list(model.adapter_optimizer_parameters()))


if __name__ == "__main__":
    unittest.main()
