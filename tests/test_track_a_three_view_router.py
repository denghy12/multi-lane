from __future__ import annotations

import unittest

import numpy as np
import torch

from multi_lane.track_a.search_constrained_gated_fusion import Geometry
from multi_lane.track_a.three_view_router import (
    INVALID_PRIOR,
    R2_FEATURE_NAMES,
    VALID_PRIOR,
    FaceMetadata,
    ThreeViewRouter,
    _fuse,
    _probability_features,
    _standardize_fit_apply,
)


class ThreeViewRouterTest(unittest.TestCase):
    def test_initial_weights_match_priors_and_invalid_face_is_exact_zero(self) -> None:
        router = ThreeViewRouter(feature_dim=5, initialization_seed=7)
        features = torch.randn(4, 5)
        reliable = torch.tensor([True, False, True, False])
        weights = router(features, reliable).detach().numpy()
        np.testing.assert_allclose(weights[reliable.numpy()], VALID_PRIOR[None], atol=1e-7)
        np.testing.assert_allclose(weights[~reliable.numpy()], INVALID_PRIOR[None], atol=1e-7)
        self.assertTrue(np.array_equal(weights[~reliable.numpy(), 2], np.zeros(2)))
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-7)

    def test_r2_features_mask_unreliable_face_evidence(self) -> None:
        sample_ids = ["a", "b"]
        full = np.asarray([[0.9, 0.1], [0.6, 0.4]], dtype=np.float32)
        person = np.asarray([[0.8, 0.2], [0.4, 0.6]], dtype=np.float32)
        face = np.asarray([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
        geometry = {
            "a": Geometry(0.2, 1.5, 1, "unused"),
            "b": Geometry(0.1, 2.0, 3, "unused"),
        }
        metadata = {
            "a": FaceMetadata(True, True, 32.0, 0.8),
            "b": FaceMetadata(True, False, 16.0, 0.9),
        }
        features, reliable = _probability_features(
            sample_ids, full, person, face, geometry, metadata
        )
        self.assertEqual(features.shape, (2, len(R2_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(np.array_equal(reliable, np.asarray([True, False])))
        for name in (
            "face_confidence",
            "face_entropy",
            "mean_disagreement_full_face",
            "mean_disagreement_person_face",
            "max_disagreement_full_face",
            "max_disagreement_person_face",
        ):
            self.assertEqual(features[1, R2_FEATURE_NAMES.index(name)], 0.0)

    def test_normalization_uses_fit_statistics_only(self) -> None:
        fit = np.asarray([[1.0, 5.0], [3.0, 5.0]], dtype=np.float32)
        apply = np.asarray([[100.0, 5.0]], dtype=np.float32)
        fit_scaled, apply_scaled, state = _standardize_fit_apply(fit, apply)
        np.testing.assert_allclose(fit_scaled[:, 0], [-1.0, 1.0])
        np.testing.assert_allclose(fit_scaled[:, 1], [0.0, 0.0])
        self.assertEqual(state["mean"], [2.0, 5.0])
        self.assertEqual(state["scale"], [1.0, 1.0])
        self.assertEqual(apply_scaled[0, 0], 98.0)

    def test_three_view_fusion_uses_one_weight_vector_per_sample(self) -> None:
        full = np.asarray([[1.0], [0.0]], dtype=np.float32)
        person = np.asarray([[0.0], [1.0]], dtype=np.float32)
        face = np.asarray([[0.5], [0.5]], dtype=np.float32)
        weights = np.asarray([[0.8, 0.2, 0.0], [0.6, 0.1, 0.3]], dtype=np.float32)
        fused = _fuse((full, person, face), weights)
        np.testing.assert_allclose(fused, [[0.8], [0.25]], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
