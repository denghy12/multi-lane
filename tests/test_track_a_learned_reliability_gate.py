from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from multi_lane.track_a.export_compact_test_scores import (
    restore_compact_model_state,
    validate_locked_selection,
)
from multi_lane.track_a.learned_reliability_gate import (
    ANCHOR_PERSON_WEIGHT,
    FEATURE_NAMES,
    MAX_PERSON_WEIGHT,
    MIN_PERSON_WEIGHT,
    ReliabilityGate,
    gate_features,
)
from multi_lane.track_a.runner import (
    compact_model_state_dict,
    fit_calibration_indices,
)
from multi_lane.track_a.search_constrained_gated_fusion import Geometry


class FakeSource:
    def __init__(self, sample_ids):
        self.sample_ids = list(sample_ids)
        self.targets = [[0] for _ in self.sample_ids]


class TinyRestorableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)
        self.restored_task = None

    def restore_task(self, task_id):
        self.restored_task = task_id


class LearnedReliabilityGateTest(unittest.TestCase):
    def test_calibration_partition_is_stable_disjoint_and_view_independent(self):
        ids = [
            f"train/image{index}.jpg#person={person}"
            for index in range(200)
            for person in (0, 1)
        ]
        full = FakeSource(ids)
        person = FakeSource(ids)
        fit_a, calibration_a = fit_calibration_indices(full, (0,), 0.10)
        fit_b, calibration_b = fit_calibration_indices(person, (0,), 0.10)
        self.assertEqual(fit_a, fit_b)
        self.assertEqual(calibration_a, calibration_b)
        self.assertFalse(set(fit_a).intersection(calibration_a))
        self.assertEqual(set(fit_a).union(calibration_a), set(range(len(ids))))
        self.assertGreater(len(fit_a), 0)
        self.assertGreater(len(calibration_a), 0)
        buckets = {
            index: "calibration" if index in set(calibration_a) else "fit"
            for index in range(len(ids))
        }
        for index in range(0, len(ids), 2):
            self.assertEqual(buckets[index], buckets[index + 1])
        full_fit, full_calibration = fit_calibration_indices(full, (0,), 0.0)
        self.assertEqual(full_fit, list(range(len(ids))))
        self.assertEqual(full_calibration, [])

    def test_compact_state_omits_only_visual_tower(self):
        model = TinyRestorableModel()
        compact = compact_model_state_dict(model)
        self.assertTrue(compact)
        self.assertTrue(all(not key.startswith("visual_encoder.") for key in compact))
        self.assertIn("head.weight", compact)
        source_git = {"commit": "abc", "dirty": False}
        restore_compact_model_state(
            model,
            {
                "schema_version": 1,
                "task_id": 3,
                "source_git": source_git,
                "model": compact,
            },
            3,
            source_git,
        )
        self.assertEqual(model.restored_task, 3)

    def test_compact_restore_rejects_missing_method_state(self):
        model = TinyRestorableModel()
        compact = compact_model_state_dict(model)
        compact.pop("head.bias")
        with self.assertRaisesRegex(ValueError, "missing method state"):
            restore_compact_model_state(
                model,
                {
                    "schema_version": 1,
                    "task_id": 0,
                    "source_git": {"commit": "abc"},
                    "model": compact,
                },
                0,
                {"commit": "abc"},
            )

    def test_gate_initializes_at_anchor_and_remains_bounded(self):
        gate = ReliabilityGate(hidden_dim=8)
        features = torch.randn(32, len(FEATURE_NAMES))
        initial = gate(features)
        self.assertTrue(
            torch.allclose(
                initial,
                torch.full_like(initial, ANCHOR_PERSON_WEIGHT),
                atol=1e-7,
                rtol=0,
            )
        )
        with torch.no_grad():
            for parameter in gate.parameters():
                parameter.normal_(mean=0.0, std=10.0)
        weights = gate(features)
        self.assertTrue(torch.all(weights >= MIN_PERSON_WEIGHT))
        self.assertTrue(torch.all(weights <= MAX_PERSON_WEIGHT))

    def test_gate_features_include_geometry_and_two_view_uncertainty(self):
        sample_ids = ["a", "b"]
        full = np.asarray([[0.9, 0.1], [0.5, 0.5]], dtype=np.float32)
        person = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
        geometry = {
            "a": Geometry(0.25, 1.5, 1, "unused"),
            "b": Geometry(0.05, 3.0, 3, "unused"),
        }
        features = gate_features(sample_ids, full, person, geometry)
        self.assertEqual(features.shape, (2, len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features[0, FEATURE_NAMES.index("is_multi_person")], 0)
        self.assertEqual(features[1, FEATURE_NAMES.index("is_multi_person")], 1)
        self.assertGreater(
            features[1, FEATURE_NAMES.index("mean_probability_disagreement")],
            features[0, FEATURE_NAMES.index("mean_probability_disagreement")],
        )

    def test_locked_selection_requires_validation_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(
                json.dumps(
                    {
                        "selection_split": "val",
                        "test_accessed": False,
                        "advance_to_locked_test": True,
                        "winner": {"candidate_id": "h8_prior1p0"},
                    }
                ),
                encoding="utf-8",
            )
            selection, digest = validate_locked_selection(path)
            self.assertEqual(selection["winner"]["candidate_id"], "h8_prior1p0")
            self.assertEqual(len(digest), 64)
            selection["test_accessed"] = True
            path.write_text(json.dumps(selection), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not lock"):
                validate_locked_selection(path)


if __name__ == "__main__":
    unittest.main()
