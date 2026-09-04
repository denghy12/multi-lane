from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
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
    TASK_SIZES,
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
    def test_original_parameter_grid_is_unchanged(self):
        import multi_lane.track_a.learned_reliability_gate as module
        self.assertEqual(module.GATE_HIDDEN_DIMS, (8, 16))
        self.assertEqual(module.PRIOR_STRENGTHS, (.1, .3, 1.0, 3.0))
        self.assertEqual((module.GATE_EPOCHS_PER_TASK, module.GATE_BATCH_SIZE), (80, 64))
        self.assertEqual((module.GATE_LEARNING_RATE, module.GATE_WEIGHT_DECAY), (1e-3, 1e-4))
        self.assertEqual((MIN_PERSON_WEIGHT, MAX_PERSON_WEIGHT, ANCHOR_PERSON_WEIGHT), (.1, .35, .2))

    @staticmethod
    def synthetic_pair(seed=0):
        from multi_lane.track_a.learned_reliability_gate import EndpointPair
        tasks = {"train": [], "val": []}
        geometry = {"train": {}, "val": {}}
        rng = np.random.RandomState(17 + seed)
        for split in tasks:
            for task in range(8):
                width = sum(TASK_SIZES[:task + 1])
                ids = np.asarray([f"{split}:image{task}_{i}#person=0" for i in range(12)])
                targets = np.asarray([[(i+j) % 3 == 0 for j in range(width)]
                                      for i in range(12)], dtype=np.float32)
                full = rng.uniform(.1, .9, targets.shape).astype(np.float32)
                person = rng.uniform(.1, .9, targets.shape).astype(np.float32)
                tasks[split].append((ids, targets, full, person))
                geometry[split].update({str(i): Geometry(.2, 1.5, 2, "unused") for i in ids})
        return EndpointPair(seed, Path("full"), Path("person"), tasks["val"],
                            tasks["train"], []), geometry["train"], geometry["val"]

    def test_selection_metadata_loads_train_and_val_but_never_test(self):
        import multi_lane.track_a.learned_reliability_gate as module
        with patch.object(module, "resolve_dataset_parent", return_value=Path("data")), \
             patch.object(module, "load_geometry", side_effect=[{"train": 1}, {"val": 2}]) as load:
            train, val = module.load_selection_geometry(Path("data/EMOTIC"))
        self.assertEqual(train, {"train": 1})
        self.assertEqual(val, {"val": 2})
        self.assertEqual([call.args[1] for call in load.call_args_list], ["train", "val"])

    def test_missing_validation_geometry_fails_before_any_gate_training(self):
        import multi_lane.track_a.learned_reliability_gate as module
        pair, train, val = self.synthetic_pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "Missing geometry for val:"):
                module.select_gate({0: pair}, train, {}, root)
            self.assertFalse((root / "gate_states").exists())

    def test_eight_task_selection_with_disjoint_split_ids(self):
        import multi_lane.track_a.learned_reliability_gate as module
        pairs = {}
        for seed in range(3):
            pairs[seed], train, val = self.synthetic_pair(seed)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(module, "GATE_EPOCHS_PER_TASK", 1), \
             patch.object(module, "GATE_HIDDEN_DIMS", (8,)), \
             patch.object(module, "PRIOR_STRENGTHS", (.1,)):
            for seed, pair in pairs.items():
                for view in ("full", "person"):
                    source = Path(directory) / f"seed{seed}_{view}"
                    source.mkdir()
                    (source / "config.json").write_text(
                        json.dumps({"seed": seed, "view": view}), encoding="utf-8"
                    )
                    setattr(pair, view + "_run", source)
            result = module.select_gate(pairs, train, val, Path(directory))
        self.assertFalse(result["test_accessed"])
        self.assertEqual(result["geometry_splits"], {"calibration": "train", "validation": "val"})
        self.assertFalse(result["diagnostics_affect_selection"])
        candidate = result["candidates"][0]
        self.assertEqual([len(row["task_metrics"]) for row in candidate["seeds"]], [8, 8, 8])
        expected = all(row["metrics"]["final_mAP"] >= anchor["metrics"]["final_mAP"]
                       for row, anchor in zip(candidate["seeds"], result["fixed_0.20_anchor"]["seeds"]))
        self.assertEqual(candidate["eligible_each_seed_vs_fixed_0.20"], expected)
        diagnostics = candidate["seeds"][0]["training"][6]["diagnostics"]
        self.assertIn("weight_std", diagnostics["validation"])
        self.assertEqual(len(diagnostics["calibration"]["current_positive_support"]), 3)

    def test_validation_labels_never_change_trained_gate_states(self):
        import multi_lane.track_a.learned_reliability_gate as module
        pair, train, val = self.synthetic_pair()
        with tempfile.TemporaryDirectory() as directory, patch.object(module, "GATE_EPOCHS_PER_TASK", 1):
            roots = [Path(directory)/name/"gate_states" for name in ("a", "b")]
            for root in roots:
                root.mkdir(parents=True)
            first = module._train_one_seed_candidate(pair, train, val, 8, .1, roots[0])
            pair.validation_tasks = [(ids, 1-targets, full, person)
                                     for ids, targets, full, person in pair.validation_tasks]
            second = module._train_one_seed_candidate(pair, train, val, 8, .1, roots[1])
            for a, b in zip(first["training"], second["training"]):
                state_a = torch.load(roots[0].parent/a["state_path"], map_location="cpu")["model"]
                state_b = torch.load(roots[1].parent/b["state_path"], map_location="cpu")["model"]
                self.assertTrue(all(torch.equal(state_a[key], state_b[key]) for key in state_a))

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
