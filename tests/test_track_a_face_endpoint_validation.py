from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from multi_lane.continual_datasets.continual_datasets import EMOTIC
from multi_lane.track_a.fuse_face_endpoint_validation import (
    load_face_reliability,
    masked_face_fusion,
)
from multi_lane.track_a.compare_face_expert_quality import (
    EQUAL_UPDATE_BUDGETS,
    _fixed_candidate,
    _validate_equal_update_candidate,
)
from multi_lane.track_a.runner import (
    PadToSquare,
    build_transforms,
    filter_face_training_indices,
    load_face_manifest_provenance,
)


class FaceEndpointInputTest(unittest.TestCase):
    @staticmethod
    def _dataset(valid: bool) -> EMOTIC:
        source = object.__new__(EMOTIC)
        source.face_records = [{"face_crop_bbox": [10, 5, 30, 25]}]
        source.face_training_valid = [valid]
        source.sample_ids = ["train:a.jpg#person=0"]
        source.face_crop_margin = 0.15
        return source

    def test_valid_face_uses_manifest_crop(self) -> None:
        image = Image.new("RGB", (80, 60), color=(255, 0, 0))
        crop = self._dataset(True)._crop_face(image, 0)
        self.assertEqual(crop.size, (20, 20))
        self.assertEqual(crop.getpixel((0, 0)), (255, 0, 0))

    def test_invalid_or_ambiguous_face_is_mean_placeholder(self) -> None:
        image = Image.new("RGB", (80, 60), color=(255, 0, 0))
        crop = self._dataset(False)._crop_face(image, 0)
        self.assertEqual(crop.size, (1, 1))
        self.assertEqual(crop.getpixel((0, 0)), (123, 117, 104))

    def test_face_transform_preserves_complete_crop(self) -> None:
        train, evaluate = build_transforms(
            "clip", (0.05, 1.0), input_mode="face_crop",
            person_color_jitter_strength=0.1,
            person_color_jitter_probability=1.0,
        )
        self.assertIsInstance(train.transforms[0], PadToSquare)
        self.assertFalse(any(isinstance(item, transforms.RandomResizedCrop) for item in train.transforms))
        self.assertFalse(any(isinstance(item, transforms.RandomApply) for item in train.transforms))
        output = evaluate(Image.new("RGB", (20, 40), color=(120, 80, 40)))
        self.assertEqual(tuple(output.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(output).all())

    def test_face_crop_margin_is_recomputed_from_raw_detection(self) -> None:
        source = self._dataset(True)
        source.face_records = [{
            "face_bbox": [20, 10, 40, 30],
            "face_crop_bbox": [17, 7, 43, 33],
        }]
        source.face_crop_margin = 0.05
        image = Image.new("RGB", (80, 60), color=(255, 0, 0))
        self.assertEqual(source._crop_face(image, 0).size, (22, 22))

    def test_face_specific_jitter_is_opt_in(self) -> None:
        train, _ = build_transforms(
            "clip", (0.05, 1.0), input_mode="face_crop",
            face_color_jitter_strength=0.1,
            face_color_jitter_probability=0.5,
        )
        self.assertTrue(any(isinstance(item, transforms.RandomApply) for item in train.transforms))

    def test_face_loss_indices_strictly_exclude_invalid_and_ambiguous(self) -> None:
        source = object.__new__(EMOTIC)
        source.file_paths = ["a", "b", "c", "d"]
        source.face_training_valid = [True, False, True, False]
        self.assertEqual(filter_face_training_indices(source, [0, 1, 3]), [0])
        with self.assertRaises(RuntimeError):
            filter_face_training_indices(source, [1, 3])


class FaceEndpointFusionTest(unittest.TestCase):
    def test_quality_comparison_uses_only_locked_face_weight(self) -> None:
        result = {"candidates": [
            {"beta": 0.0, "metrics": {"final_mAP": 1.0}},
            {"beta": 0.20, "metrics": {"final_mAP": 2.0}},
        ]}
        self.assertEqual(_fixed_candidate(result)["metrics"]["final_mAP"], 2.0)

    def test_equal_update_audit_checks_each_task_and_scheduler_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({
                "optimizer_updates_by_task": list(EQUAL_UPDATE_BUDGETS),
            }), encoding="utf-8")
            (root / "seed_summary.json").write_text(json.dumps({
                "completed_optimizer_updates": sum(EQUAL_UPDATE_BUDGETS),
            }), encoding="utf-8")
            history = {
                str(task_id): [{
                    "completed_task_optimizer_updates": budget,
                    "skipped_optimizer_steps": 0,
                    "next_learning_rate": 0.0,
                }]
                for task_id, budget in enumerate(EQUAL_UPDATE_BUDGETS)
            }
            (root / "training_history.json").write_text(
                json.dumps(history), encoding="utf-8"
            )
            audit = _validate_equal_update_candidate(root)
            self.assertEqual(audit["completed_total"], 10980)
            history["6"][0]["completed_task_optimizer_updates"] -= 1
            (root / "training_history.json").write_text(
                json.dumps(history), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                _validate_equal_update_candidate(root)

    @staticmethod
    def _manifest_root(root: Path, include_test: bool = False) -> Path:
        detector = {
            "provider": "CPUExecutionProvider",
            "det_size": [640, 640],
            "det_threshold": 0.5,
            "face_margin": 0.15,
            "ambiguity_gap": 0.08,
            "checkpoint_sha256": "detector-sha",
        }
        audit = {
            "protocol": "emotic-face-manifest-v1",
            "training_started": False,
            "detector_config": detector,
            "splits": {
                "train": {"partial": False, "duplicate_face_assignments": 0},
                "val": {"partial": False, "duplicate_face_assignments": 0},
            },
        }
        (root / "manifests").mkdir(parents=True)
        (root / "audit_summary.json").write_text(json.dumps(audit), encoding="utf-8")
        (root / "detector_config.json").write_text(json.dumps(detector), encoding="utf-8")
        train = {"sample_id": "train:a#person=0"}
        (root / "manifests" / "train.jsonl").write_text(json.dumps(train) + "\n", encoding="utf-8")
        val_rows = [
            {"sample_id": "val:a#person=0", "valid_face": True, "ambiguous_match": False,
             "face_short_side": 24.0, "face_detection_score": 0.6},
            {"sample_id": "val:b#person=0", "valid_face": True, "ambiguous_match": True,
             "face_short_side": 80.0, "face_detection_score": 0.9},
            {"sample_id": "val:c#person=0", "valid_face": True, "ambiguous_match": False,
             "face_short_side": 23.9, "face_detection_score": 0.9},
        ]
        (root / "manifests" / "val.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in val_rows), encoding="utf-8"
        )
        if include_test:
            audit["splits"]["test"] = {
                "partial": False,
                "duplicate_face_assignments": 0,
            }
            (root / "audit_summary.json").write_text(
                json.dumps(audit), encoding="utf-8"
            )
            test_rows = [
                {"sample_id": "test:a#person=0", "valid_face": True,
                 "ambiguous_match": False, "face_short_side": 30.0,
                 "face_detection_score": 0.8},
                {"sample_id": "test:b#person=0", "valid_face": False,
                 "ambiguous_match": False, "face_short_side": None,
                 "face_detection_score": None},
            ]
            (root / "manifests" / "test.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in test_rows),
                encoding="utf-8",
            )
        return root

    def test_provenance_and_reliability_are_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._manifest_root(Path(directory))
            provenance = load_face_manifest_provenance(root)
            self.assertEqual(provenance["training_policy"], "valid_face_and_not_ambiguous")
            reliability = load_face_reliability(root)
            self.assertEqual(reliability, {
                "val:a#person=0": True,
                "val:b#person=0": False,
                "val:c#person=0": False,
            })

    def test_test_manifest_is_hashed_and_reliability_is_split_specific(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self._manifest_root(Path(directory), include_test=True)
            provenance = load_face_manifest_provenance(root)
            self.assertIn("test_manifest", provenance["artifact_sha256"])
            self.assertEqual(load_face_reliability(root, "test"), {
                "test:a#person=0": True,
                "test:b#person=0": False,
            })
            with self.assertRaises(ValueError):
                load_face_reliability(root, "train")

    def test_masked_fusion_exactly_falls_back_to_full_person(self) -> None:
        full = np.asarray([[0.8, 0.2], [0.6, 0.4]], dtype=np.float32)
        person = np.asarray([[0.3, 0.7], [0.2, 0.8]], dtype=np.float32)
        face = np.asarray([[0.1, 0.9], [0.9, 0.1]], dtype=np.float32)
        mask = np.asarray([True, False], dtype=np.bool_)
        anchor = 0.8 * full + 0.2 * person
        fused = masked_face_fusion(full, person, face, mask, 0.10)
        np.testing.assert_allclose(fused[1], anchor[1], rtol=0, atol=0)
        np.testing.assert_allclose(fused[0], 0.9 * anchor[0] + 0.1 * face[0])
        np.testing.assert_allclose(
            masked_face_fusion(full, person, face, mask, 0.0), anchor,
            rtol=0, atol=0,
        )

    def test_beta_grid_cannot_be_expanded(self) -> None:
        arrays = np.zeros((1, 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            masked_face_fusion(arrays, arrays, arrays, np.asarray([True]), 0.15)


if __name__ == "__main__":
    unittest.main()
