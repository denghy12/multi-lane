from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from multi_lane.continual_datasets.continual_datasets import EMOTIC
from multi_lane.track_a.fuse_validation_scores import (
    COMMON_CONFIG_FIELDS,
    fused_scores,
    fuse_validation_runs,
    load_aligned_task_scores,
)
from multi_lane.track_a.runner import (
    CLIP_IMAGE_MEAN,
    CLASS_ORDER,
    TASK_SIZES,
    LabelView,
    PadToSquare,
    build_transforms,
    compute_metrics,
    summarize_tasks,
    write_evaluation_scores,
)


class FakeEmoticSource:
    def __init__(self) -> None:
        self.sample_ids = ("val:a.jpg#person=0", "val:a.jpg#person=1")
        self.images = (torch.zeros(3, 4, 4), torch.ones(3, 4, 4))
        self.labels = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


class DualViewInputTest(unittest.TestCase):
    def test_stable_sample_id_includes_split_path_and_person(self) -> None:
        self.assertEqual(
            EMOTIC._sample_id("val", "folder", "image.jpg", 3),
            "val:folder/image.jpg#person=3",
        )

    def test_label_view_optionally_returns_stable_id(self) -> None:
        view = LabelView(FakeEmoticSource(), (1,), (0, 1), include_sample_id=True)
        _, target, sample_id = view[0]
        self.assertEqual(sample_id, "val:a.jpg#person=1")
        self.assertTrue(torch.equal(target, torch.tensor([0.0, 1.0])))

    def test_pad_to_square_preserves_pixels_and_centers_padding(self) -> None:
        image = Image.new("RGB", (2, 4), color=(255, 0, 0))
        padded = PadToSquare(fill=(0, 255, 0))(image)
        self.assertEqual(padded.size, (4, 4))
        self.assertEqual(padded.getpixel((0, 1)), (0, 255, 0))
        self.assertEqual(padded.getpixel((1, 1)), (255, 0, 0))
        self.assertEqual(padded.getpixel((3, 1)), (0, 255, 0))

    def test_letterbox_transform_avoids_random_resized_crop(self) -> None:
        train, evaluate = build_transforms(
            "clip",
            (0.7, 1.0),
            input_mode="person_crop",
            person_transform_mode="letterbox",
            person_color_jitter_strength=0.1,
            person_color_jitter_probability=0.2,
        )
        self.assertFalse(
            any(isinstance(item, transforms.RandomResizedCrop) for item in train.transforms)
        )
        self.assertIsInstance(train.transforms[0], PadToSquare)
        self.assertEqual(
            train.transforms[0].fill,
            tuple(int(round(value * 255)) for value in CLIP_IMAGE_MEAN),
        )
        output = evaluate(Image.new("RGB", (40, 100), color=(120, 80, 40)))
        self.assertEqual(tuple(output.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(output).all())


class DualViewScoreTest(unittest.TestCase):
    @staticmethod
    def _write_validation_run(root: Path, input_mode: str, offset: float) -> None:
        config = {field: None for field in COMMON_CONFIG_FIELDS}
        config.update(
            {
                "protocol_id": "emotic_b5c3_v0.1",
                "seed": 0,
                "eval_batch_size": 64,
                "task_sizes": list(TASK_SIZES),
                "class_order": list(CLASS_ORDER),
                "reporting_split": "val",
                "max_tasks": 8,
                "save_checkpoints": False,
                "threshold": 0.5,
                "input_mode": input_mode,
                "person_transform_mode": (
                    "letterbox" if input_mode == "person_crop" else "legacy_crop"
                ),
                "save_evaluation_scores": True,
            }
        )
        rows = []
        for task_id, seen_classes in enumerate(np.cumsum(TASK_SIZES)):
            sample_ids = tuple(f"val:sample{index}#person=0" for index in range(6))
            targets = torch.zeros(6, int(seen_classes))
            for class_id in range(int(seen_classes)):
                targets[class_id % 6, class_id] = 1.0
                targets[(class_id + 2) % 6, class_id] = 1.0
            logits = (targets * 4.0 - 2.0) + offset
            write_evaluation_scores(
                root / "validation_scores" / f"task{task_id}.npz",
                task_id,
                sample_ids,
                logits,
                targets,
            )
            rows.append(
                compute_metrics(
                    task_id, torch.sigmoid(logits), targets, threshold=0.5
                )
            )
        summary = {
            "status": "complete",
            "config": config,
            "metrics": summarize_tasks(rows),
            "task_metrics": [asdict(row) for row in rows],
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (root / "seed_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

    def test_score_dump_alignment_reorders_person_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full_path = root / "full.npz"
            person_path = root / "person.npz"
            targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            write_evaluation_scores(
                full_path,
                0,
                ("a", "b"),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                targets,
            )
            write_evaluation_scores(
                person_path,
                0,
                ("b", "a"),
                torch.tensor([[30.0, 40.0], [10.0, 20.0]]),
                targets.flip(0),
            )
            sample_ids, _, person_logits, aligned_targets = load_aligned_task_scores(
                full_path, person_path
            )
            self.assertEqual(sample_ids.tolist(), ["a", "b"])
            np.testing.assert_array_equal(
                person_logits, np.asarray([[10.0, 20.0], [30.0, 40.0]])
            )
            np.testing.assert_array_equal(aligned_targets, targets.numpy())

    def test_logit_and_probability_fusion_preserve_endpoints(self) -> None:
        full = np.asarray([[0.0, 1.0]], dtype=np.float32)
        person = np.asarray([[2.0, -1.0]], dtype=np.float32)
        expected_full = torch.sigmoid(torch.from_numpy(full)).numpy()
        expected_person = torch.sigmoid(torch.from_numpy(person)).numpy()
        for mode in ("logit", "probability"):
            np.testing.assert_allclose(fused_scores(full, person, 0.0, mode), expected_full)
            np.testing.assert_allclose(fused_scores(full, person, 1.0, mode), expected_person)

    def test_complete_fusion_validates_score_dump_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full"
            person = root / "person"
            self._write_validation_run(full, "full", 0.0)
            self._write_validation_run(person, "person_crop", -0.2)
            result = fuse_validation_runs(full, person, (0.0, 0.5, 1.0))
            self.assertEqual(result["selection_split"], "val")
            self.assertEqual(result["anchors"]["full"]["alpha"], 0.0)
            self.assertEqual(result["anchors"]["person"]["alpha"], 1.0)


if __name__ == "__main__":
    unittest.main()
