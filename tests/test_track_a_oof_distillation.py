from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from test_track_a_reproduction import FakeVisual
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.compare_oof_distillation_validation import compare
from multi_lane.track_a.oof_distillation import (
    OOFTeacherBank,
    OOFTeacherLabelView,
    PERSON_FACE_WEIGHTS,
    R1_INVALID_FACE_WEIGHTS,
    R1_RELIABLE_WEIGHTS,
)
from multi_lane.track_a.runner import (
    TASK_SIZES,
    compute_oof_distillation_loss,
    task_indices,
    train_task,
)


class FakeSource:
    def __init__(self) -> None:
        self.sample_ids = [f"train:image{task}.jpg#person=0" for task in range(8)]
        self.targets = [[task_indices(task)[0]] for task in range(8)]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        target = torch.zeros(sum(TASK_SIZES), dtype=torch.float32)
        target[self.targets[index]] = 1
        return torch.full((3, 4, 4), float(index)), target


def fake_oof_tasks():
    tasks = []
    for task_id in range(8):
        seen = sum(TASK_SIZES[: task_id + 1])
        current = task_indices(task_id)
        targets = np.zeros((1, seen), dtype=np.float32)
        targets[0, current[0]] = 1
        full = np.full((1, seen), 0.4, dtype=np.float32)
        person = np.full((1, seen), 0.2, dtype=np.float32)
        face = np.full((1, seen), 0.8, dtype=np.float32)
        tasks.append((
            np.asarray([f"train:image{task_id}.jpg#person=0"]),
            targets, full, person, face,
        ))
    return tasks, {"fold_counts": {str(task): [1, 0, 0] for task in range(8)}}


class OOFTeacherBankTest(unittest.TestCase):
    def _face(self):
        return {
            f"train:image{task}.jpg#person=0": types.SimpleNamespace(
                reliable=task % 2 == 0
            )
            for task in range(8)
        }

    @patch("multi_lane.track_a.oof_distillation.load_face_metadata")
    @patch("multi_lane.track_a.oof_distillation.load_oof_tasks")
    def test_person_face_teacher_uses_locked_ratio_and_invalid_fallback(
        self, load_tasks, load_face
    ) -> None:
        load_tasks.return_value = fake_oof_tasks()
        load_face.return_value = self._face()
        source = FakeSource()
        bank = OOFTeacherBank("oof", "manifest", source, "person_face")
        person_weight, face_weight = PERSON_FACE_WEIGHTS
        reliable = bank.tasks[0].probabilities[source.sample_ids[0]]
        invalid = bank.tasks[1].probabilities[source.sample_ids[1]]
        np.testing.assert_allclose(
            reliable, person_weight * 0.2 + face_weight * 0.8
        )
        np.testing.assert_allclose(invalid, 0.2)
        view = OOFTeacherLabelView(
            source, [0], task_indices(0), bank.tasks[0]
        )
        image, target, teacher = view[0]
        self.assertEqual(tuple(image.shape), (3, 4, 4))
        self.assertEqual(tuple(target.shape), (5,))
        torch.testing.assert_close(teacher, torch.from_numpy(reliable))

    @patch("multi_lane.track_a.oof_distillation.load_face_metadata")
    @patch("multi_lane.track_a.oof_distillation.load_oof_tasks")
    def test_person_teacher_never_uses_face(self, load_tasks, load_face) -> None:
        load_tasks.return_value = fake_oof_tasks()
        load_face.return_value = self._face()
        source = FakeSource()
        bank = OOFTeacherBank("oof", "manifest", source, "person")
        for task_id, task in enumerate(bank.tasks):
            np.testing.assert_allclose(
                task.probabilities[source.sample_ids[task_id]], 0.2
            )

    @patch("multi_lane.track_a.oof_distillation.load_face_metadata")
    @patch("multi_lane.track_a.oof_distillation.load_oof_tasks")
    def test_r1_teacher_uses_locked_reliable_and_fallback_weights(
        self, load_tasks, load_face
    ) -> None:
        load_tasks.return_value = fake_oof_tasks()
        load_face.return_value = self._face()
        source = FakeSource()
        bank = OOFTeacherBank("oof", "manifest", source, "r1")
        reliable = bank.tasks[0].probabilities[source.sample_ids[0]]
        invalid = bank.tasks[1].probabilities[source.sample_ids[1]]
        np.testing.assert_allclose(
            reliable,
            R1_RELIABLE_WEIGHTS[0] * 0.4
            + R1_RELIABLE_WEIGHTS[1] * 0.2
            + R1_RELIABLE_WEIGHTS[2] * 0.8,
        )
        np.testing.assert_allclose(
            invalid,
            R1_INVALID_FACE_WEIGHTS[0] * 0.4
            + R1_INVALID_FACE_WEIGHTS[1] * 0.2,
        )


class OOFTrainingLossTest(unittest.TestCase):
    def test_soft_target_validation(self) -> None:
        logits = torch.zeros(2, sum(TASK_SIZES))
        valid = torch.full((2, TASK_SIZES[0]), 0.3)
        self.assertTrue(torch.isfinite(compute_oof_distillation_loss(
            logits, valid, task_indices(0), 1.0, "legacy_full_zero"
        )))
        invalid = valid.clone()
        invalid[0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, "\[0, 1\]"):
            compute_oof_distillation_loss(
                logits, invalid, task_indices(0), 1.0, "legacy_full_zero"
            )

    def test_training_accepts_teacher_batch_and_records_losses(self) -> None:
        torch.manual_seed(13)
        model = MultiLaneModel(
            FakeVisual(), TASK_SIZES, num_selectors=2, num_prompts=2,
            num_prompt_layers=1, adapter_mode="image_token",
            adapter_bottleneck_dim=3, adapter_layer_indices=(0,),
        )
        model.activate_task(0)
        images = torch.randn(2, 3, 4, 4)
        targets = torch.tensor(
            [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=torch.float32
        )
        teacher = torch.full_like(targets, 0.25)
        loader = DataLoader(
            TensorDataset(images, targets, teacher), batch_size=2
        )
        validation_loader = DataLoader(
            TensorDataset(images, targets), batch_size=2
        )
        history = train_task(
            model, loader, validation_loader, torch.device("cpu"),
            task_id=0, epochs=1,
            learning_rate=1e-3, weight_decay=0.0, temperature=1.0, amp=False,
            adapter_learning_rate=4e-4, loss_routing="adapter_asl",
            oof_distillation_mix=0.2,
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["oof_distillation_mix"], 0.2)
        self.assertGreater(history[0]["oof_distillation_loss"], 0)
        self.assertGreater(history[0]["supervised_bce_loss"], 0)


class OOFComparisonTest(unittest.TestCase):
    @patch("multi_lane.track_a.compare_oof_distillation_validation._row")
    def test_eligibility_precedes_raw_metric_ranking(self, row) -> None:
        row.side_effect = [
            {
                "name": "D0", "distillation": None,
                "full_metrics": {"final_mAP": 10.0, "average_mAP": 20.0},
                "locked_R1_metrics": {"final_mAP": 30.0},
            },
            {
                "name": "D1", "distillation": {"mode": "person", "mix": 0.2},
                "full_metrics": {"final_mAP": 10.1, "average_mAP": 20.1},
                "locked_R1_metrics": {"final_mAP": 30.06},
            },
            {
                "name": "D2", "distillation": {"mode": "person_face", "mix": 0.2},
                "full_metrics": {"final_mAP": 10.2, "average_mAP": 19.9},
                "locked_R1_metrics": {"final_mAP": 30.2},
            },
        ]
        result = compare(
            ("D0", Path("d0")), (("D1", Path("d1")), ("D2", Path("d2"))),
            Path("person"), Path("face"), Path("manifest"),
        )
        self.assertEqual(result["winner"]["name"], "D1")
        self.assertTrue(result["winner"]["eligible"])
        self.assertFalse(result["candidates"][1]["eligible"])


if __name__ == "__main__":
    unittest.main()
