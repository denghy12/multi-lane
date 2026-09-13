from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from test_track_a_reproduction import FakeVisual
from multi_lane.track_a.compare_oof_advantage_ranking_validation import compare
from multi_lane.track_a.fuse_validation_scores import _validate_runs
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.oof_advantage_ranking import CorrectedPairDataset, TaskPairTable
from multi_lane.track_a.runner import (
    TASK_SIZES, compute_pairwise_ranking_loss, train_task,
)


class TinySource:
    def __getitem__(self, index):
        return torch.full((3, 4, 4), float(index)), torch.zeros(sum(TASK_SIZES))


class OOFAdvantagePairTest(unittest.TestCase):
    def test_pair_dataset_is_balanced_and_uses_locked_pairs(self) -> None:
        table = TaskPairTable(
            class_indices=(0, 2),
            positive_indices=(np.asarray([1, 3]), np.asarray([5, 7])),
            negative_indices=(np.asarray([2, 4]), np.asarray([6, 8])),
            pair_counts=(2, 2),
            fold_ap_gains={},
        )
        dataset = CorrectedPairDataset(TinySource(), table, length=4, seed=9)
        classes = [int(dataset[index][2]) for index in range(4)]
        self.assertEqual(classes, [0, 2, 0, 2])

    def test_pairwise_loss_rewards_correct_margin(self) -> None:
        poor = compute_pairwise_ranking_loss(
            torch.tensor([0.0, -1.0]), torch.tensor([1.0, 0.0])
        )
        good = compute_pairwise_ranking_loss(
            torch.tensor([2.0, 1.0]), torch.tensor([0.0, -1.0])
        )
        self.assertLess(good, poor)

    def test_ranking_gradient_never_changes_adapter_objective(self) -> None:
        torch.manual_seed(29)
        plain = MultiLaneModel(
            FakeVisual(), TASK_SIZES, num_selectors=2, num_prompts=2,
            num_prompt_layers=1, adapter_mode="image_token",
            adapter_bottleneck_dim=3, adapter_layer_indices=(0,),
        )
        ranked = copy.deepcopy(plain)
        plain.activate_task(0)
        ranked.activate_task(0)
        images = torch.randn(2, 3, 4, 4)
        targets = torch.tensor(
            [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=torch.float32
        )
        loader = DataLoader(TensorDataset(images, targets), batch_size=2)
        pair_loader = DataLoader(TensorDataset(
            images[:1], images[1:], torch.tensor([0], dtype=torch.long)
        ), batch_size=1)
        common = dict(
            validation_loader=loader, device=torch.device("cpu"), task_id=0,
            epochs=1, learning_rate=1e-3, weight_decay=0.0,
            temperature=1.0, amp=False, adapter_learning_rate=4e-4,
            loss_routing="adapter_asl",
        )
        train_task(plain, loader, **common)
        history = train_task(
            ranked, loader, ranking_loader=pair_loader,
            ranking_loss_weight=0.05, **common
        )
        for left, right in zip(
            plain.adapter_optimizer_parameters(), ranked.adapter_optimizer_parameters()
        ):
            torch.testing.assert_close(left, right)
        self.assertGreater(history[0]["pairwise_ranking_loss"], 0)
        self.assertEqual(history[0]["pairwise_ranking_loss_weight"], 0.05)


class OOFAdvantageComparisonTest(unittest.TestCase):
    def test_fusion_allows_only_explicit_training_objective_difference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = {
                "input_normalization": "clip",
                "reporting_split": "val",
                "save_evaluation_scores": True,
            }
            for name, mode, objective in (
                ("full", "full", "hard_bce_plus_oof_pairwise_ranking"),
                ("person", "person_crop", "bce"),
            ):
                run = root / name
                run.mkdir()
                config = {
                    **shared,
                    "input_mode": mode,
                    "model_parameter_objective": objective,
                }
                if name == "person":
                    config["person_transform_mode"] = "letterbox"
                (run / "config.json").write_text(json.dumps(config))
                (run / "seed_summary.json").write_text(json.dumps({
                    "status": "complete",
                    "task_metrics": [{} for _ in TASK_SIZES],
                }))
            with self.assertRaisesRegex(ValueError, "model_parameter_objective"):
                _validate_runs(root / "full", root / "person")
            _validate_runs(
                root / "full",
                root / "person",
                allow_full_training_objective_difference=True,
            )

    @patch("multi_lane.track_a.compare_oof_advantage_ranking_validation._row")
    def test_gate_requires_all_three_metrics(self, row) -> None:
        row.side_effect = [
            {
                "name": "D0", "distillation": None,
                "full_metrics": {"final_mAP": 10.0, "average_mAP": 20.0},
                "locked_R1_metrics": {"final_mAP": 30.0},
            },
            {
                "name": "E1", "distillation": None,
                "full_metrics": {"final_mAP": 10.1, "average_mAP": 19.9},
                "locked_R1_metrics": {"final_mAP": 30.1},
            },
        ]
        config = {
            "oof_advantage_ranking": {
                "loss": "softplus(negative_logit-positive_logit)",
                "loss_weight": 0.05,
                "hard_bce_weight": 1.0,
                "adapter_receives_ranking_gradient": False,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("d0", {}), ("e1", config)):
                (root / name).mkdir()
                (root / name / "config.json").write_text(json.dumps(value))
            result = compare(
                ("D0", root / "d0"), ("E1", root / "e1"),
                Path("person"), Path("face"), Path("manifest"),
            )
        self.assertFalse(result["candidate"]["eligible"])
        self.assertTrue(result["decision"]["end_oof_distillation_if_ineligible"])


if __name__ == "__main__":
    unittest.main()
