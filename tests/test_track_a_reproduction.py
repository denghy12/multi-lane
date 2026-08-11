from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import (
    CLASS_ORDER,
    TASK_SIZES,
    TaskMetrics,
    compute_metrics,
    summarize_tasks,
)


class FakeResidualBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        attended = self.attn(self.ln_1(value), self.ln_1(value), self.ln_1(value))[0]
        value = value + attended
        return value + self.mlp(self.ln_2(value))


class FakeTransformer(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([FakeResidualBlock(width, heads)])


class FakeVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        width = 8
        self.conv1 = nn.Conv2d(3, width, kernel_size=2, stride=2, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(width))
        self.positional_embedding = nn.Parameter(torch.randn(5, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = FakeTransformer(width, heads=2)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, 4))
        self.output_dim = 4


class TrackAModelTest(unittest.TestCase):
    def test_task_copy_concat_and_frozen_visual(self) -> None:
        torch.manual_seed(3)
        model = MultiLaneModel(
            FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1,
        )
        model.activate_task(0)
        images = torch.randn(3, 3, 4, 4)
        train_logits = model.current_logits(images)
        self.assertEqual(tuple(train_logits.shape), (3, 2))
        first_selectors = model.selectors[0].detach().clone()
        first_prompts = model.prompts[0][:, 0].detach().clone()
        model.activate_task(1)
        self.assertTrue(torch.equal(model.selectors[1], first_selectors))
        self.assertTrue(torch.equal(model.prompts[0][:, 1], first_prompts))
        model.eval()
        seen_logits = model.seen_logits(images)
        self.assertEqual(tuple(seen_logits.shape), (3, 3))
        model.assert_visual_frozen()

    def test_parameter_names_cover_optimizer_parameters(self) -> None:
        model = MultiLaneModel(
            FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1,
        )
        named = dict(model.named_parameters())
        expected = sum(named[name].numel() for name in model.optimizer_parameter_names())
        actual = sum(parameter.numel() for parameter in model.optimizer_parameters())
        self.assertEqual(actual, expected)


class TrackAMetricsTest(unittest.TestCase):
    def test_fixed_threshold_metrics(self) -> None:
        scores = torch.tensor([[0.9, 0.1], [0.7, 0.8], [0.2, 0.6]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        row = compute_metrics(0, scores, targets, threshold=0.5)
        self.assertAlmostEqual(row.mAP, 100.0, places=5)
        self.assertAlmostEqual(row.cF1, (2.0 / 3.0 + 1.0) * 50.0)
        self.assertAlmostEqual(row.oF1, 6.0 / 7.0 * 100.0)

    def test_forgetting_uses_old_class_ap_history(self) -> None:
        rows = []
        for task_id, seen in enumerate(np.cumsum(TASK_SIZES)):
            values = [80.0] * int(seen)
            if task_id == len(TASK_SIZES) - 1:
                values[: sum(TASK_SIZES[:-1])] = [75.0] * sum(TASK_SIZES[:-1])
            rows.append(TaskMetrics(
                task_id=task_id, seen_classes=int(seen), samples=1,
                threshold=0.5, mAP=80.0, cPrecision=0.0, cRecall=0.0,
                cF1=0.0, oPrecision=0.0, oRecall=0.0, oF1=0.0,
                per_class_ap=values,
            ))
        summary = summarize_tasks(rows)
        self.assertAlmostEqual(summary["forgetting"], 5.0)
        self.assertEqual(len(summary["per_class_forgetting"]), len(CLASS_ORDER) - 3)


if __name__ == "__main__":
    unittest.main()
