from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from multi_lane.track_a.adapter import TaskLaneTransformerAdapterBank
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

    def test_zero_initialized_adapter_preserves_baseline_logits(self) -> None:
        torch.manual_seed(7)
        baseline = MultiLaneModel(
            FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1,
        )
        torch.manual_seed(7)
        adapted = MultiLaneModel(
            FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1, adapter_mode="task_lane",
            adapter_bottleneck_dim=3, adapter_layer_indices=(0,),
            adapter_residual_scale=0.1,
        )
        baseline.activate_task(0)
        adapted.activate_task(0)
        baseline.eval()
        adapted.eval()
        images = torch.randn(3, 3, 4, 4)
        baseline_logits = baseline.current_all_logits(images)
        adapted_logits = adapted.current_all_logits(images)
        self.assertTrue(torch.equal(baseline_logits, adapted_logits))

        with torch.no_grad():
            bias = adapted.adapter_bank.task_adapters[0]["0"].up.bias
            bias[0] = 1.0
        changed_logits = adapted.current_all_logits(images)
        self.assertFalse(torch.equal(baseline_logits, changed_logits))
        adapted.set_adapter_runtime_enabled(False)
        self.assertTrue(
            torch.equal(baseline_logits, adapted.current_all_logits(images))
        )

    def test_adapter_initialization_preserves_global_rng_state(self) -> None:
        torch.manual_seed(19)
        MultiLaneModel(
            FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1,
        )
        baseline_rng_state = torch.get_rng_state().clone()

        torch.manual_seed(19)
        MultiLaneModel(
            FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1, adapter_mode="task_lane",
            adapter_bottleneck_dim=3, adapter_layer_indices=(0,),
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), baseline_rng_state))

    def test_adapter_parameter_names_cover_active_optimizer_parameters(self) -> None:
        model = MultiLaneModel(
            FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1, adapter_mode="task_lane",
            adapter_bottleneck_dim=3, adapter_layer_indices=(0,),
        )
        model.activate_task(0)
        named = dict(model.named_parameters())
        expected = sum(named[name].numel() for name in model.optimizer_parameter_names())
        actual = sum(parameter.numel() for parameter in model.optimizer_parameters())
        self.assertEqual(actual, expected)


class TrackAAdapterBankTest(unittest.TestCase):
    def test_identity_routing_freezing_and_parameter_count(self) -> None:
        bank = TaskLaneTransformerAdapterBank(
            num_tasks=2,
            hidden_dim=8,
            bottleneck_dim=3,
            layer_indices=(0,),
            residual_scale=0.5,
        )
        tokens = torch.randn(2, 2, 3, 8)
        self.assertTrue(torch.equal(bank.delta_for_layer(0, tokens, (0, 1)), torch.zeros_like(tokens)))
        self.assertTrue(torch.equal(bank.delta_for_layer(1, tokens, (0, 1)), torch.zeros_like(tokens)))
        self.assertEqual(bank.per_task_parameter_count(), 2 * 8 * 3 + 3 + 8)
        self.assertEqual(bank.total_parameter_count(), 2 * (2 * 8 * 3 + 3 + 8))

        with torch.no_grad():
            bank.task_adapters[0]["0"].up.bias[0] = 2.0
            bank.task_adapters[1]["0"].up.bias[1] = 4.0
        delta = bank.delta_for_layer(0, tokens, (0, 1))
        self.assertTrue(torch.all(delta[0, :, :, 0] == 1.0))
        self.assertTrue(torch.all(delta[1, :, :, 1] == 2.0))
        self.assertTrue(torch.all(delta[0, :, :, 1:] == 0.0))

        future_down = bank.task_adapters[1]["0"].down.weight.detach().clone()
        bank.activate_task(0)
        self.assertTrue(all(parameter.requires_grad for parameter in bank.task_adapters[0].parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in bank.task_adapters[1].parameters()))
        bank.activate_task(1)
        self.assertTrue(
            torch.equal(bank.task_adapters[1]["0"].down.weight, future_down)
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in bank.task_adapters[0].parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in bank.task_adapters[1].parameters()))

    def test_copy_previous_warm_starts_only_the_new_task(self) -> None:
        bank = TaskLaneTransformerAdapterBank(
            num_tasks=2,
            hidden_dim=8,
            bottleneck_dim=3,
            layer_indices=(0,),
            task_initialization="copy_previous",
        )
        bank.activate_task(0)
        with torch.no_grad():
            for index, parameter in enumerate(
                bank.task_adapters[0].parameters(), start=1
            ):
                parameter.fill_(float(index))
        expected = {
            name: value.detach().clone()
            for name, value in bank.task_adapters[0].state_dict().items()
        }

        bank.activate_task(1)
        for name, value in bank.task_adapters[1].state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]))
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in bank.task_adapters[0].parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in bank.task_adapters[1].parameters()
            )
        )


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

    def test_single_task_summary_has_zero_forgetting(self) -> None:
        row = TaskMetrics(
            task_id=0, seen_classes=TASK_SIZES[0], samples=1,
            threshold=0.5, mAP=80.0, cPrecision=0.0, cRecall=0.0,
            cF1=0.0, oPrecision=0.0, oRecall=0.0, oF1=0.0,
            per_class_ap=[80.0] * TASK_SIZES[0],
        )
        summary = summarize_tasks([row])
        self.assertEqual(summary["forgetting"], 0.0)
        self.assertEqual(summary["per_class_forgetting"], {})


if __name__ == "__main__":
    unittest.main()
