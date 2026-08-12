from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from multi_lane.track_a.adapter import (
    TaskImageTokenAdapterBank,
    TaskLaneTransformerAdapterBank,
)
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import (
    CLASS_ORDER,
    TASK_SIZES,
    TaskMetrics,
    backward_routed_training_losses,
    build_transforms,
    compute_asymmetric_training_loss,
    compute_training_loss,
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

        for adapter_mode in ("task_lane", "image_token"):
            with self.subTest(adapter_mode=adapter_mode):
                torch.manual_seed(19)
                MultiLaneModel(
                    FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
                    num_prompt_layers=1, adapter_mode=adapter_mode,
                    adapter_bottleneck_dim=3, adapter_layer_indices=(0,),
                )
                self.assertTrue(
                    torch.equal(torch.get_rng_state(), baseline_rng_state)
                )

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

    def test_image_token_adapter_preserves_then_changes_selector_input(self) -> None:
        torch.manual_seed(29)
        baseline = MultiLaneModel(
            FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1,
        )
        torch.manual_seed(29)
        adapted = MultiLaneModel(
            FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1, adapter_mode="image_token",
            adapter_bottleneck_dim=3, adapter_layer_indices=(0,),
            adapter_residual_scale=0.1,
        )
        baseline.activate_task(0)
        adapted.activate_task(0)
        images = torch.randn(3, 3, 4, 4)
        baseline_logits = baseline.current_all_logits(images)
        initial_logits = adapted.current_all_logits(images)
        self.assertTrue(torch.allclose(baseline_logits, initial_logits, atol=1e-7))

        with torch.no_grad():
            adapted.adapter_bank.task_adapters[0]["0"].up.bias[0] = 1.0
        changed_logits = adapted.current_all_logits(images)
        self.assertFalse(torch.equal(baseline_logits, changed_logits))
        adapted.set_adapter_runtime_enabled(False)
        self.assertTrue(
            torch.equal(baseline_logits, adapted.current_all_logits(images))
        )

        adapted.set_adapter_runtime_enabled(True)
        adapted.zero_grad(set_to_none=True)
        adapted.current_all_logits(images).sum().backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in adapted.adapter_optimizer_parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in adapted.visual_encoder.parameters()
            )
        )


class TrackAAdapterBankTest(unittest.TestCase):
    def test_image_token_bank_routes_each_lane_without_visual_writeback(self) -> None:
        bank = TaskImageTokenAdapterBank(
            num_tasks=2,
            hidden_dim=8,
            bottleneck_dim=3,
            layer_indices=(0,),
            residual_scale=0.5,
        )
        frozen = torch.randn(2, 5, 8)
        original = frozen.clone()
        initial = bank.adapted_tokens_for_layer(0, frozen, (0, 1))
        self.assertTrue(torch.equal(initial[0], frozen))
        self.assertTrue(torch.equal(initial[1], frozen))

        with torch.no_grad():
            bank.task_adapters[0]["0"].up.bias[0] = 2.0
            bank.task_adapters[1]["0"].up.bias[1] = 4.0
        adapted = bank.adapted_tokens_for_layer(0, frozen, (0, 1))
        self.assertTrue(torch.all(adapted[0, :, :, 0] == frozen[:, :, 0] + 1.0))
        self.assertTrue(torch.all(adapted[1, :, :, 1] == frozen[:, :, 1] + 2.0))
        self.assertTrue(torch.equal(frozen, original))
        bypassed = bank.adapted_tokens_for_layer(1, frozen, (0, 1))
        self.assertTrue(torch.equal(bypassed[0], frozen))
        self.assertTrue(torch.equal(bypassed[1], frozen))

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
    def test_asl_zero_gamma_zero_clip_matches_bce(self) -> None:
        logits = torch.tensor([[0.2, -0.4, 0.8, -0.1]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0]])
        indices = (1, 3)
        bce = compute_training_loss(
            logits, targets, indices, 1.0, "current_only"
        )
        asl = compute_asymmetric_training_loss(
            logits,
            targets,
            indices,
            1.0,
            "current_only",
            gamma_neg=0.0,
            gamma_pos=0.0,
            clip=0.0,
        )
        self.assertTrue(torch.allclose(asl, bce, atol=1e-7))

    def test_asl_suppresses_easy_negative_gradient(self) -> None:
        bce_logits = torch.tensor([[-5.0]], requires_grad=True)
        targets = torch.zeros(1, 1)
        bce = compute_training_loss(
            bce_logits, targets, (0,), 1.0, "current_only"
        )
        bce.backward()
        asl_logits = torch.tensor([[-5.0]], requires_grad=True)
        asl = compute_asymmetric_training_loss(
            asl_logits,
            targets,
            (0,),
            1.0,
            "current_only",
            gamma_neg=9.8,
            gamma_pos=0.0,
            clip=0.05,
        )
        asl.backward()
        self.assertLess(asl_logits.grad.abs().item(), bce_logits.grad.abs().item())

    def test_mixed_loss_routing_isolates_parameter_group_gradients(self) -> None:
        model_parameter = nn.Parameter(torch.tensor(2.0))
        adapter_parameter = nn.Parameter(torch.tensor(3.0))
        model_loss = (model_parameter + 2.0 * adapter_parameter).pow(2)
        adapter_loss = (3.0 * model_parameter + adapter_parameter).pow(2)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        backward_routed_training_losses(
            model_loss,
            adapter_loss,
            (model_parameter,),
            (adapter_parameter,),
            scaler,
            same_objective=False,
        )
        self.assertEqual(float(model_parameter.grad), 16.0)
        self.assertEqual(float(adapter_parameter.grad), 18.0)

    def test_current_only_loss_removes_legacy_class_count_gradient_scaling(self) -> None:
        legacy_logits = torch.tensor(
            [[0.2, -0.4, 0.8, -0.1, 0.3, -0.7]], requires_grad=True
        )
        current_logits = legacy_logits.detach().clone().requires_grad_(True)
        targets = torch.tensor([[1.0, 0.0]])
        current = (1, 4)
        legacy_loss = compute_training_loss(
            legacy_logits, targets, current, 1.0, "legacy_full_zero"
        )
        current_loss = compute_training_loss(
            current_logits, targets, current, 1.0, "current_only"
        )
        legacy_loss.backward()
        current_loss.backward()
        self.assertTrue(torch.equal(
            legacy_logits.grad[:, [0, 2, 3, 5]], torch.zeros(1, 4)
        ))
        self.assertTrue(torch.equal(
            current_logits.grad[:, [0, 2, 3, 5]], torch.zeros(1, 4)
        ))
        self.assertTrue(
            torch.allclose(
                current_logits.grad[:, list(current)],
                legacy_logits.grad[:, list(current)] * 3.0,
            )
        )

    def test_transform_options_are_independent(self) -> None:
        legacy_train, legacy_eval = build_transforms()
        clip_train, clip_eval = build_transforms("clip", (0.5, 1.0))
        self.assertEqual(tuple(legacy_train.transforms[0].scale), (0.05, 1.0))
        self.assertEqual(tuple(clip_train.transforms[0].scale), (0.5, 1.0))
        self.assertEqual(legacy_train.transforms[-1].__class__.__name__, "ToTensor")
        self.assertEqual(legacy_eval.transforms[-1].__class__.__name__, "ToTensor")
        self.assertEqual(clip_train.transforms[-1].__class__.__name__, "Normalize")
        self.assertEqual(clip_eval.transforms[-1].__class__.__name__, "Normalize")

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
