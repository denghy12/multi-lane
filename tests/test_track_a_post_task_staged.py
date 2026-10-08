from __future__ import annotations

import copy
import random
import unittest

import numpy as np
import torch

from multi_lane.track_a.adapter import ParaXImageAdapterBank
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.post_task_calibration import fit_cached_routes, isolated_rng, base_hash
from test_track_a_reproduction import FakeVisual


class StagedPostTaskRoutingTest(unittest.TestCase):
    def make_model(self, static=False):
        return MultiLaneModel(
            FakeVisual(), task_sizes=(2, 1), num_selectors=2,
            num_prompts=2, num_prompt_layers=1, view_fusion="fixed_three_view",
            parax_mode="post_task_staged_static" if static else "post_task_staged",
            parax_layer_indices=(0,), parax_rank=2, parax_num_experts=3,
            parax_initialization="zero_output", parax_output_scale_mode="fixed",
            parax_residual_scale=1.0, parax_smooth_ratio_bound=0.02,
        )

    def test_smooth_bound_has_finite_zero_gradient_and_limits_every_token(self):
        bank = ParaXImageAdapterBank(8, 2, 3, (0,), num_tasks=2,
            task_local_router=True, task_local_projection=True,
            freeze_center_from_start=True, smooth_ratio_bound=0.02,
            initialization="zero_output", residual_scale=1.0, output_scale_mode="fixed")
        bank.activate_task(0)
        x = torch.randn(3, 4, 8)
        y, _ = bank(0, x)
        torch.testing.assert_close(y, x, atol=0, rtol=0)
        y.square().mean().backward()
        grad = bank.task_projections[0].weight.grad
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(float(grad.norm()), 0)
        with torch.no_grad():
            bank.task_projections[0].weight.fill_(1e6)
        y, _ = bank(0, x)
        ratio = (y - x).norm(dim=-1) / x.norm(dim=-1)
        self.assertLessEqual(float(ratio.max()), 0.020001)

    def test_staged_fit_freezes_base_center_old_routes_and_new_task_starts_identity(self):
        torch.manual_seed(83)
        model = self.make_model()
        model.activate_task(0)
        images = {name: torch.randn(4, 3, 4, 4) for name in ("full", "person", "face")}
        images["face_reliable"] = torch.tensor([True, False, True, True])
        model.set_parax_runtime_enabled(False)
        with torch.no_grad():
            _, features = model.encode_lanes_with_views(images, False)
        model.set_parax_runtime_enabled(True)
        cache = {"source_split": "train", "task_id": 0,
                 "features": {name: value.detach() for name, value in features.items()},
                 "targets": torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 1.]]),
                 "face_reliable": images["face_reliable"]}
        result = fit_cached_routes(model, cache, (0, 1), torch.device("cpu"),
                                   epochs=4, learning_rate=0.01, batch_size=2)
        self.assertEqual(result["initial_output_max_difference"], 0)
        self.assertEqual(result["base_hash_before"], result["base_hash_after"])
        self.assertEqual(result["center_hash_before"], result["center_hash_after"])
        self.assertGreater(float(model.parax_bank.task_projections[0].weight.norm()), 0)
        old_feature = model.encode_lanes_with_views(images, True)[1]["full"].detach()
        model.activate_task(1)
        model.set_parax_runtime_enabled(False)
        base = model.current_all_logits(images).detach()
        model.set_parax_runtime_enabled(True)
        torch.testing.assert_close(model.current_all_logits(images), base, atol=1e-6, rtol=1e-6)
        with torch.no_grad():
            _, features = model.encode_lanes_with_views(images, False)
        cache.update(task_id=1, features={name: value.detach() for name, value in features.items()},
                     targets=torch.tensor([[1.], [0.], [1.], [0.]]))
        result = fit_cached_routes(model, cache, (2,), torch.device("cpu"),
                                   epochs=3, learning_rate=0.01, batch_size=2)
        self.assertEqual(result["old_route_hash_before"], result["old_route_hash_after"])
        torch.testing.assert_close(model.encode_lanes_with_views(images, True)[1]["full"][:, :1],
                                   old_feature, atol=1e-6, rtol=1e-6)
        model.assert_visual_frozen()

    def test_disabled_route_matches_baseline_rng_and_optimizer_path(self):
        torch.manual_seed(101)
        visual = FakeVisual()
        common = dict(task_sizes=(2, 1), num_selectors=2, num_prompts=2,
                      num_prompt_layers=1, view_fusion="fixed_three_view")
        torch.manual_seed(102)
        baseline = MultiLaneModel(copy.deepcopy(visual), **common)
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(102)
        routed = MultiLaneModel(copy.deepcopy(visual), **common,
            parax_mode="post_task_staged", parax_layer_indices=(0,), parax_rank=2,
            parax_initialization="zero_output", parax_output_scale_mode="fixed",
            parax_smooth_ratio_bound=0.02)
        self.assertTrue(torch.equal(baseline_rng, torch.get_rng_state()))
        baseline.activate_task(0)
        routed.activate_task(0)
        routed.set_parax_runtime_enabled(False)
        images = {name: torch.randn(2, 3, 4, 4) for name in ("full", "person", "face")}
        images["face_reliable"] = torch.tensor([True, False])
        for model in (baseline, routed):
            optimizer = torch.optim.Adam(model.base_optimizer_parameters(), lr=0.01)
            optimizer.zero_grad()
            model.current_all_logits(images).square().mean().backward()
            optimizer.step()
        self.assertEqual(base_hash(baseline), base_hash(routed))
        torch.testing.assert_close(baseline.current_all_logits(images), routed.current_all_logits(images), atol=0, rtol=0)

    def test_rng_is_restored_after_extra_calibration(self):
        torch.manual_seed(71)
        random.seed(71)
        np.random.seed(71)
        expected_torch = torch.get_rng_state().clone()
        expected_python, expected_numpy = random.getstate(), np.random.get_state()
        with isolated_rng(torch.device("cpu")):
            torch.randn(8)
            random.random()
            np.random.randn(8)
        self.assertTrue(torch.equal(expected_torch, torch.get_rng_state()))
        self.assertEqual(expected_python, random.getstate())
        self.assertTrue(np.array_equal(expected_numpy[1], np.random.get_state()[1]))

    def test_uniform_control_has_no_trainable_router_or_center(self):
        model = self.make_model(static=True)
        model.activate_task(0)
        bank = model.parax_bank
        self.assertTrue(all(p.requires_grad for p in bank.task_projections[0].parameters()))
        self.assertFalse(any(p.requires_grad for p in bank.routers.parameters()))
        self.assertFalse(bank.expert_a.requires_grad)
        _, gate = bank(0, torch.randn(2, 1, model.output_dim))
        torch.testing.assert_close(gate, torch.full_like(gate, 1 / 3))


if __name__ == "__main__":
    unittest.main()
