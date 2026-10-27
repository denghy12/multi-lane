"""End-to-end objective routing, original trajectory and epoch recovery."""
import copy
import unittest

import torch

from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import TASK_SIZES, train_task, tensor_state_digest
from multi_lane.track_a.teacher_free_shared_learning import (
    complementary_options, optimizer_groups, configure_current, task_slices)
from multi_lane.track_a.complementary_ranking import _view_gradient_fused_logits
from multi_lane.track_a.complementary_update import adam_candidate_parameters
from multi_lane.track_a.complementary_training_probe import measure_candidate_effect
from multi_lane.track_a.task_recovery import rng_state, restore_rng
from test_track_a_reproduction import FakeVisual


class TeacherFreeRunnerTests(unittest.TestCase):
    def pair(self):
        torch.manual_seed(207)
        model = MultiLaneModel(FakeVisual(), TASK_SIZES, num_selectors=2,
            num_prompts=2, num_prompt_layers=1, view_fusion="fixed_three_view",
            adapter_mode="image_token", adapter_bottleneck_dim=4,
            adapter_layer_indices=(0,), adapter_residual_scale=.03)
        model.activate_task(0)
        configure_current(model, 0)
        x = {v: torch.randn(4, 3, 4, 4) for v in ("full", "person", "face")}
        x["face_reliable"] = torch.tensor([True, False, True, True])
        y = torch.tensor([[1.,0.,1.,0.,0.],[0.,1.,0.,0.,1.],
                          [0.,0.,1.,1.,0.],[1.,1.,0.,0.,0.]])
        return model, (x, y)

    def train(self, model, batch, **options):
        cfg = dict(epochs=1, learning_rate=.001, weight_decay=0.,
            temperature=1., amp=False, adapter_learning_rate=.0004,
            loss_routing="adapter_asl", view_auxiliary_loss_weight=.1,
            optimizer_group_builder=optimizer_groups)
        cfg.update(options)
        return train_task(model, [batch], None, torch.device("cpu"), model.current_task, **cfg)

    def test_ramp_zero_replays_entire_original_update_and_rng(self):
        base, batch = self.pair()
        original = self.train(base, batch)
        base_state = tensor_state_digest(base.state_dict())
        rng = torch.get_rng_state()
        candidate, same_batch = self.pair()
        actual = self.train(candidate, same_batch,
            complementary_learning=complementary_options("U4"))
        self.assertEqual(base_state, tensor_state_digest(candidate.state_dict()))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(original[0]["current_loss"], actual[0]["current_loss"])
        self.assertEqual(actual[0]["complementary_ramp"], 0.)

    def test_live_extra_loss_keeps_head_base_update_and_old_coordinates(self):
        base, batch = self.pair()
        self.train(base, batch, optimizer_updates_per_task=1)
        baseline_head = {k:v.clone() for k,v in base.head.state_dict().items()}
        model, batch = self.pair()
        visual = tensor_state_digest(model.visual_encoder.state_dict())
        before_future = tensor_state_digest(task_slices(model.state_dict(), 0,
                                                        model._task_sizes, current=False))
        rows = self.train(model, batch, optimizer_updates_per_task=1,
            complementary_learning=complementary_options("U4", True))
        self.assertGreater(rows[0]["complementary_loss"], 0.)
        self.assertEqual(rows[0]["complementary_active_batches"], 1.)
        for k, v in baseline_head.items():
            self.assertTrue(torch.equal(v, model.head.state_dict()[k]))
        self.assertEqual(visual, tensor_state_digest(model.visual_encoder.state_dict()))
        self.assertEqual(before_future, tensor_state_digest(task_slices(model.state_dict(), 0,
                                                                        model._task_sizes, current=False)))
        self.assertEqual(int(model.selectors.grad[1:].count_nonzero()), 0)
        for p in model.prompts:
            self.assertEqual(int(p.grad[:, 1:].count_nonzero()), 0)
        self.assertEqual(int(model.head.weight.grad[5:].count_nonzero()), 0)

    def test_extra_graph_replays_main_fused_prediction_exactly(self):
        model, (x, _) = self.pair()
        logits, _, features = model.current_all_logits_with_view_features(x)
        for name in ("person", "face"):
            extra = _view_gradient_fused_logits(
                features, model.head, model.last_fusion_weights(), name)
            self.assertTrue(torch.equal(extra, logits))

    def test_epoch_restore_replays_parameters_optimizer_and_losses(self):
        model, batch = self.pair()
        snapshots = {}
        def save(epoch, history, state):
            snapshots[epoch] = (copy.deepcopy(model.state_dict()),
                                copy.deepcopy(state), rng_state())
        expected = self.train(model, batch, epochs=3, epoch_callback=save)
        expected_model = tensor_state_digest(model.state_dict())
        for boundary in (1, 3):
            recovered, same_batch = self.pair()
            weights, state, rng = snapshots[boundary]
            recovered.load_state_dict(weights)
            restore_rng(rng)
            final = {}
            rows = self.train(recovered, same_batch, epochs=3,
                resume_training_state=state, recovery_state=final)
            self.assertEqual(expected_model, tensor_state_digest(recovered.state_dict()))
            self.assertEqual([r["current_loss"] for r in expected],
                             [r["current_loss"] for r in rows])
            for old_group,new_group in zip(snapshots[3][1]["optimizer"]["param_groups"],
                                          final["optimizer"]["param_groups"]):
                self.assertEqual(old_group,new_group)

    def test_live_old_task_bank_is_unchanged_after_new_task_update(self):
        model, (x, y) = self.pair()
        self.train(model, (x, y), optimizer_updates_per_task=1)
        model.activate_task(1)
        configure_current(model, 1)
        protected = tensor_state_digest(task_slices(model.state_dict(), 1, model._task_sizes))
        self.train(model, (x, y[:, :3]), optimizer_updates_per_task=1,
            complementary_learning=complementary_options("U4", True))
        self.assertEqual(protected, tensor_state_digest(
            task_slices(model.state_dict(), 1, model._task_sizes)))

    def test_counterfactual_probe_restores_params_rng_and_optimizer(self):
        model, (x, y) = self.pair()
        logits, _, _ = model.current_all_logits_with_view_features(x)
        optimizer = torch.optim.Adam(model.optimizer_parameters(), lr=.001)
        logits.square().sum().backward()
        baseline = adam_candidate_parameters(optimizer=optimizer)
        before = tensor_state_digest(model.state_dict())
        rng = rng_state()
        flags = [m.training for m in model.modules()]
        metrics = measure_candidate_effect(model=model, optimizer=optimizer,
            baseline_parameters=baseline, images=x, targets=y, current=range(5),
            temperature=1., loss_mode="legacy_full_zero", amp=False)
        self.assertEqual(before,tensor_state_digest(model.state_dict()))
        self.assertEqual(flags,[m.training for m in model.modules()])
        self.assertTrue(torch.equal(rng["torch_cpu"],torch.get_rng_state()))
        self.assertEqual(len(optimizer.state),0)
        self.assertIn("full_bce_extra",metrics)
        self.assertIn("fused_bce_extra",metrics)


if __name__ == "__main__":
    unittest.main()

