import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from multi_lane.track_a.evaluate_joint26_checkpoint import build_model
from multi_lane.track_a.export_compact_test_scores import restore_compact_model_state
from multi_lane.track_a.runner import (add_view_auxiliary_loss, compact_model_state_dict,
                                      evaluate_view_diagnostics, using_training_protocol)
from multi_lane.track_a.view_fusion import TaskwiseViewFusion
from test_track_a_reproduction import FakeVisual


class FixedTwoViewTest(unittest.TestCase):
    def test_deleted_view_prior_is_renormalized_and_unreliable_face_is_zero(self):
        full = torch.tensor([[[1., 0.]], [[1., 0.]]], requires_grad=True)
        auxiliary = torch.tensor([[[0., 1.]], [[0., 1.]]], requires_grad=True)
        mask = torch.tensor([True, False])
        for mode, name in (("fixed_full_person", "person"), ("fixed_full_face", "face")):
            fusion = TaskwiseViewFusion(1, 2, mode)
            out, weights = fusion({"full": full, name: auxiliary}, (0,), mask)
            expected = torch.tensor([[[.8, .2]], [[.8, .2]]]) if name == "person" else torch.tensor([
                [[.64 / .84, .20 / .84]], [[1., 0.]]])
            torch.testing.assert_close(weights, expected)
            torch.testing.assert_close(out, expected)
            self.assertEqual(fusion.parameter_count_per_task(), 0)
            self.assertEqual(fusion.prior(False)[name], .2 if name == "person" else 0.)
            if name == "face":
                with self.assertRaises(ValueError):
                    fusion({"full": full, name: auxiliary}, (0,), None)

    def test_only_active_views_encode_receive_gradients_and_export_scores(self):
        visual = FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        initial_states = []
        for mode, active, absent in (("fixed_full_person", "person", "face"), ("fixed_full_face", "face", "person")):
            torch.manual_seed(84)
            config = dict(num_selectors=2, num_prompts=2, num_prompt_layers=1,
                          selector_mode="shared", prompt_mode="shared", normalize="pre-head",
                          view_fusion=mode, view_classifier_mode="shared_post_fusion", parax_mode="disabled")
            model = build_model(config, copy.deepcopy(visual))
            initial_states.append(compact_model_state_dict(model))
            model.activate_task(0)
            inputs = {name: torch.randn(2, 3, 4, 4, requires_grad=True) for name in ("full", "person", "face")}
            inputs["face_reliable"] = torch.tensor([True, False])
            targets = torch.ones(2, 26)
            with patch.object(model, "_encode_single_lanes", wraps=model._encode_single_lanes) as encode:
                logits, views = model.current_all_logits_with_views(inputs)
                self.assertEqual([c.kwargs["image_view"] for c in encode.call_args_list], ["full", active])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
            views[active].retain_grad()
            loss = add_view_auxiliary_loss(loss, views, inputs, targets, range(26), 1., "legacy_full_zero", .1, "bce")
            loss.backward()
            self.assertIsNone(inputs[absent].grad)
            self.assertGreater(views[active].grad.abs().sum().item(), 0)
            self.assertTrue((model.head.weight.grad.norm(dim=1) > 0).all())
            self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))
            payload = dict(schema_version=1, task_id=0, source_git={"commit": "source"}, model=compact_model_state_dict(model))
            restored = build_model(config, copy.deepcopy(visual))
            restore_compact_model_state(restored, payload, 0, {"commit": "source"})
            torch.testing.assert_close(restored.seen_logits(inputs), model.seen_logits(inputs))
            with tempfile.TemporaryDirectory() as directory, using_training_protocol("joint26"):
                scores = Path(directory) / "scores.npz"
                result = evaluate_view_diagnostics(restored, [(inputs, targets, ["a", "b"])], torch.device("cpu"), 0, .5, False, scores)
                self.assertEqual(set(result["metrics"]) - {"face_reliable"}, {"fused", "full", active})
                with np.load(scores) as arrays:
                    self.assertIn(active + "_logits", arrays)
                    self.assertNotIn(absent + "_logits", arrays)
        self.assertEqual(set(initial_states[0]), set(initial_states[1]))
        for name in initial_states[0]:
            torch.testing.assert_close(initial_states[0][name], initial_states[1][name])
