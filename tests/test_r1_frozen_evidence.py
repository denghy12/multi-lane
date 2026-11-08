import copy
import unittest

import torch

from multi_lane.track_a.r1_crossvit import compact
from multi_lane.track_a.r1_frozen_cross_view import FrozenR1Receiver
from multi_lane.track_a.runner import TASK_SIZES
from test_track_a_reproduction import FakeVisual


class FrozenEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make(self, view="person", evidence="all"):
        torch.manual_seed(123)
        visual = FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        model = FrozenR1Receiver(
            visual, TASK_SIZES, num_selectors=3, num_prompts=2, num_prompt_layers=1,
            adapter_mode="image_token", adapter_bottleneck_dim=4, adapter_layer_indices=(0,),
            r1_view=view, r1_layer=0, r1_dim=8, r1_heads=2,
            frozen_cross_mode="patch", frozen_cross_evidence=evidence,
        )
        model.activate_task(0)
        images = torch.randn(4, 3, 4, 4)
        auxiliary = "face" if view == "person" else "person"
        batch = {
            "full": images, "evidence_full": images * .7,
            "evidence_" + auxiliary: images * .4,
            "evidence_face_reliable": torch.ones(4, dtype=torch.bool),
        }
        return model, batch, auxiliary

    def test_mask_choice_preserves_initial_parameters_and_rng(self):
        base, images, _ = self.make()
        rng = torch.get_rng_state().clone()
        control, other, _ = self.make(evidence="full")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(compact(base)), set(compact(control)))
        for name, tensor in compact(base).items():
            self.assertTrue(torch.equal(tensor, compact(control)[name]), name)
        torch.testing.assert_close(
            base.current_all_logits(images), control.current_all_logits(other), rtol=0, atol=0,
        )

    def test_full_only_blocks_auxiliary_evidence_but_still_reads_full(self):
        for view in ("person", "face"):
            with self.subTest(view=view):
                model, images, auxiliary = self.make(view, "full")
                with torch.no_grad():
                    model.r1_bank[0].out.weight.normal_(std=.2)
                expected = model.current_all_logits(images)
                swapped = dict(images)
                swapped["evidence_" + auxiliary] = torch.randn_like(images["full"]) * 10
                torch.testing.assert_close(
                    expected, model.current_all_logits(swapped), rtol=0, atol=0,
                )
                changed_full = dict(images, evidence_full=torch.randn_like(images["full"]))
                self.assertFalse(torch.equal(expected, model.current_all_logits(changed_full)))
                model.frozen_cross_evidence = "all"
                self.assertFalse(torch.equal(
                    model.current_all_logits(images), model.current_all_logits(swapped),
                ))

    def test_masked_source_type_gets_zero_gradient_with_the_same_two_slots(self):
        model, images, _ = self.make(evidence="full")
        with torch.no_grad():
            model.r1_bank[0].out.weight.normal_(std=.1)
        model.current_all_logits(images).square().mean().backward()
        gradient = model.r1_bank[0].types.grad
        self.assertEqual(tuple(gradient.shape), (2, model.width))
        self.assertGreater(float(gradient[0].norm()), 0)
        self.assertEqual(float(gradient[1].norm()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.visual_encoder.parameters()))

    def test_unknown_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make(evidence="unknown")
