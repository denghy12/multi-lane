import copy
import unittest

import torch

from multi_lane.track_a.evaluate_joint26_checkpoint import build_model
from multi_lane.track_a.export_compact_test_scores import restore_compact_model_state
from multi_lane.track_a.runner import compact_model_state_dict
from test_track_a_reproduction import FakeVisual


class JointCheckpointTest(unittest.TestCase):
    def test_compact_round_trip_full_three_view_with_without_parax(self):
        visual = FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        images = torch.randn(2, 3, 4, 4)
        for three in (False, True):
            for parax in (False, True):
                config = dict(num_selectors=2, num_prompts=2, num_prompt_layers=1,
                              selector_mode="shared", prompt_mode="shared", normalize="pre-head",
                              view_fusion="fixed_three_view" if three else "disabled",
                              view_classifier_mode="shared_post_fusion", parax_mode="image" if parax else "disabled",
                              parax_layer_indices=[0], parax_rank=2, parax_num_experts=3, parax_router_hidden=2)
                model = build_model(config, copy.deepcopy(visual))
                model.activate_task(0)
                # Non-default learned values must survive restoration.
                with torch.no_grad():
                    model.head.weight.add_(0.1)
                    if parax:
                        model.parax_bank.expert_a.add_(0.2)
                inputs = dict(full=images, person=images.flip(-1), face=images.flip(-2),
                              face_reliable=torch.tensor([True, False])) if three else images
                reference = model.seen_logits(inputs).detach()
                payload = dict(schema_version=1, task_id=0, source_git={"commit": "source"},
                               model=compact_model_state_dict(model))
                restored = build_model(config, copy.deepcopy(visual))
                restore_compact_model_state(restored, payload, 0, {"commit": "source"})
                restored.requires_grad_(False).eval()
                with torch.no_grad():
                    torch.testing.assert_close(restored.seen_logits(inputs), reference)
                self.assertEqual(restored.num_tasks, 1)
                self.assertEqual(restored.num_classes, 26)
                self.assertFalse(any(p.requires_grad for p in restored.parameters()))
                broken = copy.deepcopy(payload)
                del broken["model"]["head.bias"]
                with self.assertRaisesRegex(ValueError, "missing method state"):
                    restore_compact_model_state(restored, broken, 0, {"commit": "source"})
