from __future__ import annotations

import copy
import unittest

import torch

from multi_lane.track_a import runner
from multi_lane.track_a.model import MultiLaneModel
from test_track_a_reproduction import FakeVisual


class Source:
    targets = [[0], [25], []]
    sample_ids = ["a", "b", "c"]

    def __len__(self):
        return 3

    def __getitem__(self, index):
        target = torch.zeros(26)
        target[self.targets[index]] = 1
        return torch.zeros(3, 4, 4), target


class Joint26Test(unittest.TestCase):
    def test_protocol_restores_legacy_after_exception(self):
        original = runner.TASK_SIZES
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with runner.using_training_protocol("joint26"):
                self.assertEqual(runner.task_indices(0), tuple(range(26)))
                self.assertEqual(runner.seen_indices(0), tuple(range(26)))
                raise RuntimeError("stop")
        self.assertEqual(runner.TASK_SIZES, original)
        self.assertEqual(runner.task_indices(0), tuple(range(5)))
        required = ["--seed", "0", "--data-root", ".", "--clip-checkpoint", "clip.pt", "--output-root", "output"]
        self.assertEqual(runner.parse_args(required).max_tasks, 8)
        self.assertEqual(runner.parse_args(required + ["--training-protocol", "joint26"]).max_tasks, 1)

    def test_all_instances_and_all_labels_are_used(self):
        with runner.using_training_protocol("joint26"):
            view = runner.dataset_view(Source(), runner.task_indices(0))
            self.assertEqual(view.indices, (0, 1, 2))
            self.assertEqual(tuple(view[1][1].shape), (26,))
            self.assertEqual(float(view[1][1][25]), 1)
            self.assertEqual(runner.fit_calibration_indices(Source(), range(26), 0), ([0, 1, 2], []))
        self.assertEqual(runner.dataset_view(Source(), range(5)).indices, (0,))

    def test_joint_summary_has_no_incremental_forgetting(self):
        with runner.using_training_protocol("joint26"):
            row = runner.compute_metrics(0, torch.rand(3, 26), torch.ones(3, 26), 0.5)
            summary = runner.summarize_tasks([row])
            self.assertIsNone(summary["forgetting"])
            self.assertIsNone(summary["average_mAP"])
            with self.assertRaises(ValueError):
                runner.summarize_tasks([row, row])

    def test_four_arms_match_base_init_and_train_all_classifier_rows(self):
        torch.manual_seed(40)
        visual = FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        images = torch.randn(2, 3, 4, 4)
        hashes = []
        for three, parax in ((False, False), (True, False), (False, True), (True, True)):
            torch.manual_seed(41)
            model = MultiLaneModel(
                copy.deepcopy(visual), (26,), num_selectors=2, num_prompts=2,
                num_prompt_layers=1, view_fusion="fixed_three_view" if three else "disabled",
                parax_mode="image" if parax else "disabled", parax_layer_indices=(0,),
                parax_rank=2, parax_num_experts=3, parax_router_hidden=2,
                parax_initialization="official", parax_residual_scale=0.1,
            )
            hashes.append(runner.tensor_state_digest({name: value for name, value in model.state_dict().items()
                          if name == "selectors" or name.startswith(("prompts.", "head."))}))
            model.activate_task(0)
            inputs = {"full": images, "person": images.flip(-1), "face": images.flip(-2),
                      "face_reliable": torch.tensor([True, False])} if three else images
            logits = model.current_all_logits(inputs)
            self.assertEqual(tuple(logits.shape), (2, 26))
            targets = torch.ones(2, 26)
            loss = runner.compute_training_loss(logits, targets, range(26), 1, "legacy_full_zero")
            torch.testing.assert_close(loss, torch.nn.functional.binary_cross_entropy_with_logits(logits, targets))
            loss.backward()
            self.assertTrue(bool((model.head.weight.grad.norm(dim=1) > 0).all()))
            self.assertTrue(all(parameter.grad is None for parameter in model.visual_encoder.parameters()))
            if parax:
                gradients = model.parax_gradient_diagnostics()
                self.assertEqual(gradients["parax_grad_finite"], 1)
                self.assertGreater(gradients["parax_grad_total_norm"], 0)
            model.assert_visual_frozen()
        self.assertEqual(len(set(hashes)), 1)


if __name__ == "__main__":
    unittest.main()
