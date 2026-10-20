import copy
import unittest
from unittest.mock import patch

import torch

from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import build_optimizer_groups, compact_model_state_dict, tensor_state_digest
from multi_lane.track_a.export_compact_test_scores import restore_compact_model_state
from test_track_a_reproduction import FakeVisual


class JointSummaryTest(unittest.TestCase):
    def model(self, mode='fixed'):
        torch.manual_seed(71)
        visual = FakeVisual()
        visual.transformer.resblocks.extend([copy.deepcopy(visual.transformer.resblocks[0])])
        return MultiLaneModel(visual, (2, 1), num_selectors=2, num_prompts=2,
                              num_prompt_layers=1, view_fusion='fixed_three_view',
                              joint_summary_mode=mode, joint_summary_bottleneck=3)

    def images(self):
        return {**{v: torch.randn(3, 3, 4, 4) for v in ('full', 'person', 'face')},
                'face_reliable': torch.tensor([True, False, True])}

    def test_full_only_reduces_to_unmodified_single_lane(self):
        for mode in ('fixed', 'dense', 'moe'):
            model = self.model(mode); model.activate_task(0); model.eval()
            images = self.images()
            images['joint_view_valid'] = torch.tensor([[True, False, False]] * 3)
            with torch.no_grad():
                joint = model.encode_lanes(images, False)
                original = model._encode_single_lanes(images['full'], False)
            self.assertTrue(torch.allclose(joint, original, atol=1e-6), mode)

    def test_one_attention_path_reads_three_summaries_at_each_depth(self):
        model = self.model('moe'); model.activate_task(0); model.activate_task(1)
        model.eval(); calls = []; images = self.images()
        original = model._task_attention_ffn
        def record(block, lanes, lane_ids, layer, cls, selectors, selected, *args):
            calls.append((layer, tuple(lane_ids), tuple(lanes.shape), tuple(selected.shape)))
            return original(block, lanes, lane_ids, layer, cls, selectors, selected, *args)
        frozen_calls = []
        def frozen_record(module, inputs):
            frozen_calls.append(torch.is_grad_enabled())
        handle = model.visual_encoder.transformer.resblocks[0].register_forward_pre_hook(frozen_record)
        with patch.object(model, '_task_attention_ffn', side_effect=record):
            logits, views = model.seen_logits_with_views(images)
        handle.remove()
        self.assertEqual(logits.shape, (3, 3)); self.assertEqual(views, {})
        self.assertEqual(calls, [(i, (0, 1), (2, 3, 3, 8), (2, 3, 2, 8)) for i in (0, 1)])
        self.assertEqual(frozen_calls, [False] * 3)

    def test_face_mask_excludes_evidence_from_router_experts_and_output(self):
        model = self.model('moe'); model.activate_task(0); model.eval()
        with torch.no_grad():
            for module in model.joint_summary_bank.tasks[0]:
                for expert in module.experts:
                    expert.up.weight.normal_(0, .1)
        images = self.images(); images['face_reliable'].fill_(False)
        before = model.seen_logits(images)
        images['face'] = torch.randn_like(images['face']) * 100
        self.assertTrue(torch.equal(before, model.seen_logits(images)))
        images['face_reliable'].fill_(True)
        self.assertFalse(torch.allclose(before, model.seen_logits(images)))

    def test_gradients_reach_joint_experts_and_base_but_never_backbone(self):
        for mode in ('dense', 'moe'):
            model = self.model(mode); model.activate_task(0)
            before = tensor_state_digest(model.visual_encoder.state_dict())
            optimizer = torch.optim.Adam(build_optimizer_groups(model, 0)[2], lr=.01)
            for _ in range(2):
                optimizer.zero_grad()
                model.current_all_logits(self.images()).square().sum().backward()
                optimizer.step()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in model.joint_summary_bank.parameters()))
            if mode == 'moe':
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in model.joint_summary_bank.tasks[0][0].router.parameters()))
            self.assertGreater(model.selectors.grad.abs().sum(), 0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.visual_encoder.parameters()))
            self.assertEqual(before, tensor_state_digest(model.visual_encoder.state_dict()))

    def test_old_task_survives_new_task_update_and_compact_restore(self):
        model = self.model('moe'); model.activate_task(0); model.eval(); images = self.images()
        with torch.no_grad(): old = model.seen_logits(images).clone()
        model.activate_task(1)
        old_bank = tensor_state_digest(model.joint_summary_bank.tasks[0].state_dict())
        groups = build_optimizer_groups(model, 0)[2]
        optimizer = torch.optim.Adam(groups, lr=.01)
        for _ in range(2):
            optimizer.zero_grad(); model.current_logits(images).square().sum().backward()
            optimizer.step()
        self.assertEqual(old_bank, tensor_state_digest(model.joint_summary_bank.tasks[0].state_dict()))
        self.assertFalse(any(p.requires_grad for p in model.joint_summary_bank.tasks[0].parameters()))
        model.eval()
        with torch.no_grad(): full = model.seen_logits(images)
        self.assertTrue(torch.allclose(old, full[:, :2], atol=1e-6))
        restored = self.model('moe')
        payload = dict(schema_version=1, task_id=1, source_git={'commit': 'unit'}, model=compact_model_state_dict(model))
        restore_compact_model_state(restored, payload, 1, {'commit': 'unit'}); restored.eval()
        with torch.no_grad(): actual = restored.seen_logits(images)
        self.assertTrue(torch.equal(full, actual))

    def test_base_rng_initialization_and_optimizer_membership(self):
        original = self.model('disabled'); state = torch.get_rng_state()
        base = tensor_state_digest({n: v for n, v in original.state_dict().items()
                                   if n == 'selectors' or n.startswith(('prompts.', 'head.'))})
        for mode in ('fixed', 'dense', 'moe'):
            model = self.model(mode)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))
            self.assertEqual(base, tensor_state_digest({n: v for n, v in model.state_dict().items()
                             if n == 'selectors' or n.startswith(('prompts.', 'head.'))}))
            model.activate_task(0)
            actual = {id(p) for g in build_optimizer_groups(model, 0)[2] for p in g['params']}
            named = dict(model.named_parameters())
            self.assertEqual(actual, {id(named[n]) for n in model.optimizer_parameter_names()})


if __name__ == '__main__':
    unittest.main()
