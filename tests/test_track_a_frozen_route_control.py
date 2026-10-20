import copy
import unittest
from unittest.mock import patch

import torch

from multi_lane.track_a.expert_rollback import build
from multi_lane.track_a.late_route_control import GateCalibration, fixed_routing, validate_route_config
from multi_lane.track_a.runner import tensor_state_digest
from test_track_a_reproduction import FakeVisual


class FrozenRouteControlTest(unittest.TestCase):
    def visual(self):
        visual = FakeVisual()
        visual.transformer.resblocks.extend([
            copy.deepcopy(visual.transformer.resblocks[0]) for _ in range(2)
        ])
        return visual

    def config(self):
        return dict(num_selectors=2, num_prompts=2, num_prompt_layers=1,
            selector_mode='shared', prompt_mode='shared', normalize='pre-head',
            adapter_mode='moe_frozen_ffn_activate_freeze', adapter_bottleneck_dim=4,
            adapter_layer_indices=[1], adapter_residual_scale=.1, moe_num_experts=22,
            moe_top_k=3, moe_freeze_count=3, moe_freeze_metric='gate_mass',
            moe_dropout=0, moe_noisy_gating=False, view_fusion='fixed_three_view',
            view_classifier_mode='shared_post_fusion', parax_mode='disabled')

    def images(self):
        return {**{v: torch.randn(3, 3, 4, 4) for v in ('full', 'person', 'face')},
                'face_reliable': torch.tensor([True, False, True])}

    def test_locked_config_accepts_both_controls_and_rejects_unmatched_sources(self):
        config = self.config()
        validate_route_config(config)
        late = dict(config, adapter_mode='moe_task_ffn_activate_freeze',
                    adapter_layer_indices=[11], moe_top_k=2, moe_freeze_count=2)
        validate_route_config(late)
        for changed in (dict(adapter_layer_indices=[5]), dict(moe_top_k=2),
                        dict(moe_freeze_count=2), dict(adapter_mode='image_token'),
                        dict(moe_freeze_metric='hard_count'), dict(parax_mode='post_task_router')):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_route_config(dict(config, **changed))

    def test_builder_restores_three_frozen_expert_slots_without_schema_drift(self):
        config = self.config(); visual = self.visual()
        first = build(config, copy.deepcopy(visual)); first.activate_task(0)
        images = self.images()
        first.train()
        # Build a finalized synthetic training state; the bank deliberately does
        # not count no-grad diagnostic forwards as training activations.
        first.current_all_logits_with_views(images)
        first.adapter_bank.finish_task(0); first.eval()
        state = first.state_dict()
        self.assertEqual(state['adapter_bank.freeze_topk'].shape, (8, 1, 3))
        second = build(config, copy.deepcopy(visual))
        second.load_state_dict(state, strict=True); second.restore_task(0); second.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(first.seen_logits(images), second.seen_logits(images)))

    def test_full_frozen_path_fixed_top3_and_off_leave_all_weights_unchanged(self):
        torch.manual_seed(40)
        model = build(self.config(), self.visual()); model.activate_task(0)
        with torch.no_grad():
            for expert in model.adapter_bank.experts[0]:
                expert.up.weight.normal_(0, .1)
        model.requires_grad_(False).eval(); images = self.images()
        before = tensor_state_digest(model.state_dict()); capture = GateCalibration(model.adapter_bank)
        with torch.no_grad(), capture.capture(): dynamic = model.seen_logits(images)
        self.assertEqual(capture.samples[0, 0].tolist(), [3, 3, 2])
        table, rows = capture.table()
        self.assertTrue(torch.equal((table > 0).sum(-1), torch.full((1, 1, 3), 3)))
        self.assertEqual(len(rows['0']['1']['full']['selected']), 3)
        # Change only the routing function, on the actual Frozen Forward path.
        table.zero_(); table[..., :3] = torch.tensor([.2, .3, .5])
        with torch.no_grad(), fixed_routing(model.adapter_bank, table):
            fixed = model.seen_logits(images)
        self.assertFalse(torch.allclose(dynamic, fixed, atol=1e-8, rtol=0))
        model.set_adapter_runtime_enabled(False)
        with torch.no_grad(), patch.object(model.adapter_bank, 'delta_for_layer',
                                           wraps=model.adapter_bank.delta_for_layer) as calls:
            off = model.seen_logits(images)
        self.assertEqual(calls.call_count, 0)
        self.assertFalse(torch.allclose(dynamic, off, atol=1e-8, rtol=0))
        model.set_adapter_runtime_enabled(True)
        self.assertNotIn('_route', model.adapter_bank.__dict__)
        self.assertEqual(before, tensor_state_digest(model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__': unittest.main()
