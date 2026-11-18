"""A true task tap, detached fusion supervision, identical native logits and zero-prior gates."""

import copy
import unittest
from unittest.mock import patch

import torch

from multi_lane.track_a.d9_relation_fusion import D9RelationFusion, SampleFusionGate, sample_readouts
from test_d9_protected_multi_model import branches


class SampleFusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        native = branches()
        before = torch.get_rng_state()
        self.model = D9RelationFusion(native, 'sample_holdout')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.model.activate_task(0)
        self.inputs = {key: torch.randn(4, 3, 4, 4) for key in
                       ('full', 'person', 'face', 'evidence_full', 'evidence_person', 'evidence_face')}
        self.inputs['evidence_face_reliable'] = torch.tensor([True, False, True, False])

    def test_available_equal_start_fixed_replay_and_protected_gradient(self):
        native = D9RelationFusion(copy.deepcopy(dict(self.model.branches.items())), 'fixed')
        self.model.activate_task(1)
        native.activate_task(1)
        self.assertTrue(all(not p.requires_grad for p in self.model.gates[0].parameters()))
        reference = native(self.inputs, all_seen=True)
        out = self.model(self.inputs, all_seen=True)
        self.assertEqual(tuple(out['readouts']), sample_readouts('sample_holdout'))
        for view in native.branches:
            torch.testing.assert_close(out['branches'][view], reference['branches'][view], atol=0, rtol=0)
        torch.testing.assert_close(out['fused'], reference['fused'], atol=0, rtol=0)
        expected = torch.tensor([[1/3, 1/3, 1/3], [.5, .5, 0.]]).repeat(2, 1)
        for name in ('equal', 'mid_bce', 'mid_asl', 'late_bce'):
            torch.testing.assert_close(out['readouts'][name]['weights'], expected[:, None].expand(-1, 2, -1))
        for name in ('mid_bce', 'mid_asl', 'late_bce'):
            self.model.zero_grad(set_to_none=True)
            with torch.no_grad():
                self.model.gates[1][name].linear.normal_()
            readout = self.model(self.inputs)['readouts'][name]
            readout['fused'].square().sum().backward()
            self.assertTrue(self.model.gates[1][name].linear.grad.abs().sum() > 0)
            self.assertTrue(all(p.grad is None for p in self.model.branches.parameters()))
            self.assertTrue(all(p.grad is None for key in self.model.gates[1] if key != name
                                for p in self.model.gates[1][key].parameters()))
            self.assertFalse(torch.equal(readout['weights'][0], readout['weights'][2]))

    def test_task10_selector_pool_reads_matching_frozen9_query_positions(self):
        person = self.model.branches['person']
        captured = {}
        original = person._lane_block

        def capture_weights(block, stream, state, lanes, layer, **kwargs):
            if layer == 9:
                # In the native D9 recipe the image-token adapter acts at block 2.
                q = block.ln_1(state)[:, :, 1:]
                k = block.ln_1(stream)
                read = torch.einsum('tbsc,bnc->tbsn', q, k).mul(person.width ** -.5).softmax(-1)
                spatial = read[0, :, :, 1:].mean(1)
                captured['spatial'] = spatial / spatial.sum(-1, keepdim=True)
            return original(block, stream, state, lanes, layer, **kwargs)

        def capture_patch(module, inputs, output):
            captured['patches'] = output[1]

        def capture_descriptor(module, inputs):
            captured['descriptor'] = inputs[1]['person']

        hook_patch = person.cross_agent(0, 8).register_forward_hook(capture_patch)
        hook_gate = self.model.gates[0]['mid_bce'].register_forward_pre_hook(capture_descriptor)
        try:
            with patch.object(person, '_lane_block', side_effect=capture_weights):
                self.model(self.inputs)
        finally:
            hook_patch.remove()
            hook_gate.remove()
        expected = (captured['spatial'].unsqueeze(-1) * captured['patches']).sum(1)
        torch.testing.assert_close(captured['descriptor'], expected)
        self.assertFalse(torch.allclose(expected, captured['patches'].mean(1)))

    def test_face_is_masked_before_its_features_can_influence_any_score(self):
        valid = torch.tensor([True, False])
        for dims in (None, 6):
            gate = SampleFusionGate(dims)
            with torch.no_grad():
                gate.linear.normal_()
            features = {v: torch.randn(2, dims or 4) for v in ('full', 'person', 'face')}
            original = gate(valid, features)
            features['face'][1] = float('nan')
            torch.testing.assert_close(gate(valid, features), original, rtol=0, atol=0)
            self.assertEqual(original[1, 2].item(), 0.)
        self.assertNotIn('mid_asl', sample_readouts('sample_full'))


if __name__ == '__main__':
    unittest.main()
