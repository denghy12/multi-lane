"""Direction readouts preserve branch learning and compose one learned center."""
import copy
import unittest

import torch

from multi_lane.track_a.d9_relation_fusion import (
    D9RelationFusion, DIRECTION_READOUTS, LinearDirectionGate, view_weights,
)
from test_d9_protected_multi_model import branches


class DirectionModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.branches = branches()
        state = torch.get_rng_state()
        self.model = D9RelationFusion(self.branches, 'direction')
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.model.activate_task(0)
        self.inputs = {key: torch.randn(4, 3, 4, 4) for key in
                       ('full', 'person', 'face', 'evidence_full', 'evidence_person', 'evidence_face')}
        self.inputs['evidence_face_reliable'] = torch.tensor([True, False, True, False])

    def test_initial_readouts_and_composed_center(self):
        fixed = D9RelationFusion(copy.deepcopy(self.branches), 'fixed')
        output = self.model(self.inputs)
        reference = fixed(self.inputs)
        self.assertEqual(tuple(output['readouts']), DIRECTION_READOUTS)
        for name in ('fixed', 'a', 'b', 'c'):
            torch.testing.assert_close(output['readouts'][name]['fused'], reference['fused'], rtol=0, atol=0)
        expected = torch.tensor([[1/3, 1/3, 1/3], [.5, .5, 0.], [1/3, 1/3, 1/3], [.5, .5, 0.]])
        torch.testing.assert_close(output['readouts']['center']['weights'][:, 0], expected)
        with torch.no_grad():
            self.model.gates[0]['center'].offset.copy_(torch.tensor([.7, -.3]))
        output = self.model(self.inputs)
        for key in ('weights', 'fused'):
            torch.testing.assert_close(output['readouts']['center_b'][key],
                                       output['readouts']['center'][key], rtol=0, atol=0)
        # The learned-center direction probe owns only its eight linear weights.
        self.assertEqual(sum(p.numel() for p in self.model.gates[0]['b'].parameters()), 8)
        self.assertEqual(sum(p.numel() for p in self.model.gates[0]['center_b'].parameters()), 8)
        self.assertIsNot(self.model.gates[0]['b'], self.model.gates[0]['center_b'])

    def test_learned_losses_cannot_change_branches_or_sibling_gates(self):
        with torch.no_grad():
            for name in ('a', 'b', 'c', 'center_b'):
                for scorer in self.model.gates[0][name].scorers.values():
                    scorer.weight.fill_(.2)
        for name in ('center', 'a', 'b', 'c', 'center_b'):
            self.model.zero_grad(set_to_none=True)
            self.model(self.inputs)['readouts'][name]['fused'].sum().backward()
            self.assertFalse(any(p.grad is not None for p in self.model.branches.parameters()))
            for gate_name in ('center', 'a', 'b', 'c', 'center_b'):
                parameters = list(self.model.gates[0][gate_name].parameters())
                if gate_name == name:
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters))
                else:
                    self.assertTrue(all(p.grad is None for p in parameters))

    def test_all_task_output_and_fixed_branch_gradients(self):
        self.model.activate_task(1)
        self.assertTrue(all(not p.requires_grad for p in self.model.gates[0].parameters()))
        fixed = D9RelationFusion(copy.deepcopy(self.branches), 'fixed')
        output = self.model(self.inputs, all_seen=True)
        for name in ('a', 'b', 'c', 'center_b'):
            self.assertEqual(output['readouts'][name]['direction_scores'].shape, (4, 2, 2))
        current = self.model(self.inputs)
        original = fixed(self.inputs)
        for name in DIRECTION_READOUTS:
            torch.testing.assert_close(output['readouts'][name]['fused'][:, -1:],
                                       current['readouts'][name]['fused'], atol=1e-5, rtol=1e-5)
        for view in self.branches:
            torch.testing.assert_close(current['all_columns'][view], original['all_columns'][view], rtol=0, atol=0)
        current['fused'].sum().backward()
        original['fused'].sum().backward()
        old = dict(fixed.branches.named_parameters())
        for name, param in self.model.branches.named_parameters():
            if param.grad is not None or old[name].grad is not None:
                torch.testing.assert_close(param.grad, old[name].grad, rtol=0, atol=0)

    def test_raw_scores_detach_features_and_bound_only_fusion_offset(self):
        gate = LinearDirectionGate()
        with torch.no_grad():
            for scorer in gate.scorers.values():
                scorer.weight.fill_(2.)
        person = torch.ones(2, 4, requires_grad=True)
        face = torch.tensor([[1., 1., 1., 1.], [float('nan')] * 4], requires_grad=True)
        reliable = torch.tensor([True, False])
        weights, raw = gate(reliable, person, face)
        torch.testing.assert_close(raw, torch.tensor([[8., 8.], [8., 0.]]))
        torch.testing.assert_close(weights, view_weights(gate.prior + raw.tanh(), reliable))
        self.assertEqual(weights[1, 2].item(), 0.)
        self.assertTrue(torch.isfinite(weights).all())
        raw.sum().backward()
        self.assertIsNone(person.grad)
        self.assertIsNone(face.grad)
        self.assertTrue(all(p.grad is not None for p in gate.parameters()))


if __name__ == '__main__':
    unittest.main()
