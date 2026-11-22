"""Fill-table readouts must leave both the trained branches and old heads alone."""
import copy
import unittest
import torch
from multi_lane.track_a.d9_relation_fusion import D9RelationFusion, FinalSimilarityGate, FILL_GATES
from scripts.emotic.d9_fill_online import held_batches
from test_d9_protected_multi_model import branches


class FillTableTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        native = branches()
        self.reference = D9RelationFusion(native, 'sample_holdout')
        state = torch.get_rng_state()
        self.model = D9RelationFusion(copy.deepcopy(dict(native.items())), 'sample_holdout', FILL_GATES)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        for model in (self.reference, self.model):
            model.activate_task(0)
        self.inputs = {key: torch.randn(4, 3, 4, 4) for key in
                       ('full', 'person', 'face', 'evidence_full', 'evidence_person', 'evidence_face')}
        self.inputs['evidence_face_reliable'] = torch.tensor([True, False, True, False])

    def test_old_outputs_exact_and_auxiliary_gradients_do_not_reach_branches(self):
        reference = self.reference(self.inputs, sample_gate_only=True)
        output = self.model(self.inputs, sample_gate_only=True)
        for name, readout in reference['readouts'].items():
            torch.testing.assert_close(output['readouts'][name]['fused'], readout['fused'], atol=0, rtol=0)
        for view in reference['branches']:
            torch.testing.assert_close(output['branches'][view], reference['branches'][view], atol=0, rtol=0)
        for name in FILL_GATES:
            self.model.zero_grad(set_to_none=True)
            output = self.model(self.inputs, sample_gate_only=True)
            output['readouts'][name]['fused'].square().sum().backward()
            self.assertTrue(all(p.grad is None for p in self.model.branches.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in self.model.gates[0][name].parameters()))
        expected = torch.tensor([[1/3, 1/3, 1/3], [.5, .5, 0.]]).repeat(2, 1)
        torch.testing.assert_close(output['readouts']['final_cosine']['weights'][:, 0], expected)
        self.assertEqual(sum(p.numel() for p in self.model.gates[0]['final_cosine'].parameters()), 15)

    def test_global_statistics_are_spatial_means_not_selector_weighted(self):
        captured = {}
        hook = self.model.branches['person'].cross_agent(0, 8).register_forward_hook(
            lambda module, inputs, output: captured.update(patches=output[1]))
        try:
            output = self.model(self.inputs, sample_gate_only=True)
        finally:
            hook.remove()
        torch.testing.assert_close(output['fill_features']['global']['person'], captured['patches'].mean(1))
        self.assertIn('summary', output['fill_features'])

    def test_final_similarity_formula_and_missing_face(self):
        r = torch.tensor([True, False])
        features = {'full': torch.tensor([[1., 0.], [1., 0.]]),
                    'person': torch.tensor([[0., 1.], [0., 1.]]),
                    'face': torch.tensor([[1., 1.], [float('nan'), float('nan')]])}
        x = FinalSimilarityGate.features(r, features)
        torch.testing.assert_close(x, torch.tensor([[0., 2**-.5, 2**-.5], [0., 0., 0.]]))
        gate = FinalSimilarityGate()
        gate.linear.data.normal_()
        self.assertTrue(torch.isfinite(gate(r, features)).all())
        self.assertEqual(gate(r, features)[1, 2].item(), 0.)

    def test_held_batches_deterministic_without_duplicates_and_private_rng(self):
        before = torch.get_rng_state()
        groups = held_batches(62, 10, 64, 97)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(groups, held_batches(62, 10, 64, 97))
        self.assertTrue(all(len(row) == len(set(row)) == 62 for row in groups))
        self.assertTrue(all(len(row) == len(set(row)) == 64 for row in held_batches(558, 84, 64, 97)))


if __name__ == '__main__':
    unittest.main()
