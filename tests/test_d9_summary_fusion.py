import unittest
from unittest.mock import patch

import torch

from multi_lane.track_a.d9_relation_fusion import D9RelationFusion
from multi_lane.track_a.summary_logit_fusion import (
    METHODS, DIMENSIONS, SummaryLogitGate, pair_statistics, summary_statistics,
    image_descriptors, masked_weights, fused_logits, image_bce,
)
from test_d9_protected_multi_model import branches


class SummaryFusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_selector_numbering_and_zero_norm_boundaries(self):
        left, right = torch.randn(3, 4, 7), torch.randn(3, 5, 7)
        expected = pair_statistics(left, right)
        actual = pair_statistics(left[:, [3, 1, 0, 2]], right[:, [2, 4, 1, 0, 3]])
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(pair_statistics(left*0, right*0)).all())
        singleton = pair_statistics(left[:, :1], right[:, :1])
        torch.testing.assert_close(singleton[:, [2, 4, 5]], torch.zeros(3, 3))
        with self.assertRaises(ValueError):
            pair_statistics(left[:, :0], right)

    def test_only_seen_summaries_and_reliable_faces(self):
        valid = torch.tensor([True, False, True])
        capture = {v: {t: torch.randn(3, 4, 7, requires_grad=True) for t in (0, 1)}
                   for v in ('full', 'person', 'face')}
        expected = summary_statistics(capture, valid, 1)
        for x in capture['face'].values():
            with torch.no_grad():
                x[1] = torch.nan
        torch.testing.assert_close(summary_statistics(capture, valid, 1), expected, rtol=0, atol=0)
        self.assertFalse(expected.requires_grad)
        self.assertEqual(expected[1, 6:].abs().sum(), 0)
        with self.assertRaises(ValueError):
            summary_statistics(capture, valid, 0)

    def test_complete_roster_pooling_is_order_and_batch_independent(self):
        values, flags = torch.randn(5, 12), torch.tensor([True, False, True, False, False])
        whole = image_descriptors(values, flags, [3, 2])
        order = torch.tensor([2, 0, 1, 4, 3])
        shuffled = image_descriptors(values[order], flags[order], [3, 2])
        parts = [image_descriptors(values[:3], flags[:3], [3]), image_descriptors(values[3:], flags[3:], [2])]
        for method in whole:
            torch.testing.assert_close(whole[method], shuffled[method])
            torch.testing.assert_close(whole[method], torch.cat([p[method] for p in parts]))
            self.assertFalse(whole[method].requires_grad)
        torch.testing.assert_close(whole['summary_mean'][:, [0, 1]], whole['summary_correspondence'][:, [0, 6]])
        torch.testing.assert_close(whole['summary_mean'][:, [2, 3]], whole['summary_correspondence'][:, [12, 18]])
        self.assertEqual(whole['summary_mean'][1, 1], 0)
        self.assertEqual(whole['summary_mean'][1, 3], 0)

    def test_zero_initialization_full_one_and_person_face_mask(self):
        for method in METHODS:
            gate = SummaryLogitGate(method)
            scores = gate(torch.randn(2, DIMENSIONS[method]))
            torch.testing.assert_close(scores[:, 0], torch.ones(2), rtol=0, atol=0)
            weights = masked_weights(scores, torch.tensor([True, False]))
            torch.testing.assert_close(weights, torch.tensor([[.64, .16, .20], [.8, .2, 0.]]))
            self.assertEqual(sum(p.numel() for p in gate.parameters() if p.requires_grad),
                             {'fixed': 0, 'shared': 2, 'summary_mean': 14, 'summary_correspondence': 54}[method])

    def test_gradients_only_update_gate_and_missing_face_is_zero(self):
        x = torch.randn(2, 6, requires_grad=True)
        z = torch.randn(3, 2, 3, requires_grad=True)
        gate = SummaryLogitGate('summary_mean')
        inverse, valid = torch.tensor([0, 0, 1]), torch.tensor([False, False, False])
        with torch.no_grad():
            gate.matrix.normal_()
        result = fused_logits(z, gate(x), inverse, valid)
        image_bce(result, torch.tensor([[0., 1.], [1., 0.], [1., 1.]]), inverse, 2).backward()
        self.assertIsNone(z.grad)
        self.assertIsNone(x.grad)
        self.assertGreater(gate.bias.grad[0].abs().item(), 0)
        self.assertEqual(gate.bias.grad[1].item(), 0)
        self.assertEqual(gate.matrix.grad[:, 1].abs().sum().item(), 0)

    def test_image_balancing_and_state_round_trip(self):
        z, y = torch.tensor([[1., -1.], [-2., 2.]]), torch.tensor([[1., 0.], [0., 1.]])
        original = image_bce(z, y, torch.tensor([0, 1]), 2)
        duplicated = image_bce(z[[0, 0, 1]], y[[0, 0, 1]], torch.tensor([0, 0, 1]), 2)
        torch.testing.assert_close(original, duplicated)
        gate = SummaryLogitGate('summary_mean')
        x = torch.randn(3, 6)
        gate.set_center(x)
        with torch.no_grad():
            gate.bias.normal_()
            gate.matrix.normal_()
        restored = SummaryLogitGate.restore(gate.export())
        torch.testing.assert_close(gate(x), restored(x), rtol=0, atol=0)

    def test_capture_is_exact_task_msa_input_without_changing_native_logits(self):
        for mode in ('sample_holdout', 'sample_full'):
            model = D9RelationFusion(branches(), mode)
            model.activate_task(0)
            model.activate_task(1)
            model.requires_grad_(False).eval()
            inputs = {v: torch.randn(4, 3, 4, 4) for v in
                      ('full', 'person', 'face', 'evidence_full', 'evidence_person', 'evidence_face')}
            inputs['evidence_face_reliable'] = torch.tensor([True, False, True, False])
            captured, mocks = {}, []
            for view, branch in model.branches.items():
                original = branch._prompt_attention
                def tap(block, tokens, lanes, layer, view=view, original=original):
                    if layer == 9:
                        captured.update({(view, t): tokens[i, :, 1:].detach().float().clone()
                                         for i, t in enumerate(lanes)})
                    return original(block, tokens, lanes, layer)
                mocks.append(patch.object(branch, '_prompt_attention', side_effect=tap))
            with torch.inference_mode():
                expected = model(inputs, all_seen=True)
                try:
                    for mock in mocks:
                        mock.start()
                    tokens = {}
                    actual = model(inputs, all_seen=True, summary_tokens=tokens)
                finally:
                    for mock in mocks:
                        mock.stop()
            for v in ('full', 'person', 'face'):
                torch.testing.assert_close(actual['branches'][v], expected['branches'][v], atol=0, rtol=0)
                self.assertEqual(set(tokens[v]), {0, 1})
                for task in (0, 1):
                    torch.testing.assert_close(tokens[v][task], captured[v, task], atol=0, rtol=0)
                    self.assertFalse(tokens[v][task].requires_grad)
            with self.assertRaises(ValueError):
                model(inputs, all_seen=True, summary_tokens={})


if __name__ == '__main__':
    unittest.main()
