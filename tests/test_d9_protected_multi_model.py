"""Protected readouts must consume the exact summaries and leave each native lane alone."""
import copy
import unittest
from unittest.mock import patch

import torch

from multi_lane.track_a.d9_relation_fusion import D9RelationFusion, READOUTS
from multi_lane.track_a.r1_crossvit import R1Receiver
from multi_lane.track_a.r1_frozen_cross_view import FrozenR1Receiver
from test_track_a_reproduction import FakeVisual


def branches():
    visual = FakeVisual()
    for _ in range(9):
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
    args = dict(task_sizes=(2, 1), num_selectors=3, num_prompts=2, num_prompt_layers=1,
                adapter_mode='image_token', adapter_bottleneck_dim=4, adapter_layer_indices=(1,))
    full = R1Receiver(visual, r1_kind='native', r1_view='full', **args)
    result = {'full': full}
    for v in ('person', 'face'):
        result[v] = FrozenR1Receiver(visual, r1_layer=8, r1_view=v, r1_dim=8, r1_heads=2,
                                     frozen_cross_mode='patch', **args)
    return result


class ProtectedReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3)
        self.branches = branches()
        state = torch.get_rng_state()
        self.model = D9RelationFusion(self.branches, 'multi')
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.model.activate_task(0)
        self.inputs = {k: torch.randn(4, 3, 4, 4) for k in
                       ('full', 'person', 'face', 'evidence_full', 'evidence_person', 'evidence_face')}
        self.inputs['evidence_face_reliable'] = torch.tensor([True, False, True, False])

    def test_all_readouts_replay_fixed_and_detach_both_gate_inputs(self):
        fixed = D9RelationFusion(copy.deepcopy(self.branches), 'fixed')
        xx = self.model(self.inputs)
        original = fixed(self.inputs)
        for name in READOUTS:
            torch.testing.assert_close(xx['readouts'][name]['fused'], original['fused'], rtol=0, atol=0)
        # Nonzero gate output layers ensure backward would traverse evidence if attached.
        with torch.no_grad():
            gate = self.model.gates[0]
            gate['summary'].readout[-1].weight.normal_()
            for scorer in gate['relation'].scorers.values():
                scorer[-1].weight.normal_()
        xx = self.model(self.inputs)
        for name in ('static', 'relation', 'summary'):
            xx['readouts'][name]['fused'].sum().backward()
            self.assertFalse(any(p.grad is not None for p in self.model.branches.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.norm() > 0 for p in gate[name].parameters()))
        # Compare true native branch gradients in the live fixed graph after the extra readouts.
        for view in self.branches:
            torch.testing.assert_close(xx['all_columns'][view], original['all_columns'][view], rtol=0, atol=0)
        xx['fused'].sum().backward()
        original['fused'].sum().backward()
        old = dict(fixed.branches.named_parameters())
        for name, param in self.model.branches.named_parameters():
            if param.grad is not None or old[name].grad is not None:
                torch.testing.assert_close(param.grad, old[name].grad, rtol=0, atol=0)

    def test_summary_reads_actual_task10_inputs_and_excludes_unreliable_faces(self):
        self.model.activate_task(1)
        self.assertFalse(any(p.requires_grad for p in self.model.gates[0].parameters()))
        for bank in self.model.gates:
            with torch.no_grad():
                bank['summary'].readout[-1].weight.normal_()
        # The input of the task attention is [CLS, selected summaries]. Capture it
        # independently of the gate tap on the active task of each view.
        captured = {}
        patches = []
        for view, branch in self.branches.items():
            original = branch._prompt_attention
            def tap(block, tokens, lanes, layer, view=view, original=original):
                if layer == 9:
                    captured.update({(view, lane): tokens[i, :, 1:].detach().float().mean(1)
                                     for i, lane in enumerate(lanes)})
                return original(block, tokens, lanes, layer)
            patches.append(patch.object(branch, '_prompt_attention', side_effect=tap))
        try:
            for mock in patches:
                mock.start()
            out = self.model(self.inputs, all_seen=True)
        finally:
            for mock in patches:
                mock.stop()
        reliable = self.inputs['evidence_face_reliable']
        for task in range(2):
            summary = {v: captured[v, task] for v in self.branches}
            gate = self.model.gates[task]['summary']
            got = out['readouts']['summary']['weights'][:, task]
            torch.testing.assert_close(got, gate(reliable, summary), rtol=0, atol=0)
            summary['face'][~reliable] = torch.nan
            torch.testing.assert_close(got, gate(reliable, summary), rtol=0, atol=0)
            self.assertTrue(got[~reliable, 2].eq(0).all())
        current = self.model(self.inputs)
        for name in READOUTS:
            torch.testing.assert_close(out['readouts'][name]['fused'][:, -1:],
                                       current['readouts'][name]['fused'], atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
