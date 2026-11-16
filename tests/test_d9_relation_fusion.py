import copy
import unittest

import torch

from multi_lane.track_a.d9_relation_fusion import D9RelationFusion, RelationGate
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.r1_frozen_cross_view import FrozenCrossAgent, FrozenR1Receiver
from test_track_a_reproduction import FakeVisual


class RelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_readout_preserves_patch_update_and_backpropagates_into_qkv(self):
        torch.manual_seed(7)
        agent = FrozenCrossAgent(8, dim=8, heads=2, mode='patch')
        own = torch.randn(3, 5, 8)
        sources = [torch.randn(3, 4, 8, requires_grad=True) for _ in range(2)]
        masks = [torch.ones(3, dtype=torch.bool)] * 2
        baseline = agent(own, sources, masks)
        patch, features = agent(own, sources, masks, return_full_relation=True)
        self.assertTrue(torch.equal(patch, baseline))
        self.assertEqual(features.shape, (3, 4))
        self.assertFalse(torch.equal(features[0], features[1]))
        features.sum().backward()
        for parameter in [agent.q.weight, agent.k.weight, agent.v.weight]:
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.norm().item(), 0)
        self.assertTrue(all(x.grad is None for x in sources))

    def test_prior_gradient_and_invalid_face_exclusion(self):
        reliable = torch.tensor([True, False, False, True])
        expected = torch.tensor([[.64, .16, .20], [.80, .20, 0.], [.80, .20, 0.], [.64, .16, .20]])
        person, face = torch.randn(4, 4), torch.randn(4, 4, requires_grad=True)
        for mode in ('fixed', 'static', 'person'):
            g = RelationGate(mode)
            w = g(reliable, person, face)
            torch.testing.assert_close(w, expected)
            if mode == 'person':
                w[:, 2].sum().backward()
                self.assertTrue(g.scorers['face'][-1].weight.grad.norm() > 0)
                with torch.no_grad():
                    g.scorers['face'][-1].weight.fill_(.3)
                    g.scorers['person'][-1].weight.fill_(.2)
                w = g(reliable, person, face)
                self.assertFalse(torch.equal(w, expected))
                other = face.detach().clone()
                other[~reliable] = torch.nan
                self.assertTrue(torch.equal(w, g(reliable, person, other)))
                w[:, 2].sum().backward()
                self.assertTrue(torch.equal(face.grad[~reliable], torch.zeros(2, 4)))

    def test_stages_include_task_specific_gates_and_replay_native_branch_outputs(self):
        torch.manual_seed(3)
        vis = FakeVisual()
        vis.transformer.resblocks.append(copy.deepcopy(vis.transformer.resblocks[0]))
        args = dict(task_sizes=(2, 1), num_selectors=3, num_prompts=2, num_prompt_layers=1,
                    adapter_mode='image_token', adapter_bottleneck_dim=4, adapter_layer_indices=(0,))
        branches = {'full': MultiLaneModel(copy.deepcopy(vis), **args)}
        for view in ('person', 'face'):
            branches[view] = FrozenR1Receiver(copy.deepcopy(vis), r1_layer=0, r1_view=view,
                                              r1_dim=8, r1_heads=2, frozen_cross_mode='patch', **args)
        rng = torch.get_rng_state()
        model = D9RelationFusion(branches, 'person')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        model.activate_task(0)
        model.activate_task(1)
        self.assertTrue(all(not p.requires_grad for p in model.gates[0].parameters()))
        with torch.no_grad():
            model.gates[0].offset.copy_(torch.tensor([1., -1.]))
        inputs = {k: torch.randn(3, 3, 4, 4) for k in
                  ('full', 'person', 'face', 'evidence_full', 'evidence_person', 'evidence_face')}
        inputs['evidence_face_reliable'] = torch.tensor([True, False, True])
        xx = model(inputs, all_seen=True)
        self.assertEqual(xx['weights'].shape, (3, 2, 3))
        self.assertFalse(torch.equal(xx['weights'][:, 0], xx['weights'][:, 1]))
        ys = []
        for view, branch in branches.items():
            image = inputs['full'] if view == 'full' else {**inputs, 'full': inputs[view]}
            y = branch.seen_logits(image)
            torch.testing.assert_close(xx['branches'][view], y)
            torch.testing.assert_close(xx['all_columns'][view], branch.current_all_logits(image))
            ys.append(y)
        w = torch.repeat_interleave(xx['weights'], torch.tensor([2, 1]), dim=1)
        torch.testing.assert_close(xx['fused'], (torch.stack(ys, dim=-1) * w).sum(dim=-1))
        torch.testing.assert_close(model(inputs)['fused'], xx['fused'][:, -1:])


if __name__ == '__main__':
    unittest.main()
