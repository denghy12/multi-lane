import copy
import unittest

import torch
from torch import nn

from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.r1_frozen_cross_view import FrozenCrossAgent, FrozenR1Receiver
from multi_lane.track_a.r1_crossvit import compact
from multi_lane.track_a.runner import TASK_SIZES, train_task, build_optimizer_groups
from test_track_a_reproduction import FakeVisual


class FrozenCrossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make(self, mode='image_cls', view='face', native=False):
        torch.manual_seed(20261105)
        visual = FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        cls = MultiLaneModel if native else FrozenR1Receiver
        extra = {} if native else dict(r1_view=view, r1_layer=0, r1_dim=8, r1_heads=2,
                                      frozen_cross_mode=mode)
        model = cls(visual, TASK_SIZES, num_selectors=3, num_prompts=2, num_prompt_layers=1,
            adapter_mode='image_token', adapter_bottleneck_dim=4, adapter_layer_indices=(0,),
            adapter_residual_scale=.1, **extra)
        model.activate_task(0)
        x = torch.randn(4, 3, 4, 4)
        y = torch.tensor([[1., 0., 0., 1., 0.], [0., 1., 0., 0., 1.],
                          [0., 0., 1., 0., 1.], [1., 0., 1., 0., 0.]])
        inputs = dict(full=x, evidence_full=x*.7, evidence_person=x*.8,
                      evidence_face=x*.6, evidence_face_reliable=torch.tensor([True, False, True, False]))
        return model, inputs, y

    def train(self, model, inputs, y, task=0, amp=False):
        return train_task(model, [(inputs, y)], [(inputs, y)], next(model.parameters()).device,
                          task, 3, .001, 0., 1., amp,
                          adapter_learning_rate=.0004, loss_routing='adapter_asl')

    def test_initialization_rng_and_native_parameters_are_paired(self):
        base, inputs, _ = self.make(native=True)
        rng = torch.get_rng_state().clone()
        for mode in ('image_cls', 'patch'):
            model, xx, _ = self.make(mode)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(torch.equal(inputs['full'], xx['full']))
            for n, value in compact(base).items():
                self.assertTrue(torch.equal(value, compact(model)[n]), n)

    def test_zero_initial_output_and_off_replay_native(self):
        base, inputs, _ = self.make(native=True)
        expected = base.current_all_logits(inputs['full'])
        for mode in ('image_cls', 'patch'):
            for view in ('person', 'face'):
                model, xx, _ = self.make(mode, view)
                self.assertTrue(torch.equal(expected, model.current_all_logits(xx)), (mode, view))
                model.r1_enabled = False
                self.assertTrue(torch.equal(expected, model.current_all_logits(xx)))

    def test_update_positions_total_budget_and_no_inplace_changes(self):
        for mode in ('image_cls', 'patch'):
            agent = FrozenCrossAgent(8, 8, 2, mode)
            nn.init.normal_(agent.out.weight, std=3)
            x = torch.randn(3, 5, 8)
            original = x.clone()
            memory = [torch.randn(3, 4, 8), torch.randn(3, 4, 8)]
            output = agent(x, memory, [torch.ones(3, dtype=torch.bool)]*2)
            delta = output-x
            self.assertTrue(torch.equal(x, original))
            unchanged = slice(1, None) if mode == 'image_cls' else slice(0, 1)
            self.assertTrue(torch.equal(output[:, unchanged], x[:, unchanged]))
            self.assertTrue(torch.all(delta.norm(dim=(1, 2)) <= .050001*x[:, :1].norm(dim=(1, 2))))

    def test_input_gradient_reaches_new_module_but_not_senders_or_visual_weights(self):
        for mode in ('image_cls', 'patch'):
            model, inputs, y = self.make(mode)
            for key in ('evidence_full', 'evidence_person'):
                inputs[key].requires_grad_(True)
            rows = self.train(model, inputs, y)
            self.assertTrue(all(r['skipped_optimizer_steps'] == 0 for r in rows))
            for name, parameter in model.r1_bank[0].named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.norm()), 0, name)
            self.assertTrue(all(inputs[k].grad is None for k in ('evidence_full', 'evidence_person')))
            self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))
            model.assert_visual_frozen()

    def test_unreliable_face_cannot_change_person_receiver(self):
        for mode in ('image_cls', 'patch'):
            model, inputs, _ = self.make(mode, 'person')
            nn.init.normal_(model.r1_bank[0].out.weight, std=.3)
            inputs['evidence_face_reliable'].zero_()
            expected = model.current_all_logits(inputs)
            inputs['evidence_face'] = torch.randn_like(inputs['full'])*20
            self.assertTrue(torch.equal(expected, model.current_all_logits(inputs)))

    def test_old_task_path_and_parameters_stay_unchanged(self):
        for mode in ('image_cls', 'patch'):
            model, inputs, y = self.make(mode)
            self.train(model, inputs, y)
            before = model.seen_logits(inputs).detach()
            bank = copy.deepcopy(model.r1_bank[0].state_dict())
            model.activate_task(1)
            self.train(model, inputs, y[:, :3], task=1)
            self.assertTrue(torch.allclose(before, model.seen_logits(inputs)[:, :5], atol=1e-6, rtol=0))
            self.assertTrue(all(not p.requires_grad for p in model.r1_bank[0].parameters()))
            self.assertTrue(all(torch.equal(v, model.r1_bank[0].state_dict()[n]) for n, v in bank.items()))

    def test_optimizer_contains_each_parameter_once(self):
        model, _, _ = self.make()
        main, adapter, groups = build_optimizer_groups(model, 0., adapter_learning_rate=.0004)
        ids = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in (*main, *adapter)})

    def test_matched_parameter_count_and_query_shapes(self):
        for mode, count in (('image_cls', 1), ('patch', 196)):
            agent = FrozenCrossAgent(768, mode=mode)
            self.assertEqual(sum(p.numel() for p in agent.parameters()), 198144)
            agent.diagnostic_enabled = True
            agent(torch.randn(1, 197, 768), [torch.randn(1, 196, 768)]*2,
                  [torch.ones(1, dtype=torch.bool)]*2)
            self.assertEqual(agent.last_diagnostic['query_count'], count)
            self.assertEqual(agent.last_diagnostic['memory_count'], 393)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_amp_real_suffix_and_routed_losses(self):
        for mode in ('image_cls', 'patch'):
            for view in ('person', 'face'):
                model, inputs, y = self.make(mode, view)
                model.cuda();inputs = {k:v.cuda() for k, v in inputs.items()};y = y.cuda()
                rows = self.train(model, inputs, y, amp=True)
                self.assertTrue(all(r['skipped_optimizer_steps'] == 0 for r in rows))


if __name__ == '__main__':
    unittest.main()
