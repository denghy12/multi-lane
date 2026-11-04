import copy
import unittest

import torch
from torch.nn import functional as F

from multi_lane.track_a.r1_crossvit import CLSAgent, R1Receiver, compact
from multi_lane.track_a.runner import TASK_SIZES
from test_track_a_reproduction import FakeVisual


MODES = ('cls', 'selector_mean', 'selectors')


class SelectorQueryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(36)
        self.cls = torch.randn(4, 1, 16)
        self.selectors = torch.randn(1, 10, 16).expand(4, -1, -1).clone()
        self.memory = [torch.randn(4, 7, 16), torch.randn(4, 5, 16)]
        self.masks = [torch.ones(4, dtype=torch.bool), torch.tensor([True, False, True, False])]

    def agent(self, mode):
        torch.manual_seed(91)
        return CLSAgent(16, dim=8, heads=2, query_mode=mode)

    def test_identical_parameters_and_zero_output(self):
        modules = [self.agent(mode) for mode in MODES]
        for module in modules:
            self.assertEqual(sum(p.numel() for p in module.parameters()), 544)
            for key, value in modules[0].state_dict().items():
                self.assertTrue(torch.equal(value, module.state_dict()[key]))
            self.assertTrue(torch.equal(module(self.cls, self.memory, self.masks, self.selectors), self.cls))

    def test_cls_mode_replays_legacy_formula(self):
        m = self.agent('cls')
        m.out.weight.data.normal_()
        c = F.layer_norm(self.cls, (16,))
        sources = [F.layer_norm(x + m.types[i], (16,)) for i, x in enumerate(self.memory)]
        memory = torch.cat([c, *sources], 1)
        q = m.q(c).reshape(4, 1, 2, 4).transpose(1, 2)
        k = m.k(memory).reshape(4, -1, 2, 4).transpose(1, 2)
        v = m.v(memory).reshape(4, -1, 2, 4).transpose(1, 2)
        valid = torch.cat([torch.ones(4, 1, dtype=torch.bool), self.masks[0][:, None].expand(-1, 7), self.masks[1][:, None].expand(-1, 5)], 1)
        a = ((q @ k.transpose(-2, -1)) * .5).masked_fill(~valid[:, None, None], float('-inf')).softmax(-1)
        expected = self.cls + m.out((a @ v).transpose(1, 2).reshape(4, 1, 8))
        self.assertTrue(torch.equal(m(self.cls, self.memory, self.masks), expected))

    def test_query_counts_and_selector_batch_independence(self):
        for mode in MODES:
            m = self.agent(mode)
            m.diagnostic_enabled = True
            result = m(self.cls, self.memory, self.masks, self.selectors)
            self.assertEqual(result.shape, self.cls.shape)
            self.assertEqual(m.last_diagnostic['query_count'], 10 if mode == 'selectors' else 1)
            if mode != 'cls':
                self.assertEqual(m.last_diagnostic['query_batch_std'], 0)
            else:
                self.assertGreater(m.last_diagnostic['query_batch_std'], .01)

    def test_repeated_queries_and_permutation_mean_readout(self):
        one, many = self.agent('selector_mean'), self.agent('selectors')
        one.out.weight.data.normal_()
        many.load_state_dict(one.state_dict())
        repeated = self.selectors[:, :1].repeat(1, 10, 1)
        self.assertTrue(torch.allclose(one(self.cls, self.memory, self.masks, repeated), many(self.cls, self.memory, self.masks, repeated), atol=1e-6))
        self.assertTrue(torch.allclose(many(self.cls, self.memory, self.masks, self.selectors), many(self.cls, self.memory, self.masks, self.selectors.flip(1)), atol=1e-6))

    def test_invalid_source_mask_and_external_stop_gradient(self):
        for mode in MODES:
            m = self.agent(mode)
            m.out.weight.data.normal_()
            memory = [x.clone().requires_grad_() for x in self.memory]
            selectors = self.selectors.clone().requires_grad_()
            a = m(self.cls, memory, self.masks, selectors)
            changed = [memory[0], memory[1].detach().clone()]
            changed[1][~self.masks[1]] *= 100
            self.assertTrue(torch.equal(a, m(self.cls, changed, self.masks, selectors)))
            a.square().mean().backward()
            self.assertTrue(all(x.grad is None for x in memory))
            self.assertGreater(float(m.q.weight.grad.norm()), 0)
            if mode != 'cls':
                self.assertGreater(float(selectors.grad.norm()), 0)

    def model(self, mode):
        torch.manual_seed(2026)
        m = R1Receiver(FakeVisual(), TASK_SIZES, num_selectors=10, num_prompts=2,
                      num_prompt_layers=1, adapter_mode='image_token', adapter_bottleneck_dim=4,
                      adapter_layer_indices=(0,), adapter_residual_scale=.1,
                      r1_kind='attention', r1_view='face', r1_layer=0,
                      r1_dim=4, r1_heads=2, r1_query_mode=mode)
        m.activate_task(0)
        return m

    def test_model_initialization_and_old_task_protection(self):
        a = self.model('cls')
        reference = compact(a)
        for mode in MODES:
            m = self.model(mode)
            self.assertTrue(all(torch.equal(v, compact(m)[key]) for key, v in reference.items()))
            m.activate_task(1)
            self.assertTrue(all(not p.requires_grad for p in m.r1_bank[0].parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_actual_amp_query_gradients_after_zero_warmup(self):
        for mode in MODES:
            m = self.agent(mode).cuda()
            optimizer = torch.optim.Adam(m.parameters(), lr=.0004)
            for step in range(2):
                optimizer.zero_grad()
                with torch.autocast('cuda'):
                    out = m(self.cls.cuda(), [x.cuda() for x in self.memory], [x.cuda() for x in self.masks], self.selectors.cuda())
                    loss = out.square().mean()
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
                if step == 1:
                    self.assertGreater(float(m.q.weight.grad.norm()), 0)
                    self.assertGreater(float(m.v.weight.grad.norm()), 0)
                optimizer.step()


if __name__ == '__main__':
    unittest.main()
