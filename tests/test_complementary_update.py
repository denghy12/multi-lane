import copy
import unittest

import torch

from multi_lane.track_a.complementary_update import (
    adam_candidate_parameters, adam_candidate_step, select_auxiliary_gradients,
)


def assert_state_equal(test, expected, actual):
    test.assertEqual(set(expected), set(actual))
    for key, value in expected.items():
        if isinstance(value, dict):
            assert_state_equal(test, value, actual[key])
        elif isinstance(value, torch.Tensor):
            test.assertTrue(torch.equal(value, actual[key]), key)
        else:
            test.assertEqual(value, actual[key])


class ComplementaryUpdateTest(unittest.TestCase):
    def fixture(self, dtype=torch.float64):
        p = torch.nn.Parameter(torch.tensor([.3, -.7], dtype=dtype))
        a = torch.nn.Parameter(torch.tensor([0., .1], dtype=dtype))
        head = torch.nn.Parameter(torch.tensor([.2], dtype=dtype))
        opt = torch.optim.Adam([dict(params=[p, head], lr=.01), dict(params=[a], lr=.0004)],
                               foreach=False)
        for i in range(4):
            p.grad = torch.tensor([.25 + .03*i, -.5], dtype=dtype)
            a.grad = torch.tensor([1e-9, -2e-9], dtype=dtype)
            head.grad = torch.tensor([.7], dtype=dtype)
            opt.step()
        p.grad = torch.tensor([1., 2.], dtype=dtype)
        a.grad = torch.tensor([1e-9, 1e-9], dtype=dtype)
        head.grad = torch.tensor([-.2], dtype=dtype)
        return p, a, head, opt

    def test_candidate_matches_real_first_adam_step_without_initializing_state(self):
        p = torch.nn.Parameter(torch.tensor([.25, -.6], dtype=torch.float32))
        opt = torch.optim.Adam([p], lr=.0004, foreach=False)
        p.grad = torch.tensor([2e-10, -3e-10])
        before = p.detach().clone()
        predicted = adam_candidate_parameters(optimizer=opt)
        self.assertEqual(len(opt.state), 0)
        self.assertTrue(torch.equal(p, before))
        opt.step()
        self.assertTrue(torch.equal(p, predicted[p]))

    def test_candidate_matches_real_adam_with_moments_and_multiple_groups(self):
        for dtype in (torch.float32, torch.float64):
            for amsgrad in (False, True):
                p = torch.nn.Parameter(torch.tensor([.2, -.3], dtype=dtype))
                a = torch.nn.Parameter(torch.tensor([.01, .04], dtype=dtype))
                opt = torch.optim.Adam([dict(params=[p], lr=.0125), dict(params=[a], lr=.0004)],
                                       amsgrad=amsgrad, weight_decay=.02, foreach=False)
                for i in range(3):
                    p.grad = torch.tensor([.3 + .1*i, -.1], dtype=dtype)
                    a.grad = torch.tensor([1e-10, -2e-10], dtype=dtype)
                    opt.step()
                saved = copy.deepcopy(opt.state_dict())
                parameters = [p.detach().clone(), a.detach().clone()]
                predicted = adam_candidate_parameters(optimizer=opt)
                assert_state_equal(self, saved, opt.state_dict())
                self.assertTrue(torch.equal(p, parameters[0]))
                self.assertTrue(torch.equal(a, parameters[1]))
                opt.step()
                self.assertTrue(torch.equal(p, predicted[p]))
                self.assertTrue(torch.equal(a, predicted[a]))

    def test_foreach_auto_and_explicit_preserve_gradient_inputs(self):
        for foreach in (None, True):
            p = torch.nn.Parameter(torch.tensor([.4, -.2]))
            opt = torch.optim.Adam([p], lr=.01, weight_decay=.03, foreach=foreach)
            p.grad = torch.tensor([.7, -.1])
            opt.step()
            gradient = p.grad.clone()
            predicted = adam_candidate_parameters(optimizer=opt)[p]
            self.assertTrue(torch.equal(p.grad, gradient))
            opt.step()
            self.assertTrue(torch.equal(p, predicted))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA comparison requires an available GPU')
    def test_cuda_auto_foreach_matches_full_optimizer_with_head(self):
        p = torch.nn.Parameter(torch.tensor([.3, -.8], device='cuda'))
        head = torch.nn.Parameter(torch.tensor([.2], device='cuda'))
        opt = torch.optim.Adam([p, head], lr=.0125)
        p.grad = torch.tensor([.1, -.2], device='cuda')
        head.grad = torch.tensor([.4], device='cuda')
        opt.step()
        p.grad = torch.tensor([-.3, .5], device='cuda')
        head.grad = torch.tensor([-.7], device='cuda')
        state = copy.deepcopy(opt.state_dict())
        predicted = adam_candidate_parameters(optimizer=opt, parameters=(p,))[p]
        assert_state_equal(self, state, opt.state_dict())
        opt.step()
        self.assertTrue(torch.equal(p, predicted))

    def test_selector_never_mutates_state_or_parameters_or_head_gradient(self):
        p, a, head, opt = self.fixture()
        before = [x.detach().clone() for x in (p, a, head)]
        head_gradient = head.grad.clone()
        saved = copy.deepcopy(opt.state_dict())
        record = select_auxiliary_gradients(optimizer=opt,
            groups=dict(representation=[p], adapter=[a]),
            auxiliary=dict(representation=[torch.tensor([.1, -.1], dtype=p.dtype)],
                           adapter=[torch.tensor([1., -1.], dtype=a.dtype)]),
            guards={}, mode="none", update_ratio=100.)
        self.assertGreater(record['alpha'], 0.)
        self.assertTrue(record['effective_nonzero_update'])
        self.assertTrue(all(torch.equal(x, y) for x, y in zip((p, a, head), before)))
        self.assertTrue(torch.equal(head.grad, head_gradient))
        assert_state_equal(self, saved, opt.state_dict())
        self.assertLessEqual(record['groups']['adapter']['capped_auxiliary_gradient_norm'],
                             .250001 * record['groups']['adapter']['base_gradient_norm'])
        self.assertGreater(record['groups']['adapter']['capped_auxiliary_gradient_norm'], 1e-12)

    def test_guard_accepts_helpful_direction_and_rejects_harmful_direction(self):
        p, a, head, opt = self.fixture()
        base = p.grad.clone()
        q = torch.tensor([.1, -.1], dtype=p.dtype)
        default = adam_candidate_parameters(optimizer=opt, parameters=(p,))[p]
        trial = adam_candidate_parameters(optimizer=opt, parameters=(p,), gradients={p: base+q})[p]
        u = trial-default
        self.assertGreater(float(u.square().sum()), 0.)
        groups = dict(representation=(p,), adapter=(a,))
        auxiliary = dict(representation=(q,), adapter=(None,))
        safe = select_auxiliary_gradients(optimizer=opt, groups=groups, auxiliary=auxiliary,
            guards={'full':dict(representation=(-u,), adapter=(None,)),
                    'fused':dict(representation=(-u,), adapter=(None,))},
            mode='full_fusion', update_ratio=100.)
        self.assertEqual(safe['alpha'], 1.)
        self.assertLess(safe['selected_first_order_extra_loss']['full'], 0.)
        p.grad = base.clone()
        unsafe = select_auxiliary_gradients(optimizer=opt, groups=groups, auxiliary=auxiliary,
            guards={'full':dict(representation=(u,), adapter=(None,)),
                    'fused':dict(representation=(u,), adapter=(None,))},
            mode='full_fusion', update_ratio=100.)
        self.assertEqual(unsafe['alpha'], 0.)
        self.assertTrue(torch.equal(p.grad, base))

    def test_update_size_limit_applies_even_without_direction_guards(self):
        p, a, head, opt = self.fixture()
        original = p.grad
        record = select_auxiliary_gradients(optimizer=opt,
            groups=dict(representation=[p], adapter=[a]),
            auxiliary=dict(representation=[torch.tensor([.1, -.1], dtype=p.dtype)], adapter=[None]),
            guards={}, mode='none', update_ratio=0.)
        self.assertEqual(record['alpha'], 0.)
        self.assertFalse(record['effective_nonzero_update'])
        self.assertIs(p.grad, original)

    def test_rounded_zero_delta_is_not_counted_as_effective_auxiliary_update(self):
        p = torch.nn.Parameter(torch.tensor([1e9], dtype=torch.float32))
        opt = torch.optim.Adam([p], lr=.001, foreach=False)
        p.grad = torch.ones_like(p)
        record = select_auxiliary_gradients(optimizer=opt,
            groups=dict(representation=[p]), auxiliary=dict(representation=[torch.full_like(p, .1)]),
            guards={}, mode='none')
        self.assertEqual(record['alpha'], 1.)
        self.assertFalse(record['effective_nonzero_update'])
        self.assertEqual(record['status'], 'accepted_zero_parameter_delta')

    def test_base_none_and_zero_signal_groups_are_not_started(self):
        p = torch.nn.Parameter(torch.tensor([0.]))
        a = torch.nn.Parameter(torch.tensor([0.]))
        opt = torch.optim.Adam([p, a], foreach=False)
        p.grad = None
        a.grad = torch.zeros_like(a)
        record = select_auxiliary_gradients(optimizer=opt,
            groups=dict(representation=[p], adapter=[a]),
            auxiliary=dict(representation=[torch.ones_like(p)], adapter=[torch.ones_like(a)]),
            guards={}, mode='none')
        self.assertIsNone(p.grad)
        self.assertTrue(torch.equal(a.grad, torch.zeros_like(a)))
        self.assertEqual(record['base_none_auxiliary_ignored'], 1)
        self.assertEqual(record['alpha'], 0.)
        self.assertEqual(record['status'], 'no_auxiliary_signal')
        self.assertEqual(len(opt.state), 0)

    def test_ramp_zero_and_nonfinite_extra_preserve_baseline_gradients(self):
        for nonfinite in (False, True):
            p, a, head, opt = self.fixture()
            base = p.grad
            record = select_auxiliary_gradients(optimizer=opt,
                groups=dict(representation=[p], adapter=[a]),
                auxiliary=dict(representation=[torch.full_like(p, float('nan')) if nonfinite else torch.ones_like(p)],
                               adapter=[None]), guards={}, mode='none', ramp=1. if nonfinite else 0.)
            self.assertEqual(record['alpha'], 0.)
            self.assertIs(p.grad, base)
            self.assertEqual(record['status'], 'nonfinite_auxiliary' if nonfinite else 'inactive')

    def test_subset_candidate_skips_none_and_unknown_parameters_fail(self):
        p, a, head, opt = self.fixture()
        before = a.detach().clone()
        predicted = adam_candidate_step(optimizer=opt, gradients={a:None}, parameters=(a,))
        self.assertTrue(torch.equal(predicted[a], torch.zeros_like(a)))
        self.assertTrue(torch.equal(a, before))
        unknown = torch.nn.Parameter(torch.ones(1))
        with self.assertRaises(ValueError):
            adam_candidate_step(optimizer=opt, parameters=(unknown,))

    def test_unsupported_or_incomplete_configuration_fails_closed(self):
        p, a, head, opt = self.fixture()
        args = dict(optimizer=opt, groups=dict(representation=[p], adapter=[a]),
                    auxiliary=dict(representation=[None], adapter=[None]), guards={})
        with self.assertRaises(ValueError):select_auxiliary_gradients(**args)
        with self.assertRaises(ValueError):select_auxiliary_gradients(**args, mode='none', alphas=(1., .5))
        opt.param_groups[0]['fused'] = True
        with self.assertRaises(ValueError):adam_candidate_step(optimizer=opt)


if __name__ == '__main__':
    unittest.main()
