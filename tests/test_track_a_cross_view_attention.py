import copy
import unittest
from unittest.mock import patch
import torch
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import build_optimizer_groups, compact_model_state_dict, tensor_state_digest
from multi_lane.track_a.export_compact_test_scores import restore_compact_model_state
from multi_lane.track_a.cross_view_attention import TaskCrossViewAttentionBank
import test_track_a_reproduction as fixtures


class CrossViewAttentionTest(unittest.TestCase):
    def model(self, mode='person_full'):
        torch.manual_seed(71)
        visual = fixtures.FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        return MultiLaneModel(visual, (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1, view_fusion='fixed_three_view', cross_view_mode=mode,
            cross_view_layer=1, cross_view_dim=4, cross_view_heads=2)

    def images(self):
        return {**{v: torch.randn(3, 3, 4, 4) for v in ('full', 'person', 'face')},
                'face_reliable': torch.tensor([True, False, True])}

    def test_zero_output_preserves_baseline_and_initialization_rng(self):
        base = self.model('disabled'); rng = torch.get_rng_state(); base.activate_task(0)
        images = self.images()
        with torch.no_grad(): expected = base.seen_logits(images)
        for mode in TaskCrossViewAttentionBank.MODES[1:]:
            model = self.model(mode)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            model.activate_task(0)
            for n, v in base.state_dict().items():
                self.assertTrue(torch.equal(v, model.state_dict()[n]), n)
            with torch.no_grad(): actual = model.seen_logits(images)
            self.assertTrue(torch.equal(expected, actual), mode)
            self.assertEqual(model.cross_view_bank.parameter_count_per_task(), 128)

    def test_optimizer_membership_and_two_step_gradients(self):
        for mode in TaskCrossViewAttentionBank.MODES[1:]:
            model = self.model(mode); model.activate_task(0)
            groups = build_optimizer_groups(model, 0)[2]
            params = {id(p) for g in groups for p in g['params']}
            named = dict(model.named_parameters())
            self.assertEqual(params, {id(named[n]) for n in model.optimizer_parameter_names()})
            before = tensor_state_digest(model.visual_encoder.state_dict())
            opt = torch.optim.Adam(groups, lr=.01)
            for step in range(2):
                opt.zero_grad(); model.current_logits(self.images()).square().sum().backward()
                for name, p in model.cross_view_bank.tasks[0].named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all())
                    if step == 0 and name != 'out.weight': self.assertEqual(p.grad.abs().sum(), 0)
                    else: self.assertGreater(p.grad.abs().sum(), 0, (mode, name))
                opt.step()
            self.assertGreater(model.selectors.grad.abs().sum(), 0)
            self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))
            self.assertEqual(before, tensor_state_digest(model.visual_encoder.state_dict()))

    def test_full_source_remains_in_gradient_graph(self):
        model = self.model(); model.activate_task(0)
        with torch.no_grad(): model.cross_view_bank.tasks[0].out.weight.normal_(0, .1)
        captures = []
        def capture(module, inputs):
            inputs[1].retain_grad(); captures.append(inputs[1])
        handle = model.cross_view_bank.register_forward_pre_hook(capture)
        _, features = model.encode_lanes_with_views(self.images(), False)
        features['person'].square().sum().backward(); handle.remove()
        self.assertEqual(len(captures), 1)
        self.assertGreater(captures[0].grad.abs().sum(), 0)

    def test_source_causality_and_invalid_face_mask(self):
        for mode in TaskCrossViewAttentionBank.MODES[1:]:
            model = self.model(mode); model.activate_task(0); model.eval()
            with torch.no_grad(): model.cross_view_bank.tasks[0].out.weight.normal_(0, .3)
            images = self.images(); images['face_reliable'].fill_(False)
            with torch.no_grad():
                _, a = model.encode_lanes_with_views(images, False)
                images['full'] = torch.randn_like(images['full']) * 2
                _, b = model.encode_lanes_with_views(images, False)
                self.assertEqual(torch.equal(a['person'], b['person']), mode == 'person_self')
                old = model.seen_logits(images)
                images['face'] = torch.randn_like(images['face']) * 100
                self.assertTrue(torch.equal(old, model.seen_logits(images)))

    def test_original_three_paths_and_only_one_interaction_layer(self):
        model = self.model(); model.activate_task(0); calls = []; crossings = []
        original = model._task_attention_ffn
        def record(*args, **kwargs):
            calls.append((args[3], args[7] if len(args) > 7 else 'full'))
            return original(*args, **kwargs)
        handle = model.cross_view_bank.register_forward_pre_hook(lambda module, inputs: crossings.append(inputs[0].shape))
        with patch.object(model, '_task_attention_ffn', side_effect=record): model.seen_logits(self.images())
        handle.remove()
        self.assertEqual(calls, [(i, v) for v in ('full', 'person', 'face') for i in (0, 1)])
        self.assertEqual(crossings, [torch.Size([1, 3, 2, 8])])

    def test_old_task_protection_and_compact_restore(self):
        for mode in TaskCrossViewAttentionBank.MODES[1:]:
            model = self.model(mode); model.activate_task(0); images = self.images()
            with torch.no_grad():
                model.cross_view_bank.tasks[0].out.weight.normal_(0, .1)
                old = model.seen_logits(images).clone()
            digest = tensor_state_digest(model.cross_view_bank.tasks[0].state_dict())
            model.activate_task(1); opt = torch.optim.Adam(build_optimizer_groups(model, 0)[2], lr=.01)
            for _ in range(2):
                opt.zero_grad(); model.current_logits(images).square().sum().backward(); opt.step()
            self.assertEqual(digest, tensor_state_digest(model.cross_view_bank.tasks[0].state_dict()))
            with torch.no_grad(): actual = model.seen_logits(images)
            self.assertTrue(torch.allclose(old, actual[:, :2], atol=1e-6), mode)
            restored = self.model(mode)
            payload = dict(schema_version=1, task_id=1, source_git={'commit': 'unit'}, model=compact_model_state_dict(model))
            restore_compact_model_state(restored, payload, 1, {'commit': 'unit'})
            with torch.no_grad(): self.assertTrue(torch.equal(actual, restored.seen_logits(images)))

    def test_mixed_precision_and_image_dependent_query(self):
        for mode in TaskCrossViewAttentionBank.MODES[1:]:
            model = self.model(mode); model.activate_task(0)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                loss = model.current_logits(self.images()).float().square().sum()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            row = model.cross_view_bank.statistics()['task0/layer1']
            qstd = max(row['query_std_across_images'])
            self.assertLess(qstd, 1e-6) if mode == 'task_full' else self.assertGreater(qstd, 1e-3)
            self.assertEqual(row['samples'], 3)

    def test_cache_is_per_forward_and_can_recover_after_exception(self):
        model = self.model(); model.activate_task(0); images = self.images()
        with patch.object(model.cross_view_bank, 'forward', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'): model.seen_logits(images)
        self.assertTrue(torch.isfinite(model.seen_logits(images)).all())
        self.assertFalse(any(isinstance(v, dict) and 'full' in v for v in vars(model).values()))
        model.cross_view_runtime_enabled = False
        with torch.no_grad(): before = model.seen_logits(images)
        model.cross_view_runtime_enabled = True
        with torch.no_grad(): self.assertTrue(torch.equal(before, model.seen_logits(images)))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA RNG audit')
    def test_initialization_preserves_cuda_rng(self):
        torch.cuda.manual_seed_all(345)
        cuda = torch.cuda.get_rng_state().clone(); cpu = torch.get_rng_state().clone()
        TaskCrossViewAttentionBank(2, 8, 'person_full', inner_dim=4, heads=2)
        self.assertTrue(torch.equal(cuda, torch.cuda.get_rng_state()))
        self.assertTrue(torch.equal(cpu, torch.get_rng_state()))

    def test_invalid_configurations_rejected(self):
        for kwargs in (dict(mode='bad'), dict(inner_dim=3, heads=2), dict(scale=float('nan'))):
            args = dict(num_tasks=2, width=8, mode='person_full'); args.update(kwargs)
            with self.assertRaises(ValueError): TaskCrossViewAttentionBank(**args)


if __name__ == '__main__': unittest.main()
