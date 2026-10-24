import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import test_track_a_reproduction as fixtures
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import (parse_args, build_optimizer_groups, compute_training_loss,
    compute_asymmetric_training_loss, compute_view_objective_losses, train_task)
from multi_lane.track_a.historical_feature_fusion import ARMS, modes, training_arguments, validate_arguments, validate_config
from multi_lane.track_a.run_historical_feature_fusion import evaluation_job


class HistoricalFeatureFusionTest(unittest.TestCase):
    def test_evaluation_uses_old_worker_count_and_cohort(self):
        args = SimpleNamespace(data_root=Path('data'), clip_checkpoint=Path('clip'), face_manifest_root=Path('face'), smoke=True)
        job = evaluation_job(args, Path('out'), 'THREE_ASL_AUX', 'val', 'historical_val')
        self.assertEqual(job['arguments'][job['arguments'].index('--workers') + 1], '2')
        self.assertNotIn('--full-cohort', job['arguments'])

    def test_serialized_worker_default_is_resolved(self):
        inputs = SimpleNamespace(data_root=Path('data'), clip_checkpoint=Path('clip'), face_manifest_root=Path('face'), smoke=True)
        args = parse_args(training_arguments(inputs, Path('output/demo'), 'THREE_ASL_AUX'))
        self.assertIsNone(args.eval_workers)
        config = dict(vars(args), eval_workers=args.workers, learning_rate=.0125, optimizer='Adam_reset_per_task',
            adapter_task_initialization='independent', amp=True, tf32=True, fixed_view_sampler_protocol=None)
        self.assertEqual(validate_config(config), 'THREE_ASL_AUX')
        config['eval_workers'] = None
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_locked_recipes_preserve_old_numerics_and_exclude_other_modules(self):
        for smoke in [True, False]:
            inputs = SimpleNamespace(data_root=Path('data'), clip_checkpoint=Path('clip'),
                face_manifest_root=Path('face'), smoke=smoke)
            for arm in ARMS:
                flags = training_arguments(inputs, Path('output/demo'), arm)
                args = parse_args(flags)
                validate_arguments(args)
                self.assertEqual(args.workers, 2)
                self.assertEqual(args.amp_initial_scale, 65536.)
                self.assertEqual(args.amp_growth_interval, 2000)
                self.assertFalse(args.strong_full_asl or args.moe_paired_audit)
                self.assertNotIn('--full-cohort', flags)
                args.cross_view_mode = 'task_full'
                with self.assertRaises(ValueError):
                    validate_arguments(args)

    def test_real_training_routes_auxiliary_bce_and_asl_to_the_correct_parameters(self):
        torch.manual_seed(12)
        model = MultiLaneModel(fixtures.FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1, view_fusion='fixed_three_view', adapter_mode='image_token',
            adapter_layer_indices=(0,), adapter_bottleneck_dim=3)
        model.activate_task(0)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.startswith('adapter_bank.'):
                    parameter.normal_(0, .15)
        reference = copy.deepcopy(model)
        images = dict(full=torch.randn(3, 3, 4, 4), person=torch.randn(3, 3, 4, 4),
            face=torch.randn(3, 3, 4, 4), face_reliable=torch.tensor([True, False, True]))
        targets = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        logits, views, _ = reference.current_all_logits_with_view_features(images)
        bce = compute_training_loss(logits, targets, [0, 1], 1., 'legacy_full_zero')
        asl = compute_asymmetric_training_loss(logits, targets, [0, 1], 1., 'legacy_full_zero', 9.8, 0., .05, 1e-8)
        view_bce = compute_view_objective_losses(views, images, targets, [0, 1], 1., 'legacy_full_zero', 'bce')
        view_asl = compute_view_objective_losses(views, images, targets, [0, 1], 1., 'legacy_full_zero', 'asl')
        bce = bce + .1 * torch.stack(list(view_bce.values())).mean()
        asl = asl + .1 * torch.stack(list(view_asl.values())).mean()
        main, adapter, _ = build_optimizer_groups(reference, 0., adapter_learning_rate=.0004, adapter_weight_decay=0.)
        gradients = torch.autograd.grad(bce, main, retain_graph=True) + torch.autograd.grad(asl, adapter)
        expected = {id(p): g for p, g in zip((*main, *adapter), gradients)}
        expected_named = {name: expected[id(p)] for name, p in reference.named_parameters() if id(p) in expected}
        with patch('multi_lane.track_a.runner.TASK_SIZES', (2, 1)):
            history = train_task(model, [(images, targets)], None, torch.device('cpu'), 0, 1, .0125, 0., 1., False,
                adapter_learning_rate=.0004, adapter_weight_decay=0., loss_routing='adapter_asl',
                view_auxiliary_loss_weight=.1, loss_mode='legacy_full_zero')
        self.assertAlmostEqual(history[0]['model_objective_loss'], float(bce), places=7)
        self.assertAlmostEqual(history[0]['adapter_objective_loss'], float(asl), places=7)
        for name, parameter in model.named_parameters():
            if name in expected_named:
                self.assertTrue(torch.equal(parameter.grad, expected_named[name]), name)
        self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))

    def test_unreliable_face_cannot_enter_auxiliary_objectives(self):
        y = torch.tensor([[1., 0.], [0., 1.]])
        values = dict(full=torch.randn(2, 2), person=torch.randn(2, 2), face=torch.randn(2, 2))
        images = dict(face_reliable=torch.tensor([True, False]))
        for objective in ['bce', 'asl']:
            a = compute_view_objective_losses(values, images, y, [0, 1], 1., 'legacy_full_zero', objective)
            changed = {**values, 'face': values['face'].clone()}
            changed['face'][1] = 1000
            b = compute_view_objective_losses(changed, images, y, [0, 1], 1., 'legacy_full_zero', objective)
            self.assertTrue(torch.equal(a['face'], b['face']))


if __name__ == '__main__':
    unittest.main()
