import copy
import unittest
from unittest.mock import patch
import torch
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.module_stacking import ARMS, modes, runtime_bypass, factorial_contrasts, validate_arguments
from multi_lane.track_a.runner import build_optimizer_groups, compact_model_state_dict, parse_args
from multi_lane.track_a.spatial_resume import restore_boundary_model
from multi_lane.track_a.export_compact_test_scores import restore_compact_model_state
import test_track_a_reproduction as fixtures


class ModuleStackingTest(unittest.TestCase):
    def model(self, arm='F111'):
        torch.manual_seed(71)
        visual = fixtures.FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        return MultiLaneModel(visual, (2, 1), num_selectors=2, num_prompts=2,
            num_prompt_layers=1, view_fusion='fixed_three_view', module_stacking=True,
            adapter_bottleneck_dim=3, adapter_layer_indices=(0,),
            cross_view_layer=1, cross_view_dim=4, cross_view_heads=2, **modes(arm))

    def images(self):
        return {**{name: torch.randn(3, 3, 4, 4) for name in ('full', 'person', 'face')},
                'face_reliable': torch.tensor([True, False, True]),
                'spatial_bbox': torch.tensor([[0., 0., .5, 1.], [.5, 0., 1., 1.], [0., 0., 1., 1.]]),
                'spatial_visibility': torch.tensor([1., 1., 0.])}

    def nonzero(self, model):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n.startswith(('adapter_bank.', 'cross_view_bank.', 'spatial_summary_bank.')):
                    p.normal_(0, .35)

    def test_eight_cells_identical_initial_outputs_common_weights_and_rng(self):
        base = self.model('F000'); rng = torch.get_rng_state(); base.activate_task(0)
        images = self.images()
        with torch.no_grad(): expected, ev = base.seen_logits_with_views(images)
        for arm in ARMS:
            model = self.model(arm)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()), arm)
            model.activate_task(0)
            for n, v in base.state_dict().items(): self.assertTrue(torch.equal(v, model.state_dict()[n]), (arm, n))
            with torch.no_grad(): actual, views = model.seen_logits_with_views(images)
            self.assertTrue(torch.equal(expected, actual), arm)
            self.assertTrue(all(torch.equal(ev[n], v) for n, v in views.items()), arm)

    def test_all_active_parameters_covered_once_and_receive_finite_gradients(self):
        for arm in ARMS:
            model = self.model(arm); model.activate_task(0)
            groups = build_optimizer_groups(model, 0, adapter_learning_rate=.0004)[2]
            listed = [p for g in groups for p in g['params']]
            self.assertEqual(len(listed), len({id(p) for p in listed}))
            self.assertEqual({id(p) for p in listed}, {id(p) for p in model.optimizer_parameters()})
            optimizer = torch.optim.Adam(groups, lr=.01)
            for _ in range(2):
                optimizer.zero_grad(); model.current_logits(self.images()).square().sum().backward(); optimizer.step()
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.assertIsNotNone(p.grad, (arm, n)); self.assertTrue(torch.isfinite(p.grad).all(), (arm, n))
            if model.spatial_summary_bank is not None:
                self.assertGreater(model.spatial_summary_bank.tasks[0].grad.abs().sum(), 0)
                self.assertIsNone(model.spatial_summary_bank.tasks[1].grad)
            self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))

    def test_spatial_bias_changes_adapter_path_and_propagates_to_person_via_current_full_cache(self):
        model = self.model(); model.activate_task(0); self.nonzero(model); images = self.images()
        with torch.no_grad():
            on, views = model.seen_logits_with_views(images)
            runtime_bypass(model, ['C']); off, other = model.seen_logits_with_views(images)
        self.assertFalse(torch.equal(views['full'], other['full']))
        self.assertFalse(torch.equal(views['person'], other['person']))
        self.assertTrue(torch.equal(views['face'], other['face']))
        self.assertFalse(torch.equal(on, off))

    def test_person_gradient_reaches_full_geometry_and_no_stale_cache(self):
        model = self.model(); model.activate_task(0); self.nonzero(model)
        _, features = model.encode_lanes_with_views(self.images(), False)
        features['person'].square().sum().backward()
        self.assertGreater(model.spatial_summary_bank.tasks[0].grad.abs().sum(), 0)
        model.zero_grad(set_to_none=True); runtime_bypass(model, ['B'])
        _, features = model.encode_lanes_with_views(self.images(), False)
        features['person'].square().sum().backward()
        self.assertTrue(model.spatial_summary_bank.tasks[0].grad is None or
                        model.spatial_summary_bank.tasks[0].grad.abs().sum() == 0)

    def test_full_bypass_uses_original_values_and_reenables_deterministically(self):
        base = self.model('F000'); base.activate_task(0)
        model = self.model(); model.activate_task(0); self.nonzero(model); images = self.images()
        with torch.no_grad():
            on = model.seen_logits(images)
            runtime_bypass(model, ['A', 'B', 'C'])
            self.assertTrue(torch.equal(base.seen_logits(images), model.seen_logits(images)))
            runtime_bypass(model, [])
            self.assertTrue(torch.equal(on, model.seen_logits(images)))

    def test_adapter_never_writes_back_into_frozen_stream(self):
        model = self.model(); model.activate_task(0); self.nonzero(model); images = self.images()
        frozen = {n: v.clone() for n, v in model.visual_encoder.state_dict().items()}
        streams = []
        original = model._lane_block
        def capture(block, image_tokens, *args, **kwargs):
            streams.append(image_tokens.detach().clone()); return original(block, image_tokens, *args, **kwargs)
        with patch.object(model, '_lane_block', side_effect=capture): model.seen_logits(images)
        before = streams[:]; streams.clear(); runtime_bypass(model, ['A'])
        with patch.object(model, '_lane_block', side_effect=capture): model.seen_logits(images)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, streams)))
        self.assertTrue(all(torch.equal(v, model.visual_encoder.state_dict()[n]) for n, v in frozen.items()))

    def test_task_freezing_compact_and_complete_boundary_restore(self):
        model = self.model(); model.activate_task(0); self.nonzero(model); images = self.images()
        before = copy.deepcopy(compact_model_state_dict(model))
        model.activate_task(1)
        optimizer = torch.optim.Adam(build_optimizer_groups(model, 0, adapter_learning_rate=.0004)[2], lr=.01)
        optimizer.zero_grad(); model.current_logits(images).square().sum().backward(); optimizer.step()
        after = compact_model_state_dict(model)
        for n in before:
            if n.startswith(('adapter_bank.task_adapters.0.', 'cross_view_bank.tasks.0.', 'spatial_summary_bank.tasks.0')):
                self.assertTrue(torch.equal(before[n], after[n]), n)
        self.assertTrue(torch.equal(before['selectors'][0], after['selectors'][0]))
        self.assertTrue(torch.equal(before['head.weight'][:2], after['head.weight'][:2]))
        other = self.model()
        restore_compact_model_state(other, dict(schema_version=1, task_id=1, source_git={'commit':'unit'}, model=after), 1, {'commit':'unit'})
        with torch.no_grad(): self.assertTrue(torch.equal(model.seen_logits(images), other.seen_logits(images)))
        complete = self.model()
        restore_boundary_model(complete, dict(task_id=1, model=after, model_training=False,
            requires_grad={n:p.requires_grad for n,p in model.named_parameters()}))
        self.assertTrue(all(torch.equal(v, compact_model_state_dict(complete)[n]) for n,v in after.items()))

    def test_strict_profile_and_interactions(self):
        args = parse_args(['--seed','0','--data-root','data','--clip-checkpoint','clip','--output-root','out',
            '--module-stacking','--moe-paired-audit','--view-fusion','fixed_three_view',
            '--workers','0','--eval-workers','0','--calibration-fraction','.1',
            '--view-auxiliary-loss-weight','.1','--save-pre-evaluation-recovery',
            '--amp-initial-scale','1024','--amp-growth-interval','1000000000',
            '--person-transform-mode','letterbox','--person-crop-margin','.15',
            '--reporting-split','val',
            '--skip-validation-eval','--save-compact-checkpoints','--adapter-layer-indices','1',
            '--adapter-bottleneck-dim','32','--adapter-learning-rate','.0004'])
        validate_arguments(args)
        args.cross_view_mode = 'person_full'
        with self.assertRaises(ValueError): validate_arguments(args)
        linear = {arm: 30 + int(arm[1]) + 2*int(arm[2]) + 3*int(arm[3]) for arm in ARMS}
        contrasts = factorial_contrasts(linear)
        self.assertEqual(contrasts['ABC'], 0); self.assertEqual(contrasts['AB_given_C0'], 0)
        self.assertEqual(contrasts['C_given_AB'], 3)
        with self.assertRaises(ValueError): factorial_contrasts({'F000':30})


if __name__ == '__main__': unittest.main()
