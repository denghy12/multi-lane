import copy
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from multi_lane.track_a.fixed_baseline_routes import restore_base, cached_identity, validate_source, build_model, source_dataset
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.post_task_calibration import fit_cached_routes, base_hash
from multi_lane.track_a.runner import compact_model_state_dict
from test_track_a_reproduction import FakeVisual


class FixedBaselineRoutesTest(unittest.TestCase):
    def test_dataset_factory_resolves_string_paths_and_restores_null_face_defaults(self):
        config = json.loads((Path(__file__).parent / 'fixtures/protected_full_runner_config.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'EMOTIC'; root.mkdir()
            (root / 'CVPR17_Annotations.mat').touch()
            with patch('multi_lane.track_a.fixed_baseline_routes.EMOTIC') as factory:
                for split in ('train', 'val', 'test'):
                    source_dataset(config, str(root), './face_manifest', split)
                    args, kwargs = factory.call_args
                    self.assertEqual(args[0], directory)
                    self.assertEqual(kwargs['train'], split == 'train')
                    for name in ('face_crop_margin', 'face_min_training_short_side', 'face_min_training_score'):
                        self.assertNotIn(name, kwargs)
                    if split != 'train':
                        self.assertEqual(kwargs['eval_splits'], (split,))

    def test_real_serialized_config_handles_inactive_null_options(self):
        config = json.loads((Path(__file__).parent / 'fixtures/protected_full_runner_config.json').read_text())
        config.update(num_prompt_layers=1, prompt_private_layers=1, parax_layer_indices=[0])
        for fit in (False, True):
            model = build_model(config, FakeVisual(), frozen_pool=fit)
            self.assertEqual(model.view_fusion, 'fixed_three_view')

    def models(self):
        args = dict(task_sizes=(2, 1), num_selectors=2, num_prompts=2,
                    num_prompt_layers=1, view_fusion='fixed_three_view')
        base = MultiLaneModel(FakeVisual(), **args)
        model = MultiLaneModel(copy.deepcopy(base.visual_encoder), **args,
             parax_mode='post_task_protected_frozen', parax_rank=2, parax_num_experts=4,
             parax_layer_indices=(0,), parax_initialization='zero_output',
             parax_output_scale_mode='fixed', parax_residual_scale=1., parax_smooth_ratio_bound=.02)
        return base, model

    def test_restore_keeps_trained_current_path_and_bank(self):
        base, model = self.models()
        base.activate_task(0); base.activate_task(1)
        with torch.no_grad():
            base.selectors[1].fill_(7.)
            base.prompts[0][:, 1].fill_(3.)
            model.parax_bank.expert_a[0].fill_(2.)
        payload = dict(schema_version=1, task_id=1, source_git={'commit': 'x'}, model=compact_model_state_dict(base))
        restore_base(model, payload, 1, {'commit': 'x'})
        self.assertTrue(torch.equal(base.selectors, model.selectors))
        self.assertTrue(torch.equal(base.prompts[0], model.prompts[0]))
        self.assertTrue(bool((model.parax_bank.expert_a[0] == 2.).all()))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        bad = copy.deepcopy(payload); del bad['model']['head.bias']
        with self.assertRaisesRegex(ValueError, 'keys'):
            restore_base(model, bad, 1, {'commit': 'x'})

    def test_static_is_constant_but_learnable_and_old_pool_stays_frozen(self):
        _, model = self.models()
        model.restore_task(0); bank = model.parax_bank
        bank.router_input_mode = 'constant'
        x = torch.randn(6, 1, model.output_dim)
        a = bank._router_gates('0_0', torch.randn(6, model.output_dim), 0)
        b = bank._router_gates('0_0', torch.randn(6, model.output_dim), 0)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(a, a[:1].expand_as(a), atol=0, rtol=0)
        cache = {'source_split': 'train', 'task_id': 0,
                 'features': {v: x.clone() for v in ('full', 'person', 'face')},
                 'targets': torch.randint(0,2,(6,2)).float(), 'face_reliable': torch.ones(6, dtype=torch.bool)}
        model.requires_grad_(False)
        cached_identity(model, cache)
        before = base_hash(model)
        report = fit_cached_routes(model, cache, (0,1), torch.device('cpu'), 5, .01, 3, consistency_weight=0.)
        self.assertEqual(before, base_hash(model))
        self.assertGreater(report['history'][-1]['component_gradient_rms_norm_before_clip']['router'], 0.)
        bank.seal_task(0)
        old = {k:v.clone() for k,v in bank.protected_state(1).items()}
        model.restore_task(1)
        cache['task_id']=1; cache['targets']=torch.randint(0,2,(6,1)).float()
        cached_identity(model, cache)
        fit_cached_routes(model, cache, (2,), torch.device('cpu'), 2, .01, 3, consistency_weight=0.)
        for k,v in old.items():
            self.assertTrue(torch.equal(v, bank.protected_state(1)[k]), k)

    def test_reject_joint_source_and_validation_cache(self):
        cfg = dict(training_protocol='incremental', view_fusion='fixed_three_view',
                   parax_mode='disabled', adapter_mode='disabled', calibration_fraction=0.)
        validate_source(cfg, dict(status='complete',config=cfg), True)
        bad = dict(cfg, parax_mode='post_task_protected_frozen')
        with self.assertRaisesRegex(ValueError, 'actual unmodified baseline'):
            validate_source(bad, dict(status='complete',config=bad), True)
        _, model = self.models(); model.restore_task(0)
        with self.assertRaisesRegex(ValueError, 'train features'):
            fit_cached_routes(model, {'source_split':'val','task_id':0}, (0,1), torch.device('cpu'), 1,.01)

if __name__ == '__main__':
    unittest.main()
