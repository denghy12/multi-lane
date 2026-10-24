from pathlib import Path
from types import SimpleNamespace
import unittest

from multi_lane.track_a.runner import parse_args
from multi_lane.track_a.historical_feature_fusion import validate_arguments, validate_config
from multi_lane.track_a.historical_b32_search import experiment_cells, recipe, training_arguments
from multi_lane.track_a.run_historical_b32_search import evaluation_job


class HistoricalB32SearchTest(unittest.TestCase):
    def inputs(self, smoke):
        return SimpleNamespace(data_root=Path('datasets/EMOTIC'), clip_checkpoint=Path('models/clip.pt'),
            face_manifest_root=Path('output/faces'), smoke=smoke)

    def test_all_eighteen_cells_are_declared_single_factor_changes(self):
        cells = experiment_cells()
        self.assertEqual(len(cells), 18)
        self.assertEqual(len(set(cells)), 18)
        base = recipe('THREE_BASE')
        for seed in (0, 3):
            profiles = [p for s, p in cells if s == seed]
            self.assertEqual(len(profiles), 8)
            for profile in profiles:
                if profile not in {'THREE_BASE', 'FULL_BASE'}:
                    self.assertEqual(sum(v != base[k] for k, v in recipe(profile).items()), 1)
        self.assertEqual([p for s, p in cells if s == 1], ['THREE_BASE', 'FULL_BASE'])

    def test_arguments_and_serialized_configs_match_every_cell(self):
        for smoke in (False, True):
            inputs = self.inputs(smoke)
            for seed, profile in experiment_cells(smoke):
                flags = training_arguments(inputs, Path('output/demo'), seed, profile)
                args = parse_args(flags)
                validate_arguments(args)
                self.assertEqual(args.seed, seed)
                self.assertEqual(args.adapter_residual_scale, recipe(profile)['adapter_residual_scale'])
                self.assertEqual(args.person_color_jitter_strength, 0.)
                self.assertEqual(args.person_color_jitter_probability, 0.)
                self.assertTrue(args.skip_validation_eval)
                config = dict(vars(args), eval_workers=2, learning_rate=.0125, optimizer='Adam_reset_per_task',
                    adapter_task_initialization='independent', amp=True, tf32=True,
                    fixed_view_sampler_protocol=None, evaluation_scope='samples_intersect_seen_classes',
                    parameter_group_loss_routing='adapter_asl')
                self.assertEqual(validate_config(config), profile)
                config['fixed_view_sampler_protocol'] = 'seed_plus_1009_times_task'
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_historical_recipe_cannot_silently_drift(self):
        flags = training_arguments(self.inputs(False), Path('output/demo'), 1, 'THREE_BASE')
        for key, value in [('person_color_jitter_strength', .1), ('adapter_residual_scale', .1),
                           ('adapter_learning_rate', .0008), ('cross_view_mode', 'task_full'),
                           ('view_auxiliary_loss_weight', 0.)]:
            args = parse_args(flags); setattr(args, key, value)
            with self.assertRaises(ValueError):
                validate_arguments(args)

    def test_native_flags_only_retain_old_runner_options(self):
        for profile in ('THREE_BASE', 'FULL_BASE'):
            flags = training_arguments(self.inputs(True), Path('output/native'), 1, profile, native=True)
            self.assertNotIn('--historical-feature-fusion', flags)
            self.assertNotIn('--historical-b32-profile', flags)
            self.assertNotIn('--save-pre-evaluation-recovery', flags)
            self.assertIn('--save-compact-checkpoints', flags)

    def test_complete_evaluations_keep_historical_cohort_and_small_worker_pool(self):
        for split in ('val', 'test'):
            job = evaluation_job(self.inputs(False), Path('output/demo'), 'seed3/THREE_BASE', split)
            self.assertNotIn('--full-cohort', job['arguments'])
            self.assertEqual(job['arguments'][job['arguments'].index('--expected-tasks') + 1], '8')
            self.assertEqual(job['arguments'][job['arguments'].index('--workers') + 1], '2')


if __name__ == '__main__':
    unittest.main()
