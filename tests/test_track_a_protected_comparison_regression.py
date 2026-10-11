"""Exercise the whole comparator against the config schema actually emitted by runner.

The checked-in config comes from the 20261010 eight-task smoke at eb04972;
only its three machine-specific input paths are replaced by fixture paths.
Scores/audits below are small synthetic paired records, not scientific results.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from multi_lane.track_a.protected_parax_compare import compare_seeds, METHODS
from scripts.emotic.launch_protected_parax_full_evaluation import run_comparison, wait_for_smoke, verify_reaudit_source


class FullComparisonRegressionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'source'
        self.reports = Path(self.temp.name)/'reports'
        self.config = json.loads((Path(__file__).parent/'fixtures/protected_full_runner_config.json').read_text())
        self.assertIn('parameter_group_loss_routing', self.config)
        self.assertNotIn('loss_routing', self.config)
        rows = [{'task_id': t, 'mAP': 100., 'per_class_ap': [100.]*(5+3*t)} for t in range(8)]
        metrics = {'final_mAP': 100., 'average_mAP': 100., 'forgetting': 0.}
        for method in METHODS:
            path = self.root/'seed0'/method
            path.mkdir(parents=True)
            summary = {'seed': 0, 'status': 'complete', 'config': copy.deepcopy(self.config),
                       'completed_optimizer_updates': 16, 'metrics': metrics, 'test_metrics': metrics,
                       'task_metrics': rows, 'test_task_metrics': rows,
                       'residual_disabled_metrics': metrics, 'residual_disabled_test_metrics': metrics,
                       'residual_disabled_task_metrics': rows, 'residual_disabled_test_task_metrics': rows}
            self.write(path/'seed_summary.json', summary)
            self.write(path/'training_history.json', {str(t): [{'skipped_optimizer_steps': 0,
                'parax_grad_finite': 1., 'parax_full_residual_ratio': .001}] for t in range(8)})
            self.write(path/'paired_protocol_audit.json', {'frozen_visual_unchanged': True,
                'initial_selector_prompt_classifier_sha256': self.config['initial_selector_prompt_classifier_sha256'],
                'peak_reserved_mib': 100., 'tasks': {str(t): {
                    'train_instances': 4, 'first_train_batch_ids_sha256': 'paired-first',
                    'train_sample_ids_sha256': 'paired-train', 'validation_sample_ids_sha256': 'paired-val',
                    'reporting_sample_ids_sha256': 'paired-val', 'test_loaded': True,
                    'test_sample_ids_sha256': 'paired-test'} for t in range(8)}})
            self.write(path/'protected_route_audit.json', {'tasks': {str(t): {
                'protected_tensors_unchanged': True, 'identity_audit': {'passed': True, 'schema_version': 2},
                'old_task_anchor_drift': {}} for t in range(8)}})
            for split in ('val', 'test'):
                folder = path/f'{split}_scores'
                folder.mkdir()
                for t in range(8):
                    n = 5+3*t
                    labels = np.tile(np.array([[1.], [0.], [1.], [0.]]), (1, n))
                    probabilities = .1+.8*labels
                    np.savez(folder/f'task{t}.npz', sample_ids=np.array(['a', 'b', 'c', 'd']),
                             class_indices=np.arange(n), targets=labels,
                             probabilities=probabilities, logits=probabilities)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value)+'\n')

    def mutate_summary(self, method, function):
        path = self.root/'seed0'/method/'seed_summary.json'
        value = json.loads(path.read_text())
        function(value)
        self.write(path, value)

    def compare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return compare_seeds(self.root, [0], self.reports)

    def test_real_runner_config_completes_all_eight_tasks_and_both_splits(self):
        result = self.compare()
        self.assertEqual(set(result['methods']), set(METHODS))
        self.assertEqual(result['methods']['frozen_pool']['test']['final_mAP']['mean'], 100.)
        detail = json.loads((self.reports/'seed0/comparison.json').read_text())
        self.assertEqual(set(detail['reuse_old']['fixed_old_cohorts']['test']), set(map(str, range(7))))
        self.assertEqual(len(detail['reuse_old']['test_residual_ablation']), 8)
        self.assertTrue((self.reports/'multiseed_comparison.csv').exists())
        self.assertFalse((self.root/'multiseed_comparison.json').exists())
        self.assertFalse((self.root/'seed0/comparison.json').exists())

    def test_missing_real_loss_routing_field_is_not_silently_ignored(self):
        self.mutate_summary('baseline', lambda s: s['config'].pop('parameter_group_loss_routing'))
        with self.assertRaisesRegex(RuntimeError, 'missing required configuration parameter_group_loss_routing'):
            self.compare()

    def test_different_loss_routing_is_rejected(self):
        self.mutate_summary('reuse_old', lambda s: s['config'].update(parameter_group_loss_routing='both_asl'))
        with self.assertRaisesRegex(RuntimeError, 'unpaired configuration parameter_group_loss_routing'):
            self.compare()

    def test_missing_test_endpoint_is_rejected(self):
        self.mutate_summary('fresh_only', lambda s: s['test_task_metrics'].pop())
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            self.compare()

    def test_missing_saved_test_scores_is_rejected(self):
        (self.root/'seed0/frozen_pool/test_scores/task7.npz').unlink()
        with self.assertRaises(FileNotFoundError):
            self.compare()

    def test_incomplete_optimizer_budget_is_rejected(self):
        self.mutate_summary('fresh_only', lambda s: s.update(completed_optimizer_updates=15))
        with self.assertRaisesRegex(RuntimeError, 'incomplete optimizer budget'):
            self.compare()

    def test_epoch_budget_path_uses_full_training_instance_counts(self):
        for method in METHODS:
            self.mutate_summary(method, lambda s: s['config'].update(
                epochs_per_task=30, optimizer_updates_per_task=0))
            self.mutate_summary(method, lambda s: s.update(completed_optimizer_updates=240))
        self.compare()

    def test_comparison_failure_has_specific_record_and_blocks_prerequisite(self):
        control = self.reports/'control'

        def failed(command, **kwargs):
            kwargs['stdout'].write("Traceback\nRuntimeError: fixture bad loss routing\n")
            raise subprocess.CalledProcessError(1, command)

        with mock.patch('scripts.emotic.launch_protected_parax_full_evaluation.subprocess.run', side_effect=failed):
            with self.assertRaisesRegex(RuntimeError, 'fixture bad loss routing'):
                run_comparison(self.root, [0], control)
        record = json.loads((control/'failed.json').read_text())
        self.assertEqual(record['phase'], 'comparison')
        self.assertEqual(record['exit_code'], 1)
        self.assertIn('fixture bad loss routing', record['error_summary'])
        self.assertFalse((control/'completed.txt').exists())
        self.assertFalse((control/'comparison_passed.json').exists())
        with self.assertRaisesRegex(RuntimeError, 'fixture bad loss routing'):
            wait_for_smoke(self.reports, None, 0)

    def test_reaudit_records_source_and_auditor_without_marking_old_batch_complete(self):
        control = self.reports/'control'
        with mock.patch('scripts.emotic.launch_protected_parax_full_evaluation.subprocess.run') as run, \
             mock.patch('scripts.emotic.launch_protected_parax_full_evaluation.subprocess.check_output', return_value='new-auditor\n'):
            record = run_comparison(self.root, [0], control, prerequisite_source=self.root)
        self.assertEqual(record['auditor_commit'], 'new-auditor')
        self.assertEqual(record['training_commits'], [self.config['git']['commit']])
        command = run.call_args.args[0]
        self.assertIn('--report-root', command)
        self.assertFalse((self.root/'control/completed.txt').exists())

    def test_reaudit_allows_comparator_only_changes(self):
        with mock.patch('scripts.emotic.launch_protected_parax_full_evaluation.subprocess.check_output',
                        return_value='multi_lane/track_a/protected_parax_compare.py\n'):
            self.assertTrue(verify_reaudit_source(self.root)['training_paths_unchanged'])

    def test_reaudit_refuses_model_changes_without_new_smoke(self):
        with mock.patch('scripts.emotic.launch_protected_parax_full_evaluation.subprocess.check_output',
                        return_value='multi_lane/track_a/model.py\n'):
            with self.assertRaisesRegex(RuntimeError, 'Training code changed'):
                verify_reaudit_source(self.root)


if __name__ == '__main__':
    unittest.main()
