import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('b32_concurrency', Path(__file__).resolve().parents[1] /
    'scripts/emotic/expand_historical_b32_queue.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class B32ConcurrencyTest(unittest.TestCase):
    def test_six_train_slots_are_memory_gated(self):
        active = [dict(kind='train')] * 5
        self.assertTrue(module.eligible(dict(kind='train'), active, 7000, 1800, module.DEFAULT_POLICY))
        self.assertFalse(module.eligible(dict(kind='train'), active * 2, 20000, 4000, module.DEFAULT_POLICY))
        self.assertFalse(module.eligible(dict(kind='train'), active, 5999, 4000, module.DEFAULT_POLICY))
        self.assertFalse(module.eligible(dict(kind='train'), active, 20000, 1023, module.DEFAULT_POLICY))

    def test_evaluations_have_a_separate_prefetch_limit(self):
        active = [dict(kind='eval')] * 4
        self.assertFalse(module.eligible(dict(kind='eval'), active, 20000, 4000, module.DEFAULT_POLICY))
        self.assertTrue(module.eligible(dict(kind='train'), active, 20000, 4000, module.DEFAULT_POLICY))

    def test_policy_can_shrink_without_changing_jobs(self):
        policy = dict(module.DEFAULT_POLICY, max_parallel=4)
        self.assertEqual(module.validate_policy(policy)['max_parallel'], 4)
        self.assertFalse(module.eligible(dict(kind='train'), [dict(kind='train')] * 4, 20000, 4000, policy))

    def test_unsafe_policy_changes_are_rejected(self):
        for changes in (dict(max_parallel=7), dict(minimum_free_gpu_mib=3000),
                        dict(minimum_free_shm_mib=100), dict(launch_interval_seconds=0)):
            with self.assertRaises(ValueError):
                module.validate_policy(dict(module.DEFAULT_POLICY, **changes))

    def test_a_live_process_is_never_finished_from_a_marker(self):
        r = dict(adopted_start_ticks='123')
        self.assertEqual(module.adopted_status(r, dict(start='123', state='R', wait_status=0),
            'MULTI_LANE_TRACK_A_COMPLETE', True), (None, None))

    def test_linux_exit_status_is_preserved(self):
        r = dict(adopted_start_ticks='123')
        self.assertEqual(module.adopted_status(r, dict(start='123', state='Z', wait_status=0),
            'MULTI_LANE_TRACK_A_COMPLETE', True), (0, 'linux_proc_wait_status'))
        self.assertEqual(module.adopted_status(r, dict(start='123', state='Z', wait_status=256),
            'MULTI_LANE_TRACK_A_COMPLETE', True), (1, 'linux_proc_wait_status'))

    def test_missing_process_needs_both_complete_artifacts_and_success_log(self):
        r = dict(adopted_start_ticks='123')
        self.assertEqual(module.adopted_status(r, None, 'MULTI_LANE_TRACK_A_COMPLETE', True),
            (0, 'artifacts_and_log_after_pid_exit'))
        self.assertEqual(module.adopted_status(r, None, 'MULTI_LANE_TRACK_A_COMPLETE', False)[0], 1)
        self.assertEqual(module.adopted_status(r, None, 'Traceback (most recent call last)\nMULTI_LANE_TRACK_A_COMPLETE', True)[0], 1)


if __name__ == '__main__':
    unittest.main()
