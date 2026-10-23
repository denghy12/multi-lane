import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('gpu0_handoff', Path(__file__).resolve().parents[1] /
                                             'scripts/emotic/resume_strong_full_asl_gpu0.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
recover_spec = importlib.util.spec_from_file_location('gpu0_recovery', Path(__file__).resolve().parents[1] /
                                                     'scripts/emotic/recover_strong_full_asl_gpu0.py')
recovery = importlib.util.module_from_spec(recover_spec)
recover_spec.loader.exec_module(recovery)


class HandoffTest(unittest.TestCase):
    def test_recovery_keeps_training_flags_and_only_changes_output_and_boundary(self):
        original = ['--output-root', 'old/train', '--train-batch-size', '64', '--loss-routing', 'adapter_asl']
        resumed = recovery.recovery_arguments(original, Path('new/train'), Path('old/task5.pth'))
        self.assertEqual(original[1], 'old/train')
        self.assertEqual(resumed, ['--output-root', 'new/train', '--train-batch-size', '64',
                                  '--loss-routing', 'adapter_asl', '--resume-spatial-boundary', 'old/task5.pth'])

    def test_five_training_jobs_can_share_gpu0_but_evaluation_is_limited(self):
        training = [(None, dict(name='T/train'))] * 4
        self.assertTrue(recovery.eligible(dict(name='new/train'), training, 7000))
        self.assertFalse(recovery.eligible(dict(name='new/train'), training, 6000))
        self.assertFalse(recovery.eligible(dict(name='new/train'), training + training[:1], 20000))
        evaluation = [(None, dict(name='T/full_val'))] * 2
        self.assertFalse(recovery.eligible(dict(name='new/full_test'), evaluation, 20000))

    def test_live_training_is_never_reported_complete_from_old_marker(self):
        r = dict(name='T_ASL/train', adopted_start_ticks='123')
        self.assertIsNone(module.adopted_exit(r, dict(start='123', state='R'),
                                              'MULTI_LANE_TRACK_A_COMPLETE'))
        self.assertEqual(module.adopted_exit(r, None, 'MULTI_LANE_TRACK_A_COMPLETE'), 0)
        self.assertEqual(module.adopted_exit(r, dict(start='123', state='Z'), ''), 1)
        self.assertEqual(module.adopted_exit(r, None, 'Traceback (most recent call last)\nMULTI_LANE_TRACK_A_COMPLETE'), 1)

    def test_gpu0_waits_for_its_training_and_never_allocates_other_cards(self):
        self.assertFalse(module.gpu0_eligible([dict(name='T_ASL_BOTH/train', gpu=0)], 20000))
        self.assertTrue(module.gpu0_eligible([dict(name='T_ASL_POS/train', gpu=7)], 20000))
        self.assertFalse(module.gpu0_eligible([], 6499))
        self.assertTrue(module.gpu0_eligible([dict(name='T_ASL/full_val', gpu=0)], 20000))
        self.assertFalse(module.gpu0_eligible([dict(name='T_ASL/full_val', gpu=0),
                                              dict(name='T_ASL/full_test', gpu=0)], 20000))


if __name__ == '__main__':
    unittest.main()
