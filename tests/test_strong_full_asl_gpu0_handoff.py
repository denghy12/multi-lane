import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('gpu0_handoff', Path(__file__).resolve().parents[1] /
                                             'scripts/emotic/resume_strong_full_asl_gpu0.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HandoffTest(unittest.TestCase):
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
