import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import gpu1_admission as module


class SplitTest(unittest.TestCase):
    def test_four_trainers_allowed_fifth_denied(self):
        leases=[dict(kind='train',budget=4500)]*3
        self.assertTrue(module.can_start(24564,12000,leases,'train'))
        self.assertFalse(module.can_start(24564,12000,leases+leases[:1],'train'))

    def test_evaluators_use_smaller_budget_and_eight_slot_limit(self):
        leases=[dict(kind='eval',budget=2500)]*7
        self.assertTrue(module.can_start(24564,12000,leases,'eval'))
        self.assertFalse(module.can_start(24564,12000,leases+leases[:1],'eval'))

    def test_mixed_memory_budget(self):
        leases=[dict(kind='train',budget=4500)]*4
        self.assertTrue(module.can_start(24564,8000,leases,'eval'))
        self.assertFalse(module.can_start(24564,8000,leases+[dict(kind='eval',budget=2500)],'eval'))

    def test_actual_free_guard(self):
        self.assertFalse(module.can_start(24564,6499,[],'train'))
        self.assertFalse(module.can_start(24564,4499,[],'eval'))

    def test_inherited_running_process_is_not_restarted(self):
        value=module.process_state(os.getpid())
        self.assertIsNone(module.inherited_exit(os.getpid(),value['start']))

    def test_pid_reuse_and_missing_process_rejected(self):
        with self.assertRaises(RuntimeError):module.inherited_exit(os.getpid(),'wrong')
        with self.assertRaises(RuntimeError):module.inherited_exit(999999999,'missing')

    def test_exact_zombie_exit_observation(self):
        child=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)'])
        start=module.process_state(child.pid)['start']
        try:
            for _ in range(100):
                value=module.inherited_exit(child.pid,start)
                if value is not None:break
                time.sleep(.01)
            self.assertEqual(value,7)
        finally:child.wait()

    def test_signal_exit_is_recorded(self):
        with patch.object(module,'process_state',return_value=dict(state='Z',start='x',exit_status=15)):
            self.assertEqual(module.inherited_exit(12,'x'),-15)


if __name__=='__main__':unittest.main()
