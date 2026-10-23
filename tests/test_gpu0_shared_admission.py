import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

path=Path(__file__).resolve().parents[1]/'tools/gpu0_shared_admission.py'
spec=importlib.util.spec_from_file_location('admission',path)
admission=importlib.util.module_from_spec(spec);spec.loader.exec_module(admission)


class AdmissionTest(unittest.TestCase):
    def test_legacy_reservation_allows_three_new_jobs_but_not_four(self):
        self.assertTrue(admission.fits(24564,20000,2,True))
        self.assertFalse(admission.fits(24564,20000,3,True))

    def test_after_legacy_finishes_allows_five_total_not_six(self):
        self.assertTrue(admission.fits(24564,10000,4,False))
        self.assertFalse(admission.fits(24564,10000,5,False))

    def test_actual_free_memory_guard_survives_foreign_allocation(self):
        self.assertFalse(admission.fits(24564,6499,0,False))
        self.assertTrue(admission.fits(24564,6500,0,False))

    def test_current_process_identity_and_dead_pid(self):
        self.assertIsNotNone(admission.identity(__import__('os').getpid()))
        self.assertIsNone(admission.identity(999999999))

    def test_reservation_precedes_next_start_even_if_gpu_allocation_is_delayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy=admission.SharedAdmission(tmp,'context',123,'old')
            env={}; fake=type('Child',(),{'pid':999})()
            def identity(pid):return 'old' if pid==123 else 'live'
            with patch.object(admission,'identity',side_effect=identity),patch.object(admission,'gpu_memory',return_value=(24564,20000)),patch.object(admission.subprocess,'Popen',return_value=fake) as popen:
                for index in range(3):
                    child=policy.acquire(str(index),['unit'],Path(tmp)/(str(index)+'.log'),env)
                    self.assertIsNotNone(child)
                self.assertIsNone(policy.acquire('fourth',['unit'],Path(tmp)/'fourth.log',env))
                self.assertEqual(popen.call_count,3)
            state=json.loads((Path(tmp)/'admission.json').read_text())
            self.assertEqual(len(state['leases']),3)

    def test_release_returns_budget_and_keeps_audit_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy=admission.SharedAdmission(tmp,'context',123,'old')
            admission.write(Path(tmp)/'admission.json',dict(leases={'x':dict(pid=1)},requests={},history=[]))
            policy.release('x',0)
            value=policy.state();self.assertFalse(value['leases']);self.assertEqual(value['history'][0]['exit'],0)

    def test_initialization_and_smoke_take_priority_over_new_formal_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy=admission.SharedAdmission(tmp,'context',123,'old');policy.phase='formal'
            admission.write(Path(tmp)/'admission.json',dict(leases={},history=[],requests={
                'regional/init':dict(family='regional',driver_pid=77,driver_start='live',phase='preflight',submitted_unix=1)}))
            with patch.object(admission,'identity',return_value='live'),patch.object(admission,'gpu_memory',return_value=(24564,20000)),patch.object(admission.subprocess,'Popen') as popen:
                self.assertIsNone(policy.acquire('context/formal',['unit'],Path(tmp)/'x.log',{}))
                popen.assert_not_called()

    def test_finished_leases_are_pruned_without_killing_any_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy=admission.SharedAdmission(tmp,'regional',123,'old')
            admission.write(Path(tmp)/'admission.json',dict(leases={'dead':dict(pid=999,start_ticks='old')},history=[],requests={}))
            with patch.object(admission,'identity',side_effect=lambda pid:None if pid in (999,123) else 'live'),patch.object(admission,'gpu_memory',return_value=(24564,20000)),patch.object(admission.subprocess,'Popen',return_value=type('Child',(),{'pid':777})()):
                self.assertIsNotNone(policy.acquire('new',['unit'],Path(tmp)/'new.log',{}))
            self.assertNotIn('dead',policy.state()['leases'])


if __name__=='__main__':unittest.main()
