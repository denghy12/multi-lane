import importlib.util
from pathlib import Path
import sys
import unittest

folder = Path(__file__).resolve().parents[1]/'scripts/emotic'
sys.path.insert(0, str(folder))
spec = importlib.util.spec_from_file_location('person_grounded_scheduling', folder/'expand_person_grounded_queue.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class GroundedSchedulingTests(unittest.TestCase):
    def test_live_trainers_are_not_reenqueued(self):
        protocol = {k: '../example' for k in ('source_run','data_root','clip_checkpoint','face_manifest_root')}
        rows = [dict(name=a+'_train') for a in module.ARMS[:4]]
        jobs = module.pending_train_jobs(protocol, dict(records=rows), Path('/source/run'))
        self.assertEqual([j['arm'] for j in jobs], list(module.ARMS[4:]))
        self.assertEqual(module.pending_train_jobs(protocol, dict(records=rows+jobs), Path('/source/run')), [])

    def test_train_completion_locks_train_calibration_before_evaluation(self):
        known = set(); root = Path('/source/run')
        jobs = module.followup_jobs(module.ARMS[0], 'train', root, known)
        self.assertEqual([j['kind'] for j in jobs], ['calibrate'])
        self.assertNotIn('--split', jobs[0]['command'])
        self.assertEqual(module.followup_jobs(module.ARMS[0], 'train', root, known), [])
        jobs = module.followup_jobs(module.ARMS[0], 'calibrate', root, known)
        self.assertEqual([j['name'] for j in jobs], [module.ARMS[0]+'_val', module.ARMS[0]+'_test'])
        self.assertEqual(module.followup_jobs(module.ARMS[0], 'calibrate', root, known), [])

    def test_six_jobs_and_shared_nontraining_limit(self):
        self.assertTrue(module.eligible(dict(kind='train'), [dict(kind='train')]*5, 6500, 1500, module.POLICY))
        self.assertFalse(module.eligible(dict(kind='train'), [dict(kind='train')]*6, 10000, 4000, module.POLICY))
        active = [dict(kind='calibrate'), dict(kind='eval')]
        self.assertFalse(module.eligible(dict(kind='calibrate'), active, 10000, 4000, module.POLICY))
        self.assertFalse(module.eligible(dict(kind='eval'), active, 10000, 4000, module.POLICY))
        self.assertTrue(module.eligible(dict(kind='train'), active, 10000, 4000, module.POLICY))

    def test_running_or_reused_pid_cannot_be_silently_replaced(self):
        record = dict(adopted_start_ticks='123')
        self.assertEqual(module.exit_status(record, dict(start='123', state='R'), True, ''), (None, None))
        self.assertEqual(module.exit_status(record, dict(start='999', state='R'), False, '')[0], 1)


if __name__ == '__main__': unittest.main()
