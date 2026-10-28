import importlib.util
from pathlib import Path
import sys
import unittest

folder=Path(__file__).resolve().parents[1]/'scripts/emotic'
sys.path.insert(0,str(folder))
spec=importlib.util.spec_from_file_location('protected_joint_scheduling',folder/'expand_protected_joint_queue.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ProtectedJointSchedulingTests(unittest.TestCase):
    def test_only_two_unstarted_trainers_are_enqueued(self):
        protocol={k:'../example' for k in ('source_run','data_root','clip_checkpoint','face_manifest_root')}
        snapshot=dict(records=[dict(name=a+'_train') for a in ('C0','C1','C2','C3')])
        jobs=module.pending_train_jobs(protocol,snapshot,Path('/run'))
        self.assertEqual([j['name'] for j in jobs],['C4_train','FULL_ONLY_train'])
        for j in jobs:
            self.assertIn('--historical-reference-root',j['command'])
            self.assertEqual(j['command'][j['command'].index('--tasks')+1],'8')

    def test_six_trainers_fit_policy_but_seventh_does_not(self):
        self.assertTrue(module.eligible(dict(kind='train'),[dict(kind='train')]*5,6500,1500,module.POLICY))
        self.assertFalse(module.eligible(dict(kind='train'),[dict(kind='train')]*6,20000,4000,module.POLICY))
        self.assertFalse(module.eligible(dict(kind='train'),[dict(kind='train')]*5,5999,4000,module.POLICY))

    def test_evaluation_limit_stays_two(self):
        active=[dict(kind='eval')]*2
        self.assertFalse(module.eligible(dict(kind='eval'),active,10000,3000,module.POLICY))
        self.assertTrue(module.eligible(dict(kind='train'),active,10000,3000,module.POLICY))

    def test_active_pid_is_never_completed_from_marker(self):
        record=dict(adopted_start_ticks='17')
        self.assertEqual(module.exit_status(record,dict(start='17',state='R'),True,''),(None,None))

    def test_failure_and_pid_reuse_need_completion_evidence(self):
        record=dict(adopted_start_ticks='17')
        self.assertEqual(module.exit_status(record,dict(start='17',state='Z',wait_status=256),True,'')[0],1)
        self.assertEqual(module.exit_status(record,dict(start='18',state='R'),False,'')[0],1)
        self.assertEqual(module.exit_status(record,None,True,'')[0],0)
        self.assertEqual(module.exit_status(record,None,True,'Traceback (most recent call last)')[0],1)


if __name__=='__main__': unittest.main()
