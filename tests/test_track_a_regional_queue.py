import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from multi_lane.track_a.run_regional_summary import dependency_ready, training_arguments, run_queue, job
from multi_lane.track_a.runner import parse_args


class RegionalQueueTest(unittest.TestCase):
    def test_waits_for_formal_dependency_and_rejects_failure_or_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(dependency_ready(root))
            (root/'completion.json').write_text(json.dumps(dict(status='complete',smoke=True,tasks=2)))
            with self.assertRaises(ValueError): dependency_ready(root)
            (root/'completion.json').write_text(json.dumps(dict(status='complete',smoke=False,tasks=8)))
            self.assertTrue(dependency_ready(root))
            (root/'failure.json').write_text('{}')
            with self.assertRaises(RuntimeError): dependency_ready(root)

    def test_all_three_profiles_keep_budget_supervision_and_no_other_modules(self):
        args = argparse.Namespace(data_root='data',clip_checkpoint='clip',face_manifest_root='faces')
        for arm,mode in [('baseline','disabled'),('old_bias','target'),('regional','regional')]:
            for smoke in (False,True):
                argv=training_arguments(args,Path('output/unit'),arm,smoke)
                with patch('sys.argv',['runner',*argv]): parsed=parse_args()
                self.assertEqual(parsed.spatial_summary_mode,mode)
                self.assertEqual(parsed.max_tasks,2 if smoke else 8)
                self.assertEqual(parsed.optimizer_updates_per_task,2 if smoke else None)
                self.assertEqual(parsed.view_auxiliary_loss_weight,.1)
                self.assertEqual(parsed.calibration_fraction,.1)
                self.assertEqual(parsed.train_batch_size,64)
                self.assertEqual(parsed.adapter_mode,'disabled')
                self.assertEqual(parsed.cross_view_mode,'disabled')

    def test_memory_waits_and_failure_keeps_pending_without_starting_it(self):
        class Process:
            pid=123
            def poll(self):return 1
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);status=root/'status.json'
            jobs=[job('first','runner',[]),job('second','runner',[])]
            with patch('multi_lane.track_a.run_regional_summary.free_memory',side_effect=[100,6000]), \
                 patch('multi_lane.track_a.run_regional_summary.subprocess.Popen',return_value=Process()) as launch, \
                 patch('multi_lane.track_a.run_regional_summary.time.sleep'):
                with self.assertRaises(RuntimeError):run_queue(jobs,root,status,1)
            self.assertEqual(launch.call_count,1)
            saved=json.loads(status.read_text())
            self.assertEqual(saved['status'],'failed')
            self.assertEqual(saved['pending'][0]['name'],'second')
            env=launch.call_args.kwargs['env']
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0')


if __name__=='__main__':unittest.main()
