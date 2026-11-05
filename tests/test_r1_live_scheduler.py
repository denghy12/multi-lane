import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from scripts.emotic.run_r1_late_parallel_scheduler import info,priority,startup_reserve,measured_memory


class LiveSchedulerTests(unittest.TestCase):
    def test_patch_training_has_priority_over_ready_validation(self):
        names=['layer9_image_cls_face_validation','layer9_patch_face_validation','layer9_patch_person','layer9_patch_face']
        self.assertEqual([j['name'] for j in sorted([{'name':n} for n in names],key=priority)][:2],
                         ['layer9_patch_person','layer9_patch_face'])

    def test_startup_reservation_prevents_double_admission(self):
        record=dict(steady=False,job={'command':['python','train']},started=time.time(),memory_mib=4816)
        self.assertEqual(startup_reserve(record),4816)
        self.assertLess(6500-startup_reserve(record),4816+512)
        record['steady']=True;self.assertEqual(startup_reserve(record),0)

    def test_smoke_based_budget_includes_cuda_context(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'preflight';p.mkdir()
            (p/'patch_face.json').write_text(json.dumps({'smoke_train_peak_reserved_mib':4048}))
            self.assertEqual(measured_memory({'name':'layer9_patch_face'},Path(d)),4816)

    @unittest.skipUnless(Path('/proc/self/stat').exists(),'Linux procfs required')
    def test_stopping_dispatcher_does_not_stop_child_and_exit_code_can_be_adopted(self):
        code='import subprocess,sys,time;p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(.2);raise SystemExit(3)"]);print(p.pid,flush=True);time.sleep(20)'
        parent=subprocess.Popen([sys.executable,'-u','-c',code],stdout=subprocess.PIPE,text=True)
        try:
            child=int(parent.stdout.readline());old=info(child)
            os.kill(parent.pid,signal.SIGSTOP)
            for _ in range(100):
                state=info(child)
                if state['state']=='Z':break
                time.sleep(.02)
            self.assertEqual(state['start'],old['start']);self.assertEqual(state['exit_code'],3)
            self.assertEqual(info(parent.pid)['state'],'T')
        finally:
            os.kill(parent.pid,signal.SIGKILL);parent.wait();parent.stdout.close()


if __name__=='__main__':unittest.main()
