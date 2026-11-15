import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.emotic.backfill_d9_benchmark import pick
from scripts.emotic import run_d9_benchmark_protocols as run


class BackfillTests(unittest.TestCase):
    def test_background_cards_require_observed_idle_window(self):
        jobs = [('b10c4','person')]
        history = {}
        for t, idle in [(0, True), (20, False), (30, True), (55, True)]:
            self.assertIsNone(pick(jobs, {}, [(0,3000,False),(1,14000,idle)], history, t))
        self.assertEqual(pick(jobs, {}, [(0,3000,False),(1,24000,True)], history, 61), (jobs[0],1))

    def test_only_one_extra_receiver_and_one_full_on_gpu0(self):
        pending = [('b10c4','face'),('joint26','full')]
        existing = {('b10c4','person'):(0,None,None)}
        self.assertEqual(pick(pending, existing, [(0,8000,False)], {}, 50), (pending[1],0))
        self.assertIsNone(pick(pending, existing, [(0,5000,False)], {}, 50))

    def test_duplicate_worker_skips_completed_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = {'seed':1,'protocol':'joint26','view':'person','smoke':False}
            with patch.object(run, 'branch_run', side_effect=lambda *a:run.save(root/'person/complete.json',record)) as train:
                for _ in range(2):
                    run.locked_branch(root,'joint26',1,'person',False)
                train.assert_called_once()
