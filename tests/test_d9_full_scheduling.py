from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.emotic import run_d9_branch_refinement as launch


class FullSchedulingTests(unittest.TestCase):
    def test_priority_and_queue_share_one_training_and_evaluation(self):
        job = next(job for job in launch.JOBS if job.view == 'full')
        first = threading.Event()
        release = threading.Event()

        def complete(job, root, smoke):
            first.set()
            if not release.wait(5):
                raise TimeoutError('Test failed to release Full worker')
            launch.write(job.path(root) / 'completion.json', dict(status='complete', branch=job.name, seed=1))

        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
            root = Path(temp)
            with patch.object(launch, 'run_branch', side_effect=complete) as train, ThreadPoolExecutor(2) as pool:
                priority = pool.submit(launch.worker, job, root, False)
                self.assertTrue(first.wait(5))
                queued = pool.submit(launch.worker, job, root, False)
                release.set()
                priority.result(timeout=5)
                queued.result(timeout=5)
                self.assertEqual(train.call_count, 1)

    def test_wrong_completion_record_is_rejected(self):
        job = next(job for job in launch.JOBS if job.view == 'full')
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
            root = Path(temp)
            launch.write(job.path(root) / 'completion.json', dict(status='complete', branch=job.name, seed=2))
            with self.assertRaises(ValueError), patch.object(launch, 'run_branch') as train:
                launch.worker(job, root, False)
            train.assert_not_called()
