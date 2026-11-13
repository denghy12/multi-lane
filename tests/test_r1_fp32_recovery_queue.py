"""Queue gates use completed dependencies, safe checkpoints and pinned code."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.emotic import launch_r1_fp32_recovery as queue


def population_result():
    return {split: {method: {"all": {"metrics": [{"task_id": t, "mAP": 40.} for t in range(8)]}}
                    for method in ("fixed", "current_positive", "seen_population")}
            for split in ("val", "test")}


class FP32QueueTests(unittest.TestCase):
    def test_population_marker_alone_does_not_prove_process_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root, status = Path(directory)/"population", Path(directory)/"batch.json"
            queue.write(root/"completion.json", {"status": "complete"})
            queue.write(root/"result.json", population_result())
            for job, batch in [({"status": "running"}, "running"),
                               ({"status": "complete", "exit_code": 0}, "running")]:
                queue.write(status, {"status": batch, "jobs": [{"name": "fusion_population", **job}]})
                self.assertEqual(queue.dependency_state(root, status)[0], "waiting")
            queue.write(status, {"status": "finished_with_failure", "jobs": [
                {"name": "full_extend_recovery", "status": "failed", "exit_code": 1},
                {"name": "fusion_population", "status": "complete", "exit_code": 0}]})
            self.assertEqual(queue.dependency_state(root, status)[0], "ready")

    def test_population_failure_or_incomplete_test_never_releases_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root, status = Path(directory)/"population", Path(directory)/"batch.json"
            queue.write(root/"completion.json", {"status": "complete"})
            queue.write(root/"result.json", population_result())
            job = {"name": "fusion_population", "status": "failed", "exit_code": 1}
            queue.write(status, {"status": "finished_with_failure", "jobs": [job]})
            with self.assertRaises(RuntimeError): queue.dependency_state(root, status)
            job.update(status="complete", exit_code=0)
            queue.write(status, {"status": "complete", "jobs": [job]})
            bad = population_result()
            bad["test"]["seen_population"]["all"]["metrics"].pop()
            queue.write(root/"result.json", bad)
            with self.assertRaises(ValueError): queue.dependency_state(root, status)
            queue.write(root/"result.json", population_result())
            queue.write(root/"abort.json", {"error": "failed"})
            with self.assertRaises(RuntimeError): queue.dependency_state(root, status)

    def test_gpu_memory_measurements_and_requested_gpu_are_explicit(self):
        self.assertFalse(queue.memory_ready(20479, 24576))
        self.assertTrue(queue.memory_ready(20480, 24576))
        for free, total in [(-1, 24576), (26000, 24576), (0, 0)]:
            with self.assertRaises(ValueError): queue.memory_ready(free, total)
        with patch.object(queue.subprocess, "check_output", return_value="22000, 24576\n") as command:
            self.assertEqual(queue.gpu0_memory(), {"free_mib": 22000, "total_mib": 24576})
        self.assertIn("--id=0", command.call_args[0][0])

    def test_source_must_remain_clean_and_at_locked_head(self):
        with patch.object(queue.subprocess, "check_output", side_effect=["locked\n", ""]):
            self.assertEqual(queue.check_source("locked"), "locked")
        with patch.object(queue.subprocess, "check_output", return_value="changed\n"):
            with self.assertRaises(RuntimeError): queue.check_source("locked")
        with patch.object(queue.subprocess, "check_output", side_effect=["locked\n", " M code.py\n"]):
            with self.assertRaises(RuntimeError): queue.check_source("locked")

    def test_probe_must_be_current_fp32_epoch14_and_source_must_be_safe_epoch13(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"recovery"
            source = Path(directory)/"source/recovery/task3/epoch13.pth"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"safe original epoch13")
            source_hash = queue.sha(source)
            target = root/"branches/attention/person/train/epoch_recovery/task3/epoch14.pth"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"successful FP32 epoch14")
            manifest = {"person_precision": "fp32", "smoke": False, "task": 3, "completed_epoch": 13,
                        "source_sha256": source_hash, "source_whole_checkpoint": str(source),
                        "prefix_files": {"recovery/task3/epoch13.pth": source_hash}}
            probe = {"status": "passed", "task": 3, "epoch": 14, "effective_updates": 156,
                     "skipped_updates": 0, "amp": False, "scaler_disabled": True,
                     "source_prefix_unchanged": True, "checkpoint": str(target), "sha256": queue.sha(target)}
            def store(m, p):
                queue.write(root/"recovery_provenance.json", m)
                queue.write(root/"person_epoch_probe.json", p)
            store(manifest, probe)
            with patch.object(queue, "BOUNDARY_SHA", source_hash):
                queue.validate_probe(root)
                for invalid in ("skip", "amp", "failed", "external", "sha", "imported_epoch14", "wrong_boundary"):
                    m, p = copy.deepcopy(manifest), copy.deepcopy(probe)
                    if invalid == "skip": p["skipped_updates"] = 7
                    elif invalid == "amp": p["amp"] = True
                    elif invalid == "failed": p["status"] = "failed"
                    elif invalid == "external": p["checkpoint"] = str(source.parent/"epoch14.pth")
                    elif invalid == "sha": p["sha256"] = "wrong"
                    elif invalid == "imported_epoch14": m["prefix_files"]["recovery/task3/epoch14.pth"] = "old_failed"
                    else: m["completed_epoch"] = 14
                    store(m, p)
                    with self.assertRaises(ValueError): queue.validate_probe(root)
                store(manifest, probe)
                source.write_bytes(b"source changed")
                with self.assertRaises(ValueError): queue.validate_probe(root)

    def test_launch_waits_for_population_and_three_consecutive_gpu_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            args = SimpleNamespace(recovery_root=base/"recovery", population_root=base/"population",
                                   population_batch_status=base/"population_batch.json", queue_root=base/"queue",
                                   logs=base/"logs", poll_seconds=1., max_wait_hours=1., min_free_mib=20480)
            args.recovery_root.mkdir()
            for name in ("completion.json", "numerical_recovery_complete.json"):
                queue.write(args.recovery_root/name, {"status": "complete"})
            queue.write(args.recovery_root/"result.json", population_result())
            dependency = Mock(side_effect=[("waiting", {})] + [("ready", {})] * 5)
            memory = Mock(side_effect=[{"free_mib": f, "total_mib": 24576}
                                       for f in (22000, 10000, 22000, 22000, 22000)])
            process = SimpleNamespace(pid=12345, poll=lambda: 0, wait=lambda: 0)
            def start(command, **kwargs):
                self.assertEqual(dependency.call_count, 6)
                self.assertEqual(memory.call_count, 5)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
                self.assertIn("scripts.emotic.run_r1_full_extend_recovery", command)
                return process
            with patch.object(queue, "check_source", return_value="locked") as source, \
                 patch.object(queue, "validate_probe", return_value={}) as probe, \
                 patch.object(queue, "dependency_state", dependency), \
                 patch.object(queue, "gpu0_memory", memory), \
                 patch.object(queue.time, "sleep"), \
                 patch.object(queue.subprocess, "Popen", side_effect=start) as popen:
                queue.launch(args)
            self.assertEqual(popen.call_count, 1)
            self.assertEqual(probe.call_count, 2)
            self.assertEqual(source.call_args[0], ("locked",))
            self.assertEqual(queue.read(args.queue_root/"status.json")["status"], "complete")


if __name__ == "__main__": unittest.main()
