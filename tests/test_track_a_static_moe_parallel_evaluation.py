"""Exercise concurrency, immutable sources and the active-training handoff."""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from multi_lane.track_a.evaluate_static_moe import METHODS, evaluate_batch
from multi_lane.track_a.export_compact_test_scores import _sha256
from multi_lane.track_a import static_moe_parallel_evaluation as parallel


class StaticMoEParallelEvaluationTest(unittest.TestCase):
    metadata = dict(commit="evaluator", branch="parallel", tree="tree", dirty=False)

    def sources(self, source):
        return {str(source / "checkpoint"): source.name}

    def completed_result(self, batch, split, lock_path, lock):
        return dict(status="complete", split=split, samples=2397 if split == "val" else 5368,
                    training_performed=False, optimizer_updates=0, evaluation_mode="parallel_fixed_checkpoints",
                    evaluation_source_lock_sha256=_sha256(lock_path), evaluator_git=self.metadata,
                    results={m: dict(source_sha256=lock["source_sha256"][m]) for m in METHODS})

    def test_two_real_workers_start_before_either_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch, logs = Path(tmp) / "batch", Path(tmp) / "logs"
            batch.mkdir(); logs.mkdir()
            barrier = threading.Barrier(2)
            starts = []
            def run(command, **kwargs):
                split = command[command.index("--split") + 1]
                self.assertIn("--parallel-splits", command)
                lock_path = Path(command[command.index("--source-lock") + 1])
                lock = parallel.read(lock_path)
                starts.append(split)
                barrier.wait(timeout=5)
                destination = batch / ("full_" + split)
                destination.mkdir()
                parallel.write(destination / "comparison.json", self.completed_result(batch, split, lock_path, lock))
                return SimpleNamespace(returncode=0)
            state = dict(phases={})
            with patch.object(parallel, "git_metadata", return_value=self.metadata), \
                 patch.object(parallel, "audit_pairing"), \
                 patch.object(parallel, "source_hashes", side_effect=self.sources), \
                 patch.object(parallel.subprocess, "run", side_effect=run), \
                 patch("multi_lane.track_a.static_moe_pipeline.wait_memory"):
                parallel.run_parallel_evaluations(batch, logs, "data", "clip", "face", "0", state, batch / "state.json")
            self.assertEqual(set(starts), {"val", "test"})
            self.assertEqual(state["phases"], dict(val=0, test=0))
            self.assertTrue(state["parallel_sources_verified"])

    def test_either_failure_is_recorded_after_both_workers_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch, logs = Path(tmp) / "batch", Path(tmp) / "logs"
            batch.mkdir(); logs.mkdir()
            def run(command, **kwargs):
                return SimpleNamespace(returncode=2 if command[command.index("--split") + 1] == "test" else 0)
            state = dict(phases={})
            with patch.object(parallel, "git_metadata", return_value=self.metadata), \
                 patch.object(parallel, "audit_pairing"), \
                 patch.object(parallel, "source_hashes", side_effect=self.sources), \
                 patch.object(parallel.subprocess, "run", side_effect=run), \
                 patch("multi_lane.track_a.static_moe_pipeline.wait_memory"), self.assertRaises(ValueError):
                parallel.run_parallel_evaluations(batch, logs, "data", "clip", "face", "0", state, batch / "state.json")
            self.assertEqual(state["phases"], dict(val=0, test=2))
            self.assertNotIn("parallel_sources_verified", state)

    def test_completed_splits_reject_changed_checkpoint_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            with patch.object(parallel, "source_hashes", side_effect=self.sources):
                path, lock = parallel.create_source_lock(batch, self.metadata)
            for split in ("val", "test"):
                destination = batch / ("full_" + split)
                destination.mkdir()
                parallel.write(destination / "comparison.json", self.completed_result(batch, split, path, lock))
            with patch.object(parallel, "source_hashes", return_value={"changed": "checkpoint"}), self.assertRaises(ValueError):
                parallel.validate_parallel_results(batch, path, lock)

    def test_parallel_evaluator_requires_lock_before_loading_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            with patch("multi_lane.track_a.evaluate_static_moe.git_metadata", return_value=self.metadata), \
                 self.assertRaisesRegex(ValueError, "shared immutable source lock"):
                evaluate_batch(batch, batch / "test", batch, batch, batch, split="test", parallel_splits=True)

    def legacy_fixture(self, root):
        batch = root / "output/emotic_static_moe/batch"
        batch.mkdir(parents=True)
        reservation = batch / "full_val"
        reservation.mkdir()
        parallel.write(reservation / "PARALLEL_EVALUATION_RESERVED.json", dict(purpose="handoff"))
        exits = {m: 0 for m in METHODS}
        for method in METHODS:
            parallel.write(batch / (method + ".exit.json"), dict(exit_code=0))
        parallel.write(batch / "pipeline_state.json", dict(status="failed", phases=dict(training=exits, val=1)))
        logs = root / "logs/emotic_static_moe/batch"
        logs.mkdir(parents=True)
        (logs / "val_evaluation.log").write_text("ValueError: Dirty evaluator or existing output\n")
        state = dict(phases={}, handoff=dict(controller_pid=123, controller_start_ticks="100"))
        return batch, state

    def test_handoff_archives_reservation_and_legacy_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch, state = self.legacy_fixture(Path(tmp))
            with patch.object(parallel, "process_identity", return_value=None):
                parallel.wait_for_legacy_training(batch, state)
            self.assertTrue(state["legacy_training_completed"])
            self.assertFalse((batch / "full_val").exists())
            self.assertTrue((batch / "legacy_evaluation_reservation/PARALLEL_EVALUATION_RESERVED.json").is_file())
            self.assertTrue((batch / "pipeline_legacy_handoff.json").is_file())
            self.assertTrue((Path(tmp) / "logs/emotic_static_moe/batch/legacy_val_reservation_exit.log").is_file())

    def test_handoff_refuses_unexpected_evaluation_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch, state = self.legacy_fixture(Path(tmp))
            (batch / "full_val/unexpected.npz").write_bytes(b"preserve")
            with patch.object(parallel, "process_identity", return_value=None), self.assertRaises(ValueError):
                parallel.wait_for_legacy_training(batch, state)
            self.assertEqual((batch / "full_val/unexpected.npz").read_bytes(), b"preserve")
            self.assertNotIn("legacy_training_completed", state)

    def test_failed_training_does_not_start_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch, state = self.legacy_fixture(Path(tmp))
            parallel.write(batch / "MOE_OVERLAP.exit.json", dict(exit_code=1))
            with self.assertRaisesRegex(ValueError, "training group failed"):
                parallel.wait_for_legacy_training(batch, state)
            self.assertNotIn("legacy_training_completed", state)

    def test_wrong_live_controller_does_not_reserve_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            parallel.write(batch / "pipeline_state.json", dict(status="training", smoke=False, git={}))
            with patch.object(parallel, "process_identity", return_value=("100", "S")), \
                 patch.object(Path, "read_bytes", return_value=b"unrelated-command\0"), self.assertRaises(ValueError):
                parallel.prepare_handoff(batch, 123, self.metadata)
            self.assertFalse((batch / "full_val").exists())


if __name__ == "__main__":
    unittest.main()
