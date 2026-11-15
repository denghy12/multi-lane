"""Regression checks for checkpoint-derived D benchmark scoring."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from multi_lane.track_a.runner import CLASS_ORDER, TASK_PROTOCOLS, write_evaluation_scores
from scripts.emotic import summarize_d9_benchmark_protocols as report


class BenchmarkSummaryTests(unittest.TestCase):
    def generate(self, root, protocol):
        # Two fully observed people; the reliable Face deliberately changes the
        # sign of the positive example. The fallback Face logit is much larger.
        sizes = TASK_PROTOCOLS[protocol]
        truth = np.zeros((2, 26), dtype=np.float32)
        truth[0, :3] = 1
        truth[1, 4:] = 1
        for view in ("full", "person", "face"):
            cfg = dict(seed=1, protocol_id=f"emotic_{protocol}_v0.1",
                       task_sizes=sizes, class_order=CLASS_ORDER, r1_receiver=view,
                       r1_kind="native" if view == "full" else "attention",
                       r1_frozen_mode="patch", r1_exchange_layer=8, r1_frozen_evidence="all",
                       max_tasks=len(sizes), reporting_split="test")
            report.write(root / view / "train/config.json", cfg)
            report.write(root / view / "train/seed_summary.json", {"status": "complete"})
            for split in ("val", "test"):
                for i, end in enumerate(np.cumsum(sizes)):
                    ids = np.array([f"{split}:0", f"{split}:1"])
                    z = np.full((2, end), -1, dtype=np.float32)
                    z[1, 0] = 0   # equals threshold after sigmoid; must remain negative
                    if view == "face":
                        z[0, :3] = 8
                        z[1, 0] = 90
                    # ID alignment is required before applying the reliability mask.
                    order = [0, 1] if view == "full" else [1, 0]
                    order = [j for j in order if truth[j, :end].any()]
                    path = root / view / report.score_suffix(split, i)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    write_evaluation_scores(path, i, ids[order], torch.tensor(z[order]),
                                            torch.tensor(truth[order, :end]))

    def test_joint_and_incremental_reports(self):
        for protocol in ("joint26", "b4c2", "b10c4"):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.generate(root, protocol)
                sizes = TASK_PROTOCOLS[protocol]
                counts = tuple(1 if n <= 4 else 2 for n in np.cumsum(sizes))
                def reliability(_manifest, split):
                    return {f"{split}:0": True, f"{split}:1": False}
                with patch.object(report, "load_face_reliability", side_effect=reliability), \
                     patch.dict(report.FINAL_PEOPLE, val=2, test=2), \
                     patch.dict(report.TEST_PEOPLE, {protocol: counts}):
                    output = report.summarize(root, protocol, 1, root)
                self.assertTrue((root / "benchmark_summary.json").exists())
                for values in output["splits"].values():
                    self.assertEqual(len(values["metrics"]), len(sizes))
                    self.assertEqual(values["populations"][-1]["reliable_face"], 1)
                    summary = values["summary"]
                    self.assertAlmostEqual(summary["average_mAP"], np.mean([r["mAP"] for r in values["metrics"]]))
                    self.assertAlmostEqual(summary["final_cF1"], 100 * 3 / 26)
                    self.assertAlmostEqual(summary["final_oF1"], 100 * 6 / 28)

    def test_reject_stage_with_missing_people(self):
        from types import SimpleNamespace
        final = SimpleNamespace(sample_ids=np.array(["A", "B"]), targets=np.array([[1, 0], [1, 1]]))
        stage = SimpleNamespace(sample_ids=np.array(["A"]), logits=np.array([[1]]), task_id=0)
        with self.assertRaisesRegex(ValueError, "population"):
            report.check_population(stage, final, "test")

    def test_reject_mixed_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.generate(root, "joint26")
            with self.assertRaisesRegex(ValueError, "seed"):
                report.validate_sources(root, "joint26", 2, False)
