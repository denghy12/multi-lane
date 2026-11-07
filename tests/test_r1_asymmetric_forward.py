from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from multi_lane.track_a.runner import receiver_evaluation_threads, write_evaluation_scores
from scripts.emotic.r1_asymmetric_forward import (
    AsymmetricReceivers, GROUPS, fuse_logits, verify_replay,
)


class FakeReceiver(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset
        self.r1_enabled = True
        self.last_inputs = None

    def seen_logits(self, inputs):
        self.last_inputs = inputs
        if isinstance(inputs, torch.Tensor):
            return inputs
        return inputs["full"] + (self.offset if self.r1_enabled else 0)


class AsymmetricForwardTests(unittest.TestCase):
    def setUp(self):
        self.models = {
            name: FakeReceiver(index + 1)
            for index, name in enumerate(("full", "person9", "person10", "face9", "face10"))
        }
        self.module = AsymmetricReceivers(self.models)
        self.images = {
            "full": torch.tensor([[2., 3.], [4., 5.]]),
            "evidence_full": torch.tensor([[.1, .2], [.3, .4]]),
            "evidence_face": torch.tensor([[6., 7.], [8., 9.]]),
            "evidence_face_reliable": torch.tensor([True, False]),
        }

    def test_primary_receivers_get_their_native_view_and_original_memories(self):
        streams, fused = self.module(self.images)
        self.assertEqual(set(streams), {("full", True), ("person10", True), ("face9", True)})
        self.assertIs(self.models["full"].last_inputs, self.images["evidence_full"])
        self.assertIs(self.models["person10"].last_inputs, self.images)
        face = self.models["face9"].last_inputs
        self.assertIs(face["full"], self.images["evidence_face"])
        self.assertIs(face["evidence_person"], self.images["full"])
        self.assertIs(face["evidence_full"], self.images["evidence_full"])
        expected = fuse_logits(
            streams[("full", True)], streams[("person10", True)],
            streams[("face9", True)], self.images["evidence_face_reliable"],
        )
        torch.testing.assert_close(fused[("asymmetric", "both_on")], expected, rtol=0, atol=0)

    def test_each_group_has_independent_switches_and_face_fallback_is_exact(self):
        streams, fused = self.module(self.images, attribution=True)
        self.assertEqual(len(streams), 9)
        self.assertEqual(len(fused), 12)
        for group in GROUPS:
            torch.testing.assert_close(
                fused[(group, "both_on")][1], fused[(group, "face_off")][1],
                rtol=0, atol=0,
            )
            torch.testing.assert_close(
                fused[(group, "person_off")][1], fused[(group, "both_off")][1],
                rtol=0, atol=0,
            )
        self.assertTrue(all(model.r1_enabled for model in self.models.values()))
        again = self.module(self.images)[1][("asymmetric", "both_on")]
        torch.testing.assert_close(again, fused[("asymmetric", "both_on")], rtol=0, atol=0)

    def test_fusion_matches_locked_numpy_float32_arithmetic(self):
        full = self.images["evidence_full"]
        person, face = self.images["full"], self.images["evidence_face"]
        fused = fuse_logits(full, person, face, self.images["evidence_face_reliable"])
        weights = np.array([[.64, .16, .20], [.8, .2, 0]], dtype=np.float32)
        expected = (
            weights[:, :1] * full.numpy() + weights[:, 1:2] * person.numpy()
            + weights[:, 2:] * face.numpy()
        )
        self.assertTrue(np.array_equal(fused.numpy(), expected))

    def test_replay_aligns_ids_and_rejects_changed_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.npz"
            logits = torch.tensor([[.1, .2], [.3, .4]])
            targets = torch.tensor([[1., 0.], [0., 1.]])
            write_evaluation_scores(path, 0, ["a", "b"], logits, targets)
            record = verify_replay(path, ["b", "a"], targets.flip(0), logits.flip(0))
            self.assertEqual(record["max_abs"], 0)
            with self.assertRaises(AssertionError):
                verify_replay(path, ["a", "b"], targets, logits + .01)

    def test_native_evaluation_threads_restore_after_success_and_failure(self):
        original = torch.get_num_threads()
        model = SimpleNamespace(r1_kind="native")

        @receiver_evaluation_threads
        def probe(_, fail=False):
            self.assertEqual(torch.get_num_threads(), 2)
            if fail:
                raise RuntimeError("probe")
            return 1

        self.assertEqual(probe(model), 1)
        self.assertEqual(torch.get_num_threads(), original)
        with self.assertRaises(RuntimeError):
            probe(model, fail=True)
        self.assertEqual(torch.get_num_threads(), original)
