import unittest
from contextlib import ExitStack

import torch

from multi_lane.track_a.r1_crossvit import VIEWS
from scripts.emotic.extract_d9_caran_eight import head_input_hook, pack_head_inputs


class CaptureTest(unittest.TestCase):
    def test_hook_keeps_head_input_and_masks_only_unreliable_face(self):
        capture = {}
        modules = [torch.nn.Linear(4, 2) for _ in VIEWS]
        values = [torch.full((2, 3, 4), float(i + 1)) for i in range(3)]
        with ExitStack() as hooks:
            for view, head, x in zip(VIEWS, modules, values):
                hooks.callback(head.register_forward_pre_hook(head_input_hook(view, capture)).remove)
                head(x)
            result = pack_head_inputs(capture, torch.tensor([False, True]), 3)
        self.assertEqual(tuple(result.shape), (2, 3, 3, 4))
        torch.testing.assert_close(result[:, :, 0], values[0])
        torch.testing.assert_close(result[:, :, 1], values[1])
        torch.testing.assert_close(result[1, :, 2], values[2][1])
        self.assertEqual(result[0, :, 2].count_nonzero().item(), 0)
        self.assertTrue((values[2] == 3).all())  # read-only extraction

    def test_all_seen_lanes_required(self):
        capture = {v: torch.ones(2, 1, 4) for v in VIEWS}
        with self.assertRaises(ValueError):
            pack_head_inputs(capture, torch.ones(2, dtype=torch.bool), 2)
