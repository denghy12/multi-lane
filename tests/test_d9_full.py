import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn

from multi_lane.track_a.d9_full import (
    FullBoxTransform, attach_full_box, bias_selector_scores, patch_coverage, visible_eval_box,
)
from multi_lane.track_a.r1_crossvit import R1Receiver
from multi_lane.track_a.runner import build_transforms
from test_track_a_reproduction import FakeVisual


class D9FullTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_coverage_and_only_intended_keys(self):
        box = torch.tensor([0.25, 0, 0.625, 0.5, 1, 1])
        area = patch_coverage(box, grid=2)
        torch.testing.assert_close(area, torch.tensor([.5, .25, 0, 0]))
        self.assertFalse(patch_coverage(torch.zeros(6)).any())
        logits = torch.zeros(3, 2, 10, 5, requires_grad=True)
        changed = bias_selector_scores(logits, area.expand(2, -1))
        self.assertFalse(changed[:, :, :-2].any())
        self.assertFalse(changed[..., 0].any())
        torch.testing.assert_close(changed[0, 0, -1, 1:], area * math.log(2))
        changed.sum().backward()
        self.assertTrue((logits.grad == 1).all())

    def test_full_pixels_rng_and_clipping(self):
        image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (311, 571, 3), dtype=np.uint8))
        for training in (False, True):
            native = build_transforms('clip')[0 if training else 1]
            boxed = FullBoxTransform(training)
            for seed in (1, 2, 8):
                torch.manual_seed(seed)
                expect = native(image)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(seed)
                actual = boxed(image, (150, 100, 270, 180))
                self.assertTrue(torch.equal(actual['full'], expect))
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertFalse(boxed(image, (-10,-10,-1,-1))['patch_coverage'].any())
        self.assertFalse(visible_eval_box((0, 0, 20, 300), *image.size))

    def test_same_image_swaps_only_visible_distinct_people(self):
        with tempfile.TemporaryDirectory() as temp:
            p = str(Path(temp)/'one.jpg')
            q = str(Path(temp)/'two.jpg')
            Image.new('RGB',(500,300)).save(p)
            Image.new('RGB',(500,300)).save(q)
            src = SimpleNamespace(input_mode='full', paired_transform=None,
                                  body_bboxes=[(130,30,160,80), (200,50,240,90),
                                               (0,10,10,20), (150,40,170,55)],
                                  file_paths=[p,p,p,q], sample_ids=['a','b','c','d'])
            altered = attach_full_box(src, 'clip', (.05,1), training=False, mode='swapped')
            self.assertEqual(altered.d9_swap_source_ids, {'a':'b','b':'a'})
            self.assertEqual(altered.body_bboxes, [src.body_bboxes[i] for i in (1,0,2,3)])
            self.assertIsNone(src.paired_transform)

    def test_native_path_full_box_only(self):
        torch.manual_seed(7)
        visual = FakeVisual()
        visual.transformer.resblocks = nn.ModuleList(copy.deepcopy(visual.transformer.resblocks[0])
                                                     for _ in range(10))
        model = R1Receiver(visual, (2,1), d9_full_box=True, r1_layer=0, num_prompt_layers=1,
                           adapter_mode='image_token', adapter_bottleneck_dim=4,
                           adapter_layer_indices=(1,))
        model.activate_task(0)
        image = torch.randn(2,3,4,4)
        blank = torch.zeros(2,4)
        changed = blank.clone()
        changed[:,0] = 1
        model.d9_full_box = False
        reference = model.current_logits(image)
        model.d9_full_box = True
        self.assertTrue(torch.equal(reference, model.current_logits({'full':image, 'patch_coverage':blank})))
        self.assertFalse(torch.equal(reference, model.current_logits({'full':image, 'patch_coverage':changed})))
        with self.assertRaisesRegex(ValueError, 'Missing transformed'):
            model.current_logits(image)
