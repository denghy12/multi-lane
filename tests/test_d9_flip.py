import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from multi_lane.track_a.r1_crossvit import ReceiverEvidenceFlip, attach_evidence
from multi_lane.track_a.runner import build_transforms


class Picture:
    sample_ids = ['train:one#person=0']
    face_reliable = [True]
    splits = ('train',)

    def __init__(self, transform, **unused):
        self.transform = transform
        arr = np.zeros((19, 27, 3), dtype=np.uint8)
        arr[:, :9] = (190, 100, 45)
        self.image = Image.fromarray(arr)

    def __getitem__(self, index):
        return self.transform(self.image), torch.tensor([1, 0])

    def __len__(self):
        return 1


class TestD9Flip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.addClassCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)

    def test_independent_uses_all_four_quarters(self):
        picture = Picture(transforms.ToTensor())
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        wrapped = ReceiverEvidenceFlip(transform, 'independent')
        original, _ = picture[0]
        for draw, receiver, donors in ((.125, True, False), (.375, True, True),
                                       (.625, False, False), (.875, False, True)):
            with patch('torch.rand', return_value=torch.tensor([draw])) as coin:
                image, flip = wrapped(picture.image)
            self.assertEqual(flip, donors)
            self.assertTrue(torch.equal(image, original.flip((-1,)) if receiver else original))
            coin.assert_called_once_with(1)

    def test_native_pixels_rng_and_two_donors(self):
        def dataset(parent, **kwargs):
            return Picture(**kwargs)

        for view in ('person', 'face'):
            # The native Person transform consumes additional RNG for its color jitter.
            train, _ = build_transforms('clip', input_mode=view + '_crop',
                                        person_transform_mode='letterbox',
                                        person_color_jitter_probability=0.2,
                                        person_color_jitter_strength=0.1)
            original = Picture(train)
            for policy in ('sync', 'independent'):
                with patch('multi_lane.continual_datasets.continual_datasets.EMOTIC', dataset):
                    paired = attach_evidence(original, view, Path('/data'), Path('/manifest'),
                                             train_flip=policy)
                self.assertIs(original.transform, train)
                seen = set()
                for seed in range(8):
                    torch.manual_seed(seed)
                    draw = torch.rand(1).item()
                    donor_flip = draw < .5 if policy == 'sync' else int(4 * draw) % 2 == 1
                    torch.manual_seed(seed)
                    native_pixels, native_labels = original[0]
                    expected_rng = torch.get_rng_state()
                    torch.manual_seed(seed)
                    evidence, labels = paired[0]
                    self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                    self.assertTrue(torch.equal(labels, native_labels))
                    self.assertTrue(torch.equal(evidence['full'], native_pixels))
                    self.assertTrue(evidence['evidence_face_reliable'].item())
                    for sender, source in paired.auxiliary.items():
                        original_donor, _ = source[0]
                        expected = original_donor.flip((-1,)) if donor_flip else original_donor
                        self.assertTrue(torch.equal(evidence['evidence_' + sender], expected))
                    seen.add(donor_flip)
                self.assertEqual(seen, {False, True})

    def test_eval_remains_deterministic(self):
        _, evaluate = build_transforms('clip', input_mode='face_crop')
        source = Picture(evaluate)
        source.splits = ('val',)
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            attach_evidence(source, 'face', Path('/data'), Path('/manifest'), train_flip='sync')
        with patch('multi_lane.continual_datasets.continual_datasets.EMOTIC',
                   side_effect=lambda parent, **kwargs: Picture(**kwargs)):
            wrapped = attach_evidence(source, 'face', Path('/data'), Path('/manifest'))
        state = torch.get_rng_state()
        a, _ = wrapped[0]
        b, _ = wrapped[0]
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]))


if __name__ == '__main__':
    unittest.main()
