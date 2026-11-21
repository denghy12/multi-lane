import unittest

import torch
import torch.nn.functional as F

from scripts.emotic.d9_blur_dose_extract import mild_view


class MildViewTest(unittest.TestCase):
    def test_same_view_sender_receiver_and_face_mask(self):
        values = torch.arange(2 * 3 * 224 * 224, dtype=torch.float32).reshape(2, 3, 224, 224) % 11
        images = {name: values.clone() for name in ('full','person','face','evidence_full','evidence_person','evidence_face')}
        images['evidence_face_reliable'] = torch.tensor([True, False])
        before = {key: value.clone() for key, value in images.items()}
        expected = F.interpolate(F.interpolate(values, size=(168,168), mode='area'),
                                 size=(224,224), mode='bilinear', align_corners=False)
        person = mild_view(images, 'person')
        face = mild_view(images, 'face')
        for name in ('person','evidence_person'):
            torch.testing.assert_close(person[name], expected, rtol=0, atol=0)
        for name in ('face','evidence_face'):
            torch.testing.assert_close(face[name][0], expected[0], rtol=0, atol=0)
            torch.testing.assert_close(face[name][1], values[1], rtol=0, atol=0)
        for name, original in before.items():
            torch.testing.assert_close(images[name], original, rtol=0, atol=0)
        self.assertIs(person['full'], images['full'])
        self.assertIs(face['evidence_full'], images['evidence_full'])
        self.assertFalse(torch.equal(expected, values))
