import unittest
import numpy as np
import torch

from scripts.emotic.d9_quality_extract import degrade_view, verify_replay


class ViewQualityTest(unittest.TestCase):
    def setUp(self):
        square = (torch.arange(224).remainder(2).float()[:, None]
                  + torch.arange(224).remainder(2).float()[None, :]).remainder(2)
        native = square.expand(2, 3, 224, 224)
        self.inputs = {v: native.clone() for v in ('full', 'person', 'face')}
        self.inputs.update({'evidence_' + v: (native + 1.) for v in ('full', 'person', 'face')})
        self.inputs['evidence_face_reliable'] = torch.tensor([1., 0.])

    def test_face_variant_changes_both_copies_on_reliable_people_only(self):
        x = self.inputs
        altered = degrade_view(x, 'face')
        for k in ('face', 'evidence_face'):
            self.assertFalse(torch.equal(altered[k][0], x[k][0]))
            self.assertTrue(torch.equal(altered[k][1], x[k][1]))
        for k in ('full', 'person', 'evidence_person', 'evidence_face_reliable'):
            self.assertIs(altered[k], x[k])
        torch.testing.assert_close(altered['evidence_face'] - altered['face'], torch.ones_like(x['face']))

    def test_person_variant_changes_each_person_including_missing_face(self):
        altered = degrade_view(self.inputs, 'person')
        for k in ('person', 'evidence_person'):
            self.assertTrue(torch.all(altered[k] != self.inputs[k]))
        self.assertIs(altered['face'], self.inputs['face'])
        self.assertIs(altered['evidence_face'], self.inputs['evidence_face'])

    def test_replay_rejects_a_changed_original_logit(self):
        original = {'branch_logits': np.array([1., 2., 3.])}
        verify_replay({'branch_logits': np.array([2., 3.])}, original, slice(1, 3))
        with self.assertRaises(AssertionError):
            verify_replay({'branch_logits': np.array([2., 3.1])}, original, slice(1, 3))


if __name__ == '__main__':
    unittest.main()
