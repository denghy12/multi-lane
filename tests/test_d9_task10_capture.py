import unittest

import torch

from multi_lane.track_a.d9_relation_fusion import D9RelationFusion
from scripts.emotic.extract_d9_task10_probe import check_off_path
from test_d9_protected_multi_model import branches


class Task10CaptureTests(unittest.TestCase):
    def test_taps_preserve_all_seen_and_replay_off_trajectory(self):
        torch.set_num_threads(1)
        torch.manual_seed(301)
        model = D9RelationFusion(branches(), 'sample_holdout').eval()
        model.activate_task(0)
        model.activate_task(1)
        for receiver in ('person', 'face'):
            with torch.no_grad():
                model.branches[receiver].cross_agent(1, 8).out.weight.normal_(std=.3)
        images = {v: torch.randn(4, 3, 4, 4) for v in
                  ('full', 'person', 'face', 'evidence_full', 'evidence_person', 'evidence_face')}
        images['evidence_face_reliable'] = torch.tensor([True, False, True, True])
        with torch.inference_mode():
            expected = model(images, all_seen=True)
            probe = {}
            actual = model(images, all_seen=True, task10_probe=probe)
            for view in ('full', 'person', 'face'):
                torch.testing.assert_close(actual['branches'][view], expected['branches'][view], rtol=0, atol=0)
                self.assertEqual(probe[view]['content'].shape, (4, model.branches[view].width))
            self.assertEqual(probe['relation'].shape, (4, 8))
            self.assertNotIn('delta', probe['full'])
            for view in ('person', 'face'):
                self.assertGreater(torch.linalg.vector_norm(probe[view]['delta']).item(), 0)
            check_off_path(model, images, probe, 1, True)
