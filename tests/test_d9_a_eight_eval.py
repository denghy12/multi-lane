import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from multi_lane.track_a.runner import TASK_SIZES
from scripts.emotic.evaluate_d9_relation import VIEWS, metrics_for_stage, weighted_logits
from scripts.emotic.fit_d9_task10_probe import weights as masked_weights
from scripts.emotic.d9_a_eight_eval import paired, predictions, verify_controls, verify_seal


class EightStageReadoutTest(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(11)
        sizes = TASK_SIZES[:2]
        n = 11
        valid = torch.arange(n) % 2 == 0
        z = torch.from_numpy(rng.normal(size=(n, sum(sizes), 3)).astype('float32'))
        labels = (z[:, :, 0] > 0).float()
        d = dict(targets=labels.numpy(), reliable=valid.numpy(), branch_logits=z.numpy(),
                 sample_ids=np.array([str(i) for i in range(n)]), task_sizes=sizes, view_order=VIEWS)
        for key, group in [('fixed', [[.8, .2, 0.], [.64,.16,.2]]),
                           ('equal', [[.5, .5, 0.], [1/3]*3])]:
            w = torch.tensor(group)[valid.long()][:, None, :].expand(n, 2, 3)
            d[key + '_weights'] = w.numpy()
            d[key + '_logits'] = weighted_logits(z, w, sizes).numpy()
        centers = {'0': {'center': [[4,0,100], [0,0,2]]}, '1': {'center': [[0,2,300], [0,3,0]]}}
        return d, centers

    def test_heads_stay_private_and_unavailable_face_is_masked(self):
        d, centers = self.fixture()
        logits, target, valid = predictions(d, centers, 1)
        expected = []
        for task, part in enumerate(torch.from_numpy(d['branch_logits']).split(TASK_SIZES[:2], dim=1)):
            w = masked_weights(torch.tensor(centers[str(task)]['center'], dtype=torch.float32), valid)
            self.assertEqual(w[~valid, 2].count_nonzero().item(), 0)
            expected.append(sum(part[:,:,i] * w[:,i:i+1] for i in range(3)))
        torch.testing.assert_close(logits['A'], torch.cat(expected, dim=1))
        m = metrics_for_stage(1,target,valid,logits)
        compare = paired(m,1)['all']
        self.assertAlmostEqual(compare['A_minus_fixed_map'], np.mean(list(compare['per_class_delta'].values())))
        self.assertAlmostEqual(compare['current_classes_delta'],
                               np.mean(list(compare['per_class_delta'].values())[-TASK_SIZES[1]:]))
        verify_controls(m,m)

    def test_corrupt_control_or_head_order_fails(self):
        d, centers = self.fixture()
        d['fixed_logits'] = d['fixed_logits'] + 1
        with self.assertRaises(AssertionError):
            predictions(d, centers, 1)
        d['view_order'] = tuple(reversed(VIEWS))
        with self.assertRaises(ValueError):
            predictions(d, centers, 1)

    def test_test_requires_validation_sealed_to_this_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'val_results.json').write_text('{}')
            value = dict(centers_sha256='fit', val_results_sha256=sha256_file(root/'val_results.json'))
            (root/'validation_seal.json').write_text(json.dumps(value))
            self.assertEqual(verify_seal(root,'fit'), value)
            with self.assertRaises(ValueError):
                verify_seal(root,'other')
            (root/'val_results.json').write_text('changed')
            with self.assertRaises(ValueError):
                verify_seal(root,'fit')
