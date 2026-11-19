import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from multi_lane.track_a.openai_clip_loader import sha256_file
from scripts.emotic.d9_tcp_probe import active_views, fit, gate_scores, run, true_label_confidence
from test_d9_task10_fitting import make_cache


class TcpProbeTests(unittest.TestCase):
    def test_multilabel_target_mask_and_branch_protection(self):
        z = torch.tensor([[[1., 0., 4.], [2., 3., 9.]],
                          [[0., 1., -1.], [2., 1., -3.]]], requires_grad=True)
        y = torch.tensor([[0., 1.], [1., 0.]], requires_grad=True)
        r = torch.tensor([False, True])
        x = torch.randn(2, 12, requires_grad=True)
        truth = true_label_confidence(z, y)
        torch.testing.assert_close(truth[0], torch.stack(((-z[0, 0]).sigmoid(), z[0, 1].sigmoid())).mean(0))
        self.assertFalse(truth.requires_grad)
        params = {'bias': torch.zeros(2, 3), 'coefficient': torch.zeros(3, 12)}
        raw, w = gate_scores(params, x, r)
        torch.testing.assert_close(w[0], torch.tensor([.5, .5, 0.]))
        torch.testing.assert_close(w[1], torch.tensor([1/3, 1/3, 1/3]))
        self.assertFalse(active_views(r)[0, 2])
        self.assertTrue(torch.all(raw == 0))
        for method in ('B_bce', 'C_tcp'):
            model = fit(method, x, r, z, y, 3)
            z2 = z.detach().clone()
            z2[0, :, 2] += 100
            self.assertEqual(model, fit(method, x, r, z2, y, 3))
        self.assertIsNone(x.grad)
        self.assertIsNone(z.grad)
        self.assertIsNone(y.grad)

    def test_fit_val_lock_can_precede_test_extraction(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root/'probe'

            def source(split, seed):
                path = make_cache(root, split, seed)
                metadata = dict(protocol=dict(task=0, split=split, smoke=False, checkpoint_sha256='cp',
                                              source_config_sha256='cfg'), visibility_mask_applied=True,
                                exact_eval_replay=split != 'train', features_sha256=sha256_file(path))
                path.with_suffix('.json').write_text(json.dumps(metadata))

            source('train', 1)
            source('val', 2)
            run(root, out, 'fit-val', [0], 2, 2)
            val = json.loads((out/'tcp_val_task0.json').read_text())
            self.assertEqual(len(val['C_tcp']['weight_shuffle']['differences']), 2)
            self.assertIn('mse', val['B_bce']['quality'])
            self.assertIn('mse', val['C_tcp']['quality'])
            source('test', 3)
            run(root, out, 'test', [0], 2, 2)
            result = json.loads((out/'tcp_test_results.json').read_text())
            self.assertIn('C_tcp', result['summary']['macro_current_head_mAP'])
            model = out/'tcp_model_task0.json'
            model.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'lock'):
                run(root, out, 'test', [0], 2, 2)
