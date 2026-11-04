import sys
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'emotic'))
from run_r1_cls_qk_control import Readout, gradients, save_scores
from run_r1_final_cls_exchange import correction


class QKControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.features = F.normalize(torch.randn(12, 3, 512), dim=-1)
        self.reliable = torch.tensor([True, False] * 6)
        self.heads = torch.randn(3, 5, 512)
        self.labels = torch.randint(0, 2, (12, 5)).float()

    def test_common_initialization_and_zero_output(self):
        uniform = Readout(512, 'uniform')
        normalized = Readout(512, 'normalized')
        normalized.load_state_dict(uniform.state_dict())
        self.assertEqual(sum(p.numel() for p in uniform.parameters()), 78016)
        for module in (uniform, normalized):
            self.assertTrue(torch.equal(module(self.features, self.reliable), torch.zeros_like(self.features)))
        self.assertTrue(torch.equal(uniform.qkv.weight, normalized.qkv.weight))

    def test_mask_and_single_sender(self):
        for mode in ('uniform', 'normalized'):
            module = Readout(512, mode)
            weights, _, _ = module.components(self.features, self.reliable)
            self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(12, 3)))
            self.assertTrue(torch.equal(weights.diagonal(dim1=1, dim2=2), torch.zeros(12, 3)))
            self.assertTrue(torch.equal(weights[~self.reliable, 1, 0], torch.ones(6)))
            self.assertTrue(torch.equal(weights[~self.reliable, :, 2], torch.zeros(6, 3)))

    def test_invalid_face_and_full_prediction_protection(self):
        for mode in ('uniform', 'normalized'):
            module = Readout(512, mode)
            with torch.no_grad():
                module.output.weight.normal_()
            before = correction(module, self.features, self.reliable, self.heads)
            changed = self.features.clone()
            changed[~self.reliable, 2] *= 100
            after = correction(module, changed, self.reliable, self.heads)
            self.assertTrue(torch.equal(before[:, 0], torch.zeros_like(before[:, 0])))
            self.assertTrue(torch.equal(before[~self.reliable, 2], torch.zeros_like(before[~self.reliable, 2])))
            self.assertTrue(torch.equal(before, after))
            self.assertLessEqual(float(module(self.features, self.reliable).norm(dim=-1).max()), .050001)

    def test_gradient_paths_after_zero_output_warmup(self):
        for mode in ('uniform', 'normalized'):
            torch.manual_seed(19)
            module = Readout(512, mode)
            optimizer = torch.optim.Adam(module.parameters(), lr=.0004)
            original_qk = module.qkv.weight[:128].detach().clone()
            for step in range(3):
                logits = correction(module, self.features, self.reliable, self.heads).sum(1)
                loss = F.binary_cross_entropy_with_logits(logits, self.labels)
                optimizer.zero_grad()
                loss.backward()
                grad = gradients(module)
                if step == 0:
                    self.assertEqual(grad['value'], 0)
                    self.assertEqual(grad['query'], 0)
                else:
                    self.assertGreater(grad['value'], 0)
                    if mode == 'normalized':
                        self.assertGreater(grad['query'], 0)
                        self.assertGreater(grad['key'], 0)
                    else:
                        self.assertEqual(grad['query'], 0)
                        self.assertEqual(grad['key'], 0)
                optimizer.step()
            if mode == 'uniform':
                self.assertTrue(torch.equal(original_qk, module.qkv.weight[:128]))

    def test_dynamic_scores_have_content_variation(self):
        uniform = Readout(512, 'uniform')
        normalized = Readout(512, 'normalized')
        normalized.load_state_dict(uniform.state_dict())
        fixed, _, _ = uniform.components(self.features, self.reliable)
        learned, _, _ = normalized.components(self.features, self.reliable)
        self.assertEqual(float(fixed[self.reliable, 1, 0].std()), 0)
        self.assertGreater(float(learned[self.reliable, 1, 0].std()), .001)

    def test_prediction_retry_is_idempotent(self):
        payload = dict(sample_ids=['a'], logits=torch.ones(1, 2), targets=torch.zeros(1, 2), reliable=torch.ones(1, dtype=torch.bool))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'scores.pth'
            save_scores(path, payload)
            save_scores(path, payload)
            with self.assertRaises(ValueError):
                save_scores(path, dict(payload, logits=torch.zeros(1, 2)))


if __name__ == '__main__':
    unittest.main()
