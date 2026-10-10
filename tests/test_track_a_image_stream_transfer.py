import json
import tempfile
import unittest
from pathlib import Path

import torch

from multi_lane.track_a.runner import parse_args
from multi_lane.track_a.image_stream_transfer_audit import ImageStreamTransferAudit, validate_transfer_config
from multi_lane.track_a.model import MultiLaneModel
from test_track_a_reproduction import FakeVisual


class ImageStreamTransferTest(unittest.TestCase):
    def config(self, extra=()):
        return parse_args(['--seed','0','--data-root','.', '--clip-checkpoint','clip.pt',
                           '--output-root','output','--image-stream-transfer-audit',
                           '--view-fusion','fixed_three_view','--view-auxiliary-loss-weight','0.1',
                           '--skip-validation-eval', '--reporting-split', 'val', *extra])

    def test_protocol_rejects_different_route_and_changed_center_policy(self):
        base = parse_args(['--seed','0','--data-root','.', '--clip-checkpoint','clip.pt','--output-root','output'])
        self.assertFalse(base.image_stream_transfer_audit)
        validate_transfer_config(self.config())
        valid = self.config(['--parax-mode','image','--parax-layer-indices','10',
                             '--parax-rank','32','--parax-num-experts','3','--parax-router-hidden','16',
                             '--parax-initialization','official','--parax-residual-scale','0.1',
                             '--parax-output-scale-mode','learnable'])
        validate_transfer_config(valid)
        valid.parax_freeze_center_after_task0 = True
        with self.assertRaises(ValueError):
            validate_transfer_config(valid)
        valid.parax_mode = 'post_task_protected_pool'
        with self.assertRaises(ValueError):
            validate_transfer_config(valid)

    def test_fixed_anchor_detects_shared_transform_drift_and_preserves_rng(self):
        torch.manual_seed(9)
        model = MultiLaneModel(FakeVisual(), (2,2), num_selectors=2, num_prompts=2,
                               num_prompt_layers=1, view_fusion='fixed_three_view',
                               parax_mode='image', parax_layer_indices=(0,), parax_rank=2,
                               parax_num_experts=3, parax_router_hidden=2, parax_initialization='official',
                               parax_residual_scale=.1)
        model.activate_task(0)
        images = {k: torch.randn(2,3,4,4) for k in ('full','person','face')}
        images['face_reliable'] = torch.tensor([True,False])
        loader = [(images, torch.zeros(2,4), ['a','b'])]
        with tempfile.TemporaryDirectory() as tmp:
            audit = ImageStreamTransferAudit(Path(tmp), torch.device('cpu'))
            rng = torch.get_rng_state().clone()
            audit.begin(model, loader, 0)
            audit.finish(model, 0)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            model.activate_task(1)
            audit.begin(model, loader, 1)
            with torch.no_grad():
                for parameter in model.parax_bank.parameters():
                    parameter.add_(.03)
            audit.finish(model, 1)
            rows = json.loads((Path(tmp)/'image_stream_transfer_audit.json').read_text())['tasks']
            self.assertTrue(rows['1']['old_lane_parameters_unchanged'])
            self.assertTrue(rows['1']['shared_parax_changed'])
            self.assertGreater(rows['1']['fixed_anchor_old_task_drift']['0']['logit_max_absolute_difference'], 0)
            self.assertTrue(model.training)
            audit.begin(model, loader, 1)
            with torch.no_grad():
                model.selectors[0].add_(1)
            with self.assertRaisesRegex(RuntimeError, 'old task parameters'):
                audit.finish(model, 1)


if __name__ == '__main__':
    unittest.main()
