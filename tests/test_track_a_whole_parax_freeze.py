import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.image_stream_transfer_audit import ImageStreamTransferAudit
from multi_lane.track_a.runner import build_optimizer_groups, compact_model_state_dict
from multi_lane.track_a.export_compact_test_scores import restore_compact_model_state
from multi_lane.track_a.compare_snapshot_validation import summarize
from test_track_a_reproduction import FakeVisual


class WholeParaXFreezeTest(unittest.TestCase):
    def model(self, freeze):
        torch.manual_seed(9)
        return MultiLaneModel(FakeVisual(), (2,2,2), num_selectors=2, num_prompts=2,
                              num_prompt_layers=1, view_fusion='fixed_three_view',parax_mode='image',
                              parax_layer_indices=(0,),parax_rank=2,parax_num_experts=3,
                              parax_router_hidden=2,parax_freeze_all_after_task0=freeze)

    def images(self):
        images={k:torch.randn(2,3,4,4) for k in ('full','person','face')}
        images['face_reliable']=torch.tensor([True,False])
        return images

    def test_task0_identical_then_all_components_frozen_across_training_activation_and_restore(self):
        a,b=self.model(False),self.model(True)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        b.activate_task(0);b.train();images=self.images()
        loss=b.current_logits(images).sum();loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in b.parax_bank.parameters()))
        torch.optim.Adam(b.optimizer_parameters(),lr=.001).step()
        bank=copy.deepcopy(b.parax_bank.state_dict())
        b.eval()
        with torch.no_grad():before=b.seen_logits(images)
        b.activate_task(1);b.train()
        self.assertFalse(b.parax_bank.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in b.parax_bank.parameters()))
        groups=build_optimizer_groups(b,0,adapter_learning_rate=.0004)[2]
        bank_ids={id(p) for p in b.parax_bank.parameters()}
        self.assertFalse(any(id(p) in bank_ids for g in groups for p in g['params']))
        b.current_logits(images).sum().backward()
        self.assertTrue(all(p.grad is None for p in b.parax_bank.parameters()))
        with torch.no_grad():after=b.seen_logits(images)[:,:2]
        torch.testing.assert_close(before,after,atol=1e-5,rtol=1e-5)
        self.assertTrue(all(torch.equal(v,b.parax_bank.state_dict()[k]) for k,v in bank.items()))
        payload=dict(schema_version=1,model=compact_model_state_dict(b),task_id=1,source_git={'commit':'x'})
        restored=self.model(True)
        restore_compact_model_state(restored,payload,1,{'commit':'x'});restored.train()
        self.assertFalse(restored.parax_bank.training)
        self.assertFalse(list(restored.parax_optimizer_parameters()))
        restored.activate_task(2);restored.train()
        self.assertFalse(list(restored.parax_optimizer_parameters()))

    def test_audit_rejects_mutated_buffer_even_with_no_parameter_changes(self):
        b=self.model(True);b.activate_task(0)
        b.parax_bank.register_buffer('freeze_probe',torch.tensor(0.))
        loader=[(self.images(),torch.zeros(2,6),['a','b'])]
        with tempfile.TemporaryDirectory() as tmp:
            audit=ImageStreamTransferAudit(Path(tmp),torch.device('cpu'))
            audit.begin(b,loader,0);audit.finish(b,0)
            b.activate_task(1);b.train();audit.begin(b,loader,1)
            b.parax_bank.freeze_probe.add_(1)
            with self.assertRaisesRegex(RuntimeError,'fully frozen'):
                audit.finish(b,1)

    def test_single_seed_test_summary_has_no_invented_standard_deviation_and_checks_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for method,mode in [('THREE_VIEW','disabled'),('THREE_VIEW_PARAX','image')]:
                p=root/f'seed0/{method}';p.mkdir(parents=True)
                metrics={v:dict(mAP=40) for v in ('fused','full','person','face','face_reliable')}
                d=dict(status='complete',training_performed=False,optimizer_updates=0,test_images_or_scores_opened=True,
                       evaluation_split='test',source_unchanged=True,clip_unchanged=True,samples=5368,seed=0,
                       source_config=dict(parax_mode=mode,initial_selector_prompt_classifier_sha256='same'),
                       final_metrics=metrics,snapshot_metrics=metrics)
                (p/'evaluation_summary.json').write_text(json.dumps(d))
                np.savez(p/'assembled_test.npz',sample_ids=np.arange(5368),targets=np.zeros((5368,26)),
                         face_reliable=np.ones(5368,dtype=bool),class_indices=np.arange(26))
            d=summarize(root,seeds=(0,),split='test')
            self.assertIsNone(d['metrics']['THREE_VIEW']['final_metrics']['fused']['std'])
            self.assertTrue(d['test_images_or_scores_opened'])
            p=root/'seed0/THREE_VIEW/evaluation_summary.json';x=json.loads(p.read_text());x['evaluation_split']='val';p.write_text(json.dumps(x))
            with self.assertRaises(ValueError):summarize(root,seeds=(0,),split='test')


if __name__=='__main__':unittest.main()
