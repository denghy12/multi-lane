import copy
import unittest

import numpy as np
import torch

from multi_lane.track_a.evaluate_joint26_checkpoint import build_model
from multi_lane.track_a.export_compact_test_scores import restore_compact_model_state
from multi_lane.track_a.runner import compact_model_state_dict
from multi_lane.track_a.snapshot_validation import assemble_blocks, assert_owned_equal, infer
from test_track_a_reproduction import FakeVisual


class SnapshotValidationTest(unittest.TestCase):
    def blocks(self):
        rows=[]
        for task,(classes,ids) in enumerate((([0,1],['a','b']),([2],['b','a']))):
            target=np.array([[1,0,1],[0,1,0]],dtype=np.float32)
            order=[0,1] if task==0 else [1,0]
            b=dict(sample_ids=np.array(ids),class_indices=np.array(classes),full_targets=target[order],
                   face_reliable=np.array([True,False])[order])
            for suffix in ('logits','probabilities'):
                for prefix in ('','full_','person_','face_'):
                    b[prefix+suffix]=np.array([[1,2,3],[4,5,6]],dtype=np.float32)[order][:,classes] / (10 if suffix=='probabilities' else 1)
            rows.append(b)
        return rows

    def test_assembly_joins_by_ids_and_rejects_incorrect_classes_labels_and_masks(self):
        blocks=self.blocks();result=assemble_blocks(blocks,[2,1])
        np.testing.assert_array_equal(result['logits'],[[1,2,3],[4,5,6]])
        for key in ('class_indices','full_targets','face_reliable','sample_ids'):
            bad=copy.deepcopy(blocks)
            if key=='class_indices':bad[1][key][:]=0
            elif key=='sample_ids':bad[1][key][:]=['b','b']
            else:bad[1][key][0]=0 if key=='full_targets' else True
            with self.assertRaises(ValueError):assemble_blocks(bad,[2,1])
        with self.assertRaises(ValueError):assemble_blocks(blocks[:1],[2,1])

    def test_incremental_checkpoint_restores_old_task_block_and_new_default_stays_joint(self):
        torch.manual_seed(7)
        visual=FakeVisual()
        config=dict(num_selectors=2,num_prompts=2,num_prompt_layers=1,selector_mode='shared',prompt_mode='shared',
                    normalize='pre-head',view_fusion='fixed_three_view',view_classifier_mode='shared_post_fusion',
                    parax_mode='image',parax_layer_indices=[0],parax_rank=2,parax_num_experts=3,parax_router_hidden=2)
        model=build_model(config,copy.deepcopy(visual),task_sizes=[2,1])
        model.activate_task(0)
        birth=copy.deepcopy(compact_model_state_dict(model))
        model.activate_task(1)
        with torch.no_grad():
            for p in model.parax_bank.parameters():p.add_(.01)
        final=copy.deepcopy(compact_model_state_dict(model))
        assert_owned_equal(birth,final,0,[2,1])
        final['head.weight'][0].add_(1)
        with self.assertRaises(ValueError):assert_owned_equal(birth,final,0,[2,1])
        restored=build_model(config,copy.deepcopy(visual),task_sizes=[2,1])
        restore_compact_model_state(restored,dict(schema_version=1,model=birth,task_id=0,source_git={'x':'y'}),0,{'x':'y'})
        restored.requires_grad_(False).eval()
        images={k:torch.randn(2,3,4,4) for k in ('full','person','face')};images['face_reliable']=torch.tensor([True,False])
        block,full=infer(restored,[(images,torch.ones(2,3),['a','b'])],torch.device('cpu'),False,0,[2,1])
        self.assertEqual(block['logits'].shape,(2,2));self.assertEqual(full['full_targets'].shape,(2,3))
        self.assertTrue(all(p.grad is None for p in restored.parameters()))
        self.assertEqual(build_model(config,copy.deepcopy(visual)).task_sizes,(26,))


if __name__=='__main__':unittest.main()
