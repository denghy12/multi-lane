import copy
import unittest
import torch
import test_historical_b32_modules as fixtures
from multi_lane.track_a.historical_b32_backbone_attribution import (
    component, restore_components, coordinates, comparison, FACTORIAL, method_state)
from multi_lane.track_a.runner import compact_model_state_dict, tensor_state_digest
from multi_lane.track_a.module_stacking import runtime_bypass


class BackboneAttributionTest(unittest.TestCase):
    def models(self):
        base=fixtures.HistoricalB32ModulesTest().model('BASE');joint=fixtures.HistoricalB32ModulesTest().model('ATTENTION')
        for m in (base,joint):m.activate_task(0);m.requires_grad_(False).eval()
        runtime_bypass(joint,('B','C'))
        with torch.no_grad():
            for n,p in joint.named_parameters():
                if component(n):p.add_(.02)
        return base,joint

    def test_all_groups_restore_baseline_function_and_preserve_new_modules(self):
        base,joint=self.models();images=fixtures.HistoricalB32ModulesTest().images()
        before=tensor_state_digest(method_state(joint));expected=base.current_all_logits(images)
        with restore_components(joint,compact_model_state_dict(base),'ASPH'):
            self.assertTrue(torch.equal(expected,joint.current_all_logits(images)))
        self.assertEqual(before,tensor_state_digest(method_state(joint)))

    def test_single_component_and_exception_restore_leave_other_state_untouched(self):
        base,joint=self.models();state=compact_model_state_dict(base);original=copy.deepcopy(method_state(joint))
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with restore_components(joint,state,'A'):
                for k,v in method_state(joint).items():
                    self.assertTrue(torch.equal(v.cpu(),state[k].cpu() if component(k)=='A' else original[k].cpu()))
                raise RuntimeError('intentional')
        self.assertEqual(tensor_state_digest(original),tensor_state_digest(method_state(joint)))

    def test_uncovered_common_state_and_invalid_components_are_rejected(self):
        base,joint=self.models();state=compact_model_state_dict(base)
        for condition in ('X','AA'):
            with self.assertRaises(ValueError):
                with restore_components(joint,state,condition):pass
        state['task_class_mask']=state['task_class_mask']+1
        with self.assertRaises(ValueError):
            with restore_components(joint,state,'ASPH'):pass

    def test_head_only_restoration_does_not_change_representation(self):
        base,joint=self.models();images=fixtures.HistoricalB32ModulesTest().images()
        before=joint.encode_lanes_with_views(images,False)
        with restore_components(joint,compact_model_state_dict(base),'H'):
            after=joint.encode_lanes_with_views(images,False)
        self.assertTrue(torch.equal(before[0],after[0]))
        self.assertTrue(all(torch.equal(before[1][k],after[1][k]) for k in before[1]))

    def test_task_coordinates_and_complete_factorial(self):
        state={'selectors':torch.ones(8,2,3),'prompts.0':torch.ones(2,8,2,3),
               'head.weight':torch.ones(26,3),'head.bias':torch.ones(26)}
        self.assertEqual(len(FACTORIAL),16);self.assertEqual(len(set(FACTORIAL)),16)
        self.assertEqual(coordinates(state,'S',0,True)[0].shape[0],1)
        self.assertEqual(coordinates(state,'P',0,True)[0].shape[1],1)
        self.assertEqual(coordinates(state,'H',0,True)[0].shape[0],5)
        result=comparison([torch.ones(4)],[torch.ones(4)*2])
        self.assertAlmostEqual(result['relative_l2'],.5);self.assertAlmostEqual(result['cosine'],1.)


if __name__=='__main__':unittest.main()
