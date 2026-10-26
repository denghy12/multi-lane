import copy
import unittest

import numpy as np
import torch
import test_track_a_reproduction as fixtures

from multi_lane.track_a.historical_b32_backbone_attribution import method_state
from multi_lane.track_a.historical_b32_modules import modes
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.protected_view_forward import ARMS,ProtectedViewFusion,verify_batch
from multi_lane.track_a.runner import tensor_state_digest


class ProtectedViewForwardTest(unittest.TestCase):
    def pair(self):
        models=[]
        for face in (False,True):
            torch.manual_seed(31)
            visual=fixtures.FakeVisual();visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
            model=MultiLaneModel(visual,(2,1),num_selectors=2,num_prompts=2,num_prompt_layers=1,
                adapter_mode='image_token',adapter_bottleneck_dim=3,adapter_layer_indices=(0,),adapter_residual_scale=.03,
                view_fusion='fixed_three_view',historical_b32_modules=True,cross_view_layer=1,cross_view_dim=4,
                cross_view_heads=2,cross_view_face_attention=face,**modes('ATTENTION_BOUND'))
            model.activate_task(0);model.activate_task(1);models.append(model)
        a,b=models
        with torch.no_grad():
            b.selectors.add_(torch.randn_like(b.selectors)*.2)
            for mod in b.cross_view_bank.tasks:mod.out.weight.normal_(0,.1)
            for mod in b.cross_view_bank.face_tasks:mod.out.weight.normal_(0,.1)
        b.head.load_state_dict(a.head.state_dict());return a,b

    def images(self):
        return dict(full=torch.randn(3,3,4,4),person=torch.randn(3,3,4,4),face=torch.randn(3,3,4,4),
                    face_reliable=torch.tensor([True,False,True]))

    def test_complete_sources_are_preserved_and_features_are_fused_before_head(self):
        a,b=self.pair();wrapper=ProtectedViewFusion(a,b);x=self.images()
        with torch.no_grad():
            features,sources=wrapper.sources(x)
            for arm in ARMS:
                actual,_=wrapper.compose(x,features,arm)
                selected={v:features[s][v] for v,s in zip(('full','person','face'),ARMS[arm])}
                fused,_=a.view_fusion_module(selected,a._lane_ids(True),x['face_reliable'])
                self.assertTrue(torch.equal(actual['fused'],wrapper.combine(a.head(fused))))
                for v,s in zip(('full','person','face'),ARMS[arm]):
                    self.assertTrue(torch.equal(actual[v],sources[s][v]))
            self.assertFalse(torch.equal(sources['A']['face'],sources['B']['face']))

    def test_R0_and_R1_invalid_face_are_exact(self):
        a,b=self.pair();wrapper=ProtectedViewFusion(a,b);x=self.images()
        with torch.no_grad():
            old,_=a.seen_logits_with_views(x)
            r0=wrapper(x,'R0');r1=wrapper(x,'R1')
        self.assertTrue(torch.equal(old,r0['fused']))
        self.assertTrue(torch.equal(r1['fused'][1],r0['fused'][1]))
        self.assertTrue(torch.equal(r1['full'],r0['full']))
        self.assertTrue(torch.equal(r1['person'],r0['person']))

    def test_weight_sharing_and_parameters_remain_frozen(self):
        a,b=self.pair();wrapper=ProtectedViewFusion(a,b)
        self.assertIs(a.visual_encoder,b.visual_encoder);self.assertIs(a.head,b.head)
        self.assertTrue(all(not p.requires_grad for p in wrapper.parameters()))
        before={k:tensor_state_digest(method_state(m)) for k,m in (('A',a),('B',b))}
        with torch.no_grad():wrapper(self.images(),'R3')
        self.assertTrue(all(before[k]==tensor_state_digest(method_state(m)) for k,m in (('A',a),('B',b))))

    def test_head_or_visual_mismatch_is_rejected(self):
        a,b=self.pair()
        with torch.no_grad():b.head.weight.add_(.1)
        with self.assertRaises(ValueError):ProtectedViewFusion(a,b)
        b.head.load_state_dict(a.head.state_dict())
        with torch.no_grad():next(b.visual_encoder.parameters()).add_(.1)
        with self.assertRaises(ValueError):ProtectedViewFusion(a,b)

    def test_no_grad_and_B_requirements(self):
        a,_=self.pair();wrapper=ProtectedViewFusion(a)
        with self.assertRaises(RuntimeError):wrapper(self.images(),'R0')
        with torch.no_grad():
            with self.assertRaises(ValueError):wrapper(self.images(),'R1')

    def test_task_class_ownership_does_not_select_best_lane(self):
        a,b=self.pair();wrapper=ProtectedViewFusion(a,b)
        z=torch.tensor([[[1.,2.,999.],[999.,999.,3.]]])
        self.assertTrue(torch.equal(wrapper.combine(z),torch.tensor([[1.,2.,3.]])))

    def test_saved_source_replay_checks_ids_and_each_view(self):
        a,b=self.pair();wrapper=ProtectedViewFusion(a,b);x=self.images();ids=['a','b','c'];y=torch.ones(3,3)
        with torch.no_grad():
            features,sources=wrapper.sources(x)
            outputs={arm:wrapper.compose(x,features,arm)[0] for arm in ARMS}
        ref={}
        for source,z in sources.items():
            ref[source]=dict(sample_ids=np.array(ids),targets=y.numpy(),face_reliable=x['face_reliable'].numpy())
            for view,v in z.items():
                ref[source][view+'_logits']=v.numpy().copy();ref[source][view+'_probabilities']=torch.sigmoid(v).numpy()
        verify_batch(ref,0,ids,y,x['face_reliable'],sources,outputs)
        with self.assertRaises(AssertionError):verify_batch(ref,0,ids[::-1],y,x['face_reliable'],sources,outputs)
        ref['B']['face_logits'][0,0]+=1
        with self.assertRaises(AssertionError):verify_batch(ref,0,ids,y,x['face_reliable'],sources,outputs)


if __name__=='__main__':unittest.main()
