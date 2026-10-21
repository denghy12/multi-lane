import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from multi_lane.track_a.evaluate_cross_view_attribution import attribution_inputs
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import tensor_state_digest
from multi_lane.track_a.cross_view_query_pairs import eval_geometry,make_pairs
from multi_lane.track_a.paired_transforms import PairedFullPersonTransform
import test_track_a_reproduction as fixtures


class AttributionTest(unittest.TestCase):
    def model(self, mode='person_full'):
        torch.manual_seed(71);v=fixtures.FakeVisual()
        v.transformer.resblocks.append(copy.deepcopy(v.transformer.resblocks[0]))
        m=MultiLaneModel(v,(2,1),num_selectors=2,num_prompts=2,num_prompt_layers=1,
                        view_fusion='fixed_three_view',cross_view_mode=mode,
                        cross_view_layer=1,cross_view_dim=4,cross_view_heads=2)
        m.activate_task(0);m.activate_task(1);m.eval().requires_grad_(False)
        with torch.no_grad():
            for t in m.cross_view_bank.tasks:t.out.weight.normal_(0,.3)
        return m

    def images(self):
        return {**{v:torch.randn(3,3,4,4) for v in ('full','person','face')},
                'face_reliable':torch.tensor([True,False,True])}

    def test_identity_query_replays_all_views_and_weights(self):
        m=self.model();images=self.images();before=tensor_state_digest(m.state_dict())
        with torch.no_grad():
            expected,views=m.seen_logits_with_views(images)
            alt=dict(images,cross_view_query_donor=images['person'],cross_view_query_mask=torch.ones(3,dtype=torch.bool))
            actual,actual_views=m.seen_logits_with_views(alt)
        self.assertTrue(torch.equal(actual,expected))
        self.assertTrue(all(torch.equal(views[k],actual_views[k]) for k in views))
        self.assertEqual(before,tensor_state_digest(m.state_dict()))

    def test_only_selected_person_queries_can_change(self):
        m=self.model();images=self.images();mask=torch.tensor([True,False,True])
        with torch.no_grad():
            expected,views=m.seen_logits_with_views(images)
            alt=dict(images,cross_view_query_donor=images['person'].roll(1,0),cross_view_query_mask=mask)
            actual,new=m.seen_logits_with_views(alt)
        self.assertTrue(torch.equal(views['full'],new['full']))
        self.assertTrue(torch.equal(views['face'],new['face']))
        self.assertTrue(torch.equal(expected[~mask],actual[~mask]))
        self.assertFalse(torch.equal(views['person'][mask],new['person'][mask]))

    def test_capture_queries_never_replaces_residual_or_key_values(self):
        m=self.model();images=self.images();captures=[]
        original=m.cross_view_bank.forward
        def capture(*args,**kwargs):
            captures.append(([v.clone() if isinstance(v,torch.Tensor) else v for v in args],kwargs))
            return original(*args,**kwargs)
        with torch.no_grad(),patch.object(m.cross_view_bank,'forward',side_effect=capture):
            m.seen_logits(images)
            m.seen_logits(dict(images,cross_view_query_donor=images['person'].roll(1,0),
                               cross_view_query_mask=torch.ones(3,dtype=torch.bool)))
        self.assertEqual(len(captures),2)
        self.assertTrue(all(torch.equal(captures[0][0][i],captures[1][0][i]) for i in (0,1,2)))
        self.assertFalse(torch.equal(captures[1][1]['query_override'],captures[1][0][0]))

    def test_residual_off_is_local_and_restorable(self):
        for mode in ('task_full','person_full'):
            m=self.model(mode);images=self.images();before=tensor_state_digest(m.state_dict())
            with torch.no_grad():
                on,views=m.seen_logits_with_views(images);m.cross_view_runtime_enabled=False
                off,other=m.seen_logits_with_views(images);m.cross_view_runtime_enabled=True
                restored=m.seen_logits(images)
            self.assertFalse(torch.equal(on,off));self.assertTrue(torch.equal(on,restored))
            self.assertTrue(all(torch.equal(views[v],other[v]) for v in ('full','face')))
            self.assertEqual(before,tensor_state_digest(m.state_dict()))

    def test_query_override_is_eval_only_and_requires_person_full(self):
        for mode in ('task_full','person_full'):
            m=self.model(mode);images=self.images()
            images.update(cross_view_query_donor=images['person'],cross_view_query_mask=torch.ones(3,dtype=torch.bool))
            with self.assertRaises(ValueError):m.seen_logits(images)
            if mode=='task_full':
                with torch.no_grad(),self.assertRaises(ValueError):m.seen_logits(images)
            m.train()
            with torch.no_grad(),self.assertRaises(ValueError):m.seen_logits(images)

    def test_failure_clears_donor_cache(self):
        m=self.model();images=self.images()
        alt=dict(images,cross_view_query_donor=images['person'],cross_view_query_mask=torch.ones(3,dtype=torch.bool))
        with torch.no_grad(),patch.object(m.cross_view_bank,'forward',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):m.seen_logits(alt)
        with torch.no_grad():self.assertTrue(torch.isfinite(m.seen_logits(alt)).all())
        self.assertFalse(any(isinstance(v,dict) and 'person_query' in v for v in vars(m).values()))

    def test_false_mask_ignores_arbitrary_donors(self):
        m=self.model();images=self.images()
        with torch.no_grad():
            a=m.seen_logits(images)
            b=m.seen_logits(dict(images,cross_view_query_donor=torch.randn_like(images['person'])*100,
                                cross_view_query_mask=torch.zeros(3,dtype=torch.bool)))
        self.assertTrue(torch.equal(a,b))

    def test_input_mover_preserves_query_mask_and_legacy_casts(self):
        images=self.images()
        images.update(cross_view_query_donor=images['person'],cross_view_query_mask=torch.tensor([True,False,True]))
        moved=attribution_inputs(images,torch.device('cpu'))
        self.assertEqual(moved['cross_view_query_mask'].dtype,torch.bool)
        self.assertTrue(torch.equal(moved['cross_view_query_mask'],images['cross_view_query_mask']))
        self.assertEqual(moved['face_reliable'].dtype,torch.float32)
        images['cross_view_query_mask']=images['cross_view_query_mask'].float()
        with self.assertRaises(ValueError):attribution_inputs(images,torch.device('cpu'))

    def test_geometry_matches_actual_transform_odd_image_sizes(self):
        tr=PairedFullPersonTransform(False,jitter_strength=0,jitter_probability=0)
        for size in ((321,181),(127,319),(255,256),(511,511)):
            bbox=[size[0]*.1,size[1]*.2,size[0]*.75,size[1]*.95]
            actual=tr(Image.new('RGB',size),bbox)['bbox']
            geometry,_=eval_geometry(bbox,size)
            self.assertTrue(torch.equal(actual,geometry),(size,actual,geometry))

    def test_pairing_is_label_blind_same_image_nonself_and_deterministic(self):
        ids=['a#person=1','b#person=0','a#person=0','a#person=2']
        boxes=[[45,60,80,150],[60,60,100,150],[90,60,125,150],[135,60,170,150]]
        sizes=[(224,224)]*4
        a=make_pairs(ids,boxes,sizes);b=make_pairs(ids,boxes,sizes)
        self.assertEqual(a,b);self.assertEqual(a['eligible_people'],3)
        for i,row in enumerate(a['rows']):
            j=row['donor_index']
            if row['eligible']:
                self.assertNotEqual(i,j);self.assertEqual(ids[i].split('#')[0],ids[j].split('#')[0])
                self.assertTrue(a['rows'][j]['sufficiently_visible'])
            else:self.assertEqual(i,j)

    def test_partial_visibility_single_visible_and_duplicate_boxes_excluded(self):
        ids=['a#person=0','a#person=1','b#person=0','b#person=1']
        boxes=[[90,50,140,150],[0,0,25,224],[60,60,100,160],[60,60,100,160]]
        p=make_pairs(ids,boxes,[(224,224)]*4)
        self.assertEqual(p['eligible_people'],0)
        self.assertTrue(all(r['donor_index']==i for i,r in enumerate(p['rows'])))

if __name__=='__main__':unittest.main()
