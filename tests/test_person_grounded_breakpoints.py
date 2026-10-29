import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from multi_lane.track_a.person_grounded_summary import GroundedReadout
from multi_lane.track_a.person_grounded_breakpoints import (
    FIELDS, compare_fields, full_population, head_projection, pair_geometry, spatial_response, trace_module)


class BreakpointTests(unittest.TestCase):
    def inputs(self, device='cpu'):
        torch.manual_seed(948)
        tokens={v:torch.randn(4,5,8,device=device) for v in ('full','person','face')}
        anchors={v:torch.randn(4,10,8,device=device) for v in tokens}
        return (tokens,anchors,torch.randn(4,8,device=device),torch.randn(4,8,device=device),
            torch.tensor([[0.,0.,.5,1.],[.5,0.,1.,1.]]*2,device=device),
            torch.ones(4,device=device),torch.tensor([True,False,True,False],device=device))

    def test_tracing_preserves_original_native_and_parameters(self):
        for arm in ('PERSON_CONST_POOL','PERSON_CONTENT_ATTN','PERSON_GROUND_ATTN'):
            m=GroundedReadout(8,8,10,arm,4,2);inputs=self.inputs()
            with torch.no_grad(): m.out.weight.normal_();m.condition_output.normal_()
            before={k:v.clone() for k,v in m.state_dict().items()}
            trace,summaries=trace_module(m,*inputs)
            with torch.no_grad(): expected=m(inputs[2],inputs[3],summaries,inputs[-1])
            self.assertTrue(torch.equal(trace['delta'],expected))
            self.assertTrue(all(torch.equal(v,before[k]) for k,v in m.state_dict().items()))
            self.assertTrue(all(p.grad is None for p in m.parameters()))
            self.assertTrue(all(not t.requires_grad for t in trace.values()))
            self.assertEqual(len(m.out._forward_hooks),0)

    def test_person_intervention_isolated_from_other_memories_and_late_query(self):
        m=GroundedReadout(8,8,10,'PERSON_CONTENT_ATTN',4,2);x=self.inputs();donor=torch.tensor([1,0,3,2])
        with torch.no_grad(): m.out.weight.normal_();m.condition_output.normal_()
        own,ms=trace_module(m,*x);swap,ds=trace_module(m,*x,condition_person=x[3][donor])
        self.assertFalse(torch.equal(own['query'],swap['query']))
        self.assertTrue(torch.equal(own['readout_query'],swap['readout_query']))
        for n in ('global','person','face'):self.assertTrue(torch.equal(ms[n],ds[n]))

    def test_constant_negative_control_keeps_all_internal_stages(self):
        m=GroundedReadout(8,8,10,'PERSON_CONST_ATTN',4,2);x=self.inputs()
        with torch.no_grad():m.out.weight.normal_();m.condition_output.normal_()
        own,_=trace_module(m,*x);swap,_=trace_module(m,*x,condition_person=x[3].flip(0))
        for name in FIELDS:
            if name!='person_ln':self.assertTrue(torch.equal(own[name],swap[name]),name)

    def test_area_and_exclusive_mass_control_overlapping_boxes(self):
        b=torch.tensor([[0.,0.,.75,1.],[.25,0.,1.,1.]])
        g=pair_geometry(b,torch.ones(2),torch.tensor([1,0]),4)
        self.assertTrue(torch.allclose(g['own_area'],torch.full((2,),.75)))
        self.assertTrue(torch.allclose(g['own_exclusive_area'],torch.full((2,),.25)))
        self.assertTrue(torch.allclose(g['bbox_iou'],torch.full((2,),.5)))
        alpha=torch.tensor([[[.5,0.,.5,0.]],[[0.,.5,0.,.5]]])
        r=spatial_response({'alpha':alpha},{'alpha':alpha.flip(0)},g)
        self.assertTrue((r['binding_mass_gain']>0).all())

    def test_head_projection_uses_owned_classes_and_excludes_bias(self):
        head=torch.nn.Linear(8,26);model=SimpleNamespace(head=head,_task_sizes=(5,3,3,3,3,3,3,3))
        h=torch.randn(4,3,8);d=torch.randn_like(h)*.01
        b,w=head_projection(model,h,d,(0,1,2))
        self.assertEqual(w.shape,(4,11))
        expected=torch.cat([torch.nn.functional.linear(d[:,i].double(),head.weight[a:z].double())
            for i,(a,z) in enumerate(((0,5),(5,8),(8,11)))],-1)
        self.assertTrue(torch.equal(w,expected))
        with torch.no_grad():head.bias.add_(100)
        _,again=head_projection(model,h,d,(0,1,2));self.assertTrue(torch.equal(w,again))

    def test_comparison_keeps_small_differences_and_saturation_separate(self):
        m=GroundedReadout(8,8,10,'PERSON_CONTENT_POOL',4,2);x=self.inputs()
        own,_=trace_module(m,*x);swap={k:v.clone() for k,v in own.items()}
        swap['query'][0,0,0]+=1e-5
        r=compare_fields(own,swap)
        self.assertGreater(float(r['query_diff_rms'][0]),0)
        self.assertEqual(float(r['alpha_total_variation'].sum()),0)

    def test_small_tail_keeps_every_official_score_and_marks_projection_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            r=Path(directory);ids=np.array(['a','b','c','d']);y=torch.ones(4,5)
            z=np.arange(20,dtype=np.float32).reshape(4,5);p=np.full((4,5),.6,dtype=np.float32)
            for mode in ('on','off'):
                folder=r/mode/'view_scores';folder.mkdir(parents=True)
                np.savez_compressed(folder/'task0.npz',sample_ids=ids,targets=y.numpy(),
                    fused_logits=z,fused_probabilities=p)
            x=dict(face_reliable=torch.zeros(4,dtype=torch.bool),spatial_visibility=torch.ones(4),spatial_bbox=torch.zeros(4,4))
            result=full_population(None,[(x,y,list(ids))],r,0,r)
            self.assertEqual(result['samples'],4);self.assertEqual(result['tensor_forward_samples'],0)
            with np.load(r/'full.npz') as data:
                self.assertTrue(np.array_equal(data['native_probabilities'],p))
                self.assertTrue(np.array_equal(data['sample_ids'],ids))
                self.assertFalse(data['projection_available'].any())
                self.assertTrue(np.isnan(data['w_delta_fp64']).all())

    @unittest.skipUnless(torch.cuda.is_available(),'actual CUDA autocast required')
    def test_trace_inside_real_cuda_autocast_stays_fp32(self):
        m=GroundedReadout(8,8,10,'PERSON_GROUND_ATTN',4,2).cuda();x=self.inputs('cuda')
        with torch.cuda.amp.autocast(enabled=True):trace,_=trace_module(m,*x)
        for name in FIELDS:self.assertEqual(trace[name].dtype,torch.float32,name)


if __name__=='__main__':unittest.main()
