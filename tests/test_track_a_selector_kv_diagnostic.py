import unittest
from unittest.mock import patch
import torch

from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.selector_kv_diagnostic import selector_sample_diagnostics
from multi_lane.track_a.static_moe_kv_pipeline import assess_value_only_gate
from test_track_a_reproduction import FakeVisual


class SelectorKeyValueTest(unittest.TestCase):
    def model(self):
        torch.manual_seed(13)
        return MultiLaneModel(FakeVisual(), (2, 1), num_selectors=2, num_prompts=2,
                              num_prompt_layers=1, adapter_mode="image_token",
                              adapter_bottleneck_dim=6, adapter_layer_indices=(0,),
                              static_moe_mode="overlap", view_fusion="fixed_three_view")

    def test_modes_match_explicit_key_value_equations(self):
        model=self.model();model.activate_task(0)
        block=model.visual_encoder.transformer.resblocks[0]
        width=model.width
        image=torch.randn(2,5,width);lane=torch.randn(1,2,3,width)
        frozen=block.ln_1(image).detach();query=block.ln_1(lane)[:,:,1:]
        delta=torch.randn_like(frozen)*2
        adapted=frozen.unsqueeze(0)+delta.unsqueeze(0)
        for mode in ('both','key_only','value_only','none'):
            model.set_selector_evidence_mode(mode)
            model.enable_selector_sample_diagnostics(0)
            capture=[]
            def prompt(block,summarized,*args,**kwargs):
                capture.append(summarized)
                return torch.zeros_like(summarized)
            with patch.object(model.adapter_bank,'adapted_tokens_for_layer',return_value=adapted), \
                 patch.object(model,'_prompt_attention',side_effect=prompt):
                model._lane_block(block,image,lane,(0,),0)
            key=adapted if mode in ('both','key_only') else frozen.unsqueeze(0)
            value=adapted if mode in ('both','value_only') else frozen.unsqueeze(0)
            weights=torch.softmax(torch.einsum('tbsc,tbnc->tbsn',query,key)*width**-.5,-1)
            expected=torch.einsum('tbnc,tbsn->tbsc',value,weights)
            torch.testing.assert_close(capture[0][:,:,1:],expected,rtol=0,atol=0)
            records=model.pop_selector_sample_diagnostics()
            self.assertEqual(len(records),1)
            kl=records[0]['values']['attention_kl_current_to_frozen']
            if mode in ('none','value_only'):
                self.assertTrue(torch.equal(kl,torch.zeros_like(kl)))
            else:
                self.assertGreater(float(kl.max()),0)

    def test_residual_energy_decomposition_and_attention_token_overlap(self):
        h=torch.randn(2,20,8)
        d=torch.randn_like(h)
        q=torch.softmax(torch.arange(20).float().view(1,1,-1).expand(2,3,-1),-1)
        p=q.flip(-1)
        s=selector_sample_diagnostics(q,p,h,h+d)
        self.assertTrue(torch.equal(s['attention_top10_overlap'],torch.zeros(2)))
        self.assertTrue((s['attention_kl_current_to_frozen']>0).all())
        torch.testing.assert_close(s['residual_energy_ratio'],s['centered_residual_energy_ratio']+s['constant_residual_energy_ratio'])
        c=selector_sample_diagnostics(q,q,h,h+torch.ones_like(h))
        self.assertLess(float(c['centered_residual_energy_ratio'].max()),1e-12)
        self.assertTrue(torch.equal(c['attention_top10_overlap'],torch.ones(2)))

    def test_diagnostic_observation_does_not_change_outputs_or_state(self):
        model=self.model();model.activate_task(0);model.eval()
        with torch.no_grad():
            for e in model.adapter_bank.task_adapters[0]['0'].experts:e.up.weight.normal_(0,.1)
        inputs={v:torch.randn(2,3,4,4) for v in ('full','person','face')}
        inputs['face_reliable']=torch.tensor([True,False])
        with torch.no_grad():
            expected=model.seen_logits(inputs)
            before={k:v.clone() for k,v in model.state_dict().items()}
            model.enable_selector_sample_diagnostics(0)
            actual=model.seen_logits(inputs)
            self.assertTrue(torch.equal(expected,actual))
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
            model.set_selector_evidence_mode('none');disabled=model.seen_logits(inputs)
            model.set_adapter_runtime_enabled(False);historical=model.seen_logits(inputs)
            torch.testing.assert_close(disabled,historical,rtol=1e-5,atol=1e-6)

    def test_invalid_mode_and_task_rejected(self):
        model=self.model()
        with self.assertRaises(ValueError):model.set_selector_evidence_mode('test')
        with self.assertRaises(ValueError):model.enable_selector_sample_diagnostics(2)

    def test_value_only_gate_needs_both_versions_and_average_direction(self):
        curves={mode:{'final_mAP':value,'average_mAP':value}
                for mode,value in {'both':42.,'value_only':42.3,'key_only':41.8,'none':41.9}.items()}
        source={'curve_summary':curves}
        import copy
        s={k:copy.deepcopy(source) for k in ('v1_MOE_OVERLAP','v2_MOE_OVERLAP')}
        self.assertTrue(assess_value_only_gate(s)['supports_implementing_value_only'])
        s['v2_MOE_OVERLAP']['curve_summary']['value_only']['average_mAP']=40
        self.assertFalse(assess_value_only_gate(s)['supports_implementing_value_only'])
