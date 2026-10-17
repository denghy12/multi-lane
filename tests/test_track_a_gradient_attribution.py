from __future__ import annotations
import unittest
import torch

from multi_lane.track_a.gradient_attribution import (
    ParameterView, configure_precision, geometry, grouped_batches, measure,
    objective_losses,
)
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import TASK_SIZES, tensor_state_digest
from test_track_a_reproduction import FakeVisual


class GradientAttributionTest(unittest.TestCase):
    def make_model(self, mode="dynamic"):
        model = MultiLaneModel(
            FakeVisual(), TASK_SIZES, num_selectors=2, num_prompts=2,
            num_prompt_layers=1, view_fusion="fixed_three_view",
            adapter_mode="disabled" if mode=="baseline" else "moe_task_ffn_v4",
            adapter_bottleneck_dim=4, adapter_layer_indices=(0,),
            moe_num_experts=4, moe_noise_std=0, moe_route_mode=mode if mode!="baseline" else "dynamic",
        )
        model.activate_task(0)
        if model.adapter_bank is not None:
            with torch.no_grad():
                for expert in model.adapter_bank.tasks[0][0].experts:
                    expert.up.weight.normal_(std=.1)
        return model

    @staticmethod
    def inputs():
        images = {v:torch.randn(4,3,4,4) for v in ("full","person","face")}
        images["face_reliable"] = torch.tensor([True,False,True,False])
        return images, torch.tensor([[1.,0.,1.,0.,0.],[0.,1.,0.,0.,1.],
                                     [0.,0.,1.,1.,0.],[1.,1.,0.,0.,0.]])

    @staticmethod
    def config(mode="dynamic"):
        return dict(temperature=1.,training_loss_mode="legacy_full_zero",
                    view_auxiliary_loss_weight=.1,moe_balance_weight=.001 if mode!="baseline" else None)

    def test_image_groups_are_disjoint_and_repeatable(self):
        ids=[f"train:image{i}#person={j}" for i in range(12) for j in range(2)]
        rows=grouped_batches(ids,7,3,0)
        self.assertEqual(rows,grouped_batches(ids,7,3,0))
        seen=set()
        for row in rows:
            self.assertLessEqual(len(row),7)
            groups={ids[i].split('#')[0] for i in row}
            self.assertFalse(seen & groups)
            seen |= groups
            self.assertTrue(all(sum(ids[i].split('#')[0]==g for i in row)==2 for g in groups))
        with self.assertRaises(ValueError):
            grouped_batches(['same#person='+str(i) for i in range(9)],8,1,0)

    def test_zero_missing_and_path_head_are_not_false_nonconflicts(self):
        values={"left":{"selector":torch.tensor([1.,0.]),"base":torch.tensor([1.,0.]),"classifier":torch.tensor([1.])},
                "right":{"selector":torch.tensor([-1.,0.]),"base":torch.tensor([-1.,0.]),"classifier":torch.tensor([-1.])},
                "zero":{"selector":torch.zeros(2),"base":torch.zeros(2),"classifier":torch.zeros(1)},
                "fused_path_full":{"selector":torch.tensor([1.,0.]),"base":torch.tensor([1.,0.]),"classifier":torch.zeros(1)}}
        names=list(values);g=geometry(values,names)
        self.assertAlmostEqual(g['selector']['cosines'][0][1],-1.)
        self.assertIsNone(g['selector']['cosines'][0][2])
        self.assertIsNone(g['base']['cosines'][0][3])
        self.assertIsNone(g['base']['cosines'][3][0])
        self.assertIsNone(g['classifier']['norms'][3])

    def test_face_auxiliary_excludes_unreliable_and_masks_hidden_classes(self):
        model=self.make_model();ParameterView(model,0);configure_precision(model,'deterministic_fp32')
        images,y=self.inputs();loss,_,n,_=objective_losses(model,images,y,tuple(range(5)),self.config())
        self.assertEqual(n,3)
        expected=.1*(loss['view_full']+loss['view_person']+loss['view_face_reliable'])/3
        self.assertTrue(torch.allclose(expected,loss['auxiliary']))
        self.assertIn('face_unreliable_shadow',loss)
        self.assertTrue(torch.allclose(.5*loss['fused_reliable']+.5*loss['fused_unreliable'],loss['fused']))
        grad=torch.autograd.grad(loss['total'],model.head.weight,retain_graph=True)[0]
        self.assertEqual(grad[5:].count_nonzero(),0)
        images['face_reliable'].zero_()
        loss,_,n,_=objective_losses(model,images,y,tuple(range(5)),self.config())
        self.assertEqual(n,2);self.assertNotIn('view_face_reliable',loss)
        self.assertTrue(torch.allclose(loss['auxiliary'],.05*(loss['view_full']+loss['view_person'])))

    def test_vjp_reconstruction_readonly_and_counterfactual_anchor(self):
        model=self.make_model();before=tensor_state_digest(model.state_dict());pack=ParameterView(model,0)
        images,y=self.inputs()
        record,_=measure(model,pack,images,y,self.config(),'deterministic_fp32',0)
        self.assertEqual(record['face_unreliable_fused_feature_gradient_max'],0)
        self.assertTrue(all(x<1e-4 for x in record['fused_path_reconstruction_relative_error'].values()))
        self.assertIn('supervised_on_vs_same_parameter_moe_off',record)
        self.assertEqual(before,tensor_state_digest(model.state_dict()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.visual_encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.adapter_bank.tasks[1].parameters()))

    def test_scale_is_unscaled_and_active_coordinates_are_current_only(self):
        model=self.make_model();pack=ParameterView(model,0);images,y=self.inputs()
        configure_precision(model,'deterministic_fp32')
        loss,_,_,_=objective_losses(model,images,y,tuple(range(5)),self.config())
        a,b=pack.grad(loss['supervised'],1),pack.grad(loss['supervised'],1024)
        for group in a:
            self.assertTrue(torch.allclose(a[group],b[group],atol=1e-7,rtol=1e-5),group)
        self.assertEqual(a['selector'].numel(),2*8)
        self.assertEqual(a['prompt'].numel(),2*2*8)
        self.assertEqual(a['classifier'].numel(),5*(4+1))
        self.assertGreater(a['shared_expert'].norm(),0)

    def test_baseline_has_no_fictitious_expert_gradients(self):
        model=self.make_model('baseline');pack=ParameterView(model,0);images,y=self.inputs()
        r,_=measure(model,pack,images,y,self.config('baseline'),'deterministic_fp32',0)
        self.assertEqual(r['geometry']['shared_expert']['dimension'],0)
        self.assertNotIn('supervised_on_vs_same_parameter_moe_off',r)


if __name__=='__main__':unittest.main()
