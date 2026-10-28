"""Mechanism tests for read-only epoch and classifier attribution."""
import copy
import unittest
import torch
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import TASK_SIZES, tensor_state_digest
from multi_lane.track_a.teacher_free_epoch_head_attribution import (
    ARMS, EPOCHS, CONDITIONS, VIEWS, label, checkpoint_path, readout,
    probability_rows, bce_summary, factorial, verify_replay, install_state,
    validate_payload, summarize_case, state_drift)
from multi_lane.track_a.run_teacher_free_epoch_head import command
from test_track_a_reproduction import FakeVisual
from pathlib import Path
from types import SimpleNamespace


class EpochHeadAttributionTest(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(70)
        m=MultiLaneModel(FakeVisual(),TASK_SIZES,num_selectors=2,num_prompts=2,
            num_prompt_layers=1,adapter_mode="image_token",adapter_bottleneck_dim=4,
            adapter_layer_indices=(0,),adapter_residual_scale=.03,view_fusion="fixed_three_view")
        m.activate_task(0);m.requires_grad_(False).eval()
        x={v:torch.randn(4,3,4,4) for v in ("full","person","face")}
        x["face_reliable"]=torch.tensor([True,False,True,False])
        return m,x

    def test_seven_conditions_are_two_complete_two_by_twos(self):
        expected={("U0","U0"),("U0","U2"),("U2","U0"),("U2","U2"),
                  ("U0","U4"),("U4","U0"),("U4","U4")}
        self.assertEqual(set(CONDITIONS),expected)
        self.assertEqual(EPOCHS,(1,5,10,20,30))
        self.assertEqual(len({label(*c) for c in CONDITIONS}),7)
        with self.assertRaises(ValueError):label("U2","U4")
        with self.assertRaises(ValueError):checkpoint_path(Path("."),"U4",15)

    def test_native_feature_readout_is_bitwise_model_prediction(self):
        m,x=self.fixture();before=tensor_state_digest(m.state_dict())
        with torch.no_grad():
            fused,views=m.encode_lanes_with_views(x,True)
            got=readout(dict(fused=fused,**views),m.head.weight,m.head.bias,
                m.task_class_mask[[0]].to(fused.dtype))
            original,branches=m.seen_logits_with_views(x)
        for k,v in dict(fused=original,**branches).items():
            self.assertTrue(torch.equal(got[k],v),k)
        self.assertEqual(before,tensor_state_digest(m.state_dict()))
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_cross_head_uses_same_features_without_mutating_native_head(self):
        m,x=self.fixture()
        with torch.no_grad():
            fused,views=m.encode_lanes_with_views(x,True);features=dict(fused=fused,**views)
            before=tensor_state_digest(m.state_dict())
            head=m.head.weight.clone();bias=m.head.bias.clone()+.25
            native=readout(features,head,m.head.bias,m.task_class_mask[[0]])
            changed=readout(features,head,bias,m.task_class_mask[[0]])
        for v in VIEWS:
            self.assertTrue(torch.allclose(changed[v]-native[v],torch.full_like(native[v],.25),atol=1e-6))
        self.assertEqual(before,tensor_state_digest(m.state_dict()))

    def test_probability_rows_and_reference_replay_detect_wrong_population(self):
        values={v:torch.tensor([[.2,-.3],[.5,.9]]) for v in VIEWS}
        z,p=probability_rows(values)
        ref={"sample_ids":["a","b"],"targets":torch.tensor([[1.,0.],[0.,1.]]).numpy()}
        actual={}
        for v in VIEWS:
            actual[v+"_logits"]=z[v].numpy();actual[v+"_probabilities"]=p[v].numpy()
            ref.update({v+"_logits":z[v].numpy(),v+"_probabilities":p[v].numpy()})
        y=torch.tensor([[1.,0.],[0.,1.]])
        self.assertTrue(verify_replay(actual,ref,["a","b"],y)["all_logits_and_probabilities_exact"])
        with self.assertRaises(ValueError):verify_replay(actual,ref,["b","a"],y)
        actual["full_logits"]=actual["full_logits"].copy()+.01
        with self.assertRaises(AssertionError):verify_replay(actual,ref,["a","b"],y)

    def test_bce_is_stable_and_masks_reliable_face(self):
        z=torch.tensor([[100.,-100.],[0.,0.]])
        y=torch.tensor([[1.,0.],[1.,0.]])
        b=bce_summary(z,y)
        expected=torch.nn.functional.binary_cross_entropy_with_logits(z.double(),y.double())
        self.assertAlmostEqual(b["mean"],float(expected),places=12)
        self.assertEqual(len(b["per_class"]),2)
        values={v:torch.randn(4,5) for v in VIEWS}
        probabilities={v:torch.sigmoid(z) for v,z in values.items()}
        labels=torch.eye(5)[:4]
        mask=torch.tensor([True,False,True,False])
        got=summarize_case(values,probabilities,labels,mask)
        self.assertEqual(got["face_reliable"]["metrics"]["samples"],2)
        self.assertAlmostEqual(got["face_reliable"]["bce"]["mean"],
            bce_summary(values["face"][mask],labels[mask])["mean"])

    def test_factorial_contrasts_sum_to_total_and_zero_under_identity(self):
        values={}
        for i,(r,h) in enumerate(CONDITIONS):
            a=float(i+1)
            values[label(r,h)]={v:{"metrics":{"mAP":a},"bce":{"mean":a/10}}
                               for v in (*VIEWS,"face_reliable")}
        rows=factorial(values)
        for row in rows:
            self.assertAlmostEqual(row["representation_at_baseline_head"]+
                row["head_at_baseline_representation"]+row["interaction"],row["total"])
        same={k:values[label("U0","U0")] for k in values}
        self.assertTrue(all(row["total"]==0 for row in factorial(same)))

    def test_install_state_and_provenance_reject_missing_or_wrong_epoch(self):
        m,x=self.fixture()
        state={k:v.clone() for k,v in m.state_dict().items() if not k.startswith("visual_encoder.")}
        visual=tensor_state_digest(m.visual_encoder.state_dict())
        install_state(m,state)
        self.assertEqual(visual,tensor_state_digest(m.visual_encoder.state_dict()))
        bad=dict(state);del bad["head.weight"]
        with self.assertRaises(ValueError):install_state(m,bad)
        cfg={"arm":"U4","git":{"commit":"source"}}
        payload={"config":cfg,"task_id":0,"training_state":{"next_epoch":1,
            "history":[{"complementary_ramp":0}]},"provenance":{"next_epoch":1},
            "source_git":cfg["git"],"model":state}
        self.assertIs(validate_payload(payload,cfg,1),state)
        with self.assertRaises(ValueError):validate_payload(payload,cfg,5)
        payload["training_state"]["history"][0]["complementary_ramp"]=1.
        with self.assertRaises(ValueError):validate_payload(payload,cfg,1)

    def test_command_is_eval_only_and_smoke_keeps_fixed_protocol(self):
        args=SimpleNamespace(source_root=Path("source"),data_root=Path("data"),
            clip_checkpoint=Path("clip"),face_manifest_root=Path("face"))
        cmd=command(args,"test",Path("out"))
        self.assertNotIn("train",cmd)
        self.assertNotIn("--smoke",cmd)
        self.assertIn("--smoke",command(args,"val",Path("smoke"),True))


if __name__=="__main__":
    unittest.main()
