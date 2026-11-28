import os,json,tempfile
from pathlib import Path
from unittest import TestCase,main
from unittest.mock import patch
from dataclasses import replace
import numpy as np
import torch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from multi_lane.track_a.task_patch_fusion import (TaskPatchFusion,candidates,bounded_score,
    restore_gate,prediction_key,control_key,LEGACY)
from multi_lane.track_a.final_patch_fusion import FinalPatchFusion
from multi_lane.track_a.summary_logit_fusion import masked_weights
from multi_lane.track_a.d9_relation_fusion import D9RelationFusion
from scripts.emotic.d9_task_patch_run import CONFIG,FIT,check
from scripts.emotic.d9_task_patch_readout import fit_one,mix_heads,train,evaluate,METHODS
from scripts.emotic.d9_a_eight_eval import read
from scripts.emotic.r1_asymmetric_forward import fuse_logits
from test_d9_final_patch import population,args
from test_d9_protected_multi_model import branches


class TaskPatchTests(TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_all_task_capture_matches_head_inputs_and_individual_old_path(self):
        model=D9RelationFusion(branches(),'sample_holdout')
        model.activate_task(0);model.activate_task(1);model.requires_grad_(False).eval()
        data={k:torch.randn(3,3,4,4) for k in ('full','person','face','evidence_full','evidence_person','evidence_face')}
        data['evidence_face_reliable']=torch.tensor([True,False,True])
        captured={};current={};head_inputs={};handles=[]
        for v,b in model.branches.items():
            handles.append(b.head.register_forward_pre_hook(lambda m,x,v=v:head_inputs.update({v:x[0].clone()})))
        original={k:v.clone() for k,v in model.state_dict().items()}
        with torch.inference_mode():
            expected=model(data,all_seen=True)
            actual=model(data,all_seen=True,task_final_tokens=captured,final_tokens=current)
            for v in captured:
                self.assertEqual(set(captured[v]),{0,1})
                torch.testing.assert_close(expected['branches'][v],actual['branches'][v],rtol=0,atol=0)
                for t in (0,1):
                    torch.testing.assert_close(captured[v][t]['cls'],head_inputs[v][:,t],rtol=0,atol=0)
                    self.assertFalse(captured[v][t]['patch'].requires_grad)
                for k in ('cls','patch'):torch.testing.assert_close(current[v][k],captured[v][1][k],rtol=0,atol=0)
            model.restore_task(0);model.requires_grad_(False)
            old={};model(data,all_seen=True,final_tokens=old)
            for v in old:
                for k in ('cls','patch'):torch.testing.assert_close(old[v][k],captured[v][0][k],rtol=1e-6,atol=1e-6)
        for h in handles:h.remove()
        for k,v in original.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
        with self.assertRaises(ValueError):model(data,all_seen=True,task_final_tokens={})

    def test_both_intervals_preserve_zero_initialization_and_gradient_scale(self):
        for lower in (.25,.5):
            raw=torch.tensor([-100.,-.001,0.,.001,100.],requires_grad=True)
            s=bounded_score(raw,lower)
            self.assertEqual(float(s[2]),0)
            self.assertGreaterEqual(float(s.min()),-lower)
            self.assertLessEqual(float(s.max()),.25)
            s.sum().backward()
            self.assertAlmostEqual(float(raw.grad[2]),.25,places=7)
            self.assertAlmostEqual(float(raw.grad[1]),float(raw.grad[3]),places=6)
            w=masked_weights(torch.stack((torch.ones_like(s),s,s),1),torch.ones(len(s),dtype=torch.bool))
            self.assertTrue((w[:,0]>.5).all())

    def test_all_candidates_start_aux_zero_and_roundtrip_features_only(self):
        stage=population()
        for spec in candidates():
            gate=TaskPatchFusion(spec)
            before=gate(*args(stage),train_center=True)
            torch.testing.assert_close(before,torch.tensor([[1.,0.,0.]]).repeat(len(stage.base.ids),1),rtol=0,atol=0)
            restored=restore_gate(gate.export())
            torch.testing.assert_close(restored(*args(stage),train_center=True),before,rtol=0,atol=0)
            self.assertNotIn('prior',gate.state_dict())

    def test_variation_penalty_does_not_penalize_common_bias(self):
        spec=next(s for s in candidates() if s['key']=='cls__variation')
        gate=TaskPatchFusion(spec);stage=population()
        with torch.no_grad():gate.bias.fill_(2.)
        desc,_=gate.describe(*args(stage))
        u=gate.image_term(desc,True)
        self.assertEqual(float(u.square().mean()),0.)
        self.assertTrue((gate.scores(desc,train_center=True)[:,1:]>0).all())

    def test_old_heads_do_not_use_current_task_scores(self):
        stage=population('test',7).base
        scores=torch.zeros(len(stage.ids),8,3);scores[:,:,0]=1
        old=mix_heads(stage,scores,'own')
        scores[:,-1,1]=.2
        new=mix_heads(stage,scores,'own')
        torch.testing.assert_close(old[:,:23],new[:,:23],rtol=0,atol=0)
        self.assertFalse(torch.equal(old[:,23:],new[:,23:]))
        broadcast=mix_heads(stage,scores[:,-1:],'current')
        self.assertFalse(torch.equal(old[:,:23],broadcast[:,:23]))

    def test_archive_method_and_control_names_cannot_collide(self):
        keys=[prediction_key(m) for m in METHODS]+[control_key(m,c) for m in METHODS for c in ('mean','patch_swap0','spatial')]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertNotEqual(prediction_key('cls_mean'),control_key('cls','mean'))

    def test_declared_config_has_twenty_rows_per_source(self):
        cfg=read(CONFIG);check(cfg)
        self.assertEqual(len(METHODS),20)
        self.assertEqual(sum(not s['reuse'] for s in candidates()),10)
        cfg['candidates'][0]['lower']=.75
        with self.assertRaises(ValueError):check(cfg)

    def test_cpu_fit_reader_receives_supervision(self):
        stage=population()
        spec=next(s for s in candidates() if s['key']=='cls_patch__common')
        result=fit_one(stage,spec,dict(FIT,steps=3),'cpu')
        self.assertEqual(result['first_reader_gradient_step'],1)
        self.assertTrue(np.isfinite(result['bce_history']).all())

    @__import__('unittest').skipUnless(torch.cuda.is_available(),'CUDA runtime')
    def test_all_new_candidates_fit_under_strict_cuda_determinism(self):
        prior=torch.are_deterministic_algorithms_enabled();torch.use_deterministic_algorithms(True)
        try:
            for spec in candidates():
                if not spec['reuse']:
                    result=fit_one(population(),spec,dict(FIT,steps=2),'cuda')
                    self.assertTrue(np.isfinite(result['bce_history']).all())
        finally:torch.use_deterministic_algorithms(prior)

    def test_realistic_pipeline_preserves_head_concat_masks_and_val_seal(self):
        cfg=read(CONFIG);fitting=dict(FIT,steps=2,permutations=2,patch_swaps=1)
        def load(cfg,name,out,split,task,head=None,smoke=False):
            stage=population(split,task)
            stage.patch=stage.patch.repeat(1,1,13,1)[:,:,:196]
            return stage
        def fixed(source,features,split,task,base):
            return fuse_logits(*base.logits.unbind(-1),base.reliable)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);parent=p/'parent/equal90';parent.mkdir(parents=True)
            (parent/'fit.json').write_text('{}')
            for split in ('val','test'):(parent/f'{split}_results.json').write_text('{}')
            cfg['parent']=str(p/'parent');out=p/'out'
            with patch('scripts.emotic.d9_task_patch_readout.load_features',side_effect=load),patch('scripts.emotic.d9_task_patch_readout.canonical_fixed',side_effect=fixed):
                train(cfg,'equal90',out,fitting,True,'cpu')
                with self.assertRaises(FileNotFoundError):evaluate(cfg,'equal90',out,'test',[0,7],fitting,True,'cpu')
                evaluate(cfg,'equal90',out,'val',[0,7],fitting,True,'cpu')
                result=evaluate(cfg,'equal90',out,'test',[0,7],fitting,True,'cpu')
                self.assertEqual(len(result['stages']),2)
                with np.load(out/'test/task7.npz') as z:
                    self.assertEqual(z[prediction_key('cls__own')].shape[1],26)
                    self.assertEqual(z['score__cls__own'].shape[1],8)
                    self.assertEqual(z['score__cls__current'].shape[1],1)
                    for m in METHODS[2:]:
                        base=population('test',7).base
                        actual=mix_heads(base,torch.from_numpy(z['score__'+m]),'current' if m.endswith('__current') else 'own')
                        np.testing.assert_array_equal(actual.numpy(),z[prediction_key(m)])
                (out/'selection.json').write_text('{}')
                with self.assertRaises(ValueError):evaluate(cfg,'equal90',out,'test',[0,7],fitting,True,'cpu')

if __name__=='__main__':main()

