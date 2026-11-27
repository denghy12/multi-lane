import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from multi_lane.track_a.feature_bound_fusion import FeatureBoundFusion, full_bounds
from multi_lane.track_a.summary_logit_fusion import masked_weights, fused_logits, image_bce
from scripts.emotic.d9_feature_bounds import (CONFIG, check, candidates, fit_one, train, evaluate, summarize, select_on_val)
from scripts.emotic.d9_a_eight_eval import read
from scripts.emotic.evaluate_d9_relation import metrics_for_stage
from scripts.emotic.r1_asymmetric_forward import fuse_logits
from tests.test_d9_summary_eight import stage_population


class FeatureBoundTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.cfg=read(CONFIG)

    def test_configuration_and_zero_initialization(self):
        check(self.cfg);self.assertEqual(len(candidates(self.cfg)),8)
        for spec in candidates(self.cfg).values():
            gate=FeatureBoundFusion(**spec)
            x=torch.randn(9,len(gate.center));gate.set_center(x)
            scores=gate(x)
            torch.testing.assert_close(scores,torch.tensor([[1.,0.,0.]]).repeat(9,1),rtol=0,atol=0)
            self.assertEqual(set(gate.state_dict()),{'center','matrix'})
            self.assertEqual(list(dict(gate.named_parameters())),['matrix'])
            weights=masked_weights(scores,torch.ones(9,dtype=torch.bool))
            self.assertAlmostEqual(weights[0,0].item(),.5761168848,places=6)
        self.cfg['amplitudes']=[.1,1.25]
        with self.assertRaises(ValueError):check(self.cfg)

    def test_full_bounds_and_face_mask_at_extreme_features(self):
        for a in [.1,.25,.5,.75]:
            gate=FeatureBoundFusion('summary_mean',a)
            with torch.no_grad():gate.matrix.fill_(100)
            scores=gate(torch.tensor([[1.]*6,[-1.]*6,[1.]*6,[-1.]*6]))
            weights=masked_weights(scores,torch.tensor([True,True,False,False]))
            lo,hi=full_bounds(a)
            self.assertAlmostEqual(weights[0,0].item(),lo,places=6)
            self.assertAlmostEqual(weights[1,0].item(),hi,places=6)
            lo,hi=full_bounds(a,False)
            self.assertAlmostEqual(weights[2,0].item(),lo,places=6)
            self.assertAlmostEqual(weights[3,0].item(),hi,places=6)
            self.assertTrue((weights[:,0,None]>weights[:,1:]).all())
            self.assertEqual(weights[2:,2].tolist(),[0.,0.])

    def test_zero_matrix_receives_feature_gradient_only(self):
        x=torch.tensor([[-1.]*6,[0.]*6,[1.]*6],requires_grad=True)
        branches=torch.tensor([[[0.,2.,-2.]]]*3,requires_grad=True)
        y=torch.tensor([[0.],[0.],[1.]])
        gate=FeatureBoundFusion('summary_mean',.25)
        logits=fused_logits(branches,gate(x),torch.arange(3),torch.ones(3,dtype=torch.bool))
        image_bce(logits,y,torch.arange(3),3).backward()
        self.assertGreater(gate.matrix.grad.abs().sum().item(),0)
        self.assertIsNone(x.grad);self.assertIsNone(branches.grad)

    def test_export_train_center_and_initialization_not_historical(self):
        stage=stage_population('train',0)
        config=dict(self.cfg['fit'],steps=3)
        fitted=fit_one(stage,dict(feature='summary_correspondence',amplitude=.75),config)
        gate=FeatureBoundFusion.restore(fitted['gate'])
        torch.testing.assert_close(gate.center,stage.descriptors[gate.feature].mean(0))
        self.assertEqual(fitted['parameters'],52)
        center=gate.center.clone();gate(stage.descriptors[gate.feature]+100)
        torch.testing.assert_close(gate.center,center)
        payload=gate.export();payload['state']['bias']=[0,0]
        with self.assertRaises(ValueError):FeatureBoundFusion.restore(payload)

    def test_validation_selection_and_partial_rejection(self):
        summary={'first':{'all':{'average_mAP':10,'final_mAP':8}},'second':{'all':{'average_mAP':9,'final_mAP':100}}}
        self.assertEqual(select_on_val(summary,['first','second']),'first')
        with self.assertRaises(ValueError):summarize([],['fixed'],False)

    def test_eight_stage_pipeline_masks_shared_all_seen_weights_and_seal(self):
        cfg=self.cfg;config=dict(cfg['fit'],steps=2,permutations=2)
        def load(cfg,name,split,task,smoke):return stage_population(split,task,split=='train')
        def fixed(source,features,split,task,stage):return fuse_logits(*stage.logits.unbind(-1),stage.reliable)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'out';source=root/'source'
            cfg['sources']['equal90']['path']=str(source)
            for split in ('val','test'):
                rows=[]
                for task in range(8):
                    stage=stage_population(split,task,False)
                    m=metrics_for_stage(task,stage.targets,stage.reliable,{'fixed':fixed(None,None,split,task,stage)})
                    rows.append(dict(task=task,scores_sha256=split,metrics=m))
                p=source/'evaluation'/split/'result.json';p.parent.mkdir(parents=True)
                p.write_text(json.dumps(dict(stages=rows)))
            with patch('scripts.emotic.d9_feature_bounds.cached_stage',side_effect=load),patch('scripts.emotic.d9_feature_bounds.canonical_fixed',side_effect=fixed):
                train(cfg,'equal90',out,list(range(8)),config,False)
                with self.assertRaises(FileNotFoundError):evaluate(cfg,'equal90',out,'test',list(range(8)),config,False)
                evaluate(cfg,'equal90',out,'val',list(range(8)),config,False)
                result=evaluate(cfg,'equal90',out,'test',list(range(8)),config,False)
                self.assertTrue(result['target_comparable']);self.assertEqual(len(result['summary']),10)
                with np.load(out/'test/task7.npz',allow_pickle=False) as z:
                    self.assertEqual(z['targets'].shape[1],26)
                    native=stage_population('test',7,False).logits.numpy()
                    for method in candidates(cfg):
                        expected=(native*z[method+'_weights'][:,None,:]).sum(2)
                        np.testing.assert_array_equal(z[method+'_logits'],expected)
                (out/'selection.json').write_text((out/'selection.json').read_text()+'\n')
                with self.assertRaises(ValueError):evaluate(cfg,'equal90',out,'test',list(range(8)),config,False)

if __name__=='__main__':unittest.main()
