from dataclasses import replace
import unittest
from unittest.mock import patch as mock_patch

import numpy as np
import torch

from multi_lane.track_a.final_patch_fusion import (
    FinalPatchFusion, PatchReader, METHODS, LEARNED, WIDTH, project_tokens, roster_mean,
)
from multi_lane.track_a.d9_relation_fusion import D9RelationFusion
from multi_lane.track_a.summary_logit_fusion import fused_logits, masked_weights, image_bce
from scripts.emotic.d9_final_patch_readout import PatchStage, fit_one, predict, subset, patch_donors
from scripts.emotic.d9_final_patch_run import FIT
from tests.test_d9_summary_eight import stage_population
from test_d9_protected_multi_model import branches


def population(split='train', task=0):
    base=stage_population(split,task,split=='train')
    counts=2+np.arange(len(base.ids))%3
    base=replace(base,counts=counts)
    ids=[];flags=[]
    for i,name in enumerate(base.ids):
        ids.extend(f'{name}#person={p}' for p in range(counts[i]))
        flags.extend(base.reliable[2*i:2*i+2].tolist()+[False]*(counts[i]-2))
    generator=torch.Generator().manual_seed(17+task)
    cls=torch.randn(len(ids),3,WIDTH,generator=generator)
    patch=torch.randn(len(ids),3,16,WIDTH,generator=generator)
    flags=torch.tensor(flags)
    cls[~flags,2]=0;patch[~flags,2]=0
    return PatchStage(base,cls,patch,np.asarray(ids),np.repeat(np.arange(len(base.ids)),counts),flags,'synthetic_features')


def args(stage):
    return stage.cls,stage.patch,torch.tensor(stage.base.counts),stage.roster_reliable


class FinalPatchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(3)

    def test_capture_exact_native_cls_and_current_task_spatial_tokens(self):
        model=D9RelationFusion(branches(),'sample_holdout');model.activate_task(0);model.activate_task(1)
        model.requires_grad_(False).eval()
        data={k:torch.randn(4,3,4,4) for k in ('full','person','face','evidence_full','evidence_person','evidence_face')}
        data['evidence_face_reliable']=torch.tensor([True,False,True,False])
        original={k:v.clone() for k,v in model.state_dict().items()};current=[None];events={v:[] for v in ('full','person','face')};heads={}
        wrappers=[]
        for view,branch in model.branches.items():
            fn=branch.encode_lanes
            def wrapped(*args,view=view,fn=fn,**kwargs):
                current[0]=view
                return fn(*args,**kwargs)
            wrappers.append(mock_patch.object(branch,'encode_lanes',side_effect=wrapped))
            def head_hook(module,inputs,view=view):heads[view]=inputs[0].detach().clone()
            wrappers.append(branch.head.register_forward_pre_hook(head_hook))
        block=model.branches['full'].visual_encoder.transformer.resblocks[-1]
        def hook(module,inputs,output):events[current[0]].append(output.permute(1,0,2).detach().clone())
        handle=block.register_forward_hook(hook)
        with torch.inference_mode():
            handle.remove();expected=model(data,all_seen=True)
            handle=block.register_forward_hook(hook)
            for wrapper in wrappers:
                if hasattr(wrapper,'start'):wrapper.start()
            captured={};actual=model(data,all_seen=True,final_tokens=captured)
        for wrapper in wrappers:
            if hasattr(wrapper,'stop'):wrapper.stop()
            else:wrapper.remove()
        handle.remove()
        for view in events:
            torch.testing.assert_close(actual['branches'][view],expected['branches'][view],rtol=0,atol=0)
            torch.testing.assert_close(captured[view]['cls'],heads[view][:,-1],rtol=0,atol=0)
            torch.testing.assert_close(captured[view]['patch'],events[view][-1][:,1:],rtol=0,atol=0)
            self.assertNotEqual(captured[view]['patch'].shape[1],model.branches[view].num_selectors)
            self.assertFalse(captured[view]['patch'].requires_grad)
        for k,v in original.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
        with self.assertRaises(ValueError):model(data,all_seen=True,final_tokens={})

    def test_projection_is_frozen_tokenwise_and_does_not_consume_rng(self):
        x=torch.randn(4,196,48,requires_grad=True);rng=torch.get_rng_state().clone()
        y=project_tokens(x,'person','patch')
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        torch.testing.assert_close(y[:,:7],project_tokens(x[:,:7],'person','patch'))
        self.assertFalse(y.requires_grad);self.assertEqual(y.shape,(4,196,32))
        self.assertFalse(torch.equal(y,project_tokens(x,'face','patch')))

    def test_zero_initialization_full_bounds_and_parameter_counts(self):
        stage=population();expected={'initial':0,'shared':2,'cls':196,'cls_mean':388,'cls_regions':1732,'cls_patch':1948,'cls_patch_full':2460}
        for method in METHODS[1:]:
            gate=FinalPatchFusion(method);self.assertEqual(sum(p.numel() for p in gate.parameters()),expected[method])
            scores=gate(*args(stage),train_center=True)
            torch.testing.assert_close(scores,torch.tensor([[1.,0.,0.]]).repeat(len(stage.base.ids),1),rtol=0,atol=0)
            self.assertNotIn('prior',gate.state_dict())
            if method not in ('initial','shared'):self.assertNotIn('bias',gate.state_dict())
            with torch.no_grad():
                if hasattr(gate,'matrix'):gate.matrix.normal_(0,100)
                if hasattr(gate,'bias'):gate.bias.fill_(100)
            scores=gate(*args(stage));weights=masked_weights(scores[stage.base.inverse],stage.base.reliable)
            self.assertTrue((scores[:,1:].abs()<=.25).all());self.assertTrue((weights[:,0,None]>weights[:,1:]).all())
            self.assertTrue((weights[~stage.base.reliable,2]==0).all())

    def test_uniform_patch_reading_equals_same_layer_mean_and_order_invariance(self):
        stage=population();mean=FinalPatchFusion('cls_mean');learned=FinalPatchFusion('cls_patch')
        with torch.no_grad():
            for reader in learned.readers.values():reader.score.weight.zero_()
        a,_=mean.describe(*args(stage));b,_=learned.describe(*args(stage));torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        shuffled=replace(stage,patch=stage.patch[:,:,torch.randperm(16)])
        for gate in (mean,learned):
            original,_=gate.describe(*args(stage));wrong,_=gate.describe(*args(shuffled));torch.testing.assert_close(original,wrong,rtol=1e-5,atol=1e-6)
        regions=FinalPatchFusion('cls_regions');a,_=regions.describe(*args(stage));b,_=regions.describe(*args(shuffled))
        self.assertGreater((a-b).abs().max().item(),.1)

    def test_full_condition_changes_auxiliary_patch_weights_only_when_enabled(self):
        c=torch.randn(4,32);g=torch.randn(4,16,32);full=torch.randn(4,32)
        for enabled in (False,True):
            reader=PatchReader(32,8,enabled)
            _,a=reader(c,g,full);_,b=reader(c,g,full+2*torch.randn_like(full))
            if enabled:self.assertGreater((a-b).abs().max().item(),1e-5)
            else:torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_complete_roster_including_unsupervised_people_and_missing_face(self):
        stage=population();gate=FinalPatchFusion('cls_mean')
        before,_=gate.describe(*args(stage));extra=np.flatnonzero(stage.roster_ids==f'{stage.base.ids[1]}#person=2')[0]
        changed=replace(stage,cls=stage.cls.clone(),patch=stage.patch.clone())
        changed.cls[extra,1]+=10
        after,_=gate.describe(*args(changed));self.assertGreater((after[1]-before[1]).abs().sum().item(),0)
        changed=replace(stage,cls=stage.cls.clone(),patch=stage.patch.clone())
        changed.cls[~stage.roster_reliable,2]=torch.nan;changed.patch[~stage.roster_reliable,2]=torch.nan
        after,_=gate.describe(*args(changed));torch.testing.assert_close(before,after,rtol=0,atol=0)
        chosen=subset(stage,np.asarray([1,4,7]));self.assertEqual(len(chosen.roster_ids),int(chosen.base.counts.sum()))
        self.assertEqual(len(chosen.base.person_ids),6)
        donors,info=patch_donors(stage,np.random.default_rng(1))
        np.testing.assert_array_equal(stage.roster_reliable[donors],stage.roster_reliable)
        self.assertGreater(info['swapped_images'],0)
        self.assertTrue(np.all(stage.roster_inverse[donors]!=stage.roster_inverse))

    def test_supervision_reaches_readers_after_zero_output_step_and_center_is_train_only(self):
        stage=population();stage.cls.requires_grad_();stage.patch.requires_grad_();stage.base.logits.requires_grad_()
        gate=FinalPatchFusion('cls_patch')
        scores=gate(*args(stage),train_center=True)
        image_bce(fused_logits(stage.base.logits,scores,stage.base.inverse,stage.base.reliable),stage.base.targets,stage.base.inverse,len(stage.base.ids)).backward()
        self.assertGreater(gate.matrix.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum()==0 for p in gate.readers.parameters()))
        self.assertIsNone(stage.cls.grad);self.assertIsNone(stage.patch.grad);self.assertIsNone(stage.base.logits.grad)
        result=fit_one(stage,'cls_patch',dict(FIT,steps=3),'cpu');self.assertEqual(result['first_reader_gradient_step'],1)
        fitted=FinalPatchFusion.restore(result['gate']);desc,_=fitted.describe(*args(stage));torch.testing.assert_close(fitted.center,desc.mean(0))
        old=fitted.center.clone();predict(population('val'),fitted,'cpu');torch.testing.assert_close(old,fitted.center,rtol=0,atol=0)
        bad=fitted.export();bad['state']['bias']=[0.,0.]
        with self.assertRaises(ValueError):FinalPatchFusion.restore(bad)

if __name__=='__main__':unittest.main()
