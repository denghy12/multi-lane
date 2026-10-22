import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import test_track_a_reproduction as fixtures
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.runner import build_optimizer_groups, compact_model_state_dict, tensor_state_digest
from multi_lane.track_a.export_compact_test_scores import restore_compact_model_state
from multi_lane.track_a.cross_view_attention import TaskCrossViewAttentionBank
from multi_lane.track_a.summary_context import SummaryContextReader, mix_context_attention
from multi_lane.track_a.run_summary_context import dependency_ready, training_arguments, ARMS


class SummaryContextTest(unittest.TestCase):
    def model(self, mode='person_full_context'):
        torch.manual_seed(71)
        visual=fixtures.FakeVisual()
        visual.transformer.resblocks.append(copy.deepcopy(visual.transformer.resblocks[0]))
        return MultiLaneModel(visual,(2,1),num_selectors=2,num_prompts=2,num_prompt_layers=1,
            view_fusion='fixed_three_view',cross_view_mode=mode,cross_view_layer=1,
            cross_view_dim=4,cross_view_heads=2,cross_view_context_max_mass=.1)

    def images(self):
        return {**{v:torch.randn(3,3,4,4) for v in ('full','person','face')},
                'face_reliable':torch.tensor([True,False,True])}

    def test_original_parameters_rng_and_exact_runtime_off_baseline(self):
        base=self.model('disabled'); before=torch.get_rng_state();base.activate_task(0)
        images=self.images()
        with torch.no_grad(): expected=base.seen_logits(images)
        for mode in TaskCrossViewAttentionBank.CONTEXT_MODES:
            model=self.model(mode)
            self.assertTrue(torch.equal(before,torch.get_rng_state()))
            for name,value in base.state_dict().items():
                self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
            self.assertEqual(model.cross_view_bank.parameter_count_per_task(),65)
            model.activate_task(0);model.cross_view_runtime_enabled=False
            with torch.no_grad():self.assertTrue(torch.equal(expected,model.seen_logits(images)))

    def test_person_summaries_are_preserved_and_context_only_enters_one_msa(self):
        model=self.model();model.activate_task(0);images=self.images()
        original=model._task_attention_ffn;captured=[]
        def record(*args,**kwargs):
            captured.append((args[3],args[7],args[6].detach().clone(),kwargs))
            return original(*args,**kwargs)
        with patch.object(model,'_task_attention_ffn',side_effect=record):
            with torch.no_grad():on=model.seen_logits(images)
        self.assertEqual([(x[0],x[1]) for x in captured],[(i,v) for v in ('full','person','face') for i in (0,1)])
        at=[x for x in captured if 'context_tokens' in x[3]]
        self.assertEqual(len(at),1);self.assertEqual(at[0][:2],(1,'person'))
        self.assertEqual(at[0][3]['context_tokens'].shape,torch.Size([1,3,2,8]))
        expected=at[0][2];captured.clear();model.cross_view_runtime_enabled=False
        with patch.object(model,'_task_attention_ffn',side_effect=record):
            with torch.no_grad():off=model.seen_logits(images)
        actual=[x[2] for x in captured if x[:2]==(1,'person')][0]
        self.assertTrue(torch.equal(expected,actual))
        self.assertFalse(torch.equal(on,off))

    def test_context_is_convex_read_of_unprojected_source(self):
        reader=SummaryContextReader(8,4,2,.1)
        q=torch.randn(3,2,8);source=torch.randn(3,2,8)
        context,weights,_,mass=reader(q,source)
        self.assertTrue(torch.allclose(context,weights.mean(1)@source,atol=1e-6))
        self.assertTrue(torch.allclose(weights.sum(-1),torch.ones_like(weights.sum(-1))))
        self.assertEqual(set(dict(reader.named_parameters())),{'q.weight','k.weight','gate_logit'})
        self.assertAlmostEqual(float(mass),.05,places=7)
        for extreme in (-100.,100.):
            with torch.no_grad():reader.gate_logit.fill_(extreme)
            mass=reader(q,source)[3]
            self.assertGreaterEqual(float(mass),0);self.assertLessEqual(float(mass),.100001)

    def test_receiving_attention_respects_group_probability_and_gradients(self):
        attention=copy.deepcopy(fixtures.FakeVisual().transformer.resblocks[0].attn)
        query=torch.randn(1,3,attention.num_heads,3,8//attention.num_heads)
        base=torch.randn_like(query);context=torch.randn(1,3,2,8,requires_grad=True)
        mass=torch.tensor([.075],requires_grad=True)
        actual=mix_context_attention(query,base,context,attention,attention.num_heads,8//attention.num_heads,mass)
        projected=F.linear(context,attention.in_proj_weight,attention.in_proj_bias)
        projected=projected.reshape(1,3,2,3,attention.num_heads,8//attention.num_heads).permute(3,0,1,4,2,5)
        key,value=projected[1:]
        weights=(query@key.transpose(-2,-1)*(8//attention.num_heads)**-.5).softmax(-1)
        expected=.925*base+.075*(weights@value)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        actual.square().sum().backward()
        self.assertGreater(context.grad.abs().sum(),0);self.assertGreater(mass.grad.abs().sum(),0)

    def test_all_reader_parameters_and_original_training_paths_receive_gradients(self):
        for mode in TaskCrossViewAttentionBank.CONTEXT_MODES:
            model=self.model(mode);model.activate_task(0)
            groups=build_optimizer_groups(model,0)[2]
            names=dict(model.named_parameters())
            self.assertEqual({id(p) for g in groups for p in g['params']},
                             {id(names[n]) for n in model.optimizer_parameter_names()})
            inputs=[]
            def capture(module,args):
                args[1].retain_grad();inputs.append(args[1])
            handle=model.cross_view_bank.register_forward_pre_hook(capture)
            _,views=model.encode_lanes_with_views(self.images(),False)
            views['person'].square().sum().backward();handle.remove()
            for name,p in model.cross_view_bank.tasks[0].named_parameters():
                self.assertIsNotNone(p.grad,(mode,name))
                self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum(),0,(mode,name))
            self.assertGreater(inputs[0].grad.abs().sum(),0)
            self.assertTrue(all(p.grad is None for p in model.visual_encoder.parameters()))

    def test_other_endpoints_unchanged_and_query_identity_swap(self):
        model=self.model();model.activate_task(0);model.eval();images=self.images()
        with torch.no_grad():
            on,views=model.seen_logits_with_views(images)
            identity=dict(images,cross_view_query_donor=images['person'],cross_view_query_mask=torch.ones(3,dtype=torch.bool))
            same,_=model.seen_logits_with_views(identity);self.assertTrue(torch.equal(on,same))
            swap=dict(identity,cross_view_query_donor=images['person'].roll(1,0))
            swapped,swapped_views=model.seen_logits_with_views(swap)
            self.assertFalse(torch.equal(on,swapped))
            for view in ('full','face'):self.assertTrue(torch.equal(views[view],swapped_views[view]))
            model.cross_view_runtime_enabled=False
            _,off=model.seen_logits_with_views(images)
            for view in ('full','face'):self.assertTrue(torch.equal(views[view],off[view]))

    def test_old_task_frozen_compact_replay_and_amp(self):
        for mode in TaskCrossViewAttentionBank.CONTEXT_MODES:
            model=self.model(mode);model.activate_task(0);images=self.images()
            before=tensor_state_digest(model.cross_view_bank.tasks[0].state_dict())
            with torch.no_grad():old=model.seen_logits(images)
            model.activate_task(1);optimizer=torch.optim.Adam(build_optimizer_groups(model,0)[2],lr=.01)
            for _ in range(2):
                optimizer.zero_grad()
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    loss=model.current_logits(images).float().square().sum()
                loss.backward();optimizer.step();self.assertTrue(torch.isfinite(loss))
            self.assertEqual(before,tensor_state_digest(model.cross_view_bank.tasks[0].state_dict()))
            with torch.no_grad():actual=model.seen_logits(images)
            self.assertTrue(torch.allclose(old,actual[:,:2],atol=1e-6))
            restored=self.model(mode)
            payload=dict(schema_version=1,task_id=1,source_git={'commit':'unit'},model=compact_model_state_dict(model))
            restore_compact_model_state(restored,payload,1,{'commit':'unit'})
            with torch.no_grad():self.assertTrue(torch.equal(actual,restored.seen_logits(images)))
            row=model.cross_view_bank.statistics()['task1/layer1']
            self.assertFalse(row['person_summary_overwritten']);self.assertLessEqual(row['mass_mean'],.1)

    def test_dependency_never_treats_missing_or_smoke_as_finished_formal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.assertFalse(dependency_ready(root))
            (root/'completion.json').write_text(json.dumps(dict(status='complete',tasks=2,smoke=True)))
            with self.assertRaises(ValueError):dependency_ready(root)
            (root/'completion.json').write_text(json.dumps(dict(status='complete',tasks=8,smoke=False)))
            self.assertTrue(dependency_ready(root))
            (root/'failure.json').write_text('{}')
            with self.assertRaises(RuntimeError):dependency_ready(root)

    def test_all_arms_share_supervision_data_and_budget(self):
        class Args:
            data_root='datasets/EMOTIC';clip_checkpoint='pretrained/clip.pt';face_manifest_root='datasets/faces'
        values=[training_arguments(Args(),Path('output/unit'),arm,False) for arm in ARMS]
        def strip(args):
            result=list(args)
            for option in ('--cross-view-mode','--output-root'):
                index=result.index(option);del result[index:index+2]
            return result
        self.assertTrue(all(strip(values[0])==strip(row) for row in values[1:]))
        self.assertEqual(values[0][values[0].index('--max-tasks')+1],'8')
        self.assertEqual(values[0][values[0].index('--view-auxiliary-loss-weight')+1],'.1')
        self.assertNotIn('--calibration-fraction',values[0])


if __name__=='__main__':unittest.main()
