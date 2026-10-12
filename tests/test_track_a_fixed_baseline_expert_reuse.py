import copy
import unittest

import torch

from multi_lane.track_a.fixed_baseline_expert_reuse import (
    assert_pair_budget, build_expanding_model, checkpoint_payload,
    paired_banks, restore_fitted, validate_cache,
)
from multi_lane.track_a.fixed_baseline_routes import cached_identity, restore_base
from multi_lane.track_a.post_task_calibration import base_hash, fit_cached_routes
from multi_lane.track_a.runner import compact_model_state_dict
from test_track_a_reproduction import FakeVisual


class FixedBaselineExpertReuseTest(unittest.TestCase):
    def models(self):
        config = dict(task_sizes=(2,) + (1,) * 7, num_selectors=2, num_prompts=2,
                      num_prompt_layers=1, view_fusion='fixed_three_view', parax_mode='disabled')
        baseline = build_expanding_model(config, FakeVisual(), 'fresh_only')
        # Original checkpoints contain no bank at all.
        from multi_lane.track_a.fixed_baseline_routes import build_model
        original = build_model(config, copy.deepcopy(baseline.visual_encoder))
        return original, baseline

    def test_builder_rejects_policy_and_keeps_source_options_immutable(self):
        config = dict(task_sizes=(2,1), num_selectors=2, num_prompts=2, num_prompt_layers=1,
                      view_fusion='fixed_three_view', parax_mode='disabled')
        before=copy.deepcopy(config)
        for policy in ('fresh_only','reuse_old'):
            model=build_expanding_model(config,FakeVisual(),policy)
            self.assertEqual(model.parax_bank.policy,policy)
            self.assertEqual(config,before)
        with self.assertRaises(ValueError):build_expanding_model(config,FakeVisual(),'frozen_pool')

    def test_train_cache_rejects_wrong_split_checkpoint_labels_ids_or_dimensions(self):
        ids=['train:a','train:b']; targets=torch.tensor([[1.,0.],[0.,1.]])
        cache=dict(source_split='train',task_id=0,source_checkpoint_sha256='sha',sample_ids=ids,
                   targets=targets,features={v:torch.randn(2,1,4) for v in ('full','person','face')},
                   face_reliable=torch.tensor([True,False]))
        validate_cache(cache,0,'sha',ids,targets,4)
        for bad in [dict(cache,source_split='val'),dict(cache,source_checkpoint_sha256='bad'),
                    dict(cache,sample_ids=ids[::-1]),dict(cache,targets=1-targets),
                    dict(cache,features={'full':torch.randn(2,1,4)})]:
            with self.assertRaises(ValueError):validate_cache(bad,0,'sha',ids,targets,4)

    def test_eight_tasks_pair_new_budget_protect_old_functions_and_update_new_experts(self):
        torch.manual_seed(3)
        base,model=self.models();banks=paired_banks(model.parax_bank)
        x=torch.nn.functional.normalize(torch.randn(8,1,4),dim=-1)
        refs={policy:{} for policy in banks}
        first_states={}
        for task in range(8):
            base.activate_task(task)
            payload=dict(schema_version=1,task_id=task,source_git={'commit':'base'},model=compact_model_state_dict(base))
            restore_base(model,payload,task,{'commit':'base'})
            pair=assert_pair_budget(banks,task)
            self.assertEqual(len(set(pair['trainable_parameters'].values())),1)
            self.assertEqual(pair['eligible_experts']['fresh_only'],[2*task,2*task+1])
            self.assertEqual(pair['eligible_experts']['reuse_old'],list(range(2*task+2)))
            cache=dict(source_split='train',task_id=task,features={v:x.clone() for v in ('full','person','face')},
                       targets=torch.randint(0,2,(8,2 if task==0 else 1)).float(),face_reliable=torch.ones(8,dtype=torch.bool))
            for policy,bank in banks.items():
                model.parax_bank=bank;model.restore_task(task);model.requires_grad_(False)
                cached_identity(model,cache)
                old={k:v.clone() for k,v in bank.protected_state(task).items()}
                before=base_hash(model);torch.manual_seed(1009*task)
                report=fit_cached_routes(model,cache,tuple(range(sum(base.task_sizes[:task]),sum(base.task_sizes[:task+1]))),
                                         torch.device('cpu'),5,.0004,4,consistency_weight=0.)
                self.assertEqual(base_hash(model),before)
                self.assertTrue(all(report['new_experts_changed'].values()))
                self.assertEqual(report['initial_output_max_difference'],0.)
                self.assertGreater(report['history'][-1]['component_gradient_rms_norm_before_clip']['router'],0.)
                for k,v in old.items():self.assertTrue(torch.equal(v,bank.protected_state(task)[k]),k)
                for old_task,reference in refs[policy].items():
                    output,gates=bank(0,x,task_id=old_task)
                    torch.testing.assert_close(output,reference,atol=0,rtol=0)
                    self.assertEqual(torch.count_nonzero(gates[:,2*(old_task+1):]).item(),0)
                bank.seal_task(task)
                refs[policy][task]=bank(0,x,task_id=task)[0].detach().clone()
                if task==0:first_states[policy]={k:v.clone() for k,v in bank.state_dict().items() if k!='expert_access'}
            if task==0:
                for k,v in first_states['fresh_only'].items():
                    self.assertTrue(torch.equal(v,first_states['reuse_old'][k]),k)
        self.assertEqual(int(banks['reuse_old'].sealed_tasks.sum()),8)

    def test_checkpoint_restore_enforces_policy_masks_sealing_and_original_weights(self):
        base,model=self.models();base.activate_task(0);bank=model.parax_bank;bank.seal_task(0)
        model.restore_task(0)
        original=dict(task_id=0,model=compact_model_state_dict(base))
        with torch.no_grad():
            for key,value in original['model'].items():
                model.state_dict()[key].copy_(value)
        git={'commit':'fit'};payload=checkpoint_payload(model,0,git,'fresh_only')
        restored=build_expanding_model(dict(task_sizes=base.task_sizes,num_selectors=2,num_prompts=2,
                                            num_prompt_layers=1,view_fusion='fixed_three_view'),
                                       copy.deepcopy(base.visual_encoder),'fresh_only')
        restore_fitted(restored,payload,original,0,git,'fresh_only')
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
        for change in ('policy','mask','seal','base'):
            bad=copy.deepcopy(payload)
            if change=='policy':bad['expert_policy']='reuse_old'
            elif change=='mask':bad['model']['parax_bank.expert_access'][0,2]=True
            elif change=='seal':bad['model']['parax_bank.sealed_tasks'][1]=True
            else:bad['model']['head.bias'].add_(1.)
            with self.assertRaises(ValueError):restore_fitted(restored,bad,original,0,git,'fresh_only')
            restore_fitted(restored,payload,original,0,git,'fresh_only')


if __name__=='__main__':unittest.main()
