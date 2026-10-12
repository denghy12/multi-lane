import copy
import unittest
import torch
from multi_lane.track_a.fixed_baseline_staged_functions import (
    build_expanding_model,paired_banks,checkpoint_payload,restore_fitted,capture_functions)
from multi_lane.track_a.fixed_baseline_routes import build_model,restore_base,cached_identity
from multi_lane.track_a.runner import compact_model_state_dict
from multi_lane.track_a.staged_functional_experts import split_cache,held_out,image_group
from multi_lane.track_a.staged_functional_fit import fit_phase,cache_metrics
from test_track_a_reproduction import FakeVisual


class StagedFunctionTest(unittest.TestCase):
    def config(self):
        return dict(task_sizes=(2,)+(1,)*7,num_selectors=2,num_prompts=2,num_prompt_layers=1,
                    view_fusion='fixed_three_view',parax_mode='disabled')

    def cache(self,task=0):
        return dict(source_split='train',task_id=task,
                    sample_ids=[f'train/image{i//2}.jpg#person={i%2}' for i in range(100)],
                    targets=torch.randint(0,2,(100,2 if task==0 else 1)).float(),
                    features={v:torch.nn.functional.normalize(torch.randn(100,1,4),dim=-1)
                              for v in ('full','person','face')},face_reliable=torch.arange(100)%2==0)

    def test_global_partition_groups_people_across_tasks_and_rejects_leakage(self):
        data=self.cache();parts=split_cache(data)
        fit={image_group(s) for s in parts['fit']['sample_ids']}
        held={image_group(s) for s in parts['holdout']['sample_ids']}
        self.assertFalse(fit&held);self.assertEqual(len(fit|held),50)
        shuffled=dict(data,sample_ids=data['sample_ids'][::-1])
        other=split_cache(shuffled)
        self.assertEqual(fit,{image_group(s) for s in other['fit']['sample_ids']})
        with self.assertRaises(ValueError):split_cache(dict(data,source_split='test'))
        with self.assertRaises(ValueError):held_out('not-a-person-id')
        model=build_expanding_model(self.config(),FakeVisual(),'fresh_only');model.restore_task(0)
        with self.assertRaises(ValueError):fit_phase(model,parts['holdout'],(0,1),torch.device('cpu'),'new',0,epochs=1)
        leaked=dict(parts['fit'],sample_ids=parts['fit']['sample_ids'][:-1]+[parts['holdout']['sample_ids'][0]])
        with self.assertRaises(ValueError):fit_phase(model,leaked,(0,1),torch.device('cpu'),'new',0,epochs=1)

    def test_eight_tasks_common_donors_exact_old_paths_phase_freezing_and_restore(self):
        torch.manual_seed(7)
        config=self.config();base=build_model(config,FakeVisual())
        model=build_expanding_model(config,copy.deepcopy(base.visual_encoder),'fresh_only')
        banks=paired_banks(model.parax_bank);refs={a:{} for a in banks};saved=[]
        probe=torch.nn.functional.normalize(torch.randn(6,1,4),dim=-1)
        for task in range(8):
            base.activate_task(task)
            original=dict(schema_version=1,task_id=task,source_git={'commit':'base'},model=compact_model_state_dict(base))
            restore_base(model,original,task,{'commit':'base'})
            parts=split_cache(self.cache(task));classes=tuple(range(sum(base.task_sizes[:task]),sum(base.task_sizes[:task+1])))
            old_coeff=None
            for arm,bank in banks.items():
                model.parax_bank=bank;model.restore_task(task);bank.set_phase('off');model.requires_grad_(False)
                cached_identity(model,parts['fit'])
                old_state={n:x.clone() for n,x in bank.protected_state(task).items()}
                funcs=capture_functions(bank,task,{'full':probe})
                phases=['new','new'] if arm=='fresh_only' or (arm=='old_then_new' and task==0) else (['old'] if task else []) if arm=='old_only' else ['old','new']
                for pi,phase in enumerate(phases):
                    report=fit_phase(model,parts['fit'],classes,torch.device('cpu'),phase,task+100*pi,epochs=2)
                    self.assertTrue(report['phase_frozen_tensors_unchanged'])
                    if phase=='old':
                        if arm=='old_only':old_coeff=bank.old_coefficients[task].detach().clone()
                        else:torch.testing.assert_close(bank.old_coefficients[task],old_coeff,rtol=0,atol=0)
                        self.assertGreater(sum(report['history'][0]['gradient_rms'].values()),0.)
                    metrics=cache_metrics(model,parts['holdout'],classes,torch.device('cpu'))
                    self.assertEqual(metrics['optimization_updates'],0)
                for n,x in old_state.items():torch.testing.assert_close(x,bank.protected_state(task)[n],rtol=0,atol=0)
                for n,x in funcs.items():torch.testing.assert_close(x,capture_functions(bank,task,{'full':probe})[n],rtol=0,atol=0)
                for t,x in refs[arm].items():torch.testing.assert_close(bank(0,probe,task_id=t)[0],x,rtol=0,atol=0)
                bank.seal_task(task);refs[arm][task]=bank(0,probe,task_id=task)[0].detach().clone()
                ratio=(refs[arm][task]-probe).norm(dim=-1)/probe.norm(dim=-1)
                self.assertLessEqual(float(ratio.max()),.020001)
                payload=checkpoint_payload(model,task,{'commit':'fit'},arm)
                restored=build_expanding_model(config,FakeVisual(),arm)
                restore_fitted(restored,payload,original,task,{'commit':'fit'},arm)
                torch.testing.assert_close(restored.parax_bank(0,probe,task_id=task)[0],refs[arm][task],rtol=0,atol=0)
                self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
                if task==0 and arm=='old_only':torch.testing.assert_close(refs[arm][task],probe,rtol=0,atol=0)
                if task==0 and arm=='old_then_new':torch.testing.assert_close(refs[arm][task],refs['fresh_only'][task],rtol=0,atol=0)
                saved.append((arm,task,copy.deepcopy(payload),copy.deepcopy(original)))
            for bank in banks.values():
                bank.install_donor(banks['fresh_only'],task)
                for t,x in refs[bank.arm].items():torch.testing.assert_close(bank(0,probe,task_id=t)[0],x,rtol=0,atol=0)
                with self.assertRaises(ValueError):bank.install_donor(banks['fresh_only'],task)
                with self.assertRaises(ValueError):bank.donor_output(2*task,probe,task)
            for bank in banks.values():torch.testing.assert_close(bank.donor_a,banks['fresh_only'].donor_a,rtol=0,atol=0)
        arm,task,payload,original=saved[-1]
        for defect in ['arm','combination','owner','mask','future_donor','base']:
            bad=copy.deepcopy(payload)
            if defect=='arm':bad['arm']='fresh_only'
            elif defect=='combination':bad['combination']='wrong'
            elif defect=='owner':bad['model']['parax_bank.expert_owners'][0]=1
            elif defect=='mask':bad['model']['parax_bank.expert_access'][0,2]=True
            elif defect=='future_donor':bad['model']['parax_bank.donor_loaded'][task]=True
            else:bad['model']['head.weight'].add_(1)
            with self.assertRaises(ValueError):restore_fitted(build_expanding_model(config,FakeVisual(),arm),bad,original,task,{'commit':'fit'},arm)

    def test_signed_coefficients_have_direct_gradients_and_no_entropy_diagnostic(self):
        model=build_expanding_model(self.config(),FakeVisual(),'old_only');b=model.parax_bank
        b.restore_task(1);b.donor_loaded[0]=True
        b.donor_a[:2].normal_();b.donor_b[:2].normal_();b.donor_pw[0].normal_();b.donor_pb[0].normal_()
        x=torch.randn(3,1,4);b.set_phase('old');y,g=b(0,x,task_id=1)
        torch.testing.assert_close(y,x,atol=0,rtol=0);y.sum().backward()
        self.assertGreater(float(b.old_coefficients[1].grad.abs().sum()),0.)
        model.set_parax_runtime_enabled(True);model._record_parax_gates(1,'full',x,y,g)
        diagnostics=model.parax_gate_diagnostics()
        self.assertFalse(any('entropy' in k or 'top_frequency' in k for k in diagnostics))
        self.assertTrue(any('coefficient_abs_sum' in k for k in diagnostics))


if __name__=='__main__':unittest.main()
