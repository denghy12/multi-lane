from pathlib import Path
import tempfile
import numpy as np
from multi_lane.track_a.protected_parax_compare import fixed_old_cohort
import unittest
import torch
from multi_lane.track_a.runner import parse_args, validate_protected_protocol
from multi_lane.track_a.protected_parax import ProtectedParaXBank


class ProtectedFullEvaluationTest(unittest.TestCase):
    def args(self, *extra):
        return parse_args(['--seed','0','--data-root','data','--clip-checkpoint','clip.pt',
                           '--output-root','out','--reporting-split','val','--protected-parax-paired-audit',
                           '--adapter-mode','disabled','--view-fusion','fixed_three_view',
                           '--view-classifier-mode','shared_post_fusion','--loss-routing','joint_bce',
                           '--parax-mode','post_task_protected_reuse',*extra])

    def test_pilot_does_not_silently_allow_test_or_extra_tasks(self):
        for extra in [('--max-tasks','8'),('--max-tasks','3','--also-report-test'),('--max-tasks','3','--seed','1')]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_protected_protocol(self.args(*extra))
        validate_protected_protocol(self.args('--max-tasks','3'))

    def test_full_requires_explicit_locked_complete_configuration(self):
        flags=('--protected-parax-full-evaluation','--max-tasks','8','--parax-num-experts','16','--also-report-test')
        for seed in range(3):
            validate_protected_protocol(self.args(*flags,'--seed',str(seed)))
        for extra in [('--max-tasks','3'),('--parax-num-experts','6'),('--protected-parax-staged-training',),('--skip-validation-eval',),('--adapter-mode','image_token')]:
            # Invalid adapter is assigned directly to avoid coupling to CLI enum spellings.
            args=self.args(*flags)
            if extra[0]=='--adapter-mode': args.adapter_mode='not_disabled'
            else: args=self.args(*flags,*extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError): validate_protected_protocol(args)

    def test_fixed_cohort_uses_selected_old_task_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); path=root/'reuse_old/test_scores'; path.mkdir(parents=True)
            ids=np.array(['person-a','person-b','person-c'])
            labels=np.tile(np.array([[1.],[0.],[1.]]),(1,26))
            probs=np.tile(np.array([[.8],[.1],[.7]]),(1,26))
            for task,n,order in [(1,8,np.array([0,1,2])),(7,26,np.array([2,0,1]))]:
                np.savez(path/f'task{task}.npz',sample_ids=ids[order],class_indices=np.arange(n),
                         targets=labels[order,:n],probabilities=probs[order,:n],logits=probs[order,:n])
            result=fixed_old_cohort(root,'reuse_old',7,1,'test')
            self.assertEqual(result['samples'],3);self.assertEqual(result['mAP_change'],0.)
            changed=labels.copy();changed[0,0]=0
            np.savez(path/'task7.npz',sample_ids=ids,class_indices=np.arange(26),targets=changed,
                     probabilities=probs,logits=probs)
            with self.assertRaisesRegex(RuntimeError,'labels/classes'):
                fixed_old_cohort(root,'reuse_old',7,1,'test')

    def test_all_eight_tasks_protect_old_outputs_and_limit_expert_access(self):
        for policy in ('frozen_pool','fresh_only','reuse_old'):
            torch.manual_seed(5)
            bank=ProtectedParaXBank(8,2,16,(0,),policy=policy,num_tasks=8,
                                   task_local_router=True,task_local_projection=True,
                                   initialization='zero_output',output_scale_mode='fixed',
                                   residual_scale=1.,smooth_ratio_bound=.02)
            x=torch.randn(4,1,8); references={}
            for task in range(8):
                old={k:v.clone() for k,v in bank.protected_state(task).items()}
                bank.activate_task(task)
                optimizer=torch.optim.AdamW(bank.active_parameters(),lr=.01,weight_decay=.1)
                for _ in range(2):
                    optimizer.zero_grad(); y,g=bank(0,x,task_id=task)
                    (y-torch.ones_like(y)).square().mean().backward(); optimizer.step()
                expected=([0,1] if policy=='frozen_pool' else [2*task,2*task+1] if policy=='fresh_only' else list(range(2*task+2)))
                self.assertEqual(bank.expert_access[task].nonzero().flatten().tolist(),expected)
                for key,value in old.items(): self.assertTrue(torch.equal(value,bank.protected_state(task)[key]))
                bank.seal_task(task); references[task]=tuple(v.clone() for v in bank(0,x,task_id=task))
                for previous,(y,g) in references.items():
                    actual,gate=bank(0,x,task_id=previous)
                    self.assertTrue(torch.equal(actual,y));self.assertTrue(torch.equal(gate,g))
                    self.assertTrue(bool((gate[:,~bank.expert_access[previous]]==0).all()))


if __name__=='__main__':unittest.main()
