import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from multi_lane.track_a.complete_eight_task_test import require_eight, require_formal_budget, restore_cached, apply_frozen_gates
from multi_lane.track_a.fixed_baseline_routes import build_model
from multi_lane.track_a.frozen_view_ranking_fusion import BoundedFusionGate
from multi_lane.track_a.runner import compact_model_state_dict
from test_track_a_reproduction import FakeVisual


class EightTaskCompletionTest(unittest.TestCase):
    def test_actual_runner_budget_fields_accept_formal_and_reject_short_update_smoke(self):
        cfg=json.loads((Path(__file__).parent/'fixtures/protected_full_runner_config.json').read_text())
        with self.assertRaises(ValueError): require_formal_budget(cfg)
        cfg.update(training_budget_mode='epochs',epochs_per_task=30,
                   optimizer_updates_per_task=None,optimizer_updates_by_task=None)
        require_formal_budget(cfg)
        for changed in ({'epochs_per_task':1},{'optimizer_updates_per_task':2},
                        {'optimizer_updates_by_task':[2]*8},{'training_budget_mode':'optimizer_updates'}):
            with self.assertRaises(ValueError): require_formal_budget(dict(cfg,**changed))

    def test_short_joint_smoke_sources_cannot_be_formal_eight_task_results(self):
        config = dict(training_protocol='incremental', max_tasks=8, task_sizes=[5]+[3]*7)
        require_eight(config, dict(status='complete', task_metrics=[{}]*8))
        require_eight(config, dict(status='complete', completed_tasks=8), True)
        for bad in (dict(config, max_tasks=3), dict(config, training_protocol='joint26')):
            with self.assertRaises(ValueError):
                require_eight(bad, dict(status='complete', task_metrics=[{}]*8))
        for status, count in (('running', 8), ('complete', 3), ('smoke_complete', 8)):
            with self.assertRaises(ValueError):
                require_eight(config, dict(status=status, completed_tasks=count), True)

    def payloads(self):
        config = dict(task_sizes=(2,1), num_selectors=2, num_prompts=2, num_prompt_layers=1,
                      view_fusion='fixed_three_view', parax_mode='disabled')
        base = build_model(config, FakeVisual())
        model = build_model(config, copy.deepcopy(base.visual_encoder), frozen_pool=True)
        base.restore_task(1)
        state = compact_model_state_dict(model)
        state.update(compact_model_state_dict(base))
        git = dict(commit='training')
        payload = dict(schema_version=1, task_id=1, source_git=git, model=state, router_input_mode='constant')
        original = dict(task_id=1, model=compact_model_state_dict(base))
        return model, payload, original, git

    def test_restore_preserves_trained_original_and_restores_static_router_option(self):
        model, payload, original, git = self.payloads()
        original['model']['selectors'].fill_(7.)
        payload['model']['selectors'] = original['model']['selectors'].clone()
        restore_cached(model, payload, original, 1, git, 'static')
        self.assertEqual(model.current_task_id, 1)
        self.assertEqual(model.parax_bank.router_input_mode, 'constant')
        self.assertTrue(torch.equal(model.selectors, original['model']['selectors']))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        a = model.parax_bank._router_gates('1_0', torch.randn(4,model.output_dim), 1)
        b = model.parax_bank._router_gates('1_0', torch.randn(4,model.output_dim), 1)
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_restore_rejects_wrong_arm_changed_original_and_wrong_provenance(self):
        model, payload, original, git = self.payloads()
        with self.assertRaisesRegex(ValueError, 'router mode'):
            restore_cached(model, payload, original, 1, git, 'dynamic')
        altered = copy.deepcopy(payload)
        altered['model']['head.bias'].add_(1.)
        with self.assertRaisesRegex(ValueError, 'exact source baseline'):
            restore_cached(model, altered, original, 1, git, 'static')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            restore_cached(model, payload, original, 1, {'commit':'wrong'}, 'static')

    def test_frozen_gate_identity_keeps_saved_probability_and_never_reads_future_gate(self):
        sizes=[2,1]
        fixed=np.array([[20.,-20.,3.],[.2,.4,.6]],dtype=np.float32)
        data={'fused_logits':fixed,'fused_probabilities':np.full_like(fixed,.123),
              'targets':np.zeros_like(fixed),'face_reliable':np.array([False,True]),
              **{v+'_logits':fixed+1. for v in ('full','person','face')}}
        gate=BoundedFusionGate(torch.zeros(2,2,3),torch.zeros(2,2)).eval().requires_grad_(False)
        logits, probs=apply_frozen_gates(data,[gate,None],sizes,0)
        torch.testing.assert_close(logits,torch.from_numpy(fixed),rtol=0,atol=0)
        torch.testing.assert_close(probs,torch.from_numpy(data['fused_probabilities']),rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
