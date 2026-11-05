import copy
import unittest
from unittest.mock import patch
import torch
import test_track_a_reproduction as fixtures
from multi_lane.track_a.model import MultiLaneModel
from multi_lane.track_a.shared_face_cross_view import TaskFaceCLSBank
from multi_lane.track_a.runner import train_task, build_optimizer_groups


class SharedB0FaceTest(unittest.TestCase):
    def model(self, mode):
        torch.manual_seed(31)
        visual = fixtures.FakeVisual()
        visual.transformer.resblocks.extend(copy.deepcopy(visual.transformer.resblocks[0]) for _ in range(2))
        return MultiLaneModel(visual, (2,1), num_selectors=2, num_prompts=2, num_prompt_layers=1,
            adapter_mode='image_token', adapter_bottleneck_dim=3, adapter_layer_indices=(0,),
            adapter_residual_scale=.03, view_fusion='fixed_three_view', shared_face_readout=mode,
            shared_face_layer=1, shared_face_dim=4, shared_face_heads=2)

    def images(self):
        return {**{v:torch.randn(3,3,4,4) for v in ('full','person','face')},
                'face_reliable':torch.tensor([True,False,True])}

    def test_zero_output_and_random_stream_preserved(self):
        models={};states={}
        for mode in ('disabled','pool','attention'):
            models[mode]=self.model(mode);states[mode]=torch.get_rng_state()
            models[mode].activate_task(0)
        self.assertTrue(all(torch.equal(v,states['disabled']) for v in states.values()))
        images=self.images();ref=models['disabled'].current_all_logits(images)
        for m in models.values(): self.assertTrue(torch.equal(ref,m.current_all_logits(images)))
        for mode in ('pool','attention'):
            common={k:v for k,v in models[mode].state_dict().items() if not k.startswith('face_cls_bank.')}
            self.assertTrue(all(torch.equal(v,common[k]) for k,v in models['disabled'].state_dict().items()))
        self.assertTrue(torch.equal(models['pool'].face_cls_bank.tasks[0].v.weight,
                                    models['attention'].face_cls_bank.tasks[0].v.weight))

    def test_direction_reliable_mask_and_runtime_off(self):
        for mode in ('pool','attention'):
            m=self.model(mode);m.activate_task(0);x=self.images()
            base,base_views,_=m.current_all_logits_with_view_features(x)
            with torch.no_grad():m.face_cls_bank.tasks[0].out.weight.normal_(0,.1)
            on,views,_=m.current_all_logits_with_view_features(x)
            self.assertTrue(torch.equal(base_views['full'],views['full']))
            self.assertTrue(torch.equal(base_views['person'],views['person']))
            self.assertTrue(torch.equal(base[1],on[1]))
            self.assertFalse(torch.equal(base[0],on[0]))
            m.face_cls_bank.runtime_enabled=False
            self.assertTrue(torch.equal(base,m.current_all_logits(x)))

    def test_readout_keeps_receiver_gradient_and_detaches_memories(self):
        for mode in ('pool','attention'):
            bank=TaskFaceCLSBank(2,8,mode,dim=4,heads=2);bank.restore_task(0)
            reader=bank.tasks[0]
            with torch.no_grad():reader.out.weight.normal_(0,.1)
            c=torch.randn(3,1,8,requires_grad=True)
            memories=[torch.randn(3,5,8,requires_grad=True) for _ in range(2)]
            y=reader(c,memories);y.square().sum().backward()
            self.assertGreater(float(c.grad.norm()),0)
            self.assertTrue(all(p.grad is None for p in memories))
            names=['v','out']+(['q','k'] if mode=='attention' else ['mlp'])
            for name in names:
                g=getattr(reader,name).weight.grad
                self.assertTrue(torch.isfinite(g).all());self.assertGreater(float(g.norm()),0,name)
            self.assertTrue(((y-c).norm(dim=-1)<=.05001*c.norm(dim=-1)).all())

    def test_training_and_task_sealing(self):
        for mode in ('pool','attention'):
            m=self.model(mode);m.activate_task(0);images=self.images()
            with torch.no_grad():m.face_cls_bank.tasks[0].out.weight.normal_(0,.01)
            y=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
            before={k:v.clone() for k,v in m.state_dict().items()}
            with patch('multi_lane.track_a.runner.TASK_SIZES',(2,1)):
                train_task(m,[(images,y)],None,torch.device('cpu'),0,1,.0125,0.,1.,False,
                    adapter_learning_rate=.0004,view_fusion_learning_rate=.0004,
                    loss_routing='adapter_asl',view_auxiliary_loss_weight=.1,loss_mode='legacy_full_zero')
            self.assertTrue(all(p.grad is None for p in m.visual_encoder.parameters()))
            self.assertTrue(torch.equal(before['face_cls_bank.tasks.1.out.weight'],m.face_cls_bank.tasks[1].out.weight))
            # This is genuinely shared training: reliable Face loss can reach base parameters.
            self.assertGreater(float(m.selectors.grad[0].norm()),0)
            m.activate_task(1)
            self.assertTrue(all(not p.requires_grad for p in m.face_cls_bank.tasks[0].parameters()))
            self.assertTrue(all(p.requires_grad for p in m.face_cls_bank.tasks[1].parameters()))
            groups=build_optimizer_groups(m,0,adapter_learning_rate=.0004)
            active={id(p) for g in groups for p in g}
            self.assertTrue(all(id(p) not in active for p in m.face_cls_bank.tasks[0].parameters()))

    def test_no_reliable_face_training_is_finite(self):
        m=self.model('attention');m.activate_task(0);images=self.images();images['face_reliable'].fill_(False)
        with patch('multi_lane.track_a.runner.TASK_SIZES',(2,1)):
            history=train_task(m,[(images,torch.ones(3,2))],None,torch.device('cpu'),0,1,.0125,0.,1.,False,
                adapter_learning_rate=.0004,view_fusion_learning_rate=.0004,loss_routing='adapter_asl',
                view_auxiliary_loss_weight=.1,loss_mode='legacy_full_zero')
        self.assertEqual(history[0]['skipped_optimizer_steps'],0)
        self.assertTrue(all(p.grad is None for p in m.face_cls_bank.tasks[0].parameters()))

    def test_common_head_fusion_equivalence(self):
        m=self.model('disabled');m.activate_task(0);x=self.images()
        z,views,_=m.current_all_logits_with_view_features(x)
        good=z.new_tensor([.64,.16,.20]);bad=z.new_tensor([.8,.2,0.])
        weights=torch.where(x['face_reliable'][:,None],good,bad)
        separate=sum(views[v]*weights[:,i,None] for i,v in enumerate(('full','person','face')))
        self.assertTrue(torch.allclose(z,separate,atol=2e-7,rtol=2e-6))

if __name__=='__main__':unittest.main()
