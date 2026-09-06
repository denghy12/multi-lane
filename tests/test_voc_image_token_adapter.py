from types import SimpleNamespace
import unittest

import torch
from torch import nn

from multi_lane.blocks import PreT_Attention
from multi_lane.engine import compute_asymmetric_loss, backward_routed_losses
from multi_lane.track_a.adapter import TaskImageTokenAdapterBank


class VocImageTokenAdapterTest(unittest.TestCase):
    def _attention(self):
        attention = PreT_Attention(dim=8, num_heads=2, qkv_bias=True, id=0)
        attention.init(
            SimpleNamespace(
                num_selectors=2,
                detach=False,
                disable_dandr=False,
                tome=0,
            ),
            use_prompts=False,
        )
        return attention

    def test_lane_specific_tokens_preserve_shape_and_receive_gradients(self):
        torch.manual_seed(2)
        attention = self._attention()
        images = torch.randn(3, 5, 8)
        selectors = torch.randn(2, 3, 3, 8)
        shared_output = attention(images, selectors)
        lane_images = images.unsqueeze(0).expand(2, -1, -1, -1).clone()
        lane_images.requires_grad_(True)
        lane_output = attention(images, selectors, selector_image_tokens=lane_images)
        self.assertEqual(tuple(lane_output[0].shape), (3, 5, 8))
        self.assertEqual(tuple(lane_output[1].shape), (2, 3, 3, 8))
        self.assertTrue(torch.allclose(shared_output[0], lane_output[0], atol=1e-6))
        self.assertTrue(torch.allclose(shared_output[1], lane_output[1], atol=1e-6))
        lane_output[1].square().sum().backward()
        self.assertIsNotNone(lane_images.grad)
        self.assertGreater(float(lane_images.grad.abs().sum()), 0.0)

    def test_adapter_task_transition_freezes_previous_task(self):
        bank = TaskImageTokenAdapterBank(
            num_tasks=5,
            hidden_dim=8,
            bottleneck_dim=4,
            layer_indices=(1,),
            residual_scale=0.1,
            activation='relu',
            task_initialization='independent',
        )
        bank.activate_task(0)
        self.assertTrue(all(p.requires_grad for p in bank.task_adapters[0].parameters()))
        bank.activate_task(1)
        self.assertTrue(all(not p.requires_grad for p in bank.task_adapters[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in bank.task_adapters[1].parameters()))

    def test_adapter_construction_rng_can_be_isolated(self):
        torch.manual_seed(9)
        state = torch.get_rng_state()
        TaskImageTokenAdapterBank(
            num_tasks=5,
            hidden_dim=8,
            bottleneck_dim=4,
            layer_indices=(1,),
        )
        torch.set_rng_state(state)
        isolated_value = torch.rand(5)
        torch.manual_seed(9)
        control_value = torch.rand(5)
        self.assertTrue(torch.equal(isolated_value, control_value))

    def test_zero_gamma_asl_matches_bce(self):
        logits = torch.tensor([[0.2, -0.4, 0.0, 0.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
        asl = compute_asymmetric_loss(
            logits, targets, temperature=1.0,
            gamma_neg=0.0, gamma_pos=0.0, clip=0.0,
        )
        self.assertTrue(torch.allclose(asl, bce, atol=1e-7))

    def test_mixed_loss_routing_isolates_parameter_gradients(self):
        model_parameter = nn.Parameter(torch.tensor(2.0))
        adapter_parameter = nn.Parameter(torch.tensor(3.0))
        model_loss = (model_parameter + 2.0 * adapter_parameter).pow(2)
        adapter_loss = (3.0 * model_parameter + adapter_parameter).pow(2)
        backward_routed_losses(
            model_loss, adapter_loss, (model_parameter,), (adapter_parameter,)
        )
        self.assertEqual(float(model_parameter.grad), 16.0)
        self.assertEqual(float(adapter_parameter.grad), 18.0)


if __name__ == '__main__':
    unittest.main()
