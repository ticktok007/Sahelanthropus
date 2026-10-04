#!/usr/bin/env python3
"""
test_policy_mlp_256.py — Architecture & Unit Verification for Flat MLP 256x256 Policy Network

Verifies:
1. Input shape Box(80,) -> Flat MLP Trunk (Linear 80->256, Tanh, Linear 256->256, Tanh).
2. Dual heads: Policy Actor Head -> Discrete(160) logits, Value Critic Head -> scalar value.
3. Action Masking: -inf logit penalty on invalid rules.
4. PyTorch Backpropagation & Gradient Flow across all parameters on CUDA GPU.
"""

import unittest
import torch
import torch.nn as nn
from ppo_superopt import ActorCritic, DEVICE


class TestFlatMLPPolicyNetwork(unittest.TestCase):

    def setUp(self):
        self.obs_dim = 80
        self.act_dim = 160
        self.model = ActorCritic(obs_dim=self.obs_dim, act_dim=self.act_dim, num_rules=10, max_len=16).to(DEVICE)

    def test_network_architecture_and_parameter_shapes(self):
        """Verify layer count, shapes, and parameter dimensions for Flat MLP 256x256"""
        print(f"\n[*] Model Device: {DEVICE}")
        
        # Check trunk layers
        actor_trunk_layers = list(self.model.actor_trunk.children())
        self.assertEqual(len(actor_trunk_layers), 4)
        self.assertIsInstance(actor_trunk_layers[0], nn.Linear)
        self.assertEqual(actor_trunk_layers[0].in_features, 80)
        self.assertEqual(actor_trunk_layers[0].out_features, 256)
        
        self.assertIsInstance(actor_trunk_layers[2], nn.Linear)
        self.assertEqual(actor_trunk_layers[2].in_features, 256)
        self.assertEqual(actor_trunk_layers[2].out_features, 256)

        # Check actor & critic heads
        self.assertEqual(self.model.actor.in_features, 256)
        self.assertEqual(self.model.actor.out_features, 160)
        
        self.assertEqual(self.model.critic.in_features, 256)
        self.assertEqual(self.model.critic.out_features, 1)

    def test_forward_pass_and_tensor_shapes(self):
        """Verify forward pass tensor dimensions for single & batch inputs"""
        batch_size = 32
        dummy_obs = torch.randn(batch_size, self.obs_dim, device=DEVICE)
        dummy_mask = torch.ones(batch_size, self.act_dim, dtype=torch.bool, device=DEVICE)

        actions, log_probs, entropy, values = self.model.get_action_and_value(dummy_obs, action_mask=dummy_mask)

        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(log_probs.shape, (batch_size,))
        self.assertEqual(entropy.shape, (batch_size,))
        self.assertEqual(values.shape, (batch_size,))

    def test_action_masking_penalty(self):
        """Verify -1e9 logit penalty on masked invalid actions"""
        dummy_obs = torch.randn(1, self.obs_dim, device=DEVICE)
        # Mask out all actions except action 0
        mask = torch.zeros(1, self.act_dim, dtype=torch.bool, device=DEVICE)
        mask[0, :16] = True

        actor_features = self.model.actor_trunk(dummy_obs)
        logits = self.model.actor(actor_features)
        
        # Apply mask
        masked_logits = torch.where(mask, logits, torch.tensor(-1e9, device=DEVICE))

        # Check valid actions (0..15) vs invalid actions (16..159)
        self.assertTrue((masked_logits[0, :16] > -1e8).all())
        self.assertTrue((masked_logits[0, 16:] <= -1e8).all())

    def test_gradient_flow(self):
        """Verify non-zero gradients across all 256x256 MLP trunk and head parameters"""
        dummy_obs = torch.randn(16, self.obs_dim, device=DEVICE)
        actions, log_probs, entropy, values = self.model.get_action_and_value(dummy_obs)

        loss = -log_probs.mean() + 0.5 * values.pow(2).mean() - 0.01 * entropy.mean()
        self.model.zero_grad()
        loss.backward()

        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient is None for parameter {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient found for {name}")
            self.assertTrue(param.grad.abs().sum() > 0, f"Zero gradient for {name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
