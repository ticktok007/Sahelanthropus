#!/usr/bin/env python3
"""
test_ppo_gnn_actor_critic.py — Unit Tests for GNNActorCritic Network
"""

import unittest
import torch
from asm_graph_builder import asm_to_graph
from ppo_gnn_actor_critic import GNNActorCritic, HAS_PYG


class TestGNNActorCritic(unittest.TestCase):

    def setUp(self):
        if not HAS_PYG:
            self.skipTest("PyTorch Geometric is not installed.")

        self.sample_asm = """
        .section .text
        .globl _start
        _start:
            li t0, 42
            li t1, 8
            mul t2, t0, t1
            addi t3, t2, 10
            li a0, 0
            li a7, 93
            ecall
        """
        self.graph = asm_to_graph(self.sample_asm, max_len=16)

    def test_get_value(self):
        """Test value head output shape and non-nan properties."""
        model = GNNActorCritic(inst_in_dim=177, reg_in_dim=1, embed_dim=256)
        value = model.get_value(self.graph.x_dict, self.graph.edge_index_dict)

        self.assertEqual(value.shape, (1,))
        self.assertFalse(torch.isnan(value).any())
        self.assertTrue(torch.isfinite(value).all())

    def test_action_masking_enforcement(self):
        """Test that action masking strictly forces sampling from allowed actions."""
        model = GNNActorCritic(inst_in_dim=177, reg_in_dim=1, embed_dim=256)
        
        # Mask allowing only actions 2 and 3
        mask = torch.zeros((1, 160), dtype=torch.bool)
        mask[0, 2] = True
        mask[0, 3] = True

        actions = []
        for _ in range(50):
            act, _, _, _ = model.get_action_and_value(self.graph.x_dict, self.graph.edge_index_dict, action_mask=mask)
            actions.append(act.item())

        sampled_set = set(actions)
        self.assertTrue(sampled_set.issubset({2, 3}), f"Sampled actions {sampled_set} contained masked actions!")

    def test_evaluate_given_action(self):
        """Test log_prob calculation for a specified fixed action."""
        model = GNNActorCritic(inst_in_dim=177, reg_in_dim=1, embed_dim=256)
        fixed_action = torch.tensor([5], dtype=torch.long)

        action, log_prob, entropy, value = model.get_action_and_value(
            self.graph.x_dict,
            self.graph.edge_index_dict,
            action=fixed_action
        )

        self.assertEqual(action.item(), 5)
        self.assertEqual(log_prob.shape, (1,))
        self.assertEqual(entropy.shape, (1,))
        self.assertEqual(value.shape, (1,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
