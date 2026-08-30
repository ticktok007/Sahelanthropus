#!/usr/bin/env python3
"""
test_graph_ppo_rollout_buffer.py — Unit Tests for GraphRolloutBuffer and PyG DataLoader.
"""

import unittest
import torch
from asm_graph_builder import asm_to_graph
from graph_ppo_rollout_buffer import GraphRolloutBuffer, DEVICE


class TestGraphRolloutBuffer(unittest.TestCase):

    def setUp(self):
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

    def test_buffer_insert_and_dataloader(self):
        """Test inserting graph observations and iterating via PyG DataLoader."""
        num_steps = 8
        num_envs = 4
        buffer = GraphRolloutBuffer(num_steps=num_steps, num_envs=num_envs)

        for step in range(num_steps):
            obs_graphs = [self.graph for _ in range(num_envs)]
            actions = torch.randint(0, 160, (num_envs,))
            logprobs = torch.randn(num_envs)
            rewards = torch.randn(num_envs)
            dones = torch.zeros(num_envs)
            values = torch.randn(num_envs)
            action_masks = torch.ones((num_envs, 160), dtype=torch.bool)

            buffer.insert(step, obs_graphs, actions, logprobs, rewards, dones, values, action_masks)

        advantages = torch.randn(num_steps, num_envs)
        returns = torch.randn(num_steps, num_envs)

        loader = buffer.get_dataloader(advantages, returns, minibatch_size=16)

        batches = list(loader)
        self.assertEqual(len(batches), 2)  # 32 items / minibatch_size 16 = 2 batches

        for batch in batches:
            self.assertEqual(batch['inst'].x.shape, (16 * 16, 177))  # 16 graphs * 16 insts
            self.assertEqual(batch['reg'].x.shape, (16 * 32, 1))      # 16 graphs * 32 regs
            self.assertEqual(batch.action.shape, (16,))
            self.assertEqual(batch.return_val.shape, (16,))
            
            masks = batch.action_mask.reshape(-1, 160)
            self.assertEqual(masks.shape, (16, 160))


if __name__ == "__main__":
    unittest.main(verbosity=2)
