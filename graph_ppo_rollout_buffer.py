#!/usr/bin/env python3
"""
graph_ppo_rollout_buffer.py — PPO Rollout Buffer & PyG DataLoader for Graph Observations.

Handles native HeteroData graph observations directly in rollout buffer and creates mini-batches
using torch_geometric.loader.DataLoader.
"""

from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphStepItem:
    """Single transition item holding a HeteroData graph observation and RL step targets."""
    def __init__(
        self,
        graph_obs: HeteroData,
        action: int,
        logprob: float,
        advantage: float,
        return_val: float,
        value: float,
        action_mask: torch.Tensor
    ):
        # Attach transition attributes to the HeteroData graph object for automatic PyG batching
        self.data = graph_obs.clone()
        self.data.action = torch.tensor(action, dtype=torch.long)
        self.data.logprob = torch.tensor(logprob, dtype=torch.float32)
        self.data.advantage = torch.tensor(advantage, dtype=torch.float32)
        self.data.return_val = torch.tensor(return_val, dtype=torch.float32)
        self.data.value = torch.tensor(value, dtype=torch.float32)
        self.data.action_mask = action_mask.clone().detach()


class GraphRolloutBuffer:
    """
    Rollout Buffer storing PyG HeteroData graph observations and step transitions.
    """

    def __init__(self, num_steps: int = 256, num_envs: int = 4):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.batch_size = num_steps * num_envs

        self.graphs: List[List[HeteroData]] = [[None for _ in range(num_envs)] for _ in range(num_steps)]
        self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long)
        self.logprobs = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.rewards = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.dones = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.values = torch.zeros((num_steps, num_envs), dtype=torch.float32)
        self.action_masks = torch.zeros((num_steps, num_envs, 250), dtype=torch.bool)

    def insert(
        self,
        step: int,
        obs_graphs: List[HeteroData],
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        action_masks: torch.Tensor
    ):
        """Inserts a step rollout across all parallel vector environments."""
        for e in range(self.num_envs):
            self.graphs[step][e] = obs_graphs[e]

        self.actions[step] = actions.cpu()
        self.logprobs[step] = logprobs.cpu()
        self.rewards[step] = rewards.cpu()
        self.dones[step] = dones.cpu()
        self.values[step] = values.cpu()

        # Dynamic action mask padding up to 250 (max_len = 25)
        if action_masks.shape[-1] < 250:
            padded_masks = torch.zeros((self.num_envs, 250), dtype=torch.bool, device=action_masks.device)
            padded_masks[:, :action_masks.shape[-1]] = action_masks
            self.action_masks[step] = padded_masks.cpu()
        else:
            self.action_masks[step] = action_masks[:, :250].cpu()

    def get_dataloader(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        minibatch_size: int = 256
    ) -> DataLoader:
        """
        Flattens rollout transitions, attaches advantages and returns, and returns a PyG DataLoader.
        """
        items: List[HeteroData] = []
        adv_flat = advantages.cpu()
        ret_flat = returns.cpu()

        for step in range(self.num_steps):
            for e in range(self.num_envs):
                g = self.graphs[step][e].clone()
                g.action = self.actions[step, e]
                g.logprob = self.logprobs[step, e]
                g.advantage = adv_flat[step, e]
                g.return_val = ret_flat[step, e]
                g.value = self.values[step, e]
                g.action_mask = self.action_masks[step, e]
                items.append(g)

        return DataLoader(items, batch_size=minibatch_size, shuffle=True)


if __name__ == "__main__":
    from asm_graph_builder import asm_to_graph

    sample_asm = """
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

    print("=" * 70)
    print(" GraphRolloutBuffer & PyG DataLoader Verification")
    print("=" * 70)

    num_steps = 4
    num_envs = 2
    buffer = GraphRolloutBuffer(num_steps=num_steps, num_envs=num_envs)

    g_sample = asm_to_graph(sample_asm, max_len=16)

    for step in range(num_steps):
        obs_graphs = [g_sample for _ in range(num_envs)]
        actions = torch.randint(0, 160, (num_envs,))
        logprobs = torch.randn(num_envs)
        rewards = torch.randn(num_envs)
        dones = torch.zeros(num_envs)
        values = torch.randn(num_envs)
        action_masks = torch.ones((num_envs, 160), dtype=torch.bool)

        buffer.insert(step, obs_graphs, actions, logprobs, rewards, dones, values, action_masks)

    advantages = torch.randn(num_steps, num_envs)
    returns = torch.randn(num_steps, num_envs)

    loader = buffer.get_dataloader(advantages, returns, minibatch_size=4)

    for batch in loader:
        print("[*] PyG DataLoader Batch Iteration:")
        print("    Batch inst x shape :", batch['inst'].x.shape)
        print("    Batch action shape :", batch.action.shape)
        print("    Batch return shape :", batch.return_val.shape)
        print("    Batch mask shape   :", batch.action_mask.shape)

    print("=" * 70)
    print("[+] GraphRolloutBuffer PyG DataLoader Integration PASSED!")
    print("=" * 70)
