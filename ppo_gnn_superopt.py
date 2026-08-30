#!/usr/bin/env python3
"""
ppo_gnn_superopt.py — GPU-Accelerated PPO with HeteroGATEncoder + PolicyHead + ValueHead

Replaces flat MLP ActorCritic with GAT Graph Neural Network Policy.
"""

import math
import os
import time
from typing import Callable, Tuple, List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym

from superopt_env import SuperoptEnv
from peephole_rulebook import OPCODE_MAP
from asm_graph_builder import encode_instruction_177dim
from ppo_gnn_actor_critic import GNNActorCritic
from torch_geometric.data import HeteroData, Batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env(env_id: str, seed: int) -> Callable[[], gym.Env]:
    def thunk():
        env = gym.make(env_id)
        env.action_space.seed(seed)
        return env
    return thunk


def obs_vector_to_hetero_data(obs_flat: np.ndarray, max_len: int = 16) -> HeteroData:
    """
    Converts a flat 80-dim observation vector [16, 5] into a PyG HeteroData graph
    with 177-dim node features and Control/Data Flow edges.
    """
    prog = obs_flat.reshape((max_len, 5)).tolist()
    num_insts = len(prog)

    # 1. 177-dim Instruction Node Features
    inst_features = []
    for inst in prog:
        vec = encode_instruction_177dim(inst)
        inst_features.append(vec)

    inst_x = torch.stack(inst_features, dim=0)  # [16, 177]
    reg_x = torch.arange(32, dtype=torch.float32).unsqueeze(-1)  # [32, 1]

    # 2. Control Flow & Data Flow Edges
    ctrl_edges: List[Tuple[int, int]] = []
    data_edges: List[Tuple[int, int]] = []
    reads_edges: List[Tuple[int, int]] = []
    read_by_edges: List[Tuple[int, int]] = []
    writes_edges: List[Tuple[int, int]] = []
    written_by_edges: List[Tuple[int, int]] = []

    last_def: Dict[int, int] = {}

    for i, inst in enumerate(prog):
        opcode_id, rs1, rs2, rd, imm = inst

        # Sequential Control Flow Edge (i -> i+1)
        if i < num_insts - 1:
            ctrl_edges.append((i, i + 1))

        # Register Reads & Def->Use Data Flow Edges
        if rs1 > 0:
            reads_edges.append((i, rs1))
            read_by_edges.append((rs1, i))
            if rs1 in last_def:
                data_edges.append((last_def[rs1], i))

        if rs2 > 0:
            reads_edges.append((i, rs2))
            read_by_edges.append((rs2, i))
            if rs2 in last_def:
                data_edges.append((last_def[rs2], i))

        # Register Writes
        if rd > 0:
            writes_edges.append((i, rd))
            written_by_edges.append((rd, i))
            last_def[rd] = i

    def make_edge_tensor(edge_list: List[Tuple[int, int]]) -> torch.Tensor:
        if not edge_list:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data = HeteroData()

    data['inst'].x = inst_x
    data['reg'].x = reg_x

    data['inst', 'control_flow', 'inst'].edge_index = make_edge_tensor(ctrl_edges)
    data['inst', 'data_flow', 'inst'].edge_index = make_edge_tensor(data_edges)
    data['inst', 'reads', 'reg'].edge_index = make_edge_tensor(reads_edges)
    data['reg', 'read_by', 'inst'].edge_index = make_edge_tensor(read_by_edges)
    data['inst', 'writes', 'reg'].edge_index = make_edge_tensor(writes_edges)
    data['reg', 'written_by', 'inst'].edge_index = make_edge_tensor(written_by_edges)

    return data


def batch_obs_to_hetero_batch(obs_np: np.ndarray, max_len: int = 16) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Converts a batch of flat observation vectors [B, 80] into a PyG HeteroData batch representation on GPU.
    """
    batch_size = obs_np.shape[0]
    data_list = [obs_vector_to_hetero_data(obs_np[b], max_len=max_len) for b in range(batch_size)]
    batch_data = Batch.from_data_list(data_list)

    x_dict = {k: v.to(DEVICE) for k, v in batch_data.x_dict.items()}
    edge_index_dict = {k: v.to(DEVICE) for k, v in batch_data.edge_index_dict.items()}
    batch_dict = {k: v.to(DEVICE) for k, v in batch_data.batch_dict.items()} if hasattr(batch_data, 'batch_dict') else None

    if batch_dict is None:
        batch_dict = {
            'inst': batch_data['inst'].batch.to(DEVICE),
            'reg': batch_data['reg'].batch.to(DEVICE)
        }

    return x_dict, edge_index_dict, batch_dict


class PPOGNNSuperoptTrainer:
    """
    PPO Trainer utilizing GATEncoder + PolicyHead + ValueHead on GPU.
    """

    def __init__(
        self,
        num_envs: int = 4,
        num_steps: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.1,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 256
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.batch_size = num_envs * num_steps
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # Vector Environment
        env_fns = [make_env("SuperoptEnv-v0", seed=i) for i in range(num_envs)]
        self.envs = gym.vector.SyncVectorEnv(env_fns)

        # GNN Actor-Critic Network on GPU
        self.agent = GNNActorCritic(
            inst_in_dim=177,
            reg_in_dim=1,
            hidden_dim=128,
            embed_dim=256,
            num_rules=10,
            max_len=16,
            num_heads=8,
            num_layers=3
        ).to(DEVICE)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)


if __name__ == "__main__":
    print("=" * 60)
    print(" GNN PPO Superopt Trainer Initialized")
    print("=" * 60)
    trainer = PPOGNNSuperoptTrainer(num_envs=4, num_steps=256)
    print(" Compute Device :", DEVICE)
    print(" GNN Architecture : GATEncoder(177->128, 3 layers) -> PolicyHead(256->160) + ValueHead(256->1)")
