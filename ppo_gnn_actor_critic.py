#!/usr/bin/env python3
"""
ppo_gnn_actor_critic.py — GNN Actor-Critic Architecture for RISC-V Superoptimization.

Components:
    1. HeteroGATEncoder : Encodes RISC-V assembly graphs to [B, 256] graph embedding and [N, 128] node embeddings.
    2. PolicyHead       : Linear(256 -> num_rules * max_len) (Discrete 160 action logits).
    3. ValueHead        : Linear(256 -> 1) (Scalar state value V(s)).
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from gat_encoder import HeteroGATEncoder, HAS_PYG


class GNNActorCritic(nn.Module):
    """
    GNN Actor-Critic Network for PPO with Action Masking.

    Args:
        inst_in_dim : Node feature dimension for instruction nodes (177)
        reg_in_dim  : Node feature dimension for register nodes (1)
        hidden_dim  : Hidden channel dimension for GAT layers (128)
        embed_dim   : Graph embedding output dimension (256)
        num_rules   : Number of peephole rules (10)
        max_len     : Maximum instruction slot length (16)
        num_heads   : Number of GAT attention heads (8)
        num_layers  : Number of HeteroConv layers (3)
    """

    def __init__(
        self,
        inst_in_dim: int = 177,
        reg_in_dim: int = 1,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        num_rules: int = 10,
        max_len: int = 16,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        if not HAS_PYG:
            raise ImportError("torch_geometric is required for GNNActorCritic.")

        self.num_rules = num_rules
        self.max_len = max_len
        self.act_dim = num_rules * max_len  # 10 * 16 = 160

        # 1. GAT Backbone Encoder -> [B, 256] Graph Embedding
        self.encoder = HeteroGATEncoder(
            inst_in_dim=inst_in_dim,
            reg_in_dim=reg_in_dim,
            hidden_channels=hidden_dim,
            out_channels=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # 2. Policy Head: Linear(256 -> 160)
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, self.act_dim)
        )

        # 3. Value Head: Linear(256 -> 1)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal weight initialization for standard RL stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Smaller weights for final policy output layer to encourage initial high entropy exploration
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def get_value(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        batch_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Computes scalar state value V(s) of shape [batch_size]."""
        graph_embed, _ = self.encoder(x_dict, edge_index_dict, batch_dict=batch_dict)
        value = self.value_head(graph_embed).squeeze(-1)
        return value

    def get_action_and_value(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        action: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        batch_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes action, log probability, entropy, and state value V(s) with action masking.

        Returns:
            action    : Sampled action indices [batch_size]
            log_prob  : Log probabilities of actions [batch_size]
            entropy   : Policy distribution entropy [batch_size]
            value     : State values V(s) [batch_size]
        """
        # 1. Forward pass through GAT encoder
        graph_embed, _ = self.encoder(x_dict, edge_index_dict, batch_dict=batch_dict)

        # 2. Compute Policy Logits & State Value
        logits = self.policy_head(graph_embed)  # [batch_size, 160]
        value = self.value_head(graph_embed).squeeze(-1)  # [batch_size]

        # 3. Apply Action Masking (Set invalid action logits to -1e9)
        if action_mask is not None:
            if action_mask.dim() == 1:
                if action_mask.shape[0] < logits.shape[-1]:
                    logits = logits[..., :action_mask.shape[0]]
                elif action_mask.shape[0] > logits.shape[-1]:
                    action_mask = action_mask[:logits.shape[-1]]
            else:
                if action_mask.shape[-1] < logits.shape[-1]:
                    logits = logits[..., :action_mask.shape[-1]]
                elif action_mask.shape[-1] > logits.shape[-1]:
                    action_mask = action_mask[..., :logits.shape[-1]]

            if action_mask.dtype == torch.bool:
                logits = torch.where(action_mask, logits, torch.tensor(-1e9, device=logits.device))
            else:
                logits = logits + (1.0 - action_mask) * (-1e9)

        # 4. Categorical Action Distribution
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()

        return action, log_prob, entropy, value


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

    graph = asm_to_graph(sample_asm, max_len=16)
    x_dict = graph.x_dict
    edge_index_dict = graph.edge_index_dict

    model = GNNActorCritic(inst_in_dim=177, reg_in_dim=1, embed_dim=256, num_rules=10, max_len=16)
    
    # Fake boolean action mask [1, 160]
    action_mask = torch.ones((1, 160), dtype=torch.bool)
    action_mask[0, 5:] = False  # Only allow first 5 actions

    action, log_prob, entropy, value = model.get_action_and_value(x_dict, edge_index_dict, action_mask=action_mask)

    print("=" * 60)
    print(" GNNActorCritic Architecture & Heads Test")
    print("=" * 60)
    print(model)
    print("\nForward Pass Results:")
    print("  Sampled Action :", action.item())
    print("  Log Prob       :", log_prob.item())
    print("  Entropy        :", entropy.item())
    print("  State Value V  :", value.item())
