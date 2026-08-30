#!/usr/bin/env python3
"""
gat_encoder.py — Heterogeneous Graph Attention Network (HeteroConv) Encoder for RISC-V Assembly Graphs.

Handles heterogeneous edge relations separately with 100% gradient flow across inst and reg nodes:
    - ('inst', 'control_flow', 'inst') : GATConv(128, 128, heads=8)
    - ('inst', 'data_flow', 'inst')    : GATConv(128, 128, heads=8)
    - ('inst', 'reads', 'reg')         : GATConv((128, 128), 128, heads=8)
    - ('reg', 'read_by', 'inst')       : GATConv((128, 128), 128, heads=8)
    - ('inst', 'writes', 'reg')        : GATConv((128, 128), 128, heads=8)
    - ('reg', 'written_by', 'inst')    : GATConv((128, 128), 128, heads=8)
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import HeteroConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class HeteroGATEncoder(nn.Module):
    """
    Heterogeneous Graph Attention Network (HeteroConv GAT) Encoder.
    """

    def __init__(
        self,
        inst_in_dim: int = 177,
        reg_in_dim: int = 1,
        hidden_channels: int = 128,
        out_channels: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        if not HAS_PYG:
            raise ImportError("torch_geometric is required for HeteroGATEncoder.")

        self.inst_in_dim = inst_in_dim
        self.reg_in_dim = reg_in_dim
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        # 1. Input Node Feature Projections -> 128
        self.inst_input_proj = nn.Sequential(
            nn.Linear(inst_in_dim, hidden_channels),
            nn.LayerNorm(hidden_channels)
        )
        self.reg_input_proj = nn.Sequential(
            nn.Linear(reg_in_dim, hidden_channels),
            nn.LayerNorm(hidden_channels)
        )

        # 2. 3x HeteroConv Layers with separate GATConv per relation
        self.hetero_convs = nn.ModuleList()
        self.inst_norms = nn.ModuleList()
        self.reg_norms = nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {
                ('inst', 'control_flow', 'inst'): GATConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=True
                ),
                ('inst', 'data_flow', 'inst'): GATConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=True
                ),
                ('inst', 'reads', 'reg'): GATConv(
                    in_channels=(hidden_channels, hidden_channels),
                    out_channels=hidden_channels,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False
                ),
                ('reg', 'read_by', 'inst'): GATConv(
                    in_channels=(hidden_channels, hidden_channels),
                    out_channels=hidden_channels,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False
                ),
                ('inst', 'writes', 'reg'): GATConv(
                    in_channels=(hidden_channels, hidden_channels),
                    out_channels=hidden_channels,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False
                ),
                ('reg', 'written_by', 'inst'): GATConv(
                    in_channels=(hidden_channels, hidden_channels),
                    out_channels=hidden_channels,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False
                ),
            }

            self.hetero_convs.append(HeteroConv(conv_dict, aggr='sum'))
            self.inst_norms.append(nn.LayerNorm(hidden_channels))
            self.reg_norms.append(nn.LayerNorm(hidden_channels))

        # 3. Dual-Node Readout Projection: (inst_mean + inst_max + reg_mean + reg_max) = 512 -> 256
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_channels * 4, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        batch_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 1. Input Node Feature Projections -> 128
        h_dict = {
            'inst': F.elu(self.inst_input_proj(x_dict['inst'])),
            'reg': F.elu(self.reg_input_proj(x_dict['reg']))
        }

        # Setup default batch vectors if single graph
        if batch_dict is None or 'inst' not in batch_dict:
            inst_batch = torch.zeros(h_dict['inst'].size(0), dtype=torch.long, device=h_dict['inst'].device)
        else:
            inst_batch = batch_dict['inst']

        if batch_dict is None or 'reg' not in batch_dict:
            reg_batch = torch.zeros(h_dict['reg'].size(0), dtype=torch.long, device=h_dict['reg'].device)
        else:
            reg_batch = batch_dict['reg']

        valid_edges = {
            rel: edge_index
            for rel, edge_index in edge_index_dict.items()
            if rel in [
                ('inst', 'control_flow', 'inst'),
                ('inst', 'data_flow', 'inst'),
                ('inst', 'reads', 'reg'),
                ('reg', 'read_by', 'inst'),
                ('inst', 'writes', 'reg'),
                ('reg', 'written_by', 'inst')
            ] and edge_index.numel() > 0
        }

        # 2. 3x HeteroConv Layers with Residual Connections
        for i, hetero_conv in enumerate(self.hetero_convs):
            h_in_inst = h_dict['inst']
            h_in_reg = h_dict['reg']

            out_dict = hetero_conv(h_dict, valid_edges)

            if 'inst' in out_dict:
                h_inst = self.inst_norms[i](out_dict['inst'])
                h_inst = F.elu(h_inst)
                h_inst = F.dropout(h_inst, p=self.dropout, training=self.training)
                h_dict['inst'] = h_inst + h_in_inst
            else:
                h_dict['inst'] = h_in_inst

            if 'reg' in out_dict:
                h_reg = self.reg_norms[i](out_dict['reg'])
                h_reg = F.elu(h_reg)
                h_reg = F.dropout(h_reg, p=self.dropout, training=self.training)
                h_dict['reg'] = h_reg + h_in_reg
            else:
                h_dict['reg'] = h_in_reg

        # 3. Dual-Node Pooling across Instruction and Register Nodes
        inst_mean = global_mean_pool(h_dict['inst'], inst_batch)  # [batch_size, 128]
        inst_max  = global_max_pool(h_dict['inst'], inst_batch)   # [batch_size, 128]

        reg_mean  = global_mean_pool(h_dict['reg'], reg_batch)    # [batch_size, 128]
        reg_max   = global_max_pool(h_dict['reg'], reg_batch)     # [batch_size, 128]

        pooled = torch.cat([inst_mean, inst_max, reg_mean, reg_max], dim=-1)  # [batch_size, 512]

        # 4. Readout Projection -> 256
        graph_embed = self.pool_proj(pooled)  # [batch_size, 256]

        return graph_embed, h_dict['inst']


GATEncoder = HeteroGATEncoder


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

    encoder = HeteroGATEncoder()
    graph_embed, inst_embed = encoder(x_dict, edge_index_dict)

    print("=" * 60)
    print(" PyG HeteroConv HeteroGATEncoder Test")
    print("=" * 60)
    print(encoder)
    print("\nForward Pass Outputs:")
    print("  Graph Embedding Shape :", graph_embed.shape, " (Expected [1, 256])")
    print("  Inst Node Embed Shape :", inst_embed.shape, "  (Expected [16, 128])")
