#!/usr/bin/env python3
"""
verify_gat_encoder_random.py — Verification Script for HeteroGATEncoder.

Verifies:
1. Random HeteroData graph input -> forward pass -> exact [batch_size, 256] output tensor shape.
2. Zero NaN values (torch.isnan(output).sum() == 0) and all elements finite.
3. Gradient flow through 100% of trainable parameters across all layers.
"""

import sys
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from gat_encoder import HeteroGATEncoder


def generate_random_hetero_graph(num_insts: int = 16, num_regs: int = 32) -> HeteroData:
    """Generates a random RISC-V HeteroData graph for verification."""
    data = HeteroData()

    # 1. Random Node Features
    data['inst'].x = torch.randn(num_insts, 177)
    data['reg'].x = torch.randn(num_regs, 1)

    # 2. Random Control Flow Edges (0..num_insts-1)
    e_ctrl = torch.stack([
        torch.randint(0, num_insts, (20,)),
        torch.randint(0, num_insts, (20,))
    ], dim=0)
    data['inst', 'control_flow', 'inst'].edge_index = e_ctrl

    # 3. Random Data Flow Edges (0..num_insts-1)
    e_data = torch.stack([
        torch.randint(0, num_insts, (15,)),
        torch.randint(0, num_insts, (15,))
    ], dim=0)
    data['inst', 'data_flow', 'inst'].edge_index = e_data

    # 4. Random Reads & Read_by Edges (inst <-> reg)
    inst_idx = torch.randint(0, num_insts, (25,))
    reg_idx = torch.randint(0, num_regs, (25,))
    data['inst', 'reads', 'reg'].edge_index = torch.stack([inst_idx, reg_idx], dim=0)
    data['reg', 'read_by', 'inst'].edge_index = torch.stack([reg_idx, inst_idx], dim=0)

    # 5. Random Writes & Written_by Edges (inst <-> reg)
    inst_w = torch.randint(0, num_insts, (16,))
    reg_w = torch.randint(0, num_regs, (16,))
    data['inst', 'writes', 'reg'].edge_index = torch.stack([inst_w, reg_w], dim=0)
    data['reg', 'written_by', 'inst'].edge_index = torch.stack([reg_w, inst_w], dim=0)

    return data


def run_verification():
    print("=" * 70)
    print(" HETERO GAT ENCODER RANDOM GRAPH VERIFICATION")
    print("=" * 70)

    # 1. Initialize Encoder
    encoder = HeteroGATEncoder(
        inst_in_dim=177,
        reg_in_dim=1,
        hidden_channels=128,
        out_channels=256,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    )
    encoder.train()

    # 2. Generate Random HeteroData Graph
    graph = generate_random_hetero_graph(num_insts=16, num_regs=32)

    print(f"[*] Generated Random HeteroData Graph:")
    print(f"    Instruction Nodes : {graph['inst'].x.shape}")
    print(f"    Register Nodes    : {graph['reg'].x.shape}")
    for rel in graph.edge_types:
        print(f"    Relation {rel} : {graph[rel].edge_index.shape}")

    # 3. Forward Pass Verification
    graph_embed, inst_embed = encoder(graph.x_dict, graph.edge_index_dict)

    print("\n[*] Forward Pass Outputs:")
    print(f"    Graph Embedding Shape : {graph_embed.shape}")
    print(f"    Inst Node Embed Shape : {inst_embed.shape}")

    # Check 1: 256-dim Output Shape
    assert graph_embed.shape == (1, 256), f"❌ Output shape mismatch! Expected (1, 256), got {graph_embed.shape}"
    print("    ✅ CHECK 1 PASSED: Graph embedding output shape is exactly [1, 256]")

    # Check 2: No NaN or Inf Values
    nan_count = torch.isnan(graph_embed).sum().item()
    is_finite = torch.isfinite(graph_embed).all().item()
    assert nan_count == 0, f"❌ Detected {nan_count} NaN values in graph_embed!"
    assert is_finite, "❌ Detected non-finite (Inf) values in graph_embed!"
    print("    ✅ CHECK 2 PASSED: 0 NaNs detected, all values are strictly finite")

    # 4. Backward Pass & Gradient Flow Verification
    loss = graph_embed.pow(2).sum()
    loss.backward()

    print("\n[*] Checking Gradient Flow across all trainable layer parameters:")
    no_grad_count = 0
    param_count = 0

    for name, param in encoder.named_parameters():
        if param.requires_grad:
            param_count += 1
            if param.grad is None or torch.isnan(param.grad).any() or (param.grad.abs().sum() == 0):
                print(f"    ❌ Parameter missing gradient: {name}")
                no_grad_count += 1
            else:
                grad_norm = param.grad.norm().item()
                print(f"    ✓ {name:<60} | grad_norm = {grad_norm:.6f}")

    print("-" * 70)
    print(f" Total Trainable Parameters Checked : {param_count}")
    print(f" Parameters with Valid Gradients    : {param_count - no_grad_count} / {param_count}")

    assert no_grad_count == 0, f"❌ Gradient flow failed for {no_grad_count} parameters!"
    print("    ✅ CHECK 3 PASSED: 100% gradient flow verified across all layers!")
    print("=" * 70)
    print("[+] ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
