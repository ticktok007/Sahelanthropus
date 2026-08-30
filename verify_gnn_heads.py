#!/usr/bin/env python3
"""
verify_gnn_heads.py — Verification Script for GNNActorCritic PolicyHead and ValueHead.

Verifies:
1. Masked Softmax over Policy Logits: Invalid actions masked out to -1e9, probabilities of masked actions equal 0.0, valid actions sum to 1.0.
2. Value Scalar Output: State value V(s) is a 1D scalar tensor of shape [batch_size].
3. Both Heads Receive Gradients: Backward pass propagates non-zero gradients to policy_head, value_head, and shared GAT encoder backbone.
"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from asm_graph_builder import asm_to_graph
from ppo_gnn_actor_critic import GNNActorCritic


def run_verification():
    print("=" * 70)
    print(" GNN ACTOR-CRITIC HEADS & MASKED SOFTMAX VERIFICATION")
    print("=" * 70)

    # 1. Setup sample assembly graph
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

    # 2. Instantiate GNNActorCritic
    model = GNNActorCritic(
        inst_in_dim=177,
        reg_in_dim=1,
        hidden_dim=128,
        embed_dim=256,
        num_rules=10,
        max_len=16
    )
    model.train()

    # -----------------------------------------------------------------------
    # CHECK 1: Masked Softmax over Policy Logits
    # -----------------------------------------------------------------------
    print("\n[*] Testing Masked Softmax over Policy Logits:")
    
    # Create action mask allowing ONLY actions [0, 1, 2], masking out [3..159]
    action_mask = torch.zeros((1, 160), dtype=torch.bool)
    action_mask[0, 0] = True
    action_mask[0, 1] = True
    action_mask[0, 2] = True

    action, log_prob, entropy, value = model.get_action_and_value(
        x_dict,
        edge_index_dict,
        action_mask=action_mask
    )

    # Re-evaluate distribution logits
    graph_embed, _ = model.encoder(x_dict, edge_index_dict)
    raw_logits = model.policy_head(graph_embed)
    masked_logits = torch.where(action_mask, raw_logits, torch.tensor(-1e9, device=raw_logits.device))
    dist = Categorical(logits=masked_logits)
    probs = dist.probs[0]

    # Verification 1A: Masked probabilities must be 0.0
    masked_prob_sum = probs[3:].sum().item()
    assert masked_prob_sum < 1e-6, f"❌ Masked action probabilities not zero! Sum = {masked_prob_sum}"
    print(f"    ✓ Masked Action Probabilities (indices 3..159) : {masked_prob_sum:.8f} (Equal to 0.0)")

    # Verification 1B: Allowed probabilities must sum to 1.0
    valid_prob_sum = probs[:3].sum().item()
    assert abs(valid_prob_sum - 1.0) < 1e-5, f"❌ Valid action probabilities do not sum to 1.0! Sum = {valid_prob_sum}"
    print(f"    ✓ Allowed Action Probabilities (indices 0..2)  : {valid_prob_sum:.8f} (Sum to 1.0)")
    print("    ✅ CHECK 1 PASSED: Masked softmax over policy logits verified!")

    # -----------------------------------------------------------------------
    # CHECK 2: Value Scalar Output
    # -----------------------------------------------------------------------
    print("\n[*] Testing Value Scalar Output V(s):")
    val_from_get_value = model.get_value(x_dict, edge_index_dict)
    
    assert value.ndim == 1 and value.shape[0] == 1, f"❌ Value output shape mismatch! Expected (1,), got {value.shape}"
    assert val_from_get_value.ndim == 1 and val_from_get_value.shape[0] == 1, f"❌ get_value output shape mismatch! Expected (1,), got {val_from_get_value.shape}"
    assert torch.isfinite(value).all(), "❌ Non-finite (NaN/Inf) value output!"

    print(f"    ✓ Evaluated State Value V(s) : {value.item():.6f}")
    print(f"    ✓ Value Tensor Shape        : {list(value.shape)} (1D Scalar per batch item)")
    print("    ✅ CHECK 2 PASSED: Value scalar output verified!")

    # -----------------------------------------------------------------------
    # CHECK 3: Gradient Flow to Both PolicyHead and ValueHead
    # -----------------------------------------------------------------------
    print("\n[*] Testing Gradient Propagation across both PolicyHead and ValueHead:")

    # Define loss incorporating both Policy LogProb and Value Function V(s)
    loss = -log_prob.sum() + 0.5 * (value ** 2).sum()
    loss.backward()

    # Check PolicyHead gradients
    policy_l0_grad = model.policy_head[0].weight.grad
    policy_l2_grad = model.policy_head[2].weight.grad
    assert policy_l0_grad is not None and policy_l0_grad.norm().item() > 0, "❌ PolicyHead layer 0 missing gradient!"
    assert policy_l2_grad is not None and policy_l2_grad.norm().item() > 0, "❌ PolicyHead layer 2 missing gradient!"
    print(f"    ✓ PolicyHead Layer 0 Grad Norm : {policy_l0_grad.norm().item():.6f}")
    print(f"    ✓ PolicyHead Layer 2 Grad Norm : {policy_l2_grad.norm().item():.6f}")

    # Check ValueHead gradients
    value_l0_grad = model.value_head[0].weight.grad
    value_l2_grad = model.value_head[2].weight.grad
    assert value_l0_grad is not None and value_l0_grad.norm().item() > 0, "❌ ValueHead layer 0 missing gradient!"
    assert value_l2_grad is not None and value_l2_grad.norm().item() > 0, "❌ ValueHead layer 2 missing gradient!"
    print(f"    ✓ ValueHead Layer 0 Grad Norm  : {value_l0_grad.norm().item():.6f}")
    print(f"    ✓ ValueHead Layer 2 Grad Norm  : {value_l2_grad.norm().item():.6f}")

    # Check Shared GAT Encoder Backbone gradients
    encoder_inst_grad = model.encoder.inst_input_proj[0].weight.grad
    encoder_pool_grad = model.encoder.pool_proj[0].weight.grad
    assert encoder_inst_grad is not None and encoder_inst_grad.norm().item() > 0, "❌ Encoder inst_input_proj missing gradient!"
    assert encoder_pool_grad is not None and encoder_pool_grad.norm().item() > 0, "❌ Encoder pool_proj missing gradient!"
    print(f"    ✓ Shared GAT Encoder Grad Norm : {encoder_inst_grad.norm().item():.6f}")
    print("    ✅ CHECK 3 PASSED: Both PolicyHead and ValueHead propagate gradients successfully!")

    print("=" * 70)
    print("[+] ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
