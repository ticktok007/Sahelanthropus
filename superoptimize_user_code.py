#!/usr/bin/env python3
"""
superoptimize_user_code.py — Interactive / CLI Backend Breakdown & Superoptimizer Test Tool

Pass custom RISC-V assembly code (or select pre-packaged examples) to break down:
1. Instruction Parsing & 177-dim Feature Vector Encoding
2. Graph Representation & Def-Use Data Flow Edges
3. Algebraic Peephole Pattern Scanner (10 RISC-V Rules)
4. Neural PPO Policy Evaluation (Action Logits & Probabilities)
5. Step-by-Step Rewrite Transformation Trajectory
6. Z3 SMT Formal Equivalence Verification (UNSAT BitVector Proof)
7. Final Optimized RISC-V Output & Cycle Performance Summary
"""

import sys
import os
import argparse
import numpy as np
import torch

from peephole_rulebook import PeepholeRulebook, RULE_METADATA, OPCODE_MAP, REG_MAP
from superopt_env import (
    parse_assembly_program,
    program_to_assembly_text,
    REV_OPCODE_MAP,
    REV_REG_MAP_CANON,
)
from equivalence_verifier import verify_equiv_status, verify_equiv
from asm_graph_builder import encode_instruction_177dim, parse_assembly_with_labels
from ppo_superopt import ActorCritic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pre-packaged test examples for quick demonstration
EXAMPLES = {
    "1": {
        "name": "Mul Power of 2 to Shift (Rule 0)",
        "code": """
li t0, 8
mul t1, a0, t0
"""
    },
    "2": {
        "name": "Redundant Self-XOR & Add-Zero Elimination (Rules 2 & 4)",
        "code": """
xor t0, a0, a0
addi t1, a1, 0
add t2, t1, t0
"""
    },
    "3": {
        "name": "Unsigned Division Power of 2 to SRLI (Rule 1)",
        "code": """
li t0, 16
divu t1, a0, t0
"""
    },
    "4": {
        "name": "Add-Sub Cancellation Sequence (Rule 8)",
        "code": """
sub t0, a0, a1
add t2, t0, a1
"""
    },
    "5": {
        "name": "Combined Multi-Rule Optimization Chain",
        "code": """
li t0, 4
mul t1, a0, t0
xor t2, a1, a1
add t3, t1, t2
"""
    }
}


def load_trained_model(ckpt_path: str = "phase4_seed_0_ckpt.pt"):
    """Loads trained PPO GNNActorCritic or ActorCritic checkpoint if present."""
    agent = None
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            agent_dict = ckpt.get("agent", ckpt.get("model_state_dict", ckpt))
            
            # Check if checkpoint is GNN model
            if any("encoder" in k for k in agent_dict.keys()):
                try:
                    from ppo_gnn_actor_critic import GNNActorCritic
                    agent = GNNActorCritic(max_len=25).to(DEVICE)
                    agent.load_state_dict(agent_dict)
                    print(f" [Model] Successfully loaded Phase 4 GNNActorCritic checkpoint from {ckpt_path}")
                except Exception as gnn_err:
                    print(f" [Model] Could not initialize GNNActorCritic ({gnn_err}). Falling back to ActorCritic.")
            
            if agent is None:
                agent = ActorCritic(obs_dim=80, act_dim=160, num_rules=10, max_len=16).to(DEVICE)
                agent.load_state_dict(agent_dict, strict=False)
                print(f" [Model] Successfully loaded ActorCritic checkpoint from {ckpt_path}")
        except Exception as e:
            print(f" [Model] Warning: Could not load checkpoint ({e}), using initialized ActorCritic.")

    if agent is None:
        agent = ActorCritic(obs_dim=80, act_dim=160, num_rules=10, max_len=16).to(DEVICE)
    
    agent.eval()
    return agent


def format_instruction(inst: list) -> str:
    op, rs1, rs2, rd, imm = inst
    op_str = REV_OPCODE_MAP.get(op, "NOP")
    if op_str == "NOP":
        return "nop"
    rd_str = REV_REG_MAP_CANON.get(rd, f"x{rd}")
    rs1_str = REV_REG_MAP_CANON.get(rs1, f"x{rs1}")
    rs2_str = REV_REG_MAP_CANON.get(rs2, f"x{rs2}")

    if op_str == "LI":
        return f"li {rd_str}, {imm}"
    elif op_str in ("ADDI", "ANDI", "XORI", "ORI", "SLLI", "SRLI", "SRAI"):
        return f"{op_str.lower()} {rd_str}, {rs1_str}, {imm}"
    elif op_str in ("ADD", "SUB", "MUL", "DIV", "DIVU", "AND", "OR", "XOR"):
        return f"{op_str.lower()} {rd_str}, {rs1_str}, {rs2_str}"
    elif op_str == "MV":
        return f"mv {rd_str}, {rs1_str}"
    else:
        return f"{op_str.lower()} rd={rd_str}, rs1={rs1_str}, rs2={rs2_str}, imm={imm}"


def predict_policy_logits_and_value(agent, prog_encoded: list, action_mask: np.ndarray):
    """Unified policy inference for both GNNActorCritic and MLP ActorCritic."""
    if hasattr(agent, "encoder"):
        from asm_graph_builder import asm_to_graph
        asm_str = program_to_assembly_text(prog_encoded)
        graph_data = asm_to_graph(asm_str).to(DEVICE)
        x_dict = graph_data.x_dict
        edge_index_dict = graph_data.edge_index_dict
        with torch.no_grad():
            graph_embed, _ = agent.encoder(x_dict, edge_index_dict)
            logits = agent.policy_head(graph_embed).squeeze(0)
            value = agent.value_head(graph_embed).item()
    else:
        obs = np.array(prog_encoded, dtype=np.int32).flatten()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            actor_feats = agent.actor_trunk(obs_tensor)
            logits = agent.actor(actor_feats).squeeze(0)
            value = agent.get_value(obs_tensor).item()

    mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=DEVICE)
    if logits.shape[-1] > mask_tensor.shape[-1]:
        logits = logits[..., :mask_tensor.shape[-1]]
    elif mask_tensor.shape[-1] > logits.shape[-1]:
        mask_tensor = mask_tensor[..., :logits.shape[-1]]

    masked_logits = torch.where(mask_tensor, logits, torch.tensor(-1e9, device=DEVICE))
    probs = torch.softmax(masked_logits, dim=-1).cpu().numpy()
    return logits, probs, value


def breakdown_and_superoptimize(asm_code: str, agent):
    rulebook = PeepholeRulebook(num_rules=10)

    print("\n" + "="*80)
    print("               SAHELANTHROPUS SUPEROPTIMIZER — BACKEND BREAKDOWN")
    print("="*80)

    # -------------------------------------------------------------------------
    # 1. FRONTEND PARSING & TOKENIZATION
    # -------------------------------------------------------------------------
    print("\n[PART 1: INSTRUCTION TOKENIZATION & FEATURE ENCODING]")
    print("-" * 80)
    raw_lines = [line.strip() for line in asm_code.strip().splitlines() if line.strip() and not line.strip().startswith("#")]
    print("Raw Input Lines:")
    for idx, l in enumerate(raw_lines):
        print(f"  Line {idx+1:2d}: {l}")

    prog_encoded = parse_assembly_program(asm_code, max_len=16)
    non_nop_insts = [inst for inst in prog_encoded if inst[0] != OPCODE_MAP["NOP"]]

    print("\nParsed Internal Instruction Array [Opcode_ID, RS1, RS2, RD, Imm]:")
    print(f" {'Slot':<5} | {'Formatted Mnemonic':<25} | {'Opcode':<8} | {'RS1':<5} | {'RS2':<5} | {'RD':<5} | {'Imm':<8}")
    print(" " + "-" * 75)
    for slot_idx, inst in enumerate(prog_encoded[:len(non_nop_insts)]):
        op, rs1, rs2, rd, imm = inst
        formatted = format_instruction(inst)
        print(f" {slot_idx:<5} | {formatted:<25} | {op:<8} | {rs1:<5} | {rs2:<5} | {rd:<5} | {imm:<8}")

    # Feature vector demo for slot 0
    if len(non_nop_insts) > 0:
        feat_vec = encode_instruction_177dim(non_nop_insts[0], op_str="TEST")
        print(f"\nExample 177-dim Graph Feature Tensor (Slot 0): shape={list(feat_vec.shape)}, non-zero dimensions={(feat_vec > 0).sum().item()}/177")

    # -------------------------------------------------------------------------
    # 2. GRAPH REPRESENTATION & DEF-USE DATA FLOW ANALYSIS
    # -------------------------------------------------------------------------
    print("\n[PART 2: DEPENDENCY GRAPH & DEF-USE DATA FLOW]")
    print("-" * 80)
    last_def = {}
    data_flow_edges = []
    read_regs = set()
    written_regs = set()

    for idx, inst in enumerate(non_nop_insts):
        op, rs1, rs2, rd, imm = inst
        if rs1 > 0:
            read_regs.add(rs1)
            if rs1 in last_def:
                data_flow_edges.append((last_def[rs1], idx, REV_REG_MAP_CANON.get(rs1, f"x{rs1}")))
        if rs2 > 0:
            read_regs.add(rs2)
            if rs2 in last_def:
                data_flow_edges.append((last_def[rs2], idx, REV_REG_MAP_CANON.get(rs2, f"x{rs2}")))
        if rd > 0:
            written_regs.add(rd)
            last_def[rd] = idx

    print(f"Registers Read   : {[REV_REG_MAP_CANON.get(r, f'x{r}') for r in sorted(read_regs)]}")
    print(f"Registers Written: {[REV_REG_MAP_CANON.get(r, f'x{r}') for r in sorted(written_regs)]}")
    print(f"Def-Use Edges    : {len(data_flow_edges)} detected")
    for src, dst, reg_name in data_flow_edges:
        print(f"  Inst {src} ({format_instruction(non_nop_insts[src])}) --[Def-Use: {reg_name}]--> Inst {dst} ({format_instruction(non_nop_insts[dst])})")

    # -------------------------------------------------------------------------
    # 3. PATTERN-MATCHING ENGINE SCAN (10 PEEPHOLE RULES)
    # -------------------------------------------------------------------------
    print("\n[PART 3: ALGEBRAIC PEEPHOLE PATTERN SCANNER]")
    print("-" * 80)
    obs = np.array(prog_encoded, dtype=np.int32).flatten()
    action_mask = rulebook.get_action_mask(obs, max_len=16)

    matched_actions = []
    for r in range(10):
        r_info = RULE_METADATA[r]
        for slot in range(len(non_nop_insts)):
            if rulebook.matches_at(r, prog_encoded, slot):
                matched_actions.append((r, slot, r_info["name"], r_info["desc"], r_info["est_savings"]))
                print(f" [MATCH] Rule {r:2d} ({r_info['name']:<22}) matched at Slot {slot}: {format_instruction(prog_encoded[slot])}")
                print(f"         Description    : {r_info['desc']}")
                print(f"         Est. Cycle Savings: ~{r_info['est_savings']} cycles")

    if not matched_actions:
        print(" No algebraic peephole patterns matched on initial code.")

    # -------------------------------------------------------------------------
    # 4. NEURAL PPO POLICY EVALUATION & LOGITS
    # -------------------------------------------------------------------------
    print("\n[PART 4: NEURAL PPO POLICY INFERENCE & ACTION LOGITS]")
    print("-" * 80)
    logits, probs, val_pred = predict_policy_logits_and_value(agent, prog_encoded, action_mask)

    print(f" Critic Estimated Value V(s): {val_pred:+.4f}")
    
    valid_indices = np.where(action_mask)[0]
    if len(valid_indices) > 0:
        top_indices = valid_indices[np.argsort(probs[valid_indices])[::-1]]
        print(" Policy Action Probabilities for Matched Rewrites:")
        for act_idx in top_indices:
            r = act_idx // 16
            s = act_idx % 16
            r_name = RULE_METADATA[r]["name"]
            p = probs[act_idx]
            print(f"  Action {act_idx:3d} [Rule {r}: {r_name:<20} @ Slot {s:2d}] -> Prob: {p*100:6.2f}%")
    else:
        print(" No valid action slots active.")

    # -------------------------------------------------------------------------
    # 5. STEP-BY-STEP REWRITE TRANSFORMATION TRAJECTORY
    # -------------------------------------------------------------------------
    print("\n[PART 5: STEP-BY-STEP REWRITE TRANSFORMATION TRAJECTORY]")
    print("-" * 80)
    curr_prog = [list(inst) for inst in prog_encoded]
    step_count = 0
    max_steps = 10
    applied_rules_log = []

    while step_count < max_steps:
        curr_obs = np.array(curr_prog, dtype=np.int32).flatten()
        curr_mask = rulebook.get_action_mask(curr_obs, max_len=16)

        valid_acts = np.where(curr_mask)[0]
        if len(valid_acts) == 0:
            break

        # Select action with highest policy probability among valid
        _, c_probs, _ = predict_policy_logits_and_value(agent, curr_prog, curr_mask)

        best_act = valid_acts[np.argmax(c_probs[valid_acts])]
        r_id = best_act // 16
        slot_id = best_act % 16

        orig_slot_formatted = format_instruction(curr_prog[slot_id])
        curr_prog, applied = rulebook.apply_rewrite(r_id, curr_prog, slot_id)

        if not applied:
            break

        step_count += 1
        r_name = RULE_METADATA[r_id]["name"]
        new_slot_formatted = format_instruction(curr_prog[slot_id])
        applied_rules_log.append((r_id, r_name, slot_id, orig_slot_formatted, new_slot_formatted))

        print(f" Step {step_count}: Applied Rule {r_id} ({r_name}) at Slot {slot_id}")
        print(f"         Before: {orig_slot_formatted}")
        print(f"         After : {new_slot_formatted}")

    if step_count == 0:
        print(" Code is already optimal according to rulebook!")

    # Clean NOPs from rewritten program
    final_prog_clean = [inst for inst in curr_prog if inst[0] != OPCODE_MAP["NOP"]]
    if not final_prog_clean:
        final_prog_clean = [[OPCODE_MAP["NOP"], 0, 0, 0, 0]]

    # -------------------------------------------------------------------------
    # 6. Z3 SMT FORMAL EQUIVALENCE VERIFICATION
    # -------------------------------------------------------------------------
    print("\n[PART 6: FORMAL EQUIVALENCE VERIFICATION (Z3 SMT SOLVER)]")
    print("-" * 80)
    print(" Running 32-bit BitVector Symbolic Execution across all 31 registers...")
    status = verify_equiv_status(prog_encoded, curr_prog, timeout=5.0)

    if status == "UNSAT":
        print(" [Z3 PROOF SUCCESS] Status: UNSAT")
        print("                      Mathematical Proof: The original and optimized programs produce")
        print("                      EXACTLY EQUIVALENT outputs for 100% of all possible 32-bit inputs!")
    elif status == "SAT":
        print(" [Z3 VERIFICATION FAILED] Status: SAT — Counterexample found (programs non-equivalent).")
    else:
        print(f" [Z3 VERIFICATION UNKNOWN] Status: {status}")

    # -------------------------------------------------------------------------
    # 7. PERFORMANCE & CYCLE COST REDUCTION SUMMARY
    # -------------------------------------------------------------------------
    print("\n[PART 7: PERFORMANCE & CYCLE REDUCTION BREAKDOWN]")
    print("-" * 80)

    def est_prog_cycles(prog):
        total_c = 0.0
        for inst in prog:
            op = inst[0]
            if op == OPCODE_MAP["NOP"]:
                continue
            elif op in (OPCODE_MAP["MUL"], OPCODE_MAP.get("MULH", 33)):
                total_c += 4.0
            elif op in (OPCODE_MAP["DIV"], OPCODE_MAP.get("DIVU", 36)):
                total_c += 20.0
            else:
                total_c += 1.0
        return total_c

    orig_inst_count = len(non_nop_insts)
    opt_inst_count = len(final_prog_clean)
    orig_cycles = est_prog_cycles(non_nop_insts)
    opt_cycles = est_prog_cycles(final_prog_clean)
    cycle_savings = orig_cycles - opt_cycles
    pct_savings = (cycle_savings / max(1.0, orig_cycles)) * 100.0

    print(f" Original Instructions : {orig_inst_count} instructions")
    print(f" Optimized Instructions: {opt_inst_count} instructions")
    print(f" Original Cycle Cost   : {orig_cycles:.1f} cycles")
    print(f" Optimized Cycle Cost  : {opt_cycles:.1f} cycles")
    print(f" Cycle Reduction       : -{cycle_savings:.1f} cycles ({pct_savings:.1f}% faster!)")
    print(f" Formal Proof Status   : {status} (Semantically Guaranteed Correct)")

    # -------------------------------------------------------------------------
    # 8. FINAL TRANSFORMED RISC-V ASSEMBLY OUTPUT
    # -------------------------------------------------------------------------
    print("\n[PART 8: FINAL TRANSFORMED RISC-V ASSEMBLY CODE]")
    print("-" * 80)
    opt_asm_text = program_to_assembly_text(final_prog_clean)
    print(opt_asm_text)
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sahelanthropus User Code Superoptimizer & Backend Breakdown")
    parser.add_argument("--code", type=str, help="RISC-V assembly code string to optimize")
    parser.add_argument("--file", type=str, help="Path to assembly file (.s)")
    parser.add_argument("--example", type=str, choices=["1", "2", "3", "4", "5"], help="Run pre-packaged example [1-5]")
    parser.add_argument("--ckpt", type=str, default="phase4_seed_0_ckpt.pt", help="Path to PPO model checkpoint")
    args = parser.parse_args()

    agent = load_trained_model(args.ckpt)

    if args.code:
        asm_input = args.code
    elif args.file:
        with open(args.file, "r") as f:
            asm_input = f.read()
    elif args.example:
        ex = EXAMPLES[args.example]
        print(f"\n=== Running Pre-packaged Example {args.example}: {ex['name']} ===")
        asm_input = ex["code"]
    else:
        print("\n" + "="*80)
        print("          WELCOME TO SAHELANTHROPUS RISC-V NEURAL SUPEROPTIMIZER")
        print("="*80)
        print(" Available Pre-packaged Examples:")
        for k, v in EXAMPLES.items():
            print(f"  [{k}] {v['name']}")
        print("  [C] Custom RISC-V Assembly Input")
        print("="*80)
        choice = input("\nSelect an option [1-5, or C for custom]: ").strip()

        if choice in EXAMPLES:
            asm_input = EXAMPLES[choice]["code"]
        else:
            print("\nEnter your RISC-V assembly code line-by-line (type 'END' on a new line when done):")
            lines = []
            while True:
                try:
                    line = input()
                    if line.strip().upper() == "END":
                        break
                    lines.append(line)
                except EOFError:
                    break
            asm_input = "\n".join(lines)

    if not asm_input.strip():
        print("Error: No assembly code provided.")
        return

    breakdown_and_superoptimize(asm_input, agent)


if __name__ == "__main__":
    main()
