#!/usr/bin/env python3
"""
peephole_rulebook.py — RISC-V Algebraic Peephole Rulebook for SuperoptEnv (N_rules = 10)

Encapsulates 10 algebraic peephole rewrite rules:
- Rule 0: mul x, 2^k -> slli x, k
- Rule 1: sdiv x, 2^k -> srai x, k
- Rule 2: add x, 0 -> x (NOP)
- Rule 3: mul x, 0 -> 0
- Rule 4: xor x, x -> 0
- Rule 5: sub x, x -> 0
- Rule 6: and x, x -> x
- Rule 7: or x, x -> x
- Rule 8: add(sub(X, Y), Y) -> X
- Rule 9: slli then srli same k -> andi with mask
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Opcode Mapping
OPCODE_MAP = {
    "NOP": 0, "LI": 1, "ADD": 2, "ADDI": 3, "SUB": 4, "MUL": 5, "DIV": 6,
    "SLLI": 7, "SRLI": 8, "SRAI": 9, "AND": 10, "OR": 11, "XOR": 12,
    "MV": 13, "LD": 14, "SD": 15, "ANDI": 16, "ECALL": 17, "OTHER": 18
}

REG_MAP = {
    "zero": 0, "x0": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
    "t0": 5, "t1": 6, "t2": 7, "s0": 8, "fp": 8, "s1": 9,
    "a0": 10, "a1": 11, "a2": 12, "a3": 13, "a4": 14, "a5": 15,
    "a6": 16, "a7": 17, "s2": 18, "s3": 19, "s4": 20, "s5": 21,
    "s6": 22, "s7": 23, "s8": 24, "s9": 25, "s10": 26, "s11": 27,
    "t3": 28, "t4": 29, "t5": 30, "t6": 31
}

RULE_METADATA = [
    {"id": 0, "name": "mul_power2_to_slli", "desc": "mul x, 2^k -> slli x, k", "est_savings": 3.0},
    {"id": 1, "name": "sdiv_power2_to_srai", "desc": "sdiv x, 2^k -> srai x, k", "est_savings": 20.0},
    {"id": 2, "name": "add_zero_to_nop", "desc": "add x, 0 -> x", "est_savings": 1.0},
    {"id": 3, "name": "mul_zero_to_li_0", "desc": "mul x, 0 -> 0", "est_savings": 2.0},
    {"id": 4, "name": "xor_self_to_li_0", "desc": "xor x, x -> 0", "est_savings": 1.0},
    {"id": 5, "name": "sub_self_to_li_0", "desc": "sub x, x -> 0", "est_savings": 1.0},
    {"id": 6, "name": "and_self_to_mv", "desc": "and x, x -> x", "est_savings": 1.0},
    {"id": 7, "name": "or_self_to_mv", "desc": "or x, x -> x", "est_savings": 1.0},
    {"id": 8, "name": "add_sub_cancel", "desc": "add(sub(X, Y), Y) -> X", "est_savings": 2.0},
    {"id": 9, "name": "sll_srl_to_andi_mask", "desc": "slli then srli k -> andi mask", "est_savings": 1.0}
]


class PeepholeRulebook:
    """
    RISC-V Algebraic Peephole Rulebook managing pattern matching and rewrite transformations.
    """

    def __init__(self, num_rules: int = 10):
        self.num_rules = num_rules
        self.metadata = RULE_METADATA[:num_rules]

    def get_rule_info(self, rule_id: int) -> Dict[str, Any]:
        if 0 <= rule_id < len(self.metadata):
            return self.metadata[rule_id]
        return {"id": rule_id, "name": "unknown", "desc": "unknown rule", "est_savings": 0.0}

    def matches_at(self, rule_id: int, program: List[List[int]], target_idx: int) -> bool:
        """
        Checks if rule_id matches program instruction at target_idx.
        """
        if target_idx < 0 or target_idx >= len(program):
            return False

        inst = program[target_idx]
        opcode, rs1, rs2, rd, imm = inst

        # Rule 0: mul x, 2^k -> slli x, k
        if rule_id == 0 and opcode == OPCODE_MAP["MUL"] and target_idx > 0:
            prev_inst = program[target_idx - 1]
            if prev_inst[0] == OPCODE_MAP["LI"] and prev_inst[3] == rs2:
                imm_val = prev_inst[4]
                if imm_val > 0 and (imm_val & (imm_val - 1)) == 0:
                    return True

        # Rule 1: sdiv x, 2^k -> srai x, k
        elif rule_id == 1 and opcode == OPCODE_MAP["DIV"] and target_idx > 0:
            prev_inst = program[target_idx - 1]
            if prev_inst[0] == OPCODE_MAP["LI"] and prev_inst[3] == rs2:
                imm_val = prev_inst[4]
                if imm_val > 0 and (imm_val & (imm_val - 1)) == 0:
                    return True

        # Rule 2: add x, 0 -> x
        elif rule_id == 2 and opcode == OPCODE_MAP["ADDI"] and imm == 0:
            return True

        # Rule 3: mul x, 0 -> 0
        elif rule_id == 3 and opcode == OPCODE_MAP["MUL"] and rs2 == REG_MAP["zero"]:
            return True

        # Rule 4: xor x, x -> 0
        elif rule_id == 4 and opcode == OPCODE_MAP["XOR"] and rs1 == rs2:
            return True

        # Rule 5: sub x, x -> 0
        elif rule_id == 5 and opcode == OPCODE_MAP["SUB"] and rs1 == rs2:
            return True

        # Rule 6: and x, x -> x
        elif rule_id == 6 and opcode == OPCODE_MAP["AND"] and rs1 == rs2:
            return True

        # Rule 7: or x, x -> x
        elif rule_id == 7 and opcode == OPCODE_MAP["OR"] and rs1 == rs2:
            return True

        # Rule 8: add(sub(X, Y), Y) -> X
        elif rule_id == 8 and opcode == OPCODE_MAP["ADD"] and target_idx > 0:
            prev_inst = program[target_idx - 1]
            if prev_inst[0] == OPCODE_MAP["SUB"]:
                t_reg = prev_inst[3]
                y_reg = prev_inst[2]
                if (rs1 == t_reg and rs2 == y_reg) or (rs2 == t_reg and rs1 == y_reg):
                    return True

        # Rule 9: slli then srli same k -> andi mask
        elif rule_id == 9 and opcode == OPCODE_MAP["SRLI"] and target_idx > 0:
            prev_inst = program[target_idx - 1]
            if prev_inst[0] == OPCODE_MAP["SLLI"]:
                t_reg = prev_inst[3]
                k1 = prev_inst[4]
                k2 = imm
                if rs1 == t_reg and k1 == k2 and k1 > 0:
                    return True

        return False

    def rule_matches(self, rule_id: int, obs: np.ndarray) -> bool:
        """
        Evaluates whether rule_id matches any instruction slot in observation state.
        """
        if obs.ndim == 1:
            max_len = len(obs) // 5
            prog = obs.reshape((max_len, 5)).tolist()
        else:
            prog = obs.tolist()

        for i in range(len(prog)):
            if self.matches_at(rule_id, prog, i):
                return True
        return False

    def apply_rewrite(self, rule_id: int, program: List[List[int]], target_idx: int) -> Tuple[List[List[int]], bool]:
        """
        Applies algebraic rewrite rule_id to program at target_idx.
        Returns (updated_program, applied_bool).
        """
        if not self.matches_at(rule_id, program, target_idx):
            return program, False

        new_program = [list(inst) for inst in program]
        inst = new_program[target_idx]
        opcode, rs1, rs2, rd, imm = inst

        # Rule 0: mul x, 2^k -> slli x, k
        if rule_id == 0:
            prev_inst = new_program[target_idx - 1]
            imm_val = prev_inst[4]
            k = int(math.log2(imm_val))
            new_program[target_idx] = [OPCODE_MAP["SLLI"], rs1, 0, rd, k]

        # Rule 1: sdiv x, 2^k -> srai x, k
        elif rule_id == 1:
            prev_inst = new_program[target_idx - 1]
            imm_val = prev_inst[4]
            k = int(math.log2(imm_val))
            new_program[target_idx] = [OPCODE_MAP["SRAI"], rs1, 0, rd, k]

        # Rule 2: add x, 0 -> x (NOP)
        elif rule_id == 2:
            new_program[target_idx] = [OPCODE_MAP["NOP"], 0, 0, 0, 0]

        # Rule 3: mul x, 0 -> 0
        elif rule_id == 3:
            new_program[target_idx] = [OPCODE_MAP["LI"], 0, 0, rd, 0]

        # Rule 4: xor x, x -> 0
        elif rule_id == 4:
            new_program[target_idx] = [OPCODE_MAP["LI"], 0, 0, rd, 0]

        # Rule 5: sub x, x -> 0
        elif rule_id == 5:
            new_program[target_idx] = [OPCODE_MAP["LI"], 0, 0, rd, 0]

        # Rule 6: and x, x -> x
        elif rule_id == 6:
            if rd == rs1:
                new_program[target_idx] = [OPCODE_MAP["NOP"], 0, 0, 0, 0]
            else:
                new_program[target_idx] = [OPCODE_MAP["MV"], rs1, 0, rd, 0]

        # Rule 7: or x, x -> x
        elif rule_id == 7:
            if rd == rs1:
                new_program[target_idx] = [OPCODE_MAP["NOP"], 0, 0, 0, 0]
            else:
                new_program[target_idx] = [OPCODE_MAP["MV"], rs1, 0, rd, 0]

        # Rule 8: add(sub(X, Y), Y) -> X
        elif rule_id == 8:
            prev_inst = new_program[target_idx - 1]
            x_reg = prev_inst[1]
            new_program[target_idx - 1] = [OPCODE_MAP["NOP"], 0, 0, 0, 0]
            if rd == x_reg:
                new_program[target_idx] = [OPCODE_MAP["NOP"], 0, 0, 0, 0]
            else:
                new_program[target_idx] = [OPCODE_MAP["MV"], x_reg, 0, rd, 0]

        # Rule 9: slli then srli same k -> andi mask
        elif rule_id == 9:
            prev_inst = new_program[target_idx - 1]
            x_reg = prev_inst[1]
            k1 = prev_inst[4]
            mask = ((1 << (32 - k1)) - 1)
            new_program[target_idx - 1] = [OPCODE_MAP["NOP"], 0, 0, 0, 0]
            new_program[target_idx] = [OPCODE_MAP["ANDI"], x_reg, 0, rd, mask]

        return new_program, True

    def get_action_mask(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes boolean action mask vector of shape (num_rules,) = (10,).
        """
        return np.array([self.rule_matches(i, obs) for i in range(self.num_rules)], dtype=bool)
