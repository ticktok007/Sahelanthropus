#!/usr/bin/env python3
"""
test_all_10_rules.py — Unit Test Suite for All 10 RISC-V Algebraic Rewrite Rules

Rules Tested:
- Rule 0: mul x, 2^k -> slli x, k
- Rule 1: sdiv x, 2^k -> srai x, k
- Rule 2: add x, 0 -> x (NOP)
- Rule 3: mul x, 0 -> 0
- Rule 4: xor x, x -> 0
- Rule 5: sub x, x -> 0
- Rule 6: and x, x -> x
- Rule 7: or x, x -> x
- Rule 8: add(sub(X, Y), Y) -> X
- Rule 9: slli then srli same amount k -> and with mask
"""

import unittest
import numpy as np
from superopt_env import SuperoptEnv, rule_matches, OPCODE_MAP, REG_MAP


class TestAll10Rules(unittest.TestCase):

    def setUp(self):
        self.env = SuperoptEnv(max_len=8, num_rules=10)

    def test_rule_0_mul_power2(self):
        """Rule 0: mul x, 2^k -> slli x, k"""
        prog = [
            [OPCODE_MAP["LI"], 0, 0, REG_MAP["t0"], 42],
            [OPCODE_MAP["LI"], 0, 0, REG_MAP["t1"], 16],
            [OPCODE_MAP["MUL"], REG_MAP["t0"], REG_MAP["t1"], REG_MAP["t2"], 0]
        ] + [[0, 0, 0, 0, 0]] * 5
        obs = np.array(prog, dtype=np.int32).flatten()
        self.assertTrue(rule_matches(0, obs))

    def test_rule_1_sdiv_power2(self):
        """Rule 1: divu x, 2^k -> srli x, k"""
        prog = [
            [OPCODE_MAP["LI"], 0, 0, REG_MAP["t0"], 64],
            [OPCODE_MAP["LI"], 0, 0, REG_MAP["t1"], 4],
            [OPCODE_MAP["DIVU"], REG_MAP["t0"], REG_MAP["t1"], REG_MAP["t2"], 0]
        ] + [[0, 0, 0, 0, 0]] * 5
        obs = np.array(prog, dtype=np.int32).flatten()
        self.assertTrue(rule_matches(1, obs))

    def test_rule_2_add_zero(self):
        """Rule 2: add x, 0 -> x"""
        prog = [
            [OPCODE_MAP["ADDI"], REG_MAP["t0"], 0, REG_MAP["t0"], 0]
        ] + [[0, 0, 0, 0, 0]] * 7
        obs = np.array(prog, dtype=np.int32).flatten()
        self.assertTrue(rule_matches(2, obs))

    def test_rule_3_mul_zero(self):
        """Rule 3: mul x, 0 -> 0"""
        prog = [
            [OPCODE_MAP["MUL"], REG_MAP["t0"], REG_MAP["zero"], REG_MAP["t1"], 0]
        ] + [[0, 0, 0, 0, 0]] * 7
        obs = np.array(prog, dtype=np.int32).flatten()
        self.assertTrue(rule_matches(3, obs))

    def test_rule_4_xor_self(self):
        """Rule 4: xor x, x -> 0"""
        prog = [
            [OPCODE_MAP["XOR"], REG_MAP["t0"], REG_MAP["t0"], REG_MAP["t1"], 0]
        ] + [[0, 0, 0, 0, 0]] * 7
        obs = np.array(prog, dtype=np.int32).flatten()
        self.assertTrue(rule_matches(4, obs))

    def test_rule_5_sub_self(self):
        """Rule 5: sub x, x -> 0"""
        prog = [
            [OPCODE_MAP["SUB"], REG_MAP["t0"], REG_MAP["t0"], REG_MAP["t1"], 0]
        ] + [[0, 0, 0, 0, 0]] * 7
        obs = np.array(prog, dtype=np.int32).flatten()
        self.assertTrue(rule_matches(5, obs))

    def test_rule_6_and_self(self):
        """Rule 6: and x, x -> x"""
        prog = [
            [OPCODE_MAP["AND"], REG_MAP["t0"], REG_MAP["t0"], REG_MAP["t1"], 0]
        ] + [[0, 0, 0, 0, 0]] * 7
        obs = np.array(prog, dtype=np.int32).flatten()
        self.assertTrue(rule_matches(6, obs))

    def test_rule_7_or_self(self):
        """Rule 7: or x, x -> x"""
        prog = [
            [OPCODE_MAP["OR"], REG_MAP["t0"], REG_MAP["t0"], REG_MAP["t1"], 0]
        ] + [[0, 0, 0, 0, 0]] * 7
        obs = np.array(prog, dtype=np.int32).flatten()
        self.assertTrue(rule_matches(7, obs))

    def test_rule_8_add_sub_cancel(self):
        """Rule 8: add(sub(X, Y), Y) -> X"""
        # sub t2, t0, t1  (t2 = t0 - t1)
        # add t3, t2, t1  (t3 = t2 + t1 -> t0)
        prog = [
            [OPCODE_MAP["SUB"], REG_MAP["t0"], REG_MAP["t1"], REG_MAP["t2"], 0],
            [OPCODE_MAP["ADD"], REG_MAP["t2"], REG_MAP["t1"], REG_MAP["t3"], 0]
        ] + [[0, 0, 0, 0, 0]] * 6
        obs = np.array(prog, dtype=np.int32).flatten()
        self.assertTrue(rule_matches(8, obs))

    def test_rule_9_sll_srl_mask(self):
        """Rule 9: slli then srli same amount k -> and with mask"""
        # slli t2, t0, 4
        # srli t3, t2, 4
        prog = [
            [OPCODE_MAP["SLLI"], REG_MAP["t0"], 0, REG_MAP["t2"], 4],
            [OPCODE_MAP["SRLI"], REG_MAP["t2"], 0, REG_MAP["t3"], 4]
        ] + [[0, 0, 0, 0, 0]] * 6
        obs = np.array(prog, dtype=np.int32).flatten()
        self.assertTrue(rule_matches(9, obs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
