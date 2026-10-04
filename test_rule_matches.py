#!/usr/bin/env python3
"""
test_rule_matches.py — Unit Tests for rule_matches(rule_idx, obs) across all 5 rules
"""

import unittest
import numpy as np
from superopt_env import SuperoptEnv, rule_matches, OPCODE_MAP, REG_MAP


class TestRuleMatches(unittest.TestCase):

    def setUp(self):
        self.env = SuperoptEnv(max_len=8, num_rules=5)

    def test_rule_0_mul_power2(self):
        """Rule 0: mul x, 2^k -> slli x, k"""
        prog = [
            [OPCODE_MAP["LI"], 0, 0, REG_MAP["t0"], 42],
            [OPCODE_MAP["LI"], 0, 0, REG_MAP["t1"], 16],  # Power of 2 (16 = 2^4)
            [OPCODE_MAP["MUL"], REG_MAP["t0"], REG_MAP["t1"], REG_MAP["t2"], 0]
        ] + [[0, 0, 0, 0, 0]] * 5
        obs = np.array(prog, dtype=np.int32).flatten()

        self.assertTrue(rule_matches(0, obs))
        self.assertFalse(rule_matches(1, obs))
        self.assertFalse(rule_matches(2, obs))
        self.assertFalse(rule_matches(3, obs))
        self.assertFalse(rule_matches(4, obs))

    def test_rule_1_sdiv_power2(self):
        """Rule 1: divu x, 2^k -> srli x, k"""
        prog = [
            [OPCODE_MAP["LI"], 0, 0, REG_MAP["t0"], 64],
            [OPCODE_MAP["LI"], 0, 0, REG_MAP["t1"], 4],   # Power of 2 (4 = 2^2)
            [OPCODE_MAP["DIVU"], REG_MAP["t0"], REG_MAP["t1"], REG_MAP["t2"], 0]
        ] + [[0, 0, 0, 0, 0]] * 5
        obs = np.array(prog, dtype=np.int32).flatten()

        self.assertFalse(rule_matches(0, obs))
        self.assertTrue(rule_matches(1, obs))
        self.assertFalse(rule_matches(2, obs))

    def test_rule_2_add_zero(self):
        """Rule 2: add x, 0 -> x"""
        prog = [
            [OPCODE_MAP["ADDI"], REG_MAP["t0"], 0, REG_MAP["t0"], 0]  # addi t0, t0, 0
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
