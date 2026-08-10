#!/usr/bin/env python3
"""
test_peephole_rewrite.py — Apply manual peephole rewrite (MUL -> SLLI) and verify cycle reduction > 0.
"""

import unittest
from reward_env import RewardEnv, ToolchainError

# Unoptimized assembly sequence: uses register loading + MUL instruction
UNOPTIMIZED_ASM = """\
.section .text
.globl _start

_start:
    li t0, 42
    li t1, 4
    mul t2, t0, t1

    li a0, 0
    li a7, 93
    ecall
"""

# Peephole optimized assembly sequence: replaces (li t1, 4; mul t2, t0, t1) with (slli t2, t0, 2)
OPTIMIZED_ASM = """\
.section .text
.globl _start

_start:
    li t0, 42
    slli t2, t0, 2

    li a0, 0
    li a7, 93
    ecall
"""


class TestPeepholeRewrite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.env = RewardEnv(strict=True)
        except ToolchainError as exc:
            raise unittest.SkipTest(f"Required toolchain unavailable: {exc}") from exc

    def test_mul_to_slli_peephole(self):
        print("\n" + "=" * 70)
        print(" Manual Peephole Rewrite Test: (MUL -> SLLI)")
        print("=" * 70)

        # 1. Run unoptimized program
        unopt_cycles = float(self.env.compile_and_run(UNOPTIMIZED_ASM))
        unopt_score = -unopt_cycles

        # 2. Run peephole-optimized program
        opt_cycles = float(self.env.compile_and_run(OPTIMIZED_ASM))
        opt_score = -opt_cycles

        # 3. Calculate cycle reduction and reward
        cycle_reduction = unopt_cycles - opt_cycles
        reward = self.env.calculate_reward(opt_cycles, unopt_cycles)

        print(f"Unoptimized Program Cycles (MUL)  : {unopt_cycles:.1f} cycles  (Score: {unopt_score:.1f})")
        print(f"Optimized Program Cycles (SLLI)   : {opt_cycles:.1f} cycles    (Score: {opt_score:.1f})")
        print(f"Cycle Reduction (unopt - opt)     : {cycle_reduction:.1f} cycles")
        print(f"Environment Reward                : +{reward:.1f}")
        print("=" * 70)

        # Assertions
        self.assertGreater(cycle_reduction, 0, "Peephole rewrite must reduce execution cycles!")
        self.assertLess(opt_cycles, unopt_cycles, "Optimized cycle count must be strictly lower")
        self.assertGreater(opt_score, unopt_score, "Optimized score (-cycles) must be higher")
        self.assertEqual(reward, cycle_reduction, "Reward must equal cycle reduction")

        print("[+] Peephole Rewrite Verification PASSED: Cycle reduction > 0 confirmed!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
