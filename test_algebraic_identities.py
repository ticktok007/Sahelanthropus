#!/usr/bin/env python3
"""
test_algebraic_identities.py — Unit Test & QEMU Cycle Benchmark for 5 Algebraic Identities
"""

import unittest
import z3
from reward_env import RewardEnv, ToolchainError
from algebraic_identities import verify_identities_z3, RISCVAlgebraicRewriter


class TestAlgebraicIdentities(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.env = RewardEnv(strict=True)
        except ToolchainError as exc:
            raise unittest.SkipTest(f"Required toolchain unavailable: {exc}") from exc

    def test_z3_smt_verification_all_5_identities(self):
        """Formally verifies that all 5 algebraic identities hold universally in Z3 SMT."""
        z3_results = verify_identities_z3()
        for identity, result in z3_results.items():
            self.assertTrue(result, f"Z3 verification failed for identity: {identity}")
        print("\n[+] Z3 SMT Proof: All 5 algebraic identities verified UNSAT (Equivalent)!")

    def test_identity_1_mul_power_of_two(self):
        """Identity 1: mul x, 2^k -> slli x, k"""
        unopt_asm = """\
.section .text
.globl _start
_start:
    li t0, 42
    li t1, 8
    mul t2, t0, t1
    li a0, 0
    li a7, 93
    ecall
"""
        opt_asm, rewrites = RISCVAlgebraicRewriter.rewrite_assembly(unopt_asm)
        self.assertIn("Identity 1", rewrites[0])
        self.assertIn("slli t2, t0, 3", opt_asm)

        unopt_cycles = float(self.env.compile_and_run(unopt_asm))
        opt_cycles = float(self.env.compile_and_run(opt_asm))
        self.assertLess(opt_cycles, unopt_cycles)
        print(f"\nIdentity 1 (mul -> slli): Unopt {unopt_cycles:.1f} cycles -> Opt {opt_cycles:.1f} cycles (Saved {unopt_cycles - opt_cycles:.1f} cycles)")

    def test_identity_2_sdiv_power_of_two(self):
        """Identity 2: sdiv x, 2^k -> srai x, k"""
        unopt_asm = """\
.section .text
.globl _start
_start:
    li t0, 64
    li t1, 4
    div t2, t0, t1
    li a0, 0
    li a7, 93
    ecall
"""
        opt_asm, rewrites = RISCVAlgebraicRewriter.rewrite_assembly(unopt_asm)
        self.assertIn("Identity 2", rewrites[0])
        self.assertIn("srai t2, t0, 2", opt_asm)

        unopt_cycles = float(self.env.compile_and_run(unopt_asm))
        opt_cycles = float(self.env.compile_and_run(opt_asm))
        self.assertLess(opt_cycles, unopt_cycles)
        print(f"Identity 2 (sdiv -> srai): Unopt {unopt_cycles:.1f} cycles -> Opt {opt_cycles:.1f} cycles (Saved {unopt_cycles - opt_cycles:.1f} cycles)")

    def test_identity_3_add_zero(self):
        """Identity 3: add x, 0 -> mv x, y (or NOP)"""
        unopt_asm = """\
.section .text
.globl _start
_start:
    li t0, 42
    addi t0, t0, 0
    li a0, 0
    li a7, 93
    ecall
"""
        opt_asm, rewrites = RISCVAlgebraicRewriter.rewrite_assembly(unopt_asm)
        self.assertIn("Identity 3", rewrites[0])

        unopt_cycles = float(self.env.compile_and_run(unopt_asm))
        opt_cycles = float(self.env.compile_and_run(opt_asm))
        self.assertLessEqual(opt_cycles, unopt_cycles)
        print(f"Identity 3 (add x, 0 -> x): Unopt {unopt_cycles:.1f} cycles -> Opt {opt_cycles:.1f} cycles (Saved {unopt_cycles - opt_cycles:.1f} cycles)")

    def test_identity_4_mul_zero(self):
        """Identity 4: mul x, 0 -> 0"""
        unopt_asm = """\
.section .text
.globl _start
_start:
    li t0, 42
    mul t1, t0, zero
    li a0, 0
    li a7, 93
    ecall
"""
        opt_asm, rewrites = RISCVAlgebraicRewriter.rewrite_assembly(unopt_asm)
        self.assertIn("Identity 4", rewrites[0])
        self.assertIn("li t1, 0", opt_asm)

        unopt_cycles = float(self.env.compile_and_run(unopt_asm))
        opt_cycles = float(self.env.compile_and_run(opt_asm))
        self.assertLess(opt_cycles, unopt_cycles)
        print(f"Identity 4 (mul x, 0 -> 0): Unopt {unopt_cycles:.1f} cycles -> Opt {opt_cycles:.1f} cycles (Saved {unopt_cycles - opt_cycles:.1f} cycles)")

    def test_identity_5_xor_self(self):
        """Identity 5: xor x, x -> 0"""
        unopt_asm = """\
.section .text
.globl _start
_start:
    li t0, 42
    xor t1, t0, t0
    li a0, 0
    li a7, 93
    ecall
"""
        opt_asm, rewrites = RISCVAlgebraicRewriter.rewrite_assembly(unopt_asm)
        self.assertIn("Identity 5", rewrites[0])
        self.assertIn("li t1, 0", opt_asm)

        unopt_cycles = float(self.env.compile_and_run(unopt_asm))
        opt_cycles = float(self.env.compile_and_run(opt_asm))
        self.assertLessEqual(opt_cycles, unopt_cycles)
        print(f"Identity 5 (xor x, x -> 0): Unopt {unopt_cycles:.1f} cycles -> Opt {opt_cycles:.1f} cycles (Saved {unopt_cycles - opt_cycles:.1f} cycles)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
