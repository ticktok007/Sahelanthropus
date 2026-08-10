import unittest
from reward_env import RewardEnv, ToolchainError

FAST_ASM = """\
.section .text
.globl _start

_start:
    li a0, 0
    li a7, 93
    ecall
"""

SLOW_ASM = """\
.section .text
.globl _start

_start:
    li t0, 0
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1
    addi t0, t0, 1

    li a0, 0
    li a7, 93
    ecall
"""


class TestRewardOrdering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.env = RewardEnv(strict=True)
        except ToolchainError as exc:
            raise unittest.SkipTest(f"Required local toolchain or plugin unavailable: {exc}") from exc

    def test_reward_ordering(self):
        fast_cycle_count = float(self.env.compile_and_run(FAST_ASM))
        slow_cycle_count = float(self.env.compile_and_run(SLOW_ASM))

        fast_score = -float(fast_cycle_count)
        slow_score = -float(slow_cycle_count)

        print(f"Fast cycle count: {fast_cycle_count}, Fast score: {fast_score}")
        print(f"Slow cycle count: {slow_cycle_count}, Slow score: {slow_score}")

        self.assertLess(fast_cycle_count, slow_cycle_count)
        self.assertGreater(fast_score, slow_score)


if __name__ == "__main__":
    unittest.main()
