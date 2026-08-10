"""
test_reward_env.py — unittest suite for RewardEnv.

Run with:
    python3 -m unittest -v test_reward_env.py
"""

import unittest
from pathlib import Path

from reward_env import (
    RewardEnv,
    CompilationError,
    ExecutionError,
    MeasurementError,
    RewardEnvError,
    ToolchainError,
)

# ---------------------------------------------------------------------------
# Minimal RISC-V assembly programs
# ---------------------------------------------------------------------------

ASM_EXIT_ZERO = """\
.section .text
.globl _start
_start:
    li a0, 0
    li a7, 93
    ecall
"""

ASM_EXIT_ONE = """\
.section .text
.globl _start
_start:
    li a0, 1
    li a7, 93
    ecall
"""

ASM_SIMPLE_LOOP = """\
.section .text
.globl _start
_start:
    li   t0, 100
.loop:
    addi t0, t0, -1
    bnez t0, .loop
    li   a0, 0
    li   a7, 93
    ecall
"""

ASM_NO_START = """\
.section .text
    li a0, 0
"""

ASM_INVALID_SYNTAX = """\
.section .text
.globl _start
_start:
    THIS_IS_NOT_VALID_RISCV_ASSEMBLY !!@@##
"""

ASM_EMPTY = ""


# ---------------------------------------------------------------------------
# Base test case that skips if the toolchain is unavailable
# ---------------------------------------------------------------------------

class _BaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.env = RewardEnv(strict=True)
        except ToolchainError as exc:
            raise unittest.SkipTest(f"Toolchain not available: {exc}") from exc


# ---------------------------------------------------------------------------
# Test: basic compile_and_run
# ---------------------------------------------------------------------------

class TestCompileAndRun(_BaseTest):

    def test_returns_integer(self):
        result = self.env.compile_and_run(ASM_EXIT_ZERO)
        self.assertIsInstance(result, int,
            f"compile_and_run must return int, got {type(result).__name__}")

    def test_result_positive(self):
        result = self.env.compile_and_run(ASM_EXIT_ZERO)
        self.assertGreater(result, 0,
            "weighted_cost should be > 0 for a non-trivial program")

    def test_exit_zero_vs_loop(self):
        cost_simple = self.env.compile_and_run(ASM_EXIT_ZERO)
        cost_loop   = self.env.compile_and_run(ASM_SIMPLE_LOOP)
        self.assertGreater(cost_loop, cost_simple,
            "Loop program should have higher weighted_cost than exit-only program")

    def test_exit_one_is_execution_error(self):
        """Guest program exiting with code 1 → QEMU exits 1 → ExecutionError."""
        with self.assertRaises(ExecutionError):
            self.env.compile_and_run(ASM_EXIT_ONE)

    def test_deterministic(self):
        """Same assembly should produce the same cost on two runs."""
        c1 = self.env.compile_and_run(ASM_EXIT_ZERO)
        c2 = self.env.compile_and_run(ASM_EXIT_ZERO)
        self.assertEqual(c1, c2,
            "compile_and_run must be deterministic for identical input")


# ---------------------------------------------------------------------------
# Test: invalid input raises CompilationError in strict mode
# ---------------------------------------------------------------------------

class TestInvalidInput(_BaseTest):

    def test_invalid_syntax_raises(self):
        with self.assertRaises(CompilationError):
            self.env.compile_and_run(ASM_INVALID_SYNTAX)

    def test_empty_string_raises(self):
        with self.assertRaises(CompilationError):
            self.env.compile_and_run(ASM_EMPTY)

    def test_no_start_raises(self):
        """Assembly without _start should raise CompilationError (require_start=True)."""
        env_strict_start = RewardEnv(strict=True, require_start=True)
        with self.assertRaises(CompilationError):
            env_strict_start.compile_and_run(ASM_NO_START)

    def test_none_raises(self):
        with self.assertRaises((CompilationError, TypeError)):
            self.env.compile_and_run(None)  # type: ignore[arg-type]

    def test_whitespace_only_raises(self):
        with self.assertRaises(CompilationError):
            self.env.compile_and_run("   \n\t  ")


# ---------------------------------------------------------------------------
# Test: non-strict mode returns penalty instead of raising
# ---------------------------------------------------------------------------

class TestNonStrictMode(_BaseTest):

    def setUp(self):
        self.env_soft = RewardEnv(strict=False)

    def test_invalid_assembly_returns_penalty(self):
        result = self.env_soft.compile_and_run(ASM_INVALID_SYNTAX)
        self.assertEqual(result, -1_000_000,
            "Non-strict mode must return penalty_value on failure")

    def test_empty_string_returns_penalty(self):
        result = self.env_soft.compile_and_run(ASM_EMPTY)
        self.assertEqual(result, -1_000_000)

    def test_custom_penalty(self):
        env = RewardEnv(strict=False, penalty_value=-42)
        result = env.compile_and_run(ASM_INVALID_SYNTAX)
        self.assertEqual(result, -42)

    def test_valid_assembly_still_returns_int(self):
        result = self.env_soft.compile_and_run(ASM_EXIT_ZERO)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)


# ---------------------------------------------------------------------------
# Test: bad toolchain raises ToolchainError
# ---------------------------------------------------------------------------

class TestToolchainDetection(unittest.TestCase):

    def test_bad_compiler_raises(self):
        with self.assertRaises(ToolchainError):
            RewardEnv(compiler="/nonexistent/path/riscv-gcc")

    def test_bad_qemu_raises(self):
        with self.assertRaises(ToolchainError):
            RewardEnv(qemu_binary="/nonexistent/path/qemu-riscv64")

    def test_bad_plugin_raises(self):
        with self.assertRaises(ToolchainError):
            RewardEnv(plugin_path="/nonexistent/cycle_counter.so")


# ---------------------------------------------------------------------------
# Test: compile_asm and run_binary low-level methods
# ---------------------------------------------------------------------------

class TestLowLevelMethods(_BaseTest):

    def test_compile_asm_produces_file(self):
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        try:
            binary = tmp / "test_bin"
            self.env.compile_asm(ASM_EXIT_ZERO, binary)
            self.assertTrue(binary.exists(),
                "compile_asm must produce the output binary")
            self.assertGreater(binary.stat().st_size, 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_binary_returns_dict(self):
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        try:
            binary = tmp / "test_bin"
            json_out = tmp / "result.json"
            self.env.compile_asm(ASM_EXIT_ZERO, binary)
            data = self.env.run_binary(binary, json_out)
            self.assertIsInstance(data, dict)
            self.assertIn("weighted_cost", data)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_nonexistent_binary_raises(self):
        import tempfile, shutil
        tmp = Path(tempfile.mkdtemp())
        try:
            json_out = tmp / "result.json"
            with self.assertRaises(ExecutionError):
                self.env.run_binary(tmp / "no_such_binary", json_out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test: calculate_reward
# ---------------------------------------------------------------------------

class TestCalculateReward(_BaseTest):

    def test_lower_cost_positive_reward(self):
        reward = self.env.calculate_reward(
            cycle_count=100, baseline_cycle_count=200
        )
        self.assertEqual(reward, 100.0)

    def test_higher_cost_negative_reward(self):
        reward = self.env.calculate_reward(
            cycle_count=300, baseline_cycle_count=200
        )
        self.assertEqual(reward, -100.0)

    def test_equal_cost_zero_reward(self):
        reward = self.env.calculate_reward(
            cycle_count=150, baseline_cycle_count=150
        )
        self.assertEqual(reward, 0.0)

    def test_returns_float(self):
        reward = self.env.calculate_reward(100, 200)
        self.assertIsInstance(reward, float)


# ---------------------------------------------------------------------------
# Test: RL step / reset interface
# ---------------------------------------------------------------------------

class TestRLInterface(_BaseTest):

    def test_reset_returns_dict(self):
        obs = self.env.reset()
        self.assertIsInstance(obs, dict)

    def test_step_returns_tuple(self):
        self.env.reset()
        result = self.env.step(ASM_EXIT_ZERO)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_step_obs_contains_cycle_count(self):
        self.env.reset()
        obs, reward, done, info = self.env.step(ASM_EXIT_ZERO)
        self.assertIn("cycle_count", obs)
        self.assertIsInstance(obs["cycle_count"], int)

    def test_step_baseline_set_on_first_call(self):
        self.env.reset()
        obs1, _, _, _ = self.env.step(ASM_EXIT_ZERO)
        # First call: reward should be 0 (same as baseline)
        _, reward, _, _ = self.env.step(ASM_EXIT_ZERO)
        self.assertEqual(reward, 0.0,
            "Second call with identical assembly should give 0 reward")

    def test_step_done_is_bool(self):
        self.env.reset()
        _, _, done, _ = self.env.step(ASM_EXIT_ZERO)
        self.assertIsInstance(done, bool)


# ---------------------------------------------------------------------------
# Test: timeout enforcement
# ---------------------------------------------------------------------------

class TestTimeout(_BaseTest):

    def test_short_timeout_raises_on_infinite_loop(self):
        """An infinite loop should eventually hit the timeout."""
        asm_infinite = """\
.section .text
.globl _start
_start:
    j _start
"""
        env_tight = RewardEnv(strict=True, timeout_seconds=2.0)
        # QEMU may time out OR exit with non-zero (SIGALRM etc.)
        with self.assertRaises((ExecutionError, CompilationError)):
            env_tight.compile_and_run(asm_infinite)


# ---------------------------------------------------------------------------
# Test: require_start option
# ---------------------------------------------------------------------------

class TestRequireStart(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.env_no_check = RewardEnv(strict=True, require_start=False)
        except ToolchainError as exc:
            raise unittest.SkipTest(f"Toolchain not available: {exc}") from exc

    def test_no_start_check_disabled(self):
        """When require_start=False, missing _start should not raise at validation."""
        # It may still fail at compile time, but not at the Python validation step.
        try:
            self.env_no_check.compile_and_run(ASM_NO_START)
        except CompilationError as exc:
            # Acceptable: failed at compiler, not at Python validation
            self.assertNotIn("does not contain '_start'", str(exc))
        except (ExecutionError, MeasurementError):
            pass  # Also acceptable


if __name__ == "__main__":
    unittest.main(verbosity=2)
