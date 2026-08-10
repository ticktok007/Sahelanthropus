"""
run_reward_env_test.py — Standalone smoke-test for RewardEnv.

Run with:
    python3 run_reward_env_test.py
"""

from __future__ import annotations

import sys
import traceback

from reward_env import (
    RewardEnv,
    CompilationError,
    ExecutionError,
    MeasurementError,
    ToolchainError,
)

# ANSI colours (disabled if not a tty)
_tty = sys.stdout.isatty()
GREEN  = "\033[32m" if _tty else ""
RED    = "\033[31m" if _tty else ""
YELLOW = "\033[33m" if _tty else ""
RESET  = "\033[0m"  if _tty else ""

_passed = 0
_failed = 0
_skipped = 0


def _ok(name: str, detail: str = "") -> None:
    global _passed
    _passed += 1
    msg = f"  {GREEN}PASS{RESET}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


def _fail(name: str, detail: str = "") -> None:
    global _failed
    _failed += 1
    msg = f"  {RED}FAIL{RESET}  {name}"
    if detail:
        msg += f"\n        {detail}"
    print(msg)


def _skip(name: str, reason: str = "") -> None:
    global _skipped
    _skipped += 1
    msg = f"  {YELLOW}SKIP{RESET}  {name}"
    if reason:
        msg += f"  ({reason})"
    print(msg)


# ---------------------------------------------------------------------------
# Assembly fixtures
# ---------------------------------------------------------------------------

ASM_EXIT_ZERO = """\
.section .text
.globl _start
_start:
    li a0, 0
    li a7, 93
    ecall
"""

ASM_SIMPLE_LOOP = """\
.section .text
.globl _start
_start:
    li   t0, 50
.loop:
    addi t0, t0, -1
    bnez t0, .loop
    li   a0, 0
    li   a7, 93
    ecall
"""

ASM_INVALID = """\
.section .text
.globl _start
_start:
    OBVIOUSLY_NOT_RISCV @!#$
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_toolchain_detection(env: RewardEnv) -> None:
    name = "toolchain_detection"
    try:
        assert env.compiler,    "compiler must be set"
        assert env.qemu_binary, "qemu_binary must be set"
        assert env.plugin_path, "plugin_path must be set"
        _ok(name, f"cc={env.compiler}")
    except AssertionError as exc:
        _fail(name, str(exc))


def test_basic_compile_and_run(env: RewardEnv) -> None:
    name = "basic_compile_and_run"
    try:
        cycle_count = env.compile_and_run(ASM_EXIT_ZERO)
        assert isinstance(cycle_count, int), \
            f"Expected int, got {type(cycle_count).__name__}"
        assert cycle_count > 0, \
            f"Expected positive cycle_count, got {cycle_count}"
        _ok(name, f"cycle_count={cycle_count}")
    except Exception as exc:
        _fail(name, traceback.format_exc(limit=3))


def test_loop_costs_more(env: RewardEnv) -> None:
    name = "loop_costs_more_than_exit"
    try:
        c_simple = env.compile_and_run(ASM_EXIT_ZERO)
        c_loop   = env.compile_and_run(ASM_SIMPLE_LOOP)
        assert c_loop > c_simple, \
            f"Loop ({c_loop}) should cost more than simple exit ({c_simple})"
        _ok(name, f"exit={c_simple}  loop={c_loop}")
    except Exception as exc:
        _fail(name, traceback.format_exc(limit=3))


def test_determinism(env: RewardEnv) -> None:
    name = "determinism"
    try:
        c1 = env.compile_and_run(ASM_EXIT_ZERO)
        c2 = env.compile_and_run(ASM_EXIT_ZERO)
        assert c1 == c2, f"Results differ: {c1} vs {c2}"
        _ok(name, f"both={c1}")
    except Exception as exc:
        _fail(name, traceback.format_exc(limit=3))


def test_invalid_asm_strict(env: RewardEnv) -> None:
    name = "invalid_asm_strict_raises"
    try:
        env.compile_and_run(ASM_INVALID)
        _fail(name, "Expected CompilationError but got no exception")
    except CompilationError:
        _ok(name)
    except Exception as exc:
        _fail(name, f"Wrong exception type: {exc!r}")


def test_empty_asm_strict(env: RewardEnv) -> None:
    name = "empty_asm_strict_raises"
    try:
        env.compile_and_run("")
        _fail(name, "Expected CompilationError but got no exception")
    except CompilationError:
        _ok(name)
    except Exception as exc:
        _fail(name, f"Wrong exception type: {exc!r}")


def test_non_strict_returns_penalty() -> None:
    name = "non_strict_returns_penalty"
    try:
        env = RewardEnv(strict=False)
        result = env.compile_and_run(ASM_INVALID)
        assert result == -1_000_000, \
            f"Expected -1_000_000, got {result}"
        _ok(name)
    except ToolchainError as exc:
        _skip(name, str(exc))
    except Exception as exc:
        _fail(name, traceback.format_exc(limit=3))


def test_custom_penalty() -> None:
    name = "custom_penalty_value"
    try:
        env = RewardEnv(strict=False, penalty_value=-999)
        result = env.compile_and_run(ASM_INVALID)
        assert result == -999, f"Expected -999, got {result}"
        _ok(name)
    except ToolchainError as exc:
        _skip(name, str(exc))
    except Exception as exc:
        _fail(name, traceback.format_exc(limit=3))


def test_calculate_reward(env: RewardEnv) -> None:
    name = "calculate_reward"
    try:
        r = env.calculate_reward(cycle_count=100, baseline_cycle_count=200)
        assert r == 100.0, f"Expected 100.0, got {r}"
        r2 = env.calculate_reward(cycle_count=300, baseline_cycle_count=200)
        assert r2 == -100.0, f"Expected -100.0, got {r2}"
        _ok(name)
    except Exception as exc:
        _fail(name, traceback.format_exc(limit=3))


def test_rl_step_reset(env: RewardEnv) -> None:
    name = "rl_step_reset"
    try:
        obs = env.reset()
        assert isinstance(obs, dict), "reset() must return dict"

        obs, reward, done, info = env.step(ASM_EXIT_ZERO)
        assert isinstance(obs, dict),   "step obs must be dict"
        assert isinstance(reward, float), "step reward must be float"
        assert isinstance(done, bool),  "step done must be bool"
        assert "cycle_count" in obs,    "obs must contain cycle_count"

        # First call sets baseline; second call with same asm = 0 reward
        _, reward2, _, _ = env.step(ASM_EXIT_ZERO)
        assert reward2 == 0.0, \
            f"Identical asm should give 0 reward, got {reward2}"
        _ok(name)
    except Exception as exc:
        _fail(name, traceback.format_exc(limit=3))


def test_no_start_check(env: RewardEnv) -> None:
    name = "no_start_detection"
    env_nocheck = RewardEnv(strict=True, require_start=False)
    try:
        # Should not raise CompilationError about '_start'
        env_nocheck.compile_and_run("\n.section .text\n")
    except CompilationError as exc:
        if "_start" in str(exc):
            _fail(name, f"Incorrectly rejected for missing _start: {exc}")
        else:
            _ok(name, "Failed at compiler as expected, not at _start check")
    except (ExecutionError, MeasurementError):
        _ok(name, "Failed post-compilation as expected")
    except Exception as exc:
        _fail(name, traceback.format_exc(limit=3))


def test_bad_toolchain_raises() -> None:
    name = "bad_toolchain_raises_ToolchainError"
    try:
        RewardEnv(compiler="/nonexistent/riscv-gcc")
        _fail(name, "Expected ToolchainError")
    except ToolchainError:
        _ok(name)
    except Exception as exc:
        _fail(name, f"Wrong exception: {exc!r}")


def test_repr(env: RewardEnv) -> None:
    name = "repr"
    try:
        r = repr(env)
        assert "RewardEnv(" in r, f"Unexpected repr: {r!r}"
        _ok(name)
    except Exception as exc:
        _fail(name, str(exc))


# ---------------------------------------------------------------------------
# End-to-end demo
# ---------------------------------------------------------------------------

def demo() -> None:
    print("\n" + "=" * 60)
    print("  RewardEnv — end-to-end demo")
    print("=" * 60)
    try:
        env = RewardEnv()
    except ToolchainError as exc:
        print(f"{RED}Cannot instantiate RewardEnv: {exc}{RESET}")
        return

    print(f"\n  Compiler   : {env.compiler}")
    print(f"  QEMU       : {env.qemu_binary}")
    print(f"  Plugin     : {env.plugin_path}")

    assembly = """\
.section .text
.globl _start
_start:
    li a0, 0
    li a7, 93
    ecall
"""
    print("\n  Running minimal _start program …")
    cycle_count = env.compile_and_run(assembly)
    print(f"  cycle_count (weighted_cost) = {cycle_count}")

    baseline = cycle_count
    reward = env.calculate_reward(cycle_count, baseline)
    print(f"  reward vs itself            = {reward}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("  RewardEnv smoke tests")
    print("=" * 60)

    # Attempt to create the primary env used by most tests
    env: RewardEnv | None = None
    try:
        env = RewardEnv(strict=True)
        print(f"\n  {GREEN}Toolchain found{RESET}: {env.compiler}\n")
    except ToolchainError as exc:
        print(f"\n  {YELLOW}Toolchain not available{RESET}: {exc}")
        print("  Skipping tests that require a working toolchain.\n")

    # Tests that require a working env
    if env is not None:
        test_toolchain_detection(env)
        test_basic_compile_and_run(env)
        test_loop_costs_more(env)
        test_determinism(env)
        test_invalid_asm_strict(env)
        test_empty_asm_strict(env)
        test_calculate_reward(env)
        test_rl_step_reset(env)
        test_no_start_check(env)
        test_repr(env)
    else:
        for t in [
            "toolchain_detection", "basic_compile_and_run",
            "loop_costs_more_than_exit", "determinism",
            "invalid_asm_strict_raises", "empty_asm_strict_raises",
            "calculate_reward", "rl_step_reset",
            "no_start_detection", "repr",
        ]:
            _skip(t, "toolchain unavailable")

    # Tests that instantiate their own envs
    test_non_strict_returns_penalty()
    test_custom_penalty()
    test_bad_toolchain_raises()

    # Demo
    if env is not None:
        demo()

    # Summary
    total = _passed + _failed + _skipped
    print("=" * 60)
    print(f"  Results: {_passed}/{total} passed, "
          f"{_failed} failed, {_skipped} skipped")
    print("=" * 60)

    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
