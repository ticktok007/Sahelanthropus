#!/usr/bin/env python3
"""
test_success_termination.py — Verify episode success termination tuning:
    1. total_speedup > 15% (0.15) triggers terminated = True
    2. Bonus +1.0 added to reward
    3. Step info contains "success": True and "total_speedup"
"""

from superopt_env import SuperoptEnv


def test_success_termination():
    print("=" * 75)
    print(" TESTING SUCCESS TERMINATION TUNING IN SUPEROPTENV")
    print("=" * 75)

    env = SuperoptEnv(use_reward_shaping=False)
    obs, info = env.reset(seed=42)

    # Set baseline cycles = 100.0, current cycles = 80.0 -> Speedup = (100 - 80) / 100 = 0.20 (20% > 15%)
    env.baseline_cycles = 100.0
    env.current_cycles = 80.0

    # Set matching instructions: mul t1, a0, t0 preceded by li t0, 4 (Rule 0 match)
    env.current_program[0] = [1, 0, 0, 5, 4]   # li t0, 4
    env.current_program[1] = [5, 10, 5, 6, 0]  # mul t1, a0, t0
    action_mask = env.get_action_mask(obs)
    valid_actions = [a for a in range(160) if action_mask[a]]
    assert len(valid_actions) > 0, "Expected valid actions"

    obs, reward, term, trunc, step_info = env.step(valid_actions[0])

    print(f" Baseline Cycles : {step_info['baseline_cycles']:.1f}")
    print(f" Current Cycles  : {step_info['current_cycles']:.1f}")
    print(f" Total Speedup   : {step_info['total_speedup']*100:.1f}%")
    print(f" Success Flag    : {step_info['success']}")
    print(f" Terminated      : {term}")
    print(f" Reward          : {reward:+.4f}")

    assert step_info["success"] == True, "Expected success = True for 20% speedup"
    assert term == True, "Expected terminated = True for success condition"
    assert reward >= 1.0, f"Expected reward >= +1.0 bonus, got {reward}"
    print(" ✓ Success termination (>15% speedup, +1.0 bonus) verified successfully!")

    print("=" * 75)
    print(" ALL SUCCESS TERMINATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    test_success_termination()
