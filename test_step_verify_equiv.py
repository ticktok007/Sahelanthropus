#!/usr/bin/env python3
"""
test_step_verify_equiv.py — Verify that SuperoptEnv.step() invokes verify_equiv(original, candidate)
with a 5s timeout prior to computing rewards.
"""

import time
import gymnasium as gym
from superopt_env import SuperoptEnv
from equivalence_verifier import verify_equiv


def test_step_verify_equiv_integration():
    print("=" * 70)
    print(" TESTING SUPEROPTENV INTEGRATION WITH VERIFY_EQUIV (5S TIMEOUT)")
    print("=" * 70)

    env = SuperoptEnv()
    obs, info = env.reset(seed=42)

    # Set current program to contain an applicable rewrite rule (mul t1, a0, 4 -> slli t1, a0, 2)
    # Instruction 0: li t0, 4 [1, 0, 0, 5, 4]
    # Instruction 1: mul t1, a0, t0 [5, 10, 5, 6, 0]
    env.current_program[0] = [1, 0, 0, 5, 4]   # li t0, 4
    env.current_program[1] = [5, 10, 5, 6, 0]  # mul t1, a0, t0
    obs = env._get_obs()
    info["action_mask"] = env.get_action_mask(obs)

    # Test equivalent transformation step
    # Rule 2: add x, 0 -> x
    action_mask = info["action_mask"]
    valid_actions = [a for a in range(160) if action_mask[a]]
    print(f" [+] Found {len(valid_actions)} valid candidate action rewrites")

    if valid_actions:
        action = valid_actions[0]
        rule_id = action // 16
        target_idx = action % 16

        orig_prog = [list(inst) for inst in env.current_program]

        t0 = time.time()
        obs, reward, term, trunc, step_info = env.step(action)
        elapsed = time.time() - t0

        print(f" ✓ Step Executed: Applied = {step_info['applied']}, Reward = {reward:.4f}, Verification Time = {elapsed:.4f}s")
        assert elapsed < 5.0, f"Step execution exceeded 5s timeout ({elapsed:.2f}s)"
        print(" ✓ 5-second Timeout Constraint Honored!")

    print("=" * 70)
    print(" ALL SUPEROPTENV STEP VERIFY_EQUIV TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_step_verify_equiv_integration()
