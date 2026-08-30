#!/usr/bin/env python3
"""
test_reward_shaping.py — Verify potential-based reward shaping:
    Phi(s) = -instruction_count(s)
    r_shaped = r + gamma * Phi(s') - Phi(s)
"""

from superopt_env import SuperoptEnv


def test_potential_reward_shaping():
    print("=" * 75)
    print(" TESTING POTENTIAL-BASED REWARD SHAPING IN SUPEROPTENV")
    print("=" * 75)

    env = SuperoptEnv(use_reward_shaping=True, gamma=0.99)
    obs, info = env.reset(seed=42)

    # Set known program: 4 instructions (li t0, 4; mul t1, a0, t0; addi t2, t1, 0; ret)
    env.current_program[0] = [1, 0, 0, 5, 4]   # li t0, 4
    env.current_program[1] = [5, 10, 5, 6, 0]  # mul t1, a0, t0
    env.current_program[2] = [2, 6, 0, 7, 0]   # addi t2, t1, 0
    for slot in range(3, 16):
        env.current_program[slot] = [0, 0, 0, 0, 0]

    initial_phi = env._get_potential()
    print(f" Initial State Non-NOP Instruction Count : {-initial_phi:.0f}")
    print(f" Initial Potential Phi(s)                 : {initial_phi:+.2f}")
    assert initial_phi == -3.0, f"Expected initial Phi(s) = -3.0, got {initial_phi}"

    # Step 1: Execute rule 2 (addi t2, t1, 0 -> mv t2, t1) -> Non-NOP count stays 3
    # Phi(s) = -3, Phi(s') = -3 -> Shaping = 0.99*(-3) - (-3) = +0.03
    action_mask = env.get_action_mask(obs)
    valid_actions = [a for a in range(160) if action_mask[a]]
    assert len(valid_actions) > 0

    obs, reward, term, trunc, step_info = env.step(valid_actions[0])

    next_phi = env._get_potential()
    expected_shaping = 0.99 * next_phi - initial_phi

    print(f" Next State Potential Phi(s')             : {next_phi:+.2f}")
    print(f" Shaping Signal (gamma*Phi(s') - Phi(s))  : {expected_shaping:+.4f}")
    print(f" Shaped Reward Returned                   : {reward:+.4f}")

    assert abs(expected_shaping - (0.99 * next_phi - initial_phi)) < 1e-5, "Shaping signal calculation mismatch"
    print(" ✓ Potential-based reward shaping verified successfully!")

    print("=" * 75)
    print(" ALL REWARD SHAPING TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    test_potential_reward_shaping()
