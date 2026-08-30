#!/usr/bin/env python3
"""
test_curriculum_integration.py — Verify CurriculumScheduler and SuperoptEnv.update_curriculum(step) integration.
"""

from superopt_env import SuperoptEnv
from curriculum_scheduler import CurriculumScheduler, compute_curriculum_max_len


def test_curriculum_integration():
    print("=" * 70)
    print(" TESTING CURRICULUM SCHEDULER INTEGRATION IN SUPEROPTENV")
    print("=" * 70)

    env = SuperoptEnv(max_len=8)
    assert env.max_len == 8, "Expected initial max_len=8"
    print(f" ✓ [Step 0] Initial SuperoptEnv max_len: {env.max_len}")

    # Test milestones
    milestones = [
        (0, 8),
        (50000, 9),
        (100000, 11),
        (250000, 16),
        (500000, 25),
        (750000, 25)
    ]

    scheduler = CurriculumScheduler(start_step=0)

    for step, expected_len in milestones:
        m_len = env.update_curriculum(step)
        sched_len, changed = scheduler.update(step)

        assert m_len == expected_len, f"Step {step}: expected env max_len={expected_len}, got {m_len}"
        assert sched_len == expected_len, f"Step {step}: expected scheduler max_len={expected_len}, got {sched_len}"
        assert env.action_space.n == 10 * expected_len, f"Step {step}: expected action space size {10 * expected_len}, got {env.action_space.n}"
        assert env.observation_space.shape[0] == 5 * expected_len, f"Step {step}: expected obs shape {5 * expected_len}, got {env.observation_space.shape[0]}"

        print(f" ✓ [Step {step:>7,d}] max_len: {m_len:>2d} | Obs Dim: {env.observation_space.shape[0]:>3d} | Action Dim: {env.action_space.n:>3d}")

    print("=" * 70)
    print(" ALL CURRICULUM SCHEDULER INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_curriculum_integration()
