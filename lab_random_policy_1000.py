#!/usr/bin/env python3
"""
lab_random_policy_1000.py — Lab Experiment: Run Random Policy for 1,000 Steps on SuperoptEnv

Target:
- Execute a uniform random policy sampling `env.action_space.sample()` across 1,000 environment steps.
- Measure reward distribution (r = 0.0 vs r > 0.0).
- Verify >90% of steps yield r = 0.0 as expected due to sparse valid algebraic patterns in random action spaces.
"""

import numpy as np
from superopt_env import SuperoptEnv


def run_random_policy_lab(total_steps: int = 1000, seed: int = 42):
    print("======================================================================")
    print(f" Lab Experiment: Uniform Random Policy Benchmark ({total_steps} Steps)")
    print("======================================================================")

    env = SuperoptEnv(corpus_path="corpus.json", max_len=16, num_rules=5)
    obs, info = env.reset(seed=seed)

    zero_reward_count = 0
    positive_reward_count = 0
    applied_count = 0
    episodes_completed = 0

    print(f" Initial Corpus Function : {info['func_name']} (ID: {info['corpus_id']})")
    print(f" Action Space            : {env.action_space}")
    print(f" Observation Space       : {env.observation_space.shape}\n")

    for step in range(1, total_steps + 1):
        # Sample uniform random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if reward == 0.0:
            zero_reward_count += 1
        elif reward > 0.0:
            positive_reward_count += 1

        if info.get("applied", False):
            applied_count += 1

        if terminated or truncated:
            episodes_completed += 1
            obs, info = env.reset()

    zero_reward_percentage = (zero_reward_count / total_steps) * 100.0
    positive_reward_percentage = (positive_reward_count / total_steps) * 100.0

    print("======================================================================")
    print(" Empirical Results: 1,000 Step Random Policy Distribution")
    print("======================================================================")
    print(f" Total Environment Steps   : {total_steps}")
    print(f" Episodes Completed        : {episodes_completed}")
    print(f" Zero Reward Steps (r=0.0) : {zero_reward_count:<6d} ({zero_reward_percentage:5.1f}%)")
    print(f" Positive Reward Steps     : {positive_reward_count:<6d} ({positive_reward_percentage:5.1f}%)")
    print(f" Rewrite Rules Applied     : {applied_count:<6d}")

    print("\n----------------------------------------------------------------------")
    print(" Verification Status")
    print("----------------------------------------------------------------------")
    print(f" Target Requirement : > 90.0% steps get r = 0.0")
    print(f" Measured Value     : {zero_reward_percentage:.1f}%")

    assert zero_reward_percentage > 90.0, f"Expected > 90% zero reward steps, got {zero_reward_percentage:.1f}%!"
    print("\n[+] LAB EXPERIMENT VERIFIED SUCCESSFUL: >90% of steps yielded r=0.0 as expected!")
    print("======================================================================\n")


if __name__ == "__main__":
    run_random_policy_lab(total_steps=1000, seed=42)
