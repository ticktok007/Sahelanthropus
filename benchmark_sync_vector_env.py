#!/usr/bin/env python3
"""
benchmark_sync_vector_env.py — Vectorized SuperoptEnv Benchmark using SyncVectorEnv

Features:
- Wraps 4 parallel instances of SuperoptEnv-v0 in gymnasium.vector.SyncVectorEnv.
- Verifies batch reset and step shapes (obs_batch shape: (4, 80), action_mask shape: (4, 5)).
- Benchmarks vector step execution throughput (steps/second).
"""

import time
from typing import Callable

import numpy as np
import gymnasium as gym
from superopt_env import SuperoptEnv  # Ensures environment registration


def make_env(env_id: str, seed: int) -> Callable[[], gym.Env]:
    def thunk():
        env = gym.make(env_id)
        env.action_space.seed(seed)
        return env
    return thunk


def run_sync_vector_benchmark(num_envs: int = 4, total_vector_steps: int = 5000):
    print("======================================================================")
    print(f" SyncVectorEnv Benchmark: {num_envs} Parallel Instances of SuperoptEnv")
    print("======================================================================")

    # Instantiate SyncVectorEnv with 4 parallel environments
    env_fns = [make_env("SuperoptEnv-v0", seed=i) for i in range(num_envs)]
    vec_envs = gym.vector.SyncVectorEnv(env_fns)

    # 1. Batch Reset Verification
    obs_batch, info_batch = vec_envs.reset(seed=42)
    print(f" Observation Batch Shape : {obs_batch.shape} (dtype: {obs_batch.dtype})")
    print(f" Action Space            : {vec_envs.action_space}")
    print(f" Single Action Space     : {vec_envs.single_action_space}")
    
    assert obs_batch.shape == (num_envs, 80), f"Expected shape ({num_envs}, 80), got {obs_batch.shape}"
    assert "action_mask" in info_batch, "info_batch must contain 'action_mask'!"
    assert info_batch["action_mask"].shape == (num_envs, 5), f"Expected action_mask shape ({num_envs}, 5), got {info_batch['action_mask'].shape}"

    print(" [+] Vectorized reset() verified: obs shape (4, 80) and action_mask (4, 5) match!\n")

    # 2. Benchmark Throughput (Steps/Second)
    print("----------------------------------------------------------------------")
    print(f" Running Throughput Benchmark ({total_vector_steps} Vector Steps)...")
    print("----------------------------------------------------------------------")

    start_time = time.perf_counter()

    for _ in range(total_vector_steps):
        # Sample random actions for all 4 parallel environments
        actions_batch = vec_envs.action_space.sample()
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = vec_envs.step(actions_batch)

    end_time = time.perf_counter()

    elapsed_sec = end_time - start_time
    total_env_steps = num_envs * total_vector_steps
    vector_sps = total_vector_steps / elapsed_sec
    env_sps = total_env_steps / elapsed_sec
    avg_latency_ms = (elapsed_sec / total_vector_steps) * 1000.0

    print("======================================================================")
    print(" Benchmark Results & Performance Metrics")
    print("======================================================================")
    print(f" Total Vector Steps     : {total_vector_steps:<8d}")
    print(f" Total Env Transitions  : {total_env_steps:<8d}")
    print(f" Total Benchmark Time   : {elapsed_sec:.4f} seconds")
    print(f" Vector Steps / Sec     : {vector_sps:10.2f} steps/sec")
    print(f" Total Env Transitions/s: {env_sps:10.2f} transitions/sec")
    print(f" Avg Latency per Batch  : {avg_latency_ms:10.4f} ms")

    vec_envs.close()

    assert env_sps > 1000.0, f"Expected > 1,000 transitions/sec throughput, got {env_sps:.2f}!"
    print("\n[+] SYNC VECTOR ENV BENCHMARK VERIFIED SUCCESSFUL!")
    print("======================================================================\n")


if __name__ == "__main__":
    run_sync_vector_benchmark(num_envs=4, total_vector_steps=5000)
