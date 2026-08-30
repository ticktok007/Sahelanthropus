#!/usr/bin/env python3
"""
run_shaping_ablation.py — Ablation Study: Reward Shaping ON vs OFF.
Measures:
    1. Episodes-to-first-positive-reward
    2. Cumulative positive reward trajectory
"""

import math
import os
import csv
import time
import argparse
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

from superopt_env import SuperoptEnv
from ppo_gnn_actor_critic import GNNActorCritic
from graph_ppo_rollout_buffer import GraphRolloutBuffer
from run_verified_gnn_ppo import GraphVectorEnv
from utils_seed import set_seed
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_shaping_env(env_id: str, seed: int, use_reward_shaping: bool) -> gym.Env:
    env = gym.make(env_id)
    env.unwrapped.use_reward_shaping = use_reward_shaping
    env.unwrapped.use_graph_obs = True
    env.action_space.seed(seed)
    return env


def run_shaping_trial(
    use_reward_shaping: bool = True,
    lr: float = 3e-4,
    num_steps: int = 256,
    seed: int = 0,
    total_steps: int = 100000,
    csv_out: str = "shaping_on.csv",
    tag: str = "Shaping ON"
) -> Tuple[int, int]:
    print("\n" + "=" * 75)
    print(f" ABLATION RUN: {tag.upper()} ({total_steps:,d} STEPS)")
    print("=" * 75)
    print(f" Reward Shaping    : {'ON (Phi(s) = -instruction_count)' if use_reward_shaping else 'OFF (Standard Rewards)'}")
    print(f" Compute Device    : {DEVICE}")

    # Set unified seeds
    set_seed(seed)

    num_envs = 4
    batch_size = num_envs * num_steps
    num_updates = math.ceil(total_steps / batch_size)

    envs = GraphVectorEnv([
        lambda idx=i: make_shaping_env("SuperoptEnv-v0", seed=seed + idx, use_reward_shaping=use_reward_shaping)
        for i in range(num_envs)
    ])

    agent = GNNActorCritic(
        inst_in_dim=177, reg_in_dim=1, hidden_dim=128, embed_dim=256,
        num_rules=10, max_len=16, num_heads=8, num_layers=3
    ).to(DEVICE)

    if os.path.exists("pretrained_gat_encoder.pt"):
        state_dict = torch.load("pretrained_gat_encoder.pt", weights_only=True)
        agent.encoder.load_state_dict(state_dict)

    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    rollout_buffer = GraphRolloutBuffer(num_steps=num_steps, num_envs=num_envs)

    csv_file = open(csv_out, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["update", "global_step", "episode_count", "mean_reward", "first_pos_episode", "first_pos_step"])

    global_step = 0
    total_episodes = 0
    first_positive_episode = None
    first_positive_step = None

    next_obs_list, infos = envs.reset(seed=seed)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=DEVICE)
    next_mask = torch.tensor(infos["action_mask"], dtype=torch.bool, device=DEVICE)

    ep_returns = [0.0] * num_envs

    for update in range(1, num_updates + 1):
        for step in range(num_steps):
            global_step += num_envs

            from torch_geometric.data import Batch
            step_batch = Batch.from_data_list(next_obs_list)
            x_dict = {k: v.to(DEVICE) for k, v in step_batch.x_dict.items()}
            edge_dict = {k: v.to(DEVICE) for k, v in step_batch.edge_index_dict.items()}
            batch_dict = {'inst': step_batch['inst'].batch.to(DEVICE), 'reg': step_batch['reg'].batch.to(DEVICE)}

            with torch.no_grad():
                act, logp, _, val = agent.get_action_and_value(x_dict, edge_dict, action_mask=next_mask, batch_dict=batch_dict)

            rollout_buffer.insert(step, next_obs_list, act, logp, torch.zeros(num_envs), next_done, val, next_mask)

            next_obs_list, reward, terms, truncs, step_info = envs.step(act.cpu().numpy())
            dones = np.logical_or(terms, truncs)
            next_done = torch.tensor(dones, dtype=torch.float32, device=DEVICE)
            rollout_buffer.rewards[step] = torch.tensor(reward, dtype=torch.float32)
            next_mask = torch.tensor(step_info["action_mask"], dtype=torch.bool, device=DEVICE)

            for e in range(num_envs):
                ep_returns[e] += reward[e]

                # Check for first positive reward condition
                if reward[e] > 0 and first_positive_episode is None:
                    first_positive_episode = total_episodes + 1
                    first_positive_step = global_step
                    print(f" 🎉 [{tag}] FIRST POSITIVE REWARD RECEIVED! Episode: {first_positive_episode:,d} | Step: {first_positive_step:,d} | Reward: {reward[e]:+.4f}")

                if dones[e]:
                    total_episodes += 1
                    ep_returns[e] = 0.0

        # Optimization Step
        with torch.no_grad():
            next_batch = Batch.from_data_list(next_obs_list)
            x_next = {k: v.to(DEVICE) for k, v in next_batch.x_dict.items()}
            edge_next = {k: v.to(DEVICE) for k, v in next_batch.edge_index_dict.items()}
            b_next = {'inst': next_batch['inst'].batch.to(DEVICE), 'reg': next_batch['reg'].batch.to(DEVICE)}
            next_val = agent.get_value(x_next, edge_next, batch_dict=b_next).cpu()

            advantages = torch.zeros_like(rollout_buffer.rewards)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done.cpu()
                    nextvalues = next_val
                else:
                    nextnonterminal = 1.0 - rollout_buffer.dones[t + 1]
                    nextvalues = rollout_buffer.values[t + 1]
                delta = rollout_buffer.rewards[t] + 0.99 * nextvalues * nextnonterminal - rollout_buffer.values[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + rollout_buffer.values

        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        loader = rollout_buffer.get_dataloader(adv_norm, returns, minibatch_size=256)

        for epoch in range(4):
            for mb in loader:
                mb_x = {k: v.to(DEVICE) for k, v in mb.x_dict.items()}
                mb_edge = {k: v.to(DEVICE) for k, v in mb.edge_index_dict.items()}
                mb_b = {'inst': mb['inst'].batch.to(DEVICE), 'reg': mb['reg'].batch.to(DEVICE)}
                mb_act = mb.action.to(DEVICE)
                mb_logp = mb.logprob.to(DEVICE)
                mb_adv = mb.advantage.to(DEVICE)
                mb_ret = mb.return_val.to(DEVICE)
                mb_val = mb.value.to(DEVICE)
                mb_mask = mb.action_mask.to(DEVICE)

                _, newlogp, entropy, newval = agent.get_action_and_value(mb_x, mb_edge, action=mb_act, action_mask=mb_mask, batch_dict=mb_b)
                logratio = newlogp - mb_logp
                ratio = logratio.exp()

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 0.9, 1.1)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((newval - mb_ret) ** 2).mean()

                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        csv_writer.writerow([update, global_step, total_episodes, rollout_buffer.rewards.mean().item(), first_positive_episode or -1, first_positive_step or -1])
        csv_file.flush()

        if update % 10 == 0:
            print(f" [{tag}] Update {update:>3d}/{num_updates} | Steps: {global_step:>7,d} | Total Ep: {total_episodes:>5,d} | First Pos Ep: {first_positive_episode}")

    csv_file.close()
    envs.close()
    return first_positive_episode, first_positive_step


def plot_ablation_results(on_ep: Optional[int], on_step: Optional[int], off_ep: Optional[int], off_step: Optional[int]):
    print("\n" + "=" * 75)
    print(" ABLATION COMPARISON: EPISODES TO FIRST POSITIVE REWARD")
    print("=" * 75)
    print(f" Reward Shaping ON  : Episode {on_ep if on_ep else 'N/A'} (Step {on_step if on_step else 'N/A'})")
    print(f" Reward Shaping OFF : Episode {off_ep if off_ep else 'N/A'} (Step {off_step if off_step else 'N/A'})")

    if on_ep and off_ep:
        ep_speedup = ((off_ep - on_ep) / off_ep) * 100.0
        print(f" Episode Speedup    : {ep_speedup:+.1f}% faster to first positive reward ({off_ep / max(1, on_ep):.2f}x speedup)")

    # Bar Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    labels = ['Shaping ON\n(Phi = -inst_count)', 'Shaping OFF\n(Standard Reward)']
    episodes = [on_ep or 0, off_ep or 0]
    colors = ['#2ca02c', '#d62728']

    bars = ax.bar(labels, episodes, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
    ax.set_ylabel("Episodes to First Positive Reward", fontsize=11, fontweight='bold')
    ax.set_title("Ablation Study: Potential-Based Reward Shaping Impact", fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, ep in zip(bars, episodes):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + max(episodes)*0.02, f"{ep:,d} Ep", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    out_png = "shaping_ablation_comparison.png"
    plt.savefig(out_png, dpi=300)
    print(f"[+] Saved ablation plot to {out_png}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50000)
    args = parser.parse_args()

    # 1. Run Shaping ON
    on_ep, on_step = run_shaping_trial(use_reward_shaping=True, total_steps=args.steps, csv_out="shaping_on.csv", tag="Shaping ON")

    # 2. Run Shaping OFF
    off_ep, off_step = run_shaping_trial(use_reward_shaping=False, total_steps=args.steps, csv_out="shaping_off.csv", tag="Shaping OFF")

    # 3. Compare & Plot
    plot_ablation_results(on_ep, on_step, off_ep, off_step)


if __name__ == "__main__":
    main()
