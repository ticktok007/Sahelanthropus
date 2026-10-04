#!/usr/bin/env python3
"""
run_phase4_final_1m.py — Phase 4 Final 1,000,000 Step Benchmark Run

Locked-In Best Hyperparameters (From Pareto Frontier & W&B Sweep Analysis):
  - Learning Rate (lr)      : 3e-4 (with 3e-5 min floor)
  - Rollout Steps (n_steps) : 256
  - Clip Coefficient        : 0.1
  - GAE Lambda (λ)          : 0.95
  - Value Loss Clipping     : Enabled
  - Observation Normalizer  : Enabled (RunningMeanStd)
  - Gradient Clipping       : 0.5 (max_grad_norm)
  - Total Steps             : 1,000,000 steps per seed
  - Seeds                   : 5 independent seeds (0, 1, 2, 3, 4)
"""

import math
import os
import time
import csv
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import matplotlib.pyplot as plt

from superopt_env import SuperoptEnv
from ppo_gnn_actor_critic import GNNActorCritic
from graph_ppo_rollout_buffer import GraphRolloutBuffer
from run_verified_gnn_ppo import GraphVectorEnv
from curriculum_scheduler import compute_curriculum_max_len
from utils_seed import set_seed
from torch_geometric.data import Batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

# Locked-In Best Hyperparameter Constants
LOCKED_LR = 3e-4
LOCKED_MIN_LR = 3e-5
LOCKED_NUM_STEPS = 256
LOCKED_CLIP_COEF = 0.1
LOCKED_GAE_LAMBDA = 0.95
LOCKED_MAX_GRAD_NORM = 0.5
LOCKED_TOTAL_STEPS = 1000000


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def make_env_for_seed(env_id: str, seed: int, idx: int) -> gym.Env:
    env = gym.make(env_id)
    env.unwrapped.use_graph_obs = True
    env.unwrapped.use_reward_shaping = True
    env.action_space.seed(seed + idx)
    return env


def train_single_seed_phase4(
    seed: int,
    total_steps: int = LOCKED_TOTAL_STEPS,
    num_steps: int = LOCKED_NUM_STEPS,
    lr: float = LOCKED_LR,
    clip_coef: float = LOCKED_CLIP_COEF,
    gae_lambda: float = LOCKED_GAE_LAMBDA,
    csv_out: str = "phase4_1m_curve_seed0.csv",
    seed_idx: int = 0,
    total_seeds: int = 5
) -> Tuple[List[int], List[float], List[int]]:
    print("\n" + "=" * 75)
    print(f" PHASE 4 FINAL 1M RUN: SEED {seed} ({seed_idx+1}/{total_seeds})")
    print(f" Locked Hyperparams: lr={lr}, num_steps={num_steps}, clip_coef={clip_coef}, λ={gae_lambda}")
    print("=" * 75)

    set_seed(seed)

    num_envs = 4
    batch_size = num_envs * num_steps
    num_updates = math.ceil(total_steps / batch_size)

    ckpt_file = f"phase4_seed_{seed}_ckpt.pt"
    start_update = 1
    global_step = 0
    total_rules = 0
    completed_returns = []
    recorded_steps = []
    recorded_rewards = []
    recorded_rules = []

    if os.path.exists(csv_out):
        try:
            steps_in, rews_in, rules_in = [], [], []
            with open(csv_out, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 6:
                        steps_in.append(int(row[1]))
                        rews_in.append(float(row[3]))
                        rules_in.append(int(row[5]))
            if len(steps_in) >= num_updates:
                print(f" [Seed {seed}] Found complete CSV {csv_out} ({len(steps_in)} updates). Skipping training.")
                return steps_in[:num_updates], rews_in[:num_updates], rules_in[:num_updates]
            elif os.path.exists(ckpt_file):
                ckpt = torch.load(ckpt_file, map_location=DEVICE, weights_only=False)
                if ckpt.get("update", 0) > 0:
                    start_update = ckpt["update"] + 1
                    global_step = ckpt["global_step"]
                    total_rules = ckpt["total_rules"]
                    completed_returns = ckpt["completed_returns"]
                    recorded_steps = steps_in[:ckpt["update"]]
                    recorded_rewards = rews_in[:ckpt["update"]]
                    recorded_rules = rules_in[:ckpt["update"]]
                    print(f" [Seed {seed}] Resuming from update {start_update}/{num_updates} (global_step: {global_step:,})")
        except Exception as e:
            print(f" [Seed {seed}] Could not parse existing CSV/checkpoint: {e}. Starting fresh.")
            start_update = 1

    initial_max_len = compute_curriculum_max_len(global_step)

    envs = GraphVectorEnv([
        lambda idx=i: make_env_for_seed("SuperoptEnv-v0", seed=seed, idx=idx)
        for i in range(num_envs)
    ])

    for sub_env in envs.envs:
        sub_env.unwrapped.max_len = initial_max_len

    agent = GNNActorCritic(
        inst_in_dim=177, reg_in_dim=1, hidden_dim=128, embed_dim=256,
        num_rules=10, max_len=25, num_heads=8, num_layers=3
    ).to(DEVICE)

    if os.path.exists("pretrained_gat_encoder.pt"):
        state_dict = torch.load("pretrained_gat_encoder.pt", weights_only=True)
        agent.encoder.load_state_dict(state_dict)

    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    rollout_buffer = GraphRolloutBuffer(num_steps=num_steps, num_envs=num_envs)

    if start_update > 1 and os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=DEVICE, weights_only=False)
        agent.load_state_dict(ckpt["agent"])
        optimizer.load_state_dict(ckpt["optimizer"])
        csv_file = open(csv_out, "a", newline="")
        csv_writer = csv.writer(csv_file)
    else:
        csv_file = open(csv_out, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["update", "global_step", "max_len", "episode_reward_mean", "explained_variance", "rules_applied"])

    next_obs_list, infos = envs.reset(seed=seed)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=DEVICE)
    next_mask = torch.tensor(infos["action_mask"], dtype=torch.bool, device=DEVICE)

    ep_returns = [0.0] * num_envs
    start_time = time.time()

    for update in range(start_update, num_updates + 1):
        frac = max(0.1, 1.0 - (update - 1.0) / num_updates)
        optimizer.param_groups[0]["lr"] = frac * lr

        # Update curriculum max_len
        curr_max_len = compute_curriculum_max_len(global_step)
        for sub_env in envs.envs:
            sub_env.unwrapped.update_curriculum(global_step)

        for step in range(num_steps):
            global_step += num_envs

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
                if step_info.get("applied", [False]*num_envs)[e]:
                    total_rules += 1
                if dones[e]:
                    completed_returns.append(ep_returns[e])
                    ep_returns[e] = 0.0

        # GAE Optimization with gae_lambda
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
                advantages[t] = lastgaelam = delta + 0.99 * gae_lambda * nextnonterminal * lastgaelam
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
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (newval - mb_ret) ** 2
                v_clipped = mb_val + torch.clamp(newval - mb_val, -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - mb_ret) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), LOCKED_MAX_GRAD_NORM)
                optimizer.step()

        mean_rew = np.mean(completed_returns[-50:]) if completed_returns else 0.0
        y_true = returns.numpy().flatten()
        y_pred = rollout_buffer.values.numpy().flatten()
        var_y = np.var(y_true)
        expl_var = 1.0 - (np.var(y_true - y_pred) / (var_y + 1e-8)) if var_y > 1e-8 else 0.0

        recorded_steps.append(global_step)
        recorded_rewards.append(mean_rew)
        recorded_rules.append(total_rules)

        csv_writer.writerow([update, global_step, curr_max_len, mean_rew, expl_var, total_rules])
        csv_file.flush()

        if update % 10 == 0 or update == num_updates:
            elapsed = time.time() - start_time
            updates_done = update - start_update + 1
            sec_per_update = elapsed / max(1, updates_done)

            rem_updates_seed = num_updates - update
            eta_seed = rem_updates_seed * sec_per_update

            rem_seeds = total_seeds - seed_idx - 1
            rem_updates_total = (rem_seeds * num_updates) + rem_updates_seed
            eta_total = rem_updates_total * sec_per_update

            print(f" [Seed {seed} ({seed_idx+1}/{total_seeds})] Update {update:>4d}/{num_updates} | Steps: {global_step:>7,d} | max_len: {curr_max_len:>2d} | Mean Ep Rew: {mean_rew:+.4f} | Rules: {total_rules:>3d} | ETA Seed: {format_time(eta_seed)} | ETA Total: {format_time(eta_total)}")
            torch.save({
                "update": update,
                "global_step": global_step,
                "total_rules": total_rules,
                "completed_returns": completed_returns[-50:],
                "agent": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_file)

    csv_file.close()
    envs.close()
    return recorded_steps, recorded_rewards, recorded_rules


def compute_statistics_and_plot_phase4(seeds_data: Dict[int, Tuple[List[int], List[float], List[int]]]):
    print("\n" + "=" * 75)
    print(" CALCULATING 95% CONFIDENCE INTERVAL STATISTICS & PLOTTING PHASE 4")
    print("=" * 75)

    steps = seeds_data[0][0]
    n_seeds = len(seeds_data)
    n_points = len(steps)

    reward_matrix = np.zeros((n_seeds, n_points))
    for s_idx, s in enumerate(seeds_data.keys()):
        reward_matrix[s_idx] = seeds_data[s][1]

    mean_rewards = np.mean(reward_matrix, axis=0)
    std_rewards = np.std(reward_matrix, axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(mean_rewards)

    rel_std = std_rewards / (np.abs(mean_rewards) + 1e-8)
    avg_rel_std = np.mean(rel_std) * 100.0
    max_rel_std = np.max(rel_std) * 100.0

    print(f" [+] Stability Diagnostic: Avg (Std / Mean) = {avg_rel_std:.2f}%, Max = {max_rel_std:.2f}%")
    if avg_rel_std > 30.0:
        print(f" [!] WARNING: Training instability detected! Avg (Std / Mean) is {avg_rel_std:.2f}% (Threshold: 30.0%).")
    else:
        print(f" [+] STABILITY VERIFIED: Avg (Std / Mean) {avg_rel_std:.2f}% <= 30.0%. Training is STABLE!")

    stderr = std_rewards / math.sqrt(n_seeds)
    ci95 = 1.96 * stderr

    out_csv = "phase4_1m_aggregated.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["global_step", "mean_reward", "std_reward", "rel_std_percent", "ci95_lower", "ci95_upper"])
        for i in range(n_points):
            writer.writerow([
                steps[i],
                mean_rewards[i],
                std_rewards[i],
                rel_std[i] * 100.0,
                mean_rewards[i] - ci95[i],
                mean_rewards[i] + ci95[i]
            ])

    print(f" [+] Saved {out_csv}!")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for s_idx, s in enumerate(seeds_data.keys()):
        ax.plot(steps, reward_matrix[s_idx], label=f"Seed {s}", color=colors[s_idx], alpha=0.35, linewidth=1.0)

    ax.plot(steps, mean_rewards, label="Phase 4 Locked-In 5-Seed Mean Reward", color="#1f77b4", linewidth=2.5)

    ax.fill_between(
        steps,
        mean_rewards - ci95,
        mean_rewards + ci95,
        color="#1f77b4",
        alpha=0.25,
        label="95% Confidence Interval (±1.96 SE)"
    )

    ax.set_title("Phase 4 Final 1M Step Benchmark: Episode Reward (5 Independent Seeds)\n(Locked Hyperparameters: lr=3e-4, n_steps=256, clip_coef=0.1, λ=0.95)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Environment Steps", fontsize=11)
    ax.set_ylabel("Episode Reward Mean", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_png = "ppo_1m_5seeds_mean_ci95.png"
    plt.savefig(out_png, dpi=300)
    print(f" [+] Saved publication plot to {out_png}!")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=LOCKED_TOTAL_STEPS)
    args = parser.parse_args()

    seeds = [0, 1, 2, 3, 4]
    seeds_data: Dict[int, Tuple[List[int], List[float], List[int]]] = {}

    for idx, s in enumerate(seeds):
        csv_out = f"phase4_1m_curve_seed{s}.csv"
        steps, rewards, rules = train_single_seed_phase4(
            seed=s,
            total_steps=args.steps,
            num_steps=LOCKED_NUM_STEPS,
            lr=LOCKED_LR,
            clip_coef=LOCKED_CLIP_COEF,
            gae_lambda=LOCKED_GAE_LAMBDA,
            csv_out=csv_out,
            seed_idx=idx,
            total_seeds=len(seeds)
        )
        seeds_data[s] = (steps, rewards, rules)

    compute_statistics_and_plot_phase4(seeds_data)


if __name__ == "__main__":
    main()
