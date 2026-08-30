#!/usr/bin/env python3
"""
run_curriculum_benchmark.py — Comparative 300K Benchmark:
Curriculum Strategy vs. No-Curriculum Strategy on RISC-V Superoptimization.
"""

import math
import os
import time
import csv
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
from curriculum_scheduler import CurriculumScheduler, compute_curriculum_max_len
from utils_seed import set_seed
from torch_geometric.data import Batch, HeteroData

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_curriculum_env(env_id: str, seed: int, max_len: int, use_curriculum: bool) -> gym.Env:
    env = gym.make(env_id)
    env.unwrapped.max_len = max_len
    env.unwrapped.use_graph_obs = True
    env.action_space.seed(seed)
    return env


def train_curriculum_run(
    use_curriculum: bool = True,
    lr: float = 3e-4,
    num_steps: int = 256,
    clip_coef: float = 0.1,
    seed: int = 0,
    total_steps: int = 300000,
    csv_out: str = "curriculum_curve.csv",
    tag: str = "Curriculum"
):
    print("\n" + "=" * 75)
    print(f" RUNNING {tag.upper()} PPO BENCHMARK ({total_steps:,d} STEPS)")
    print("=" * 75)
    print(f" Strategy          : {'Curriculum (max_len 8 -> 25)' if use_curriculum else 'No-Curriculum (Fixed max_len 25)'}")
    print(f" Compute Device    : {DEVICE}")

    # Set unified seeds
    set_seed(seed)

    num_envs = 4
    batch_size = num_envs * num_steps
    num_updates = math.ceil(total_steps / batch_size)

    # Initial max_len
    initial_max_len = compute_curriculum_max_len(0) if use_curriculum else 25

    envs = GraphVectorEnv([
        lambda idx=i: make_curriculum_env("SuperoptEnv-v0", seed=seed + idx, max_len=initial_max_len, use_curriculum=use_curriculum)
        for i in range(num_envs)
    ])

    agent = GNNActorCritic(
        inst_in_dim=177,
        reg_in_dim=1,
        hidden_dim=128,
        embed_dim=256,
        num_rules=10,
        max_len=25,  # Upper bound max_len for network heads
        num_heads=8,
        num_layers=3
    ).to(DEVICE)

    if os.path.exists("pretrained_gat_encoder.pt"):
        state_dict = torch.load("pretrained_gat_encoder.pt", weights_only=True)
        agent.encoder.load_state_dict(state_dict)
        print(" [+] Loaded pre-trained GAT Encoder weights!")

    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    rollout_buffer = GraphRolloutBuffer(num_steps=num_steps, num_envs=num_envs)

    csv_file = open(csv_out, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["update", "global_step", "max_len", "episode_reward_mean", "explained_variance", "rules_applied"])

    global_step = 0
    start_time = time.time()

    next_obs_list, infos = envs.reset(seed=seed)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=DEVICE)
    next_mask = torch.tensor(infos["action_mask"], dtype=torch.bool, device=DEVICE)

    ep_returns = [0.0] * num_envs
    completed_returns = []
    total_rules = 0

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * lr

        # Update curriculum max_len across environments if active
        if use_curriculum:
            curr_max_len = compute_curriculum_max_len(global_step)
            for sub_env in envs.envs:
                sub_env.unwrapped.update_curriculum(global_step)
        else:
            curr_max_len = 25

        for step in range(num_steps):
            global_step += num_envs

            step_batch = Batch.from_data_list(next_obs_list)
            x_dict = {k: v.to(DEVICE) for k, v in step_batch.x_dict.items()}
            edge_dict = {k: v.to(DEVICE) for k, v in step_batch.edge_index_dict.items()}
            batch_dict = {
                'inst': step_batch['inst'].batch.to(DEVICE),
                'reg': step_batch['reg'].batch.to(DEVICE)
            }

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

        # GAE
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
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_unclipped = (newval - mb_ret) ** 2
                v_clipped = mb_val + torch.clamp(newval - mb_val, -clip_coef, clip_coef)
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - mb_ret) ** 2).mean()

                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        y_true = returns.numpy().flatten()
        y_pred = rollout_buffer.values.numpy().flatten()
        var_y = np.var(y_true)
        expl_var = 1.0 - (np.var(y_true - y_pred) / (var_y + 1e-8)) if var_y > 1e-8 else 0.0
        mean_rew = np.mean(completed_returns[-50:]) if completed_returns else 0.0

        if update % 10 == 0 or update == num_updates:
            print(f" [{tag}] Update {update:>3d}/{num_updates} | Steps: {global_step:>7,d} | max_len: {curr_max_len:>2d} | ExplVar: {expl_var:+.4f} | Ep Rew: {mean_rew:+.4f} | Rules: {total_rules:>3d}")

        csv_writer.writerow([update, global_step, curr_max_len, mean_rew, expl_var, total_rules])
        csv_file.flush()

    csv_file.close()
    envs.close()
    print(f" [+] {tag} Benchmark Finished!\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300000)
    args = parser.parse_args()

    # 1. Run Curriculum Strategy
    train_curriculum_run(use_curriculum=True, total_steps=args.steps, csv_out="curriculum_curve.csv", tag="Curriculum")

    # 2. Run No-Curriculum Strategy
    train_curriculum_run(use_curriculum=False, total_steps=args.steps, csv_out="nocurriculum_curve.csv", tag="NoCurriculum")


if __name__ == "__main__":
    main()
