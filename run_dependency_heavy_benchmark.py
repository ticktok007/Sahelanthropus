#!/usr/bin/env python3
"""
run_dependency_heavy_benchmark.py — Compare MLP Baseline, GNN (Random Init), and GNN (Pre-trained)
on dependency-heavy assembly corpus (max def-use chain >= 3).
"""

import math
import os
import sys
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
from ppo_superopt import ActorCritic as MLPActorCritic
from ppo_gnn_actor_critic import GNNActorCritic
from gat_encoder import HeteroGATEncoder
from graph_ppo_rollout_buffer import GraphRolloutBuffer
from run_verified_gnn_ppo import GraphVectorEnv
from torch_geometric.data import Batch, HeteroData

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_heavy_env(env_id: str, seed: int, use_graph: bool) -> gym.Env:
    env = gym.make(env_id)
    env.unwrapped.corpus_path = "dependency_heavy_corpus.json"
    env.unwrapped.corpus = env.unwrapped._load_corpus("dependency_heavy_corpus.json")
    env.unwrapped.use_graph_obs = use_graph
    env.action_space.seed(seed)
    return env


def train_mlp_baseline(
    lr: float = 3e-4,
    num_steps: int = 256,
    clip_coef: float = 0.1,
    seed: int = 0,
    total_steps: int = 100000,
    csv_out: str = "mlp_heavy_curve.csv"
):
    print("\n" + "=" * 70)
    print(" RUNNING MLP BASELINE ON DEPENDENCY-HEAVY CORPUS")
    print("=" * 70)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_envs = 4
    batch_size = num_envs * num_steps
    num_updates = math.ceil(total_steps / batch_size)

    envs = gym.vector.SyncVectorEnv([
        lambda idx=i: make_heavy_env("SuperoptEnv-v0", seed=seed + idx, use_graph=False)
        for i in range(num_envs)
    ])

    agent = MLPActorCritic(obs_dim=80, num_rules=10, max_len=16).to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    csv_file = open(csv_out, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["update", "global_step", "episode_reward_mean", "explained_variance", "rules_applied"])

    global_step = 0
    start_time = time.time()

    obs, infos = envs.reset(seed=seed)
    mask = torch.tensor(infos["action_mask"], dtype=torch.bool, device=DEVICE)

    ep_returns = [0.0] * num_envs
    completed_returns = []
    total_rules = 0

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * lr

        obs_buf = torch.zeros((num_steps, num_envs, 80), device=DEVICE)
        act_buf = torch.zeros((num_steps, num_envs), device=DEVICE)
        logp_buf = torch.zeros((num_steps, num_envs), device=DEVICE)
        rew_buf = torch.zeros((num_steps, num_envs), device=DEVICE)
        done_buf = torch.zeros((num_steps, num_envs), device=DEVICE)
        val_buf = torch.zeros((num_steps, num_envs), device=DEVICE)
        mask_buf = torch.zeros((num_steps, num_envs, 160), dtype=torch.bool, device=DEVICE)

        for step in range(num_steps):
            global_step += num_envs
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                act, logp, _, val = agent.get_action_and_value(obs_tensor, action_mask=mask)

            obs_buf[step] = obs_tensor
            act_buf[step] = act
            logp_buf[step] = logp
            val_buf[step] = val.squeeze(-1)
            mask_buf[step] = mask

            next_obs, reward, terms, truncs, step_info = envs.step(act.cpu().numpy())
            dones = np.logical_or(terms, truncs)
            rew_buf[step] = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            done_buf[step] = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

            obs = next_obs
            mask = torch.tensor(step_info["action_mask"], dtype=torch.bool, device=DEVICE)

            for e in range(num_envs):
                ep_returns[e] += reward[e]
                if step_info.get("applied", [False]*num_envs)[e]:
                    total_rules += 1
                if dones[e]:
                    completed_returns.append(ep_returns[e])
                    ep_returns[e] = 0.0

        # GAE
        with torch.no_grad():
            next_obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            next_val = agent.get_value(next_obs_t).squeeze(-1)
            advantages = torch.zeros_like(rew_buf)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - torch.tensor(dones, dtype=torch.float32, device=DEVICE)
                    nextvalues = next_val
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = val_buf[t + 1]
                delta = rew_buf[t] + 0.99 * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + val_buf

        # Flatten for PPO update
        b_obs = obs_buf.reshape(-1, 80)
        b_act = act_buf.reshape(-1)
        b_logp = logp_buf.reshape(-1)
        b_adv = (advantages.reshape(-1) - advantages.mean()) / (advantages.std() + 1e-8)
        b_ret = returns.reshape(-1)
        b_val = val_buf.reshape(-1)
        b_mask = mask_buf.reshape(-1, 160)

        inds = np.arange(batch_size)
        for epoch in range(4):
            np.random.shuffle(inds)
            for start in range(0, batch_size, 256):
                end = start + 256
                mbinds = inds[start:end]

                _, newlogp, entropy, newval = agent.get_action_and_value(b_obs[mbinds], action=b_act[mbinds], action_mask=b_mask[mbinds])
                logratio = newlogp - b_logp[mbinds]
                ratio = logratio.exp()

                pg_loss1 = -b_adv[mbinds] * ratio
                pg_loss2 = -b_adv[mbinds] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newval.squeeze(-1) - b_ret[mbinds]) ** 2).mean()
                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        y_true = returns.cpu().numpy().flatten()
        y_pred = val_buf.cpu().numpy().flatten()
        var_y = np.var(y_true)
        expl_var = 1.0 - (np.var(y_true - y_pred) / (var_y + 1e-8)) if var_y > 1e-8 else 0.0
        mean_rew = np.mean(completed_returns[-50:]) if completed_returns else 0.0

        if update % 5 == 0 or update == num_updates:
            print(f" [MLP] Update {update:>2d}/{num_updates} | Steps: {global_step:>7,d} | ExplVar: {expl_var:+.4f} | Ep Rew: {mean_rew:+.4f} | Rules: {total_rules:>3d}")

        csv_writer.writerow([update, global_step, mean_rew, expl_var, total_rules])
        csv_file.flush()

    csv_file.close()
    envs.close()
    print(" [+] MLP Baseline Training Finished!\n")


def train_gnn_model(
    pretrained_path: Optional[str] = None,
    lr: float = 3e-4,
    num_steps: int = 256,
    clip_coef: float = 0.1,
    seed: int = 0,
    total_steps: int = 100000,
    csv_out: str = "gnn_heavy_curve.csv",
    tag: str = "GNN_Random"
):
    print("\n" + "=" * 70)
    print(f" RUNNING GNN POLICY ({tag}) ON DEPENDENCY-HEAVY CORPUS")
    print("=" * 70)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_envs = 4
    batch_size = num_envs * num_steps
    num_updates = math.ceil(total_steps / batch_size)

    envs = GraphVectorEnv([
        lambda idx=i: make_heavy_env("SuperoptEnv-v0", seed=seed + idx, use_graph=True)
        for i in range(num_envs)
    ])

    agent = GNNActorCritic(
        inst_in_dim=177,
        reg_in_dim=1,
        hidden_dim=128,
        embed_dim=256,
        num_rules=10,
        max_len=16,
        num_heads=8,
        num_layers=3
    ).to(DEVICE)

    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, weights_only=True)
        agent.encoder.load_state_dict(state_dict)
        print(f" [+] Successfully loaded pre-trained GAT Encoder weights from {pretrained_path}!")

    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)
    rollout_buffer = GraphRolloutBuffer(num_steps=num_steps, num_envs=num_envs)

    csv_file = open(csv_out, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["update", "global_step", "episode_reward_mean", "explained_variance", "rules_applied"])

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
                mb_mask = mb.action_mask.reshape(-1, 160).to(DEVICE)

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

        if update % 5 == 0 or update == num_updates:
            print(f" [{tag}] Update {update:>2d}/{num_updates} | Steps: {global_step:>7,d} | ExplVar: {expl_var:+.4f} | Ep Rew: {mean_rew:+.4f} | Rules: {total_rules:>3d}")

        csv_writer.writerow([update, global_step, mean_rew, expl_var, total_rules])
        csv_file.flush()

    csv_file.close()
    envs.close()
    print(f" [+] {tag} Training Finished!\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100000)
    args = parser.parse_args()

    total_steps = args.steps

    # 1. MLP Baseline
    train_mlp_baseline(total_steps=total_steps, csv_out="mlp_heavy_curve.csv")

    # 2. GNN (Random Init)
    train_gnn_model(pretrained_path=None, total_steps=total_steps, csv_out="gnn_random_heavy_curve.csv", tag="GNN_Random")

    # 3. GNN (Pre-trained GAT Encoder)
    train_gnn_model(pretrained_path="pretrained_gat_encoder.pt", total_steps=total_steps, csv_out="gnn_pretrained_heavy_curve.csv", tag="GNN_Pretrained")


if __name__ == "__main__":
    main()
