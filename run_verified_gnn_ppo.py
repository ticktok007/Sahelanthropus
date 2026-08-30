#!/usr/bin/env python3
"""
run_verified_gnn_ppo.py — Train PPO with GATEncoder + PolicyHead + ValueHead on GPU (100K steps).

Handles native PyG HeteroData graph observations directly in GraphRolloutBuffer
and iterates mini-batches using torch_geometric.loader.DataLoader.
"""

import math
import os
import sys
import time
import csv
import argparse
from typing import Tuple, List, Dict, Any, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import wandb

from superopt_env import SuperoptEnv
from ppo_gnn_actor_critic import GNNActorCritic
from ppo_gnn_superopt import make_env, DEVICE
from graph_ppo_rollout_buffer import GraphRolloutBuffer
from utils_seed import set_seed
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader


class GraphVectorEnv:
    """
    Vectorized environment wrapper supporting native PyG HeteroData graph observations.
    """
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)

    def reset(self, seed: int = 0) -> Tuple[List[HeteroData], Dict[str, Any]]:
        obs_list = []
        info_list = []
        for i, env in enumerate(self.envs):
            o, inf = env.reset(seed=seed + i)
            obs_list.append(o)
            info_list.append(inf)
        masks = np.stack([inf["action_mask"] for inf in info_list], axis=0)
        return obs_list, {"action_mask": masks, "infos": info_list}

    def step(self, actions: np.ndarray) -> Tuple[List[HeteroData], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs_list = []
        rewards = []
        terms = []
        truncs = []
        info_list = []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, inf = env.step(actions[i])
            if term or trunc:
                o_reset, _ = env.reset()
                obs_list.append(o_reset)
            else:
                obs_list.append(o)
            rewards.append(r)
            terms.append(term)
            truncs.append(trunc)
            info_list.append(inf)

        masks = np.stack([inf["action_mask"] for inf in info_list], axis=0)
        return (
            obs_list,
            np.array(rewards, dtype=np.float32),
            np.array(terms, dtype=bool),
            np.array(truncs, dtype=bool),
            {
                "action_mask": masks,
                "applied": [inf.get("applied", False) for inf in info_list]
            }
        )

    def close(self):
        for env in self.envs:
            env.close()


def train_gnn_ppo_100k(
    lr: float = 3e-4,
    num_steps: int = 256,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.1,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    seed: int = 0,
    total_steps_target: int = 100000,
    group_name: str = "gnn_verified_phase"
):
    run_tag = f"ppo_gnn_z3gated_{total_steps_target//1000}k_seed{seed}"
    print("=" * 70)
    print(f" PHASE 5: GNN PPO TRAINING BENCHMARK WITH Z3 GATING ({total_steps_target//1000}K STEPS)")
    print("=" * 70)
    print(f" Run Tag           : {run_tag}")
    print(f" Group             : {group_name}")
    print(f" Seed              : {seed}")
    print(f" Compute Device    : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Architecture      : HeteroGATEncoder + PolicyHead(256->160) + ValueHead(256->1)")
    print(f" Z3 Gating         : Active (UNSAT -> QEMU Reward, SAT -> r=-1.0, TIMEOUT -> r=-0.1)")
    print(f" Rollout Buffer    : GraphRolloutBuffer (Native HeteroData Graphs)")
    print(f" Batching          : PyG DataLoader (minibatch_size = 256)")
    print(f" Hyperparameters   : lr={lr}, num_steps={num_steps}, clip_coef={clip_coef}")
    print(f" Target Steps      : {total_steps_target:,}")
    print("=" * 70 + "\n")

    # Set unified seeds across random, numpy, torch, cuda, cudnn
    set_seed(seed)

    num_envs = 4
    batch_size = num_envs * num_steps
    num_updates = math.ceil(total_steps_target / batch_size)

    # Initialize GraphVectorEnv
    def make_graph_env(env_id: str, s: int) -> gym.Env:
        env = gym.make(env_id)
        env.unwrapped.use_graph_obs = True
        env.action_space.seed(s)
        return env

    env_fns = [lambda idx=i: make_graph_env("SuperoptEnv-v0", s=seed + idx) for i in range(num_envs)]
    envs = GraphVectorEnv(env_fns)

    # Initialize GNN Agent
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

    if os.path.exists("pretrained_gat_encoder.pt"):
        try:
            state_dict = torch.load("pretrained_gat_encoder.pt", weights_only=True)
            agent.encoder.load_state_dict(state_dict)
            print(" [+] Loaded pre-trained GAT Encoder weights from pretrained_gat_encoder.pt!")
        except Exception as e:
            print(f" [!] Could not load pretrained_gat_encoder.pt: {e}")

    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    # W&B Logging
    wandb.init(
        project="ppo_superopt_gnn_verified",
        group=group_name,
        name=run_tag,
        config={
            "architecture": "HeteroGATEncoder_PolicyHead_ValueHead",
            "lr": lr,
            "num_steps": num_steps,
            "clip_coef": clip_coef,
            "seed": seed,
            "num_envs": num_envs,
            "batch_size": batch_size,
            "total_steps": total_steps_target,
        },
        reinit=True
    )

    # Graph Rollout Buffer
    rollout_buffer = GraphRolloutBuffer(num_steps=num_steps, num_envs=num_envs)

    # CSV Logging
    csv_filename = f"gnn_ppo_dataloader_seed{seed}.csv"
    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["update", "global_step", "episode_reward_mean", "explained_variance", "value_loss", "policy_loss", "clip_fraction", "rules_applied"])

    global_step = 0
    start_time = time.time()

    # Reset Environment
    next_obs_list, infos = envs.reset(seed=seed)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=DEVICE)

    if isinstance(infos, dict) and "action_mask" in infos:
        next_mask = torch.tensor(infos["action_mask"], dtype=torch.bool, device=DEVICE)
    else:
        next_mask = torch.ones((num_envs, 160), dtype=torch.bool, device=DEVICE)

    ep_returns = [0.0] * num_envs
    ep_lengths = [0] * num_envs
    completed_ep_returns = []
    total_rules_applied = 0

    print("Starting PPO GNN Rollout Training Loop with PyG DataLoader...\n")

    for update in range(1, num_updates + 1):
        # Anneal Learning Rate linearly
        frac = 1.0 - (update - 1.0) / num_updates
        optimizer.param_groups[0]["lr"] = frac * lr

        # -------------------------------------------------------------------
        # 1. Rollout Phase (Collect HeteroData Graphs directly)
        # -------------------------------------------------------------------
        for step in range(num_steps):
            global_step += num_envs

            # Batch current HeteroData observations for GPU inference
            step_batch = Batch.from_data_list(next_obs_list)
            x_dict = {k: v.to(DEVICE) for k, v in step_batch.x_dict.items()}
            edge_index_dict = {k: v.to(DEVICE) for k, v in step_batch.edge_index_dict.items()}
            batch_dict = {
                'inst': step_batch['inst'].batch.to(DEVICE),
                'reg': step_batch['reg'].batch.to(DEVICE)
            }

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    x_dict,
                    edge_index_dict,
                    action_mask=next_mask,
                    batch_dict=batch_dict
                )

            # Store transition in GraphRolloutBuffer
            rollout_buffer.insert(
                step=step,
                obs_graphs=next_obs_list,
                actions=action,
                logprobs=logprob,
                rewards=torch.zeros(num_envs),  # Updated after env step
                dones=next_done,
                values=value,
                action_masks=next_mask
            )

            # Environment Step
            next_obs_list, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=DEVICE)
            rollout_buffer.rewards[step] = torch.tensor(reward, dtype=torch.float32)

            if isinstance(infos, dict) and "action_mask" in infos:
                next_mask = torch.tensor(infos["action_mask"], dtype=torch.bool, device=DEVICE)
            else:
                next_mask = torch.ones((num_envs, 160), dtype=torch.bool, device=DEVICE)

            for e in range(num_envs):
                ep_returns[e] += reward[e]
                ep_lengths[e] += 1

                if "applied" in infos and infos["applied"][e]:
                    total_rules_applied += 1

                if next_done_np[e]:
                    completed_ep_returns.append(ep_returns[e])
                    ep_returns[e] = 0.0
                    ep_lengths[e] = 0

        # -------------------------------------------------------------------
        # 2. GAE Advantage Calculation
        # -------------------------------------------------------------------
        with torch.no_grad():
            next_batch = Batch.from_data_list(next_obs_list)
            x_next = {k: v.to(DEVICE) for k, v in next_batch.x_dict.items()}
            edge_next = {k: v.to(DEVICE) for k, v in next_batch.edge_index_dict.items()}
            b_next = {
                'inst': next_batch['inst'].batch.to(DEVICE),
                'reg': next_batch['reg'].batch.to(DEVICE)
            }
            next_value = agent.get_value(x_next, edge_next, batch_dict=b_next).cpu()

            advantages = torch.zeros_like(rollout_buffer.rewards)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done.cpu()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - rollout_buffer.dones[t + 1]
                    nextvalues = rollout_buffer.values[t + 1]
                delta = rollout_buffer.rewards[t] + gamma * nextvalues * nextnonterminal - rollout_buffer.values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + rollout_buffer.values

        # -------------------------------------------------------------------
        # 3. PPO Optimization Phase using PyG DataLoader
        # -------------------------------------------------------------------
        # Normalize advantages
        advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PyG DataLoader over mini-batches
        dataloader = rollout_buffer.get_dataloader(advantages_norm, returns, minibatch_size=256)
        clipfracs = []

        for epoch in range(4):
            for mb in dataloader:
                mb_x = {k: v.to(DEVICE) for k, v in mb.x_dict.items()}
                mb_edge = {k: v.to(DEVICE) for k, v in mb.edge_index_dict.items()}
                mb_batch = {
                    'inst': mb['inst'].batch.to(DEVICE),
                    'reg': mb['reg'].batch.to(DEVICE)
                }

                mb_actions = mb.action.to(DEVICE)
                mb_logprobs = mb.logprob.to(DEVICE)
                mb_adv = mb.advantage.to(DEVICE)
                mb_ret = mb.return_val.to(DEVICE)
                mb_val = mb.value.to(DEVICE)
                mb_masks = mb.action_mask.reshape(-1, 160).to(DEVICE)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_x,
                    mb_edge,
                    action=mb_actions,
                    action_mask=mb_masks,
                    batch_dict=mb_batch
                )

                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

                # Policy Loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss with Clipping
                v_loss_unclipped = (newvalue - mb_ret) ** 2
                v_clipped = mb_val + torch.clamp(newvalue - mb_val, -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - mb_ret) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy Loss
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        # -------------------------------------------------------------------
        # 4. Metrics Logging & Diagnostics
        # -------------------------------------------------------------------
        y_true = returns.numpy().flatten()
        y_pred = rollout_buffer.values.numpy().flatten()
        var_y = np.var(y_true)
        expl_var = 1.0 - (np.var(y_true - y_pred) / (var_y + 1e-8)) if var_y > 1e-8 else 0.0

        mean_ep_rew = np.mean(completed_ep_returns[-50:]) if completed_ep_returns else 0.0
        mean_clip_frac = np.mean(clipfracs)

        sps = int(global_step / (time.time() - start_time))

        print(f"Update {update:>2d}/{num_updates} | Steps: {global_step:>7,d} | "
              f"ExplVar: {expl_var:+.4f} | ClipFrac: {mean_clip_frac:.4f} | "
              f"Ep Rew: {mean_ep_rew:+.4f} | VLoss: {v_loss.item():.6f} | "
              f"Rules: {total_rules_applied:>3d} | SPS: {sps}")

        csv_writer.writerow([update, global_step, mean_ep_rew, expl_var, v_loss.item(), pg_loss.item(), mean_clip_frac, total_rules_applied])
        csv_file.flush()

        wandb.log({
            "global_step": global_step,
            "charts/episode_reward_mean": mean_ep_rew,
            "losses/explained_variance": expl_var,
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/clip_fraction": mean_clip_frac,
            "charts/rules_applied": total_rules_applied,
            "charts/sps": sps
        })

    csv_file.close()
    envs.close()
    wandb.finish()

    print(f"\n[+] GNN PPO DataLoader Benchmark Complete! Logs saved to {csv_filename}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--clip_coef", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=100000)
    args = parser.parse_args()

    train_gnn_ppo_100k(
        lr=args.lr,
        num_steps=args.num_steps,
        clip_coef=args.clip_coef,
        seed=args.seed,
        total_steps_target=args.total_steps
    )
