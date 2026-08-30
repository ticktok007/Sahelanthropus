#!/usr/bin/env python3
"""
run_ppo_100k_benchmark.py — PPO Benchmark with All Diagnostic Fixes Applied

Fixes Applied:
  1. Per-slot action mask (160-element) — no more repeat_interleave
  2. RunningMeanStd observation normalization — prevents value explosion
  3. Stabilized critic — matched LR (3e-4), vf_coef=0.5, value clipping, separate trunks
  4. Standard RL explained variance: 1 - Var(y - ŷ) / Var(y)
  5. Proper episode return tracking — includes zero-reward (failed) episodes
  6. Diverse corpus generation — exercises all 10 rules
"""

import csv
import json
import math
import os
import random
import time
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import wandb
from ppo_superopt import PPOSuperoptTrainer, ActorCritic, make_env, DEVICE
from running_mean_std import RunningMeanStd
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

CSV_LOG_FILE = "ppo_100k_learning_curve.csv"
JSON_LOG_FILE = "ppo_100k_learning_curve.json"
TENSORBOARD_LOG_DIR = "runs/ppo_superopt_100k"


def compute_explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Standard RL explained variance: 1 - Var(y_true - y_pred) / Var(y_true).

    Returns a float in [-inf, 1.0]. A value of 1.0 means perfect prediction.
    A value of 0.0 means the predictor is no better than predicting the mean.
    """
    var_y = torch.var(y_true)
    if var_y < 1e-8:
        return 0.0
    return float((1.0 - torch.var(y_true - y_pred) / var_y).item())


def generate_diverse_corpus() -> List[Dict[str, Any]]:
    """
    Generates a diverse corpus exercising all 10 peephole rules.
    Each entry produces a unique program structure so the agent
    encounters varied observations and reward signals.
    """
    corpus = []

    # Rule 0: mul x, 2^k -> slli x, k (various powers of 2)
    for k in [1, 2, 3, 4, 5]:
        val = 2 ** k
        corpus.append({
            "id": f"rule0_mul_pow2_{k}",
            "name": f"mul_by_{val}",
            "assembly": f".section .text\n.globl _start\n_start:\n    li t0, 42\n    li t1, {val}\n    mul t2, t0, t1\n    li a0, 0\n    li a7, 93\n    ecall\n"
        })

    # Rule 1: div x, 2^k -> srai x, k
    for k in [1, 2, 3, 4]:
        val = 2 ** k
        corpus.append({
            "id": f"rule1_div_pow2_{k}",
            "name": f"div_by_{val}",
            "assembly": f".section .text\n.globl _start\n_start:\n    li t0, 100\n    li t1, {val}\n    div t2, t0, t1\n    li a0, 0\n    li a7, 93\n    ecall\n"
        })

    # Rule 2: addi x, 0 -> nop
    for reg_src, reg_dst in [("t0", "t1"), ("a0", "a1"), ("s0", "s1")]:
        corpus.append({
            "id": f"rule2_add0_{reg_src}_{reg_dst}",
            "name": f"add_zero_{reg_src}",
            "assembly": f".section .text\n.globl _start\n_start:\n    li {reg_src}, 55\n    addi {reg_dst}, {reg_src}, 0\n    li a0, 0\n    li a7, 93\n    ecall\n"
        })

    # Rule 3: mul x, zero -> li rd, 0
    for reg in ["t0", "t1", "t2"]:
        corpus.append({
            "id": f"rule3_mul0_{reg}",
            "name": f"mul_zero_{reg}",
            "assembly": f".section .text\n.globl _start\n_start:\n    li {reg}, 77\n    mul {reg}, {reg}, zero\n    li a0, 0\n    li a7, 93\n    ecall\n"
        })

    # Rule 4: xor x, x -> li rd, 0
    for reg in ["t0", "t1", "s0"]:
        corpus.append({
            "id": f"rule4_xor_self_{reg}",
            "name": f"xor_self_{reg}",
            "assembly": f".section .text\n.globl _start\n_start:\n    li {reg}, 33\n    xor {reg}, {reg}, {reg}\n    li a0, 0\n    li a7, 93\n    ecall\n"
        })

    # Rule 5: sub x, x -> li rd, 0
    for reg in ["t0", "t1", "s0"]:
        corpus.append({
            "id": f"rule5_sub_self_{reg}",
            "name": f"sub_self_{reg}",
            "assembly": f".section .text\n.globl _start\n_start:\n    li {reg}, 44\n    sub {reg}, {reg}, {reg}\n    li a0, 0\n    li a7, 93\n    ecall\n"
        })

    # Rule 6: and x, x -> mv (or nop if rd==rs1)
    for reg in ["t0", "t1", "s1"]:
        corpus.append({
            "id": f"rule6_and_self_{reg}",
            "name": f"and_self_{reg}",
            "assembly": f".section .text\n.globl _start\n_start:\n    li {reg}, 99\n    and {reg}, {reg}, {reg}\n    li a0, 0\n    li a7, 93\n    ecall\n"
        })

    # Rule 7: or x, x -> mv (or nop if rd==rs1)
    for reg in ["t0", "t1", "s1"]:
        corpus.append({
            "id": f"rule7_or_self_{reg}",
            "name": f"or_self_{reg}",
            "assembly": f".section .text\n.globl _start\n_start:\n    li {reg}, 88\n    or {reg}, {reg}, {reg}\n    li a0, 0\n    li a7, 93\n    ecall\n"
        })

    # Rule 8: add(sub(X, Y), Y) -> X
    corpus.append({
        "id": "rule8_add_sub_cancel_1",
        "name": "add_sub_cancel_t0_t1",
        "assembly": ".section .text\n.globl _start\n_start:\n    li t0, 50\n    li t1, 20\n    sub t2, t0, t1\n    add t2, t2, t1\n    li a0, 0\n    li a7, 93\n    ecall\n"
    })
    corpus.append({
        "id": "rule8_add_sub_cancel_2",
        "name": "add_sub_cancel_a0_a1",
        "assembly": ".section .text\n.globl _start\n_start:\n    li a0, 100\n    li a1, 30\n    sub a2, a0, a1\n    add a2, a2, a1\n    li a0, 0\n    li a7, 93\n    ecall\n"
    })

    # Rule 9: slli then srli same k -> andi mask
    for k in [1, 2, 4, 8]:
        corpus.append({
            "id": f"rule9_sll_srl_k{k}",
            "name": f"slli_srli_k{k}",
            "assembly": f".section .text\n.globl _start\n_start:\n    li t0, 255\n    slli t1, t0, {k}\n    srli t1, t1, {k}\n    li a0, 0\n    li a7, 93\n    ecall\n"
        })

    # Multi-rule programs (more complex, agent must pick the right one)
    corpus.append({
        "id": "multi_rule_mul_add0",
        "name": "mul_pow2_and_add0",
        "assembly": ".section .text\n.globl _start\n_start:\n    li t0, 42\n    li t1, 4\n    mul t2, t0, t1\n    addi t2, t2, 0\n    li a0, 0\n    li a7, 93\n    ecall\n"
    })
    corpus.append({
        "id": "multi_rule_xor_or",
        "name": "xor_self_and_or_self",
        "assembly": ".section .text\n.globl _start\n_start:\n    li t0, 55\n    xor t0, t0, t0\n    li t1, 77\n    or t1, t1, t1\n    li a0, 0\n    li a7, 93\n    ecall\n"
    })

    return corpus


class PPOFixedBenchmarkTrainer(PPOSuperoptTrainer):

    def __init__(self, num_envs: int = 4, num_steps: int = 512, lr: float = 3e-4, clip_coef: float = 0.2):
        super().__init__(num_envs=num_envs, num_steps=num_steps, lr=lr, clip_coef=clip_coef)

        self.lr = lr
        self.clip_coef = clip_coef
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)
        self.vf_coef = 0.5

        # Fix 2: Observation normalizer
        self.obs_rms = RunningMeanStd(shape=(80,))

    def _normalize_obs(self, obs_np: np.ndarray) -> np.ndarray:
        """Normalize observations using running mean/std."""
        self.obs_rms.update(obs_np)
        return self.obs_rms.normalize(obs_np, clip_obs=10.0)

    def train_benchmark_100k(
        self,
        seed: int = 0,
        total_steps_target: int = 100000,
        run_tag: Optional[str] = None,
        group_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # Set Random Seeds for Reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        total_updates = int(np.ceil(total_steps_target / self.batch_size))
        writer = SummaryWriter(log_dir=f"{TENSORBOARD_LOG_DIR}_seed{seed}")

        if run_tag is None:
            run_tag = f"ppo_mlp_lr{self.lr}_ns{self.num_steps}_clip{self.clip_coef}_seed{seed}"
        if group_name is None:
            group_name = f"sweep_lr{self.lr}_ns{self.num_steps}_clip{self.clip_coef}"

        tags_list = [f"lr{self.lr}", f"ns{self.num_steps}", f"clip{self.clip_coef}", f"seed{seed}"]

        # Initialize Weights & Biases with Tag & Grouping (fallback to offline if not logged in)
        try:
            run = wandb.init(
                project="ppo_superopt_100k",
                name=run_tag,
                group=group_name,
                tags=tags_list,
                config={
                    "seed": seed,
                    "total_steps_target": total_steps_target,
                    "batch_size": self.batch_size,
                    "minibatch_size": self.minibatch_size,
                    "num_envs": self.num_envs,
                    "num_steps": self.num_steps,
                    "learning_rate": self.lr,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "clip_coef": self.clip_coef,
                    "vf_coef": self.vf_coef,
                    "ent_coef": self.ent_coef,
                    "device": str(DEVICE),
                },
                sync_tensorboard=True,
                reinit=True,
            )
        except Exception as e:
            print(f"[!] W&B online mode not authenticated ({e}). Running in offline mode.")
            run = wandb.init(
                project="ppo_superopt_100k",
                name=run_tag,
                group=group_name,
                tags=tags_list,
                config={
                    "seed": seed,
                    "total_steps_target": total_steps_target,
                    "batch_size": self.batch_size,
                    "minibatch_size": self.minibatch_size,
                    "num_envs": self.num_envs,
                    "num_steps": self.num_steps,
                    "learning_rate": self.lr,
                    "gamma": self.gamma,
                    "gae_lambda": self.gae_lambda,
                    "clip_coef": self.clip_coef,
                    "vf_coef": self.vf_coef,
                    "ent_coef": self.ent_coef,
                    "device": str(DEVICE),
                },
                mode="offline",
                sync_tensorboard=True,
                reinit=True,
            )

        print("======================================================================")
        print(f" PPO Seed Benchmark Run: {run_tag}")
        print("======================================================================")
        print(f" Seed             : {seed}")
        print(f" W&B Run Tag      : {run_tag}")
        print(f" W&B Group        : {group_name}")
        print(f" Compute Device   : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        print(f" Total Updates    : {total_updates} updates ({total_updates * self.batch_size:,} total steps)")

        # Rollout Buffers on GPU
        obs_buf = torch.zeros((self.num_steps, self.num_envs, 80), dtype=torch.float32, device=DEVICE)
        act_buf = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long, device=DEVICE)
        logprob_buf = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        rew_buf = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        done_buf = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        val_buf = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        # Fix 1: mask_buf is now shape (160,) not (10,)
        mask_buf = torch.zeros((self.num_steps, self.num_envs, 160), dtype=torch.bool, device=DEVICE)

        next_obs_raw, info = self.envs.reset(seed=seed)
        # Fix 2: Normalize observations
        next_obs_norm = self._normalize_obs(next_obs_raw)
        next_obs = torch.tensor(next_obs_norm, dtype=torch.float32, device=DEVICE)
        next_mask = torch.tensor(info["action_mask"], dtype=torch.bool, device=DEVICE)
        next_done = torch.zeros(self.num_envs, dtype=torch.float32, device=DEVICE)

        start_time = time.time()

        history: List[Dict[str, Any]] = []
        # Fix 5: Track cumulative episode returns properly
        episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        episode_applied_rules = [[] for _ in range(self.num_envs)]
        all_episode_returns: List[float] = []

        for update in range(1, total_updates + 1):
            total_env_steps = update * self.batch_size
            update_episode_returns = []
            update_applied_rule_ids: List[int] = []
            update_completed_episodes_rule_firings: List[List[int]] = []
            rules_applied = 0
            entropy_list = []

            # 1. Rollout Collection
            for step in range(self.num_steps):
                obs_buf[step] = next_obs
                done_buf[step] = next_done
                mask_buf[step] = next_mask

                with torch.no_grad():
                    action, logprob, entropy, value = self.agent.get_action_and_value(next_obs, action_mask=next_mask)
                    val_buf[step] = value
                    entropy_list.append(entropy.mean().item())

                act_buf[step] = action
                logprob_buf[step] = logprob

                cpu_actions = action.cpu().numpy()
                next_obs_raw, reward_cpu, term_cpu, trunc_cpu, info_cpu = self.envs.step(cpu_actions)

                rew_buf[step] = torch.tensor(reward_cpu, dtype=torch.float32, device=DEVICE)

                # Fix 2: Normalize observations
                next_obs_norm = self._normalize_obs(next_obs_raw)
                next_obs = torch.tensor(next_obs_norm, dtype=torch.float32, device=DEVICE)
                next_mask = torch.tensor(info_cpu["action_mask"], dtype=torch.bool, device=DEVICE)

                done_cpu = np.logical_or(term_cpu, trunc_cpu)
                next_done = torch.tensor(done_cpu, dtype=torch.float32, device=DEVICE)

                rules_applied += int(sum(info_cpu.get("applied", [False]*self.num_envs)))

                # Track rule firings per env step and per completed episode
                applied_flags = info_cpu.get("applied", [False]*self.num_envs)
                rule_ids = info_cpu.get("rule_id", [0]*self.num_envs)

                # Fix 5: Track true cumulative episode returns & rule firing histograms
                episode_returns += reward_cpu
                for env_idx in range(self.num_envs):
                    if applied_flags[env_idx]:
                        r_id = int(rule_ids[env_idx])
                        update_applied_rule_ids.append(r_id)
                        episode_applied_rules[env_idx].append(r_id)

                    if done_cpu[env_idx]:
                        update_episode_returns.append(float(episode_returns[env_idx]))
                        episode_returns[env_idx] = 0.0
                        update_completed_episodes_rule_firings.append(list(episode_applied_rules[env_idx]))
                        episode_applied_rules[env_idx] = []

            all_episode_returns.extend(update_episode_returns)
            # Keep a rolling window for display
            recent_returns = all_episode_returns[-500:] if len(all_episode_returns) > 500 else all_episode_returns

            # 2. GAE-Lambda Advantage Computation
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs)
                advantages = torch.zeros_like(rew_buf, device=DEVICE)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - done_buf[t + 1]
                        nextvalues = val_buf[t + 1]
                    delta = rew_buf[t] + self.gamma * nextvalues * nextnonterminal - val_buf[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + val_buf

            b_obs = obs_buf.reshape(-1, 80)
            b_logprobs = logprob_buf.reshape(-1)
            b_actions = act_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = val_buf.reshape(-1)
            b_masks = mask_buf.reshape(-1, 160)

            # 3. PPO Optimization Epochs & Clip Fraction Tracking
            b_inds = np.arange(self.batch_size)
            clipfracs = []

            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds],
                        action_mask=b_masks[mb_inds],
                        action=b_actions[mb_inds]
                    )

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean().item()
                        clipfracs.append(((ratio - 1.0).abs() > self.clip_coef).float().mean().item())

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Fix 3: Value loss with PPO clipping
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -self.clip_coef, self.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    entropy_loss = entropy.mean()

                    loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            # Fix 4: Standard RL explained variance (not Pearson r)
            with torch.no_grad():
                updated_values = self.agent.get_value(b_obs)
                expl_var = compute_explained_variance(b_returns, updated_values)

            current_clip_frac = float(np.mean(clipfracs))
            elapsed = time.time() - start_time
            sps = total_env_steps / elapsed

            # Fix 5: episode_reward_mean now includes zero-reward episodes
            episode_reward_mean = float(np.mean(recent_returns)) if len(recent_returns) > 0 else 0.0
            mean_entropy = float(np.mean(entropy_list))

            rule_counts = {f"rules/rule_{r}_firings": update_applied_rule_ids.count(r) for r in range(10)}
            ep_firing_counts = [len(firings) for firings in update_completed_episodes_rule_firings]

            record = {
                "update": int(update),
                "total_steps": int(total_env_steps),
                "sps": float(sps),
                "episode_reward_mean": float(episode_reward_mean),
                "episodes_completed": len(update_episode_returns),
                "policy_entropy": float(mean_entropy),
                "clip_fraction": float(current_clip_frac),
                "explained_variance": float(expl_var),
                "rules_applied": int(rules_applied),
                "policy_loss": float(pg_loss.item()),
                "value_loss": float(v_loss.item()),
                "approx_kl": float(approx_kl)
            }
            for r in range(10):
                record[f"rule_{r}_firings"] = update_applied_rule_ids.count(r)
            history.append(record)

            # TensorBoard Logging
            writer.add_scalar("charts/episode_reward_mean", episode_reward_mean, total_env_steps)
            writer.add_scalar("charts/policy_entropy", mean_entropy, total_env_steps)
            writer.add_scalar("charts/rules_applied", rules_applied, total_env_steps)
            writer.add_scalar("charts/sps", sps, total_env_steps)
            writer.add_scalar("losses/clip_fraction", current_clip_frac, total_env_steps)
            writer.add_scalar("losses/explained_variance", expl_var, total_env_steps)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), total_env_steps)
            writer.add_scalar("losses/value_loss", v_loss.item(), total_env_steps)
            writer.add_scalar("losses/approx_kl", approx_kl, total_env_steps)
            for r in range(10):
                writer.add_scalar(f"rules/rule_{r}_firings", update_applied_rule_ids.count(r), total_env_steps)

            # Weights & Biases Logging
            wandb_metrics = {
                "charts/episode_reward_mean": episode_reward_mean,
                "charts/policy_entropy": mean_entropy,
                "charts/rules_applied": rules_applied,
                "charts/sps": sps,
                "losses/clip_fraction": current_clip_frac,
                "losses/explained_variance": expl_var,
                "losses/policy_loss": pg_loss.item(),
                "losses/value_loss": v_loss.item(),
                "losses/approx_kl": approx_kl,
                "rules/rule_firing_histogram": wandb.Histogram(update_applied_rule_ids) if update_applied_rule_ids else wandb.Histogram([0]),
                "rules/per_episode_firing_count_histogram": wandb.Histogram(ep_firing_counts) if ep_firing_counts else wandb.Histogram([0]),
                "rules/mean_rules_fired_per_episode": float(np.mean(ep_firing_counts)) if ep_firing_counts else 0.0,
                "global_step": total_env_steps,
            }
            wandb_metrics.update(rule_counts)
            wandb.log(wandb_metrics, step=update)

            print(f"Update {update:02d}/{total_updates:02d} | Steps: {total_env_steps:7,d} | "
                  f"ExplVar: {expl_var:+7.4f} | ClipFrac: {current_clip_frac:6.4f} | "
                  f"Ep Rew: {episode_reward_mean:+7.4f} | VLoss: {v_loss.item():.6f} | "
                  f"Rules: {rules_applied:3d} | Eps: {len(update_episode_returns)}")

        writer.close()
        wandb.finish()
        self.envs.close()

        # Save CSV & JSON logs
        seed_csv_file = f"ppo_100k_learning_curve_seed{seed}.csv"
        fieldnames = ["update", "total_steps", "sps", "episode_reward_mean", "episodes_completed",
                      "policy_entropy", "clip_fraction", "explained_variance", "rules_applied",
                      "policy_loss", "value_loss", "approx_kl"]
        with open(seed_csv_file, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer_csv.writeheader()
            writer_csv.writerows(history)

        with open(CSV_LOG_FILE, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer_csv.writeheader()
            writer_csv.writerows(history)

        with open(JSON_LOG_FILE, "w") as f:
            json.dump(history, f, indent=2)

        init_expl_var = history[0]["explained_variance"]
        final_expl_var = history[-1]["explained_variance"]
        max_expl_var = max(h["explained_variance"] for h in history)
        init_ep_rew = history[0]["episode_reward_mean"]
        final_ep_rew = history[-1]["episode_reward_mean"]
        max_vloss = max(h["value_loss"] for h in history)

        print("\n======================================================================")
        print(f" SEED {seed} BENCHMARK FINISHED ({run_tag})")
        print("======================================================================")
        print(f" Initial Explained Variance  : {init_expl_var:+.4f}")
        print(f" Final Explained Variance    : {final_expl_var:+.4f} (Target -> 1.0)")
        print(f" Max Explained Variance      : {max_expl_var:+.4f}")
        print(f" Initial Episode Reward      : {init_ep_rew:+.4f}")
        print(f" Final Episode Reward        : {final_ep_rew:+.4f}")
        print(f" Max Value Loss              : {max_vloss:.6f}")
        print(f" Total Episodes Completed    : {len(all_episode_returns)}")
        print(f" Seed CSV Saved              : {seed_csv_file}")
        print("======================================================================\n")

        return {
            "seed": seed,
            "history": history,
            "init_explained_variance": init_expl_var,
            "final_explained_variance": final_expl_var,
            "max_explained_variance": max_expl_var,
        }


def plot_3_seeds_mean_std(all_histories: Dict[int, List[Dict[str, Any]]], output_png: str = "ppo_3seeds_mean_std.png"):
    """
    Plots mean ± std across 3 random seeds for key PPO training metrics.
    """
    seeds = list(all_histories.keys())
    num_updates = len(all_histories[seeds[0]])
    steps = [all_histories[seeds[0]][i]["total_steps"] for i in range(num_updates)]

    metrics = [
        ("explained_variance", "Explained Variance (Target -> 1.0)", "lower right"),
        ("episode_reward_mean", "Episode Reward Mean", "lower right"),
        ("value_loss", "Value Function Loss", "upper right"),
        ("policy_entropy", "Policy Entropy", "upper right"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PPO Superopt Benchmark Across 3 Seeds (Mean ± Std)\nTag: ppo_mlp_lr3e-4_ns512", fontsize=15, fontweight="bold")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for ax_idx, (metric_key, metric_title, legend_loc) in enumerate(metrics):
        row, col = divmod(ax_idx, 2)
        ax = axes[row, col]

        # Stack data across seeds: shape (num_seeds, num_updates)
        matrix = np.array([
            [all_histories[s][i][metric_key] for i in range(num_updates)]
            for s in seeds
        ])

        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)

        # Plot individual seed runs (dashed thin lines)
        for s_idx, seed in enumerate(seeds):
            ax.plot(steps, matrix[s_idx], linestyle="--", alpha=0.4, label=f"Seed {seed}")

        # Plot Mean Line and Shaded Std Band
        ax.plot(steps, mean, color=colors[ax_idx], linewidth=2.5, label="Mean across 3 seeds")
        ax.fill_between(steps, mean - std, mean + std, color=colors[ax_idx], alpha=0.2, label="±1 Std Dev")

        ax.set_title(metric_title, fontsize=12, fontweight="semibold")
        ax.set_xlabel("Total Environment Steps", fontsize=10)
        ax.set_ylabel(metric_key, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc=legend_loc, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"[+] Saved 3-Seed Mean ± Std Plot to: {output_png}")


def run_3_seeds_benchmark(total_steps_target: int = 100000):
    seeds = [0, 1, 2]
    all_histories = {}

    print("======================================================================")
    print(f" STARTING 3-SEED BENCHMARK (Seeds: {seeds})")
    print(f" Tag Format: ppo_mlp_lr3e-4_ns512_seedX")
    print("======================================================================")

    for seed in seeds:
        trainer = PPOFixedBenchmarkTrainer(num_envs=4, num_steps=512)
        res = trainer.train_benchmark_100k(seed=seed, total_steps_target=total_steps_target)
        all_histories[seed] = res["history"]

    # Generate Combined Mean ± Std Plot
    plot_3_seeds_mean_std(all_histories, output_png="ppo_3seeds_mean_std.png")

    # Compute Summary Stats Across 3 Seeds
    final_expl_vars = [all_histories[s][-1]["explained_variance"] for s in seeds]
    max_expl_vars = [max(h["explained_variance"] for h in all_histories[s]) for s in seeds]
    final_ep_rews = [all_histories[s][-1]["episode_reward_mean"] for s in seeds]

    print("\n======================================================================")
    print(" 3-SEED BENCHMARK SUMMARY (Mean ± Std)")
    print("======================================================================")
    print(f" Seeds Evaluated         : {seeds}")
    print(f" Final Explained Variance: {np.mean(final_expl_vars):+.4f} ± {np.std(final_expl_vars):.4f}")
    print(f" Max Explained Variance  : {np.mean(max_expl_vars):+.4f} ± {np.std(max_expl_vars):.4f}")
    print(f" Final Episode Reward    : {np.mean(final_ep_rews):+.4f} ± {np.std(final_ep_rews):.4f}")
    print(f" Plot Graphic Saved      : ppo_3seeds_mean_std.png")
    print("======================================================================\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--clip_coef", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--single", action="store_true", help="Run single seed instead of 3-seed benchmark")
    args = parser.parse_args()

    corpus_path = "corpus_index.jsonl" if os.path.exists("corpus_index.jsonl") else "corpus.json"
    generate_toy = True
    if os.path.exists("corpus_index.jsonl"):
        print("[+] Using full 1.038M+ multi-dataset corpus index (corpus_index.jsonl)")
        generate_toy = False
    elif os.path.exists("corpus.json"):
        try:
            with open("corpus.json", "r") as f:
                existing = json.load(f)
            if len(existing) >= 100:
                print(f"[+] Using existing corpus.json with {len(existing):,} entries (skipping toy corpus generation)")
                generate_toy = False
        except Exception:
            pass

    if generate_toy:
        diverse_corpus = generate_diverse_corpus()
        with open("corpus.json", "w") as f:
            json.dump(diverse_corpus, f, indent=2)
        print(f"[+] Generated diverse corpus with {len(diverse_corpus)} entries covering all 10 rules")
        print("    Saved to: corpus.json\n")

    if args.single:
        # Single-seed run with specified hyperparameters
        tag = args.run_tag or f"ppo_mlp_lr{args.lr}_ns{args.num_steps}_clip{args.clip_coef}_seed{args.seed}"
        group = args.group or f"single_lr{args.lr}_ns{args.num_steps}_clip{args.clip_coef}"
        trainer = PPOFixedBenchmarkTrainer(num_envs=4, num_steps=args.num_steps, lr=args.lr, clip_coef=args.clip_coef)
        trainer.train_benchmark_100k(
            seed=args.seed,
            total_steps_target=args.total_steps,
            run_tag=tag,
            group_name=group
        )
    else:
        run_3_seeds_benchmark(total_steps_target=args.total_steps)
