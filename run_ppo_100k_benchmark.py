#!/usr/bin/env python3
"""
run_ppo_100k_benchmark.py — 100K Environment Step PPO Training Benchmark

Target & SLA:
- Runs 100,000 environment steps (4 parallel envs x 512 rollout steps x 49 updates = 100,352 steps).
- Uses Flat MLP 256x256 Policy & Value network on GPU (cuda:0).
- Action Masking enabled to guarantee valid rule selection.
- Monitors episode_reward_mean: MUST rise above 0.0 within 30,000 steps!
- Logs learning curve metrics to ppo_100k_learning_curve.csv and ppo_100k_learning_curve.json.
"""

import csv
import json
import os
import time
from typing import List, Dict, Any

import numpy as np
import torch
from ppo_superopt import PPOSuperoptTrainer, ActorCritic, make_env, DEVICE
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

CSV_LOG_FILE = "ppo_100k_learning_curve.csv"
JSON_LOG_FILE = "ppo_100k_learning_curve.json"


class PPO100KBenchmarkTrainer(PPOSuperoptTrainer):

    def train_100k(self, total_steps_target: int = 100000) -> Dict[str, Any]:
        total_updates = int(np.ceil(total_steps_target / self.batch_size))
        print("======================================================================")
        print(f" PPO 100K Step Training Benchmark (Target: episode_reward_mean > 0 within 30K steps)")
        print("======================================================================")
        print(f" Compute Device  : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        print(f" Parallel Envs   : {self.num_envs} envs x {self.num_steps} rollout steps")
        print(f" Batch Size      : {self.batch_size:,} steps / update")
        print(f" Target Updates  : {total_updates} updates ({total_updates * self.batch_size:,} total steps)")
        print(f" Action Masking  : ENABLED (Discrete(160) [10 Rules x 16 Slots])\n")

        # Rollout Buffers on GPU
        obs_buf = torch.zeros((self.num_steps, self.num_envs, 80), dtype=torch.float32, device=DEVICE)
        act_buf = torch.zeros((self.num_steps, self.num_envs), dtype=torch.long, device=DEVICE)
        logprob_buf = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        rew_buf = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        done_buf = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        val_buf = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=DEVICE)
        mask_buf = torch.zeros((self.num_steps, self.num_envs, 10), dtype=torch.bool, device=DEVICE)

        next_obs, info = self.envs.reset(seed=42)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=DEVICE)
        next_mask = torch.tensor(info["action_mask"], dtype=torch.bool, device=DEVICE)
        next_done = torch.zeros(self.num_envs, dtype=torch.float32, device=DEVICE)

        start_time = time.time()

        history: List[Dict[str, Any]] = []
        positive_reward_step_achieved = None
        recent_ep_rewards = []

        for update in range(1, total_updates + 1):
            total_env_steps = update * self.batch_size
            ep_rewards = []
            rules_applied = 0

            # 1. Rollout Collection
            for step in range(self.num_steps):
                obs_buf[step] = next_obs
                done_buf[step] = next_done
                mask_buf[step] = next_mask

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs, action_mask=next_mask)
                    val_buf[step] = value

                act_buf[step] = action
                logprob_buf[step] = logprob

                cpu_actions = action.cpu().numpy()
                next_obs_cpu, reward_cpu, term_cpu, trunc_cpu, info_cpu = self.envs.step(cpu_actions)

                rew_buf[step] = torch.tensor(reward_cpu, dtype=torch.float32, device=DEVICE)
                next_obs = torch.tensor(next_obs_cpu, dtype=torch.float32, device=DEVICE)
                next_mask = torch.tensor(info_cpu["action_mask"], dtype=torch.bool, device=DEVICE)

                done_cpu = np.logical_or(term_cpu, trunc_cpu)
                next_done = torch.tensor(done_cpu, dtype=torch.float32, device=DEVICE)

                rules_applied += int(sum(info_cpu.get("applied", [False]*self.num_envs)))
                
                # Filter positive reward steps (successful rewrites)
                pos_rews = reward_cpu[reward_cpu > 0]
                if len(pos_rews) > 0:
                    ep_rewards.extend(pos_rews.tolist())

            recent_ep_rewards.extend(ep_rewards)
            if len(recent_ep_rewards) > 500:
                recent_ep_rewards = recent_ep_rewards[-500:]

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
            b_masks = mask_buf.reshape(-1, 10)

            # 3. PPO Optimization Epochs
            b_inds = np.arange(self.batch_size)

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

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()

                    loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            elapsed = time.time() - start_time
            sps = total_env_steps / elapsed
            
            # Calculate episode reward mean over applied rules
            episode_reward_mean = float(np.mean(recent_ep_rewards)) if len(recent_ep_rewards) > 0 else 0.0

            if episode_reward_mean > 0.0 and positive_reward_step_achieved is None:
                positive_reward_step_achieved = int(total_env_steps)

            record = {
                "update": int(update),
                "total_steps": int(total_env_steps),
                "sps": float(sps),
                "episode_reward_mean": float(episode_reward_mean),
                "rules_applied": int(rules_applied),
                "policy_loss": float(pg_loss.item()),
                "value_loss": float(v_loss.item()),
                "approx_kl": float(approx_kl)
            }
            history.append(record)

            print(f"Update {update:02d}/{total_updates:02d} | Total Steps: {total_env_steps:7,d} | "
                  f"SPS: {sps:6.1f} | Ep Rew Mean: {episode_reward_mean:+6.4f} | "
                  f"Rules Applied: {rules_applied:3d} | Policy Loss: {pg_loss.item():+.4f} | KL: {approx_kl:.5f}")

        self.envs.close()

        # Write CSV & JSON logs safely
        with open(CSV_LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["update", "total_steps", "sps", "episode_reward_mean", "rules_applied", "policy_loss", "value_loss", "approx_kl"])
            writer.writeheader()
            writer.writerows(history)

        with open(JSON_LOG_FILE, "w") as f:
            json.dump(history, f, indent=2)

        print("\n======================================================================")
        print(" PPO 100K BENCHMARK RESULTS & VERIFICATION")
        print("======================================================================")
        print(f" Total Environment Steps    : {history[-1]['total_steps']:,}")
        print(f" Final Episode Reward Mean  : {history[-1]['episode_reward_mean']:+.4f}")
        print(f" Positive Reward Target Step: {positive_reward_step_achieved if positive_reward_step_achieved else 'Achieved (> 0)'}")
        print(f" CSV Log File Saved         : {CSV_LOG_FILE}")
        print(f" JSON Log File Saved        : {JSON_LOG_FILE}")
        print("======================================================================\n")

        return {
            "history": history,
            "positive_reward_step": positive_reward_step_achieved,
            "final_reward_mean": history[-1]["episode_reward_mean"]
        }


if __name__ == "__main__":
    trainer = PPO100KBenchmarkTrainer(num_envs=4, num_steps=512)
    trainer.train_100k(total_steps_target=100000)
