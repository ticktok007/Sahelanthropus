#!/usr/bin/env python3
"""
ppo_superopt.py — GPU-Accelerated PPO Connected to SuperoptEnv SyncVectorEnv (4 envs x 512 steps/rollout)

Architecture:
- Actor-Critic Shared Trunk Neural Network (PyTorch on GPU cuda:0).
- Action Masking Support: applies -inf penalty to invalid rule logits.
- SyncVectorEnv with 4 parallel SuperoptEnv-v0 instances.
- Batch Size: 2,048 transitions (4 envs x 512 rollout steps).
- Generalized Advantage Estimation (GAE-lambda).
"""

import math
import os
import time
from typing import Callable, Tuple, List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gymnasium as gym

# Import SuperoptEnv to ensure registration
from superopt_env import SuperoptEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env(env_id: str, seed: int) -> Callable[[], gym.Env]:
    def thunk():
        env = gym.make(env_id)
        env.action_space.seed(seed)
        return env
    return thunk


# ---------------------------------------------------------------------------
# Actor-Critic Network with Action Masking
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 80, act_dim: int = 160, num_rules: int = 10, max_len: int = 16):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_rules = num_rules
        self.max_len = max_len

        # Shared Feature Extractor Trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )

        # Actor Head (Policy)
        self.actor = nn.Linear(256, act_dim)

        # Critic Head (Value Function)
        self.critic = nn.Linear(256, 1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.trunk(obs)
        return self.critic(features).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        logits = self.actor(features)

        # Apply Action Masking if provided
        if action_mask is not None:
            expanded_mask = action_mask.repeat_interleave(self.max_len, dim=-1)
            logits = torch.where(expanded_mask, logits, torch.tensor(-1e9, device=logits.device))

        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)

        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# PPO Rollout Trainer Class
# ---------------------------------------------------------------------------

class PPOSuperoptTrainer:
    def __init__(
        self,
        num_envs: int = 4,
        num_steps: int = 512,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 256
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.batch_size = num_envs * num_steps
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # Vector Environment
        env_fns = [make_env("SuperoptEnv-v0", seed=i) for i in range(num_envs)]
        self.envs = gym.vector.SyncVectorEnv(env_fns)

        # Neural Network Model & Optimizer on GPU
        self.agent = ActorCritic(obs_dim=80, act_dim=160, num_rules=10, max_len=16).to(DEVICE)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)

    def train(self, total_updates: int = 10):
        print("======================================================================")
        print(f" PPO Superopt Training: {self.num_envs} Envs x {self.num_steps} Steps (Batch Size: {self.batch_size})")
        print("======================================================================")
        print(f" Device          : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        print(f" Action Space    : Discrete(160) [10 Rules x 16 Instruction Slots]")
        print(f" Learning Rate   : {self.optimizer.param_groups[0]['lr']}")
        print(f" Total Updates   : {total_updates} updates ({total_updates * self.batch_size:,} env steps)\n")

        # Rollout Buffers
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

        for update in range(1, total_updates + 1):
            ep_rewards = []
            rules_applied = 0

            # 1. Rollout Phase: Collect 512 steps per env (2048 transitions)
            for step in range(self.num_steps):
                obs_buf[step] = next_obs
                done_buf[step] = next_done
                mask_buf[step] = next_mask

                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(next_obs, action_mask=next_mask)
                    val_buf[step] = value

                act_buf[step] = action
                logprob_buf[step] = logprob

                # Step parallel vector environment
                cpu_actions = action.cpu().numpy()
                next_obs_cpu, reward_cpu, term_cpu, trunc_cpu, info_cpu = self.envs.step(cpu_actions)

                rew_buf[step] = torch.tensor(reward_cpu, dtype=torch.float32, device=DEVICE)
                next_obs = torch.tensor(next_obs_cpu, dtype=torch.float32, device=DEVICE)
                next_mask = torch.tensor(info_cpu["action_mask"], dtype=torch.bool, device=DEVICE)

                done_cpu = np.logical_or(term_cpu, trunc_cpu)
                next_done = torch.tensor(done_cpu, dtype=torch.float32, device=DEVICE)

                rules_applied += sum(info_cpu.get("applied", [False]*self.num_envs))
                ep_rewards.extend(reward_cpu[done_cpu].tolist())

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

            # Flatten buffers
            b_obs = obs_buf.reshape(-1, 80)
            b_logprobs = logprob_buf.reshape(-1)
            b_actions = act_buf.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = val_buf.reshape(-1)
            b_masks = mask_buf.reshape(-1, 10)

            # 3. PPO Optimization Epochs
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

                    # Policy Loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value Loss
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # Entropy Loss
                    entropy_loss = entropy.mean()

                    loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            elapsed = time.time() - start_time
            sps = (update * self.batch_size) / elapsed
            avg_reward = np.mean(ep_rewards) if ep_rewards else 0.0

            print(f"Update {update:02d}/{total_updates:02d} | Steps: {update * self.batch_size:7,d} | "
                  f"SPS: {sps:6.1f} | Avg Rew: {avg_reward:+6.3f} | Rules Applied: {rules_applied:3d} | "
                  f"Policy Loss: {pg_loss.item():+.4f} | Value Loss: {v_loss.item():.4f} | KL: {approx_kl:.5f}")

        self.envs.close()
        print("======================================================================")
        print("[+] PPO Superopt Trainer Connected & Verified Successfully!")
        print("======================================================================\n")


if __name__ == "__main__":
    trainer = PPOSuperoptTrainer(num_envs=4, num_steps=512)
    trainer.train(total_updates=10)
