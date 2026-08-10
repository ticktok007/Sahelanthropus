#!/usr/bin/env python3
"""
ppo_from_scratch.py — PPO-Clip with TensorBoard Diagnostic Logging (GPU Compatible)

Features:
- Device selection (CUDA / CPU)
- Rollout Buffer Trajectory Collection
- Generalized Advantage Estimation (GAE-lambda)
- TensorBoard diagnostics (entropy, clip_fraction, approx_kl, explained_variance)
"""

import os
import random
from collections import deque
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym


# ---------------------------------------------------------------------------
# 1. Rollout Buffer
# ---------------------------------------------------------------------------

class PPORolloutBuffer:
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def add(self, state: np.ndarray, action: int, log_prob: float, reward: float, done: bool, value: float):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)


# ---------------------------------------------------------------------------
# 2. Actor-Critic Network Architecture
# ---------------------------------------------------------------------------

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorCriticNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        value = self.critic(state).squeeze(-1)
        return dist, value

    def get_action(self, state: np.ndarray, device: torch.device) -> Tuple[int, float, float]:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            dist, value = self.forward(state_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, values = self.forward(states)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        return log_probs, values, entropies


# ---------------------------------------------------------------------------
# 3. PPO Agent (GPU Compatible)
# ---------------------------------------------------------------------------

class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        k_epochs: int = 4,
        batch_size: int = 64,
        val_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        log_dir: str = "/home/sanjay/Documents/Sahelanthropus/runs/ppo_cartpole",
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.k_epochs = k_epochs
        self.batch_size = batch_size
        self.val_coef = val_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.policy = ActorCriticNet(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.buffer = PPORolloutBuffer()
        self.writer = SummaryWriter(log_dir=log_dir)

    def compute_gae(
        self,
        rewards: List[float],
        dones: List[bool],
        values: List[float],
        next_value: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        gae = 0.0
        values_extended = values + [next_value]

        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values_extended[t + 1] * non_terminal - values_extended[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            advantages.insert(0, gae)

        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = advantages_t + torch.FloatTensor(values).to(self.device)
        advantages_norm = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        return advantages_norm, returns_t

    def update(self, next_value: float, step: int) -> Dict[str, float]:
        states_t = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions_t = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(self.buffer.log_probs).to(self.device)

        advantages_t, returns_t = self.compute_gae(
            self.buffer.rewards, self.buffer.dones, self.buffer.values, next_value
        )

        dataset_size = len(states_t)
        indices = np.arange(dataset_size)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        clip_fractions = []
        approx_kls = []
        num_updates = 0

        values_t = torch.FloatTensor(self.buffer.values).to(self.device)
        y_var = torch.var(returns_t).item()
        explained_var = 1.0 - (torch.var(returns_t - values_t).item() / (y_var + 1e-8)) if y_var > 0 else 0.0

        for _ in range(self.k_epochs):
            np.random.shuffle(indices)

            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = start_idx + self.batch_size
                mb_indices = indices[start_idx:end_idx]

                mb_states = states_t[mb_indices]
                mb_actions = actions_t[mb_indices]
                mb_old_log_probs = old_log_probs_t[mb_indices]
                mb_advantages = advantages_t[mb_indices]
                mb_returns = returns_t[mb_indices]

                mb_log_probs, mb_values, mb_entropies = self.policy.evaluate_actions(mb_states, mb_actions)

                ratios = torch.exp(mb_log_probs - mb_old_log_probs)

                with torch.no_grad():
                    clip_frac = (torch.abs(ratios - 1.0) > self.clip_eps).float().mean().item()
                    approx_kl = (mb_old_log_probs - mb_log_probs).mean().item()
                    clip_fractions.append(clip_frac)
                    approx_kls.append(approx_kl)

                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(mb_values, mb_returns)
                entropy_loss = -mb_entropies.mean()

                loss = policy_loss + self.val_coef * value_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                num_updates += 1

        self.buffer.clear()

        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = -total_entropy_loss / num_updates
        avg_clip_fraction = np.mean(clip_fractions)
        avg_approx_kl = np.mean(approx_kls)

        # TensorBoard Diagnostic Logging
        self.writer.add_scalar("losses/policy_loss", avg_policy_loss, step)
        self.writer.add_scalar("losses/value_loss", avg_value_loss, step)
        self.writer.add_scalar("diagnostics/entropy", avg_entropy, step)
        self.writer.add_scalar("diagnostics/clip_fraction", avg_clip_fraction, step)
        self.writer.add_scalar("diagnostics/approx_kl", avg_approx_kl, step)
        self.writer.add_scalar("diagnostics/explained_variance", explained_var, step)
        self.writer.flush()

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "clip_fraction": avg_clip_fraction,
            "approx_kl": avg_approx_kl,
            "explained_variance": explained_var
        }


# ---------------------------------------------------------------------------
# 4. Training Loop & Evaluation
# ---------------------------------------------------------------------------

def train_ppo(total_timesteps: int = 100000, rollout_steps: int = 2048) -> PPOAgent:
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    recent_rewards = deque(maxlen=100)

    print("==========================================================")
    print(" Training PPO with TensorBoard Diagnostic Logging (GPU Enabled)")
    print("==========================================================")
    print(f" Device : {agent.device}")
    print(" Metrics Logged: entropy, clip_fraction, approx_kl, explained_variance\n")

    state, _ = env.reset()
    current_episode_reward = 0.0
    num_timesteps = 0
    iteration = 0

    while num_timesteps < total_timesteps:
        iteration += 1

        for _ in range(rollout_steps):
            action, log_prob, value = agent.policy.get_action(state, agent.device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            num_timesteps += 1

            agent.buffer.add(state, action, log_prob, reward, done, value)
            state = next_state
            current_episode_reward += reward

            if done:
                recent_rewards.append(current_episode_reward)
                state, _ = env.reset()
                current_episode_reward = 0.0

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            _, next_value = agent.policy(state_t)
            next_value = next_value.item()

        metrics = agent.update(next_value, step=num_timesteps)
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0

        agent.writer.add_scalar("rollout/ep_rew_mean", avg_reward, num_timesteps)
        agent.writer.flush()

        print(f"Iter {iteration:03d} | Steps: {num_timesteps:6d} | Avg (100): {avg_reward:5.1f} | "
              f"Entropy: {metrics['entropy']:5.3f} | ClipFrac: {metrics['clip_fraction']:5.3f} | "
              f"KL: {metrics['approx_kl']:7.4f} | ExplVar: {metrics['explained_variance']:5.3f}")

        if avg_reward >= 475.0:
            print(f"\n[+] Environment Solved in Iteration {iteration} ({num_timesteps} timesteps)! Avg Reward: {avg_reward:.1f}")
            break

    agent.writer.close()
    env.close()
    return agent


if __name__ == "__main__":
    trained_agent = train_ppo(total_timesteps=100000, rollout_steps=2048)
