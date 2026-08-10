#!/usr/bin/env python3
"""
a2c_shared_trunk.py — Advantage Actor-Critic (A2C) with Shared Trunk Architecture (GPU Compatible)
HuggingFace Deep RL Unit 6: Actor-Critic Methods.

Features:
- Device selection (CUDA / CPU)
- Shared Feature Extractor Trunk
- Dual Heads: Actor Head (Action Logits) & Critic Head (State Value V(s))
- Advantage calculation: A(s, a) = R_t - V(s)
- Joint Optimization with Advantage Normalization
"""

import random
from collections import deque
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym


# ---------------------------------------------------------------------------
# 1. Shared Trunk Actor-Critic Network Architecture
# ---------------------------------------------------------------------------

class A2CSharedTrunkNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(A2CSharedTrunkNet, self).__init__()

        # Shared Feature Extractor Trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor Head (Policy Logits)
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic Head (State Value V(s))
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        features = self.shared_trunk(state)
        logits = self.actor_head(features)
        dist = Categorical(logits=logits)
        value = self.critic_head(features).squeeze(-1)
        return dist, value


# ---------------------------------------------------------------------------
# 2. A2C Agent (GPU Compatible)
# ---------------------------------------------------------------------------

class A2CAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1.5e-3,
        gamma: float = 0.99,
        value_loss_coef: float = 0.25,
        entropy_coef: float = 0.002,
        max_grad_norm: float = 0.5,
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.model = A2CSharedTrunkNet(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist, value = self.model(state_t)
        action = dist.sample()
        return action.item(), dist.log_prob(action).squeeze(0), value.squeeze(0)

    def update(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        dones: List[bool]
    ) -> Dict[str, float]:
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)

        dists, values = self.model(states_t)
        log_probs = dists.log_prob(actions_t)
        entropies = dists.entropy()

        # Discounted Monte Carlo Returns
        returns = []
        R = 0.0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1.0 - float(done))
            returns.insert(0, R)

        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = returns_t - values

        # Normalize advantages
        if len(advantages_t) > 1:
            advantages_norm = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        else:
            advantages_norm = advantages_t

        actor_loss = -(log_probs * advantages_norm.detach()).mean()
        critic_loss = nn.MSELoss()(values, returns_t)
        entropy_loss = -entropies.mean()

        total_loss = actor_loss + self.value_loss_coef * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy_loss": entropy_loss.item()
        }


# ---------------------------------------------------------------------------
# 3. Training Loop & Evaluation
# ---------------------------------------------------------------------------

def train_a2c(num_episodes: int = 700, max_steps: int = 500) -> A2CAgent:
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(state_dim=state_dim, action_dim=action_dim)
    recent_rewards = deque(maxlen=100)

    print("==========================================================")
    print(" Training A2C (Advantage Actor-Critic) with Shared Trunk (GPU Enabled)")
    print("==========================================================")
    print(f" Device           : {agent.device}")
    print(f" State Dimension  : {state_dim}")
    print(f" Action Dimension : {action_dim}")
    print(f" Architecture     : Shared Trunk -> Dual Heads (Actor & Critic)\n")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        states, actions, rewards, dones = [], [], [], []

        for step in range(max_steps):
            action, _, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            episode_reward += reward

            if done:
                break

        loss_dict = agent.update(states, actions, rewards, dones)
        recent_rewards.append(episode_reward)
        avg_reward = np.mean(recent_rewards)

        if episode % 50 == 0 or episode == 1:
            print(f"Episode {episode:03d}/{num_episodes} | "
                  f"Total Reward: {episode_reward:5.1f} | "
                  f"Avg (100 ep): {avg_reward:5.1f} | "
                  f"Actor Loss: {loss_dict['actor_loss']:6.3f} | "
                  f"Critic Loss: {loss_dict['critic_loss']:6.3f}")

        if avg_reward >= 450.0 and episode >= 100:
            print(f"\n[+] Environment Solved in {episode} episodes! Avg Reward: {avg_reward:.1f}")
            break

    env.close()
    return agent


def evaluate_a2c_agent(agent: A2CAgent, eval_episodes: int = 10):
    env = gym.make("CartPole-v1")
    eval_rewards = []

    print("\n==========================================================")
    print(" Evaluating Trained A2C Agent (Greedy Action Selection)")
    print("==========================================================")

    agent.model.eval()

    for ep in range(1, eval_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                dist, _ = agent.model(state_t)
                action = dist.probs.argmax(dim=-1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        eval_rewards.append(total_reward)
        print(f"Eval Episode {ep:02d}: Score = {total_reward:.1f}")

    env.close()

    avg_eval = np.mean(eval_rewards)
    print(f"\n[+] Evaluation Average Score over {eval_episodes} episodes: {avg_eval:.1f} / 500.0")
    assert avg_eval >= 250.0, f"A2C Evaluation score too low: {avg_eval}"


if __name__ == "__main__":
    trained_agent = train_a2c(num_episodes=700)
    evaluate_a2c_agent(trained_agent, eval_episodes=10)
