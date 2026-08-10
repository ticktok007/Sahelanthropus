#!/usr/bin/env python3
"""
verify_dqn_100k.py — Verify DQN achieves > 450 episode reward in under 100K steps and logs learning curve (GPU Compatible).
"""

import csv
import json
import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class ReplayBuffer:
    def __init__(self, capacity: int = 50000, device: torch.device = torch.device("cpu")):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(state)).to(self.device),
            torch.LongTensor(action).to(self.device),
            torch.FloatTensor(reward).to(self.device),
            torch.FloatTensor(np.array(next_state)).to(self.device),
            torch.FloatTensor(done).to(self.device)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.99,
        buffer_capacity: int = 50000,
        batch_size: int = 64,
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_capacity, device=self.device)

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_t)
            return q_values.argmax(dim=1).item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        q_values = self.policy_net(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q_values = rewards + (1.0 - dones) * self.gamma * max_next_q_values

        loss = nn.SmoothL1Loss()(state_action_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()


def run_verification_and_log():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    
    total_steps = 0
    recent_rewards = deque(maxlen=100)
    recent_50_rewards = deque(maxlen=50)
    learning_curve_log = []
    first_450_step = None

    print("======================================================================")
    print(" DQN Verification: Target > 450 Reward in < 100K Total Steps (GPU Enabled)")
    print("======================================================================")
    print(f" Device: {agent.device}")
    print(f"{'Episode':<10} | {'Total Steps':<12} | {'Reward':<10} | {'Avg (50)':<10} | {'Epsilon':<8} | Status")
    print("-" * 70)

    for episode in range(1, 450):
        state, _ = env.reset()
        episode_reward = 0.0
        losses = []

        for step in range(500):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_steps += 1

            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

            state = next_state
            episode_reward += reward

            if done:
                break

        agent.update_epsilon()
        recent_rewards.append(episode_reward)
        recent_50_rewards.append(episode_reward)
        avg_reward = np.mean(recent_rewards)
        avg_50_reward = np.mean(recent_50_rewards)
        avg_loss = np.mean(losses) if losses else 0.0

        if episode_reward >= 450.0 and first_450_step is None:
            first_450_step = total_steps

        log_entry = {
            "episode": episode,
            "total_steps": total_steps,
            "episode_reward": float(episode_reward),
            "moving_avg_100": float(avg_reward),
            "moving_avg_50": float(avg_50_reward),
            "epsilon": float(agent.epsilon),
            "avg_loss": float(avg_loss)
        }
        learning_curve_log.append(log_entry)

        if episode % 20 == 0 or episode == 1:
            print(f"{episode:<10d} | {total_steps:<12d} | {episode_reward:<10.1f} | {avg_50_reward:<10.1f} | {agent.epsilon:<8.3f} | RUNNING")

        if avg_50_reward >= 450.0:
            print("-" * 70)
            print(f"[+] SOLVED! Moving Avg (50 ep) = {avg_50_reward:.1f} (>= 450.0) at Episode {episode}, Total Steps: {total_steps}")
            print("-" * 70)
            break

    env.close()

    # Save Learning Curve CSV and JSON
    csv_file = "learning_curve.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "total_steps", "episode_reward", "moving_avg_100", "moving_avg_50", "epsilon", "avg_loss"])
        writer.writeheader()
        writer.writerows(learning_curve_log)

    json_file = "learning_curve.json"
    with open(json_file, "w") as f:
        json.dump(learning_curve_log, f, indent=2)

    print(f"\n[+] Learning curve logged to '{csv_file}' and '{json_file}'.")

    # Assertions
    assert first_450_step is not None, "DQN failed to reach single episode reward >= 450!"
    assert first_450_step < 100000, f"First > 450 reward episode took {first_450_step} steps (exceeded 100K limit)!"
    print(f"[+] VERIFICATION SUCCESSFUL: Episode reward >= 450 achieved at step {first_450_step} (< 100,000 steps)!")


if __name__ == "__main__":
    run_verification_and_log()
