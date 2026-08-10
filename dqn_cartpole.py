#!/usr/bin/env python3
"""
dqn_cartpole.py — Deep Q-Network (DQN) from scratch on CartPole-v1 (GPU Compatible)

Features:
- Device selection (CUDA / CPU)
- Replay Buffer (Experience Replay)
- Target Network with Soft Updates (tau = 0.005)
- Epsilon-greedy policy with exponential decay
"""

import random
from collections import deque
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# ---------------------------------------------------------------------------
# 1. Replay Buffer
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 2. Q-Network Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. DQN Agent (GPU Compatible)
# ---------------------------------------------------------------------------

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

    def train_step(self) -> Optional[float]:
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

        # Soft update of target network
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

        return loss.item()


# ---------------------------------------------------------------------------
# 4. Training Loop & Evaluation
# ---------------------------------------------------------------------------

def train_dqn(num_episodes: int = 350, max_steps: int = 500) -> DQNAgent:
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    recent_rewards = deque(maxlen=100)

    print("==========================================================")
    print(" Training DQN from scratch on CartPole-v1 (GPU Enabled)")
    print("==========================================================")
    print(f" Device : {agent.device}\n")

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        losses = []

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

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
        avg_reward = np.mean(recent_rewards)

        if episode % 20 == 0 or episode == 1:
            print(f"Episode {episode:03d}/{num_episodes} | "
                  f"Total Reward: {episode_reward:5.1f} | "
                  f"Avg (100 ep): {avg_reward:5.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

        if avg_reward >= 450.0 and episode >= 100:
            print(f"\n[+] Environment Solved in {episode} episodes! Avg Reward: {avg_reward:.1f}")
            break

    env.close()
    return agent


def evaluate_agent(agent: DQNAgent, eval_episodes: int = 10):
    env = gym.make("CartPole-v1")
    eval_rewards = []

    print("\n==========================================================")
    print(" Evaluating Trained DQN Agent (Greedy Policy, Epsilon = 0)")
    print("==========================================================")

    old_eps = agent.epsilon
    agent.epsilon = 0.0

    for ep in range(1, eval_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        eval_rewards.append(total_reward)
        print(f"Eval Episode {ep:02d}: Score = {total_reward:.1f}")

    agent.epsilon = old_eps
    env.close()

    avg_eval = np.mean(eval_rewards)
    print(f"\n[+] Evaluation Average Score over {eval_episodes} episodes: {avg_eval:.1f} / 500.0")
    assert avg_eval >= 450.0, f"DQN Evaluation score too low: {avg_eval}"


if __name__ == "__main__":
    trained_agent = train_dqn(num_episodes=350)
    evaluate_agent(trained_agent, eval_episodes=10)
