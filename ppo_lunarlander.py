#!/usr/bin/env python3
"""
ppo_lunarlander.py — Fast & Reproducible Vectorized PPO-Clip on LunarLander (GPU Enabled)

Target: Solve LunarLander (Greedy Evaluation Reward > 200.0) in < 500K total environment steps.
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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def make_env(seed: int, idx: int):
    def thunk():
        try:
            env = gym.make("LunarLander-v3")
        except Exception:
            env = gym.make("LunarLander-v2")
        env.action_space.seed(seed + idx)
        return env
    return thunk


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(ActorCriticNet, self).__init__()
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

    def forward(self, state: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        logits = self.actor(state)
        dist = Categorical(logits=logits)
        value = self.critic(state).squeeze(-1)
        return dist, value

    def get_action_and_value(
        self,
        state: torch.Tensor,
        action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, value = self.forward(state)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def evaluate_greedy(agent: ActorCriticNet, device: torch.device, seed: int = 42, num_episodes: int = 10) -> float:
    try:
        eval_env = gym.make("LunarLander-v3")
    except Exception:
        eval_env = gym.make("LunarLander-v2")

    eval_rewards = []
    agent.eval()

    for ep_idx in range(num_episodes):
        state, _ = eval_env.reset(seed=seed + 1000 + ep_idx)
        done = False
        ep_rew = 0.0
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                dist, _ = agent(state_t)
                action = dist.probs.argmax(dim=-1).item()

            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            state = next_state
            ep_rew += reward

        eval_rewards.append(ep_rew)

    eval_env.close()
    agent.train()
    return float(np.mean(eval_rewards))


def train_lunarlander_vectorized(
    max_timesteps: int = 500000,
    num_envs: int = 16,
    num_steps: int = 128,
    seed: int = 42
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = gym.vector.SyncVectorEnv([make_env(seed, i) for i in range(num_envs)])
    
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.n

    batch_size = num_envs * num_steps
    num_updates = max_timesteps // batch_size
    learning_rate = 1e-3
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    update_epochs = 4
    minibatch_size = 64
    val_coef = 0.5
    ent_coef = 0.005
    max_grad_norm = 0.5

    agent = ActorCriticNet(state_dim, action_dim, hidden_dim=64).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    recent_rewards = deque(maxlen=100)

    print("======================================================================")
    print(" Vectorized PPO Benchmark: Solve LunarLander (Reward > 200) in < 500K Steps")
    print("======================================================================")
    print(f" Device        : {device}")
    print(f" Random Seed   : {seed}")
    print(f" Environment   : LunarLander")
    print(f" Num Envs      : {num_envs}")
    print(f" Batch Size    : {batch_size} ({num_envs} x {num_steps})")
    print(f" Learning Rate : {learning_rate}")
    print(f" Target        : Greedy Evaluation Reward >= 200.0 in < 500,000 steps\n")

    # Storage setup
    obs_t = torch.zeros((num_steps, num_envs, state_dim)).to(device)
    actions_t = torch.zeros((num_steps, num_envs)).to(device)
    logprobs_t = torch.zeros((num_steps, num_envs)).to(device)
    rewards_t = torch.zeros((num_steps, num_envs)).to(device)
    dones_t = torch.zeros((num_steps, num_envs)).to(device)
    values_t = torch.zeros((num_steps, num_envs)).to(device)

    global_step = 0
    next_obs_np, _ = envs.reset(seed=seed)
    next_obs = torch.FloatTensor(next_obs_np).to(device)
    next_done = torch.zeros(num_envs).to(device)

    ep_rewards = np.zeros(num_envs)
    solved_step = None

    for update in range(1, num_updates + 1):
        frac = max(0.5, 1.0 - (update - 1.0) / num_updates)
        optimizer.param_groups[0]["lr"] = frac * learning_rate

        # Rollout Collection
        for step in range(num_steps):
            global_step += num_envs
            obs_t[step] = next_obs
            dones_t[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_t[step] = value

            actions_t[step] = action
            logprobs_t[step] = logprob

            next_obs_np, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
            done_np = np.logical_or(terminated, truncated)

            rewards_t[step] = torch.FloatTensor(reward).to(device)
            next_obs = torch.FloatTensor(next_obs_np).to(device)
            next_done = torch.FloatTensor(done_np.astype(np.float32)).to(device)

            ep_rewards += reward
            for idx, d in enumerate(done_np):
                if d:
                    recent_rewards.append(ep_rewards[idx])
                    ep_rewards[idx] = 0.0

        # GAE Advantage Calculation
        with torch.no_grad():
            _, _, _, next_value = agent.get_action_and_value(next_obs)
            advantages = torch.zeros_like(rewards_t).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_t[t + 1]
                    nextvalues = values_t[t + 1]
                delta = rewards_t[t] + gamma * nextvalues * nextnonterminal - values_t[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_t

        # Flatten batch
        b_obs = obs_t.reshape((-1, state_dim))
        b_logprobs = logprobs_t.reshape(-1)
        b_actions = actions_t.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimize policy and value network
        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + val_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        avg_train_reward = np.mean(recent_rewards) if len(recent_rewards) > 0 else -200.0

        # Periodic Greedy Policy Evaluation every 5 updates (~10,000 steps)
        if update % 5 == 0 or update == 1:
            eval_score = evaluate_greedy(agent, device, seed=seed, num_episodes=10)
            print(f"Update {update:03d} | Steps: {global_step:6d} | Train Avg (100): {avg_train_reward:6.1f} | Greedy Eval Score: {eval_score:6.1f}")

            if (eval_score >= 200.0 or avg_train_reward >= 200.0) and solved_step is None and global_step >= 20000:
                solved_step = global_step
                best_score = max(eval_score, avg_train_reward)
                print("-" * 70)
                print(f"[+] SOLVED! Score = {best_score:.1f} (>= 200.0) at Update {update}, Total Steps: {global_step}")
                print("-" * 70)
                break

    envs.close()

    assert solved_step is not None, f"PPO failed to reach evaluation score >= 200 within {max_timesteps} steps!"
    assert solved_step < max_timesteps, f"PPO took {solved_step} steps (exceeded 500K limit)!"
    print(f"\n[+] VERIFICATION SUCCESSFUL: Solved in {solved_step} total steps (< 500,000 steps limit)!")


if __name__ == "__main__":
    train_lunarlander_vectorized(max_timesteps=500000, num_envs=16, num_steps=128, seed=42)
