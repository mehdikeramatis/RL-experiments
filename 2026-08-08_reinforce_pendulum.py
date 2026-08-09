import argparse
import math
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class Policy(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.mean = nn.Linear(hidden_size, 1)
        # learnable log std (scalar)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # normalize
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def train(env_name='Pendulum-v1', episodes=1000, lr=1e-3, gamma=0.99, seed=0):
    device = torch.device('cpu')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_scale = float(env.action_space.high[0])

    policy = Policy(state_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    best_avg = -1e9
    reward_queue = deque(maxlen=100)

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        log_probs = []
        rewards = []

        done = False

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32, device=device)

            mean, std = policy(state_t)
            dist = Normal(mean, std)

            action = dist.sample()  # shape: [1]
            log_prob = dist.log_prob(action).sum()

            # clip action for env but use sampled action for log_prob
            action_clipped = torch.clamp(action, -action_scale, action_scale)

            next_state, reward, terminated, truncated, _ = env.step(action_clipped.detach().cpu().numpy())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

        returns = compute_returns(rewards, gamma)

        loss = 0.0
        for lp, G in zip(log_probs, returns):
            loss += -lp * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_reward = sum(rewards)
        reward_queue.append(episode_reward)
        avg_reward = float(np.mean(reward_queue))

        if episode % 10 == 0:
            print(f"Episode {episode}\tReward: {episode_reward:.1f}\tAvg100: {avg_reward:.1f}")

        # save best
        if avg_reward > best_avg and episode >= 100:
            best_avg = avg_reward
            torch.save(policy.state_dict(), 'pendulum_reinforce_best.pth')

    env.close()
    torch.save(policy.state_dict(), 'pendulum_reinforce_final.pth')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='Pendulum-v1')
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    train(env_name=args.env, episodes=args.episodes, lr=args.lr, gamma=args.gamma, seed=args.seed)


if __name__ == '__main__':
    main()
