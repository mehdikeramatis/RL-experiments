#!/usr/bin/env python3
"""Pendulum-v1 RL benchmark.

Compare five approaches on exactly the same Gymnasium Pendulum-v1 environment:
1. Discretized action value learning baseline
2. Vanilla REINFORCE
3. Basic Actor-Critic
4. PPO
5. SAC

This script trains each method for three seeds, saves raw CSV metrics,
and produces comparison plots for return learning curves and final performance.
"""

import argparse
import csv
import math
import os
import random
from collections import deque, namedtuple, defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'action_idx'])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_csv(filepath, headers, rows):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(headers)
        writer.writerows(rows)


def plot_mean_std(x, datas, labels, filepath, title, xlabel='Episode', ylabel='Return'):
    plt.figure(figsize=(12, 7))
    
    all_values = []
    plot_data = []
    
    for label, series in zip(labels, datas):
        if isinstance(series, np.ndarray):
            if series.ndim == 1:
                series = [series]
            else:
                series = [series[i] for i in range(series.shape[0])]
        else:
            series = list(series)

        arrays = []
        for s in series:
            arr = np.asarray(s, dtype=np.float32)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            arrays.append(arr)
            all_values.extend(arr.tolist())

        # Pad arrays to same length
        max_len = max(len(arr) for arr in arrays)
        padded = []
        for arr in arrays:
            if len(arr) < max_len:
                padded_arr = np.pad(arr, (0, max_len - len(arr)), mode='edge')
            else:
                padded_arr = arr[:max_len]
            padded.append(padded_arr)
        
        stacked = np.stack(padded, axis=0)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        plot_data.append((label, mean, std))
    
    # Plot with consistent axis scaling
    for label, mean, std in plot_data:
        plt.plot(x[: mean.shape[0]], mean, label=label, linewidth=2.5, marker='o', 
                 markersize=4, markevery=max(1, mean.shape[0]//20))
        plt.fill_between(x[: mean.shape[0]], mean - std, mean + std, alpha=0.2)
    
    # Set y-axis limits with padding
    if all_values:
        all_values = np.array(all_values)
        global_min = np.nanmin(all_values)
        global_max = np.nanmax(all_values)
        value_range = global_max - global_min
        padding = value_range * 0.1
        plt.ylim(global_min - padding, global_max + padding)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_bar(values, errors, labels, filepath, title, ylabel='Return'):
    plt.figure(figsize=(10, 6))
    x = np.arange(len(labels))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    bars = plt.bar(x, values, yerr=errors, capsize=8, color=colors, edgecolor='black', 
                   linewidth=1.5, alpha=0.8)
    plt.xticks(x, labels, rotation=15, ha='right', fontsize=11)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(last_dim, hidden), nn.Tanh()])
            last_dim = hidden
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DiscreteValueAgent:
    """DQN-style baseline using discretized Pendulum actions."""

    def __init__(self, state_dim, action_low, action_high, n_actions=11, gamma=0.99, lr=1e-3):
        self.device = torch.device('cpu')
        self.action_bins = np.linspace(action_low, action_high, n_actions)
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = 64
        self.replay = deque(maxlen=20000)
        self.update_every = 4
        self.target_update = 200
        self.steps = 0

        self.q_net = MLP(state_dim, n_actions).to(self.device)
        self.target_net = MLP(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            action_idx = random.randrange(self.n_actions)
        else:
            state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            action_idx = int(torch.argmax(q_values, dim=-1).item())
        return float(self.action_bins[action_idx]), action_idx

    def store_transition(self, transition):
        self.replay.append(transition)

    def update(self):
        self.steps += 1
        if len(self.replay) < self.batch_size or self.steps % self.update_every != 0:
            return 0.0

        batch = random.sample(self.replay, self.batch_size)
        states = torch.tensor(np.array([t.state for t in batch], dtype=np.float32), device=self.device)
        actions = torch.tensor([t.action_idx for t in batch], dtype=torch.int64, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([t.next_state for t in batch], dtype=np.float32), device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1)[0]
            q_target = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


class ReinforceAgent:
    def __init__(self, state_dim, action_scale, lr=1e-3, gamma=0.99):
        self.device = torch.device('cpu')
        self.policy = MLP(state_dim, 1).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(1, device=self.device))
        self.optimizer = optim.Adam(list(self.policy.parameters()) + [self.log_std], lr=lr)
        self.gamma = gamma
        self.action_scale = action_scale

    def get_distribution(self, state):
        mean = self.policy(state)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def select_action(self, state):
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        dist = self.get_distribution(state_t)
        action = dist.sample()
        return float(torch.clamp(action, -self.action_scale, self.action_scale).cpu().numpy().squeeze()), dist.log_prob(action).sum(), float(dist.entropy().sum().item())

    def update(self, log_probs, rewards):
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())


class ActorCriticAgent:
    def __init__(self, state_dim, action_scale, lr=1e-3, gamma=0.99, critic_coef=0.5):
        self.device = torch.device('cpu')
        self.actor = MLP(state_dim, 1).to(self.device)
        self.critic = MLP(state_dim, 1).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(1, device=self.device))
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std], lr=lr)
        self.gamma = gamma
        self.action_scale = action_scale
        self.critic_coef = critic_coef
        self.loss_fn = nn.MSELoss()

    def get_distribution(self, state):
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def select_action(self, state):
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        dist = self.get_distribution(state_t)
        action = dist.sample()
        value = self.critic(state_t)
        return float(torch.clamp(action, -self.action_scale, self.action_scale).cpu().numpy().squeeze()), float(value.item()), dist.log_prob(action).sum(), float(dist.entropy().sum().item())

    def update(self, log_probs, values, rewards):
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values = torch.stack(values).squeeze(1)
        advantages = returns - values

        actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
        critic_loss = self.loss_fn(values, returns)
        loss = actor_loss + self.critic_coef * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(actor_loss.item()), float(critic_loss.item())


class PPOAgent:
    def __init__(self, state_dim, action_scale, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=4, batch_size=64, lam=0.95):
        self.device = torch.device('cpu')
        self.actor = MLP(state_dim, 1).to(self.device)
        self.critic = MLP(state_dim, 1).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(1, device=self.device))
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()) + [self.log_std], lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.lam = lam
        self.action_scale = action_scale
        self.loss_fn = nn.MSELoss()

    def get_distribution(self, state):
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def select_action(self, state):
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        dist = self.get_distribution(state_t)
        action = dist.sample()
        return float(torch.clamp(action, -self.action_scale, self.action_scale).cpu().numpy().squeeze()), float(self.critic(state_t).item()), dist.log_prob(action).sum(), float(dist.entropy().sum().item())

    def compute_gae(self, rewards, values, dones):
        values = values + [0.0]
        gae = 0.0
        returns = []
        advantages = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device).unsqueeze(1)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device).unsqueeze(1)

        dataset = list(range(states.shape[0]))
        actor_loss_total = 0.0
        critic_loss_total = 0.0

        for _ in range(self.epochs):
            random.shuffle(dataset)
            for start in range(0, len(dataset), self.batch_size):
                batch_idx = dataset[start:start + self.batch_size]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_returns = returns[batch_idx].unsqueeze(1)
                batch_advantages = advantages[batch_idx].unsqueeze(1)
                batch_old_log_probs = old_log_probs[batch_idx]

                dist = self.get_distribution(batch_states)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=1, keepdim=True)
                entropy = dist.entropy().sum(dim=1, keepdim=True)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                actor_loss = -torch.min(ratio * batch_advantages, clipped_ratio * batch_advantages).mean()
                critic_loss = self.loss_fn(self.critic(batch_states), batch_returns)
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                actor_loss_total += float(actor_loss.item())
                critic_loss_total += float(critic_loss.item())

        num_updates = self.epochs * math.ceil(len(dataset) / self.batch_size)
        return actor_loss_total / max(num_updates, 1), critic_loss_total / max(num_updates, 1)


class SquashedNormalPolicy(nn.Module):
    def __init__(self, state_dim, action_scale):
        super().__init__()
        self.base = MLP(state_dim, 2)
        self.action_scale = action_scale

    def forward(self, state):
        params = self.base(state)
        mean, log_std = params[:, :1], params[:, 1:]
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self(state)
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z) * self.action_scale
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) / (self.action_scale ** 2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)

    def deterministic(self, state):
        mean, _ = self(state)
        return torch.tanh(mean) * self.action_scale


class SACAgent:
    def __init__(self, state_dim, action_scale, lr=3e-4, gamma=0.99, alpha=0.2, tau=0.005, buffer_size=20000):
        self.device = torch.device('cpu')
        self.action_scale = action_scale
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = 64

        self.policy = SquashedNormalPolicy(state_dim, action_scale).to(self.device)
        self.q1 = MLP(state_dim + 1, 1).to(self.device)
        self.q2 = MLP(state_dim + 1, 1).to(self.device)
        self.target_q1 = MLP(state_dim + 1, 1).to(self.device)
        self.target_q2 = MLP(state_dim + 1, 1).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        self.replay = deque(maxlen=buffer_size)
        self.loss_fn = nn.MSELoss()
        self.step_count = 0

    def select_action(self, state, deterministic=False):
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        if deterministic:
            action = self.policy.deterministic(state_t)
        else:
            action, _ = self.policy.sample(state_t)
        return float(action.detach().cpu().numpy().squeeze())

    def store_transition(self, transition):
        self.replay.append(transition)

    def update(self):
        self.step_count += 1
        if len(self.replay) < self.batch_size:
            return 0.0, 0.0

        batch = random.sample(self.replay, self.batch_size)
        states = torch.tensor(np.array([t.state for t in batch], dtype=np.float32), device=self.device)
        actions = torch.tensor(np.array([t.action for t in batch], dtype=np.float32), device=self.device).unsqueeze(1)
        rewards = torch.tensor(np.array([t.reward for t in batch], dtype=np.float32), device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array([t.next_state for t in batch], dtype=np.float32), device=self.device)
        dones = torch.tensor(np.array([t.done for t in batch], dtype=np.float32), device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample(next_states)
            q1_target = self.target_q1(torch.cat([next_states, next_action], dim=1))
            q2_target = self.target_q2(torch.cat([next_states, next_action], dim=1))
            min_q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_prob
            q_target = rewards + self.gamma * (1 - dones) * min_q_target

        q1_pred = self.q1(torch.cat([states, actions], dim=1))
        q2_pred = self.q2(torch.cat([states, actions], dim=1))
        q1_loss = self.loss_fn(q1_pred, q_target)
        q2_loss = self.loss_fn(q2_pred, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_action, log_prob = self.policy.sample(states)
        q1_new = self.q1(torch.cat([states, new_action], dim=1))
        q2_new = self.q2(torch.cat([states, new_action], dim=1))
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.soft_update(self.q1, self.target_q1)
        self.soft_update(self.q2, self.target_q2)

        return float(policy_loss.item()), float((q1_loss + q2_loss).item() / 2.0)

    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


def run_discrete_value(env_name, seed, config, save_dir=None):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])
    agent = DiscreteValueAgent(state_dim, action_low, action_high, n_actions=config['dqn_actions'], lr=config['dqn_lr'])
    epsilon_schedule = np.linspace(1.0, 0.05, config['episodes'])

    rows = []
    env_steps = 0

    for episode in range(config['episodes']):
        print(f"[DiscreteValue, seed={seed}] Episode {episode + 1}/{config['episodes']}", end='\r')
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        actions = []
        losses = []
        done = False
        episode_steps = 0

        while not done:
            epsilon = float(epsilon_schedule[episode])
            action, action_idx = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step([action])
            done = terminated or truncated
            episode_steps += 1
            env_steps += 1
            total_reward += reward
            actions.append(action)
            agent.store_transition(Transition(state, action, reward, next_state, float(done), action_idx))
            loss = agent.update()
            if loss:
                losses.append(loss)
            state = next_state

        mean_action = float(np.mean(actions)) if actions else 0.0
        std_action = float(np.std(actions)) if actions else 0.0
        rows.append([episode, total_reward, episode_steps, env_steps, mean_action, std_action, np.nan, np.nan, np.nan, np.nan])

    env.close()
    if save_dir is not None:
        torch.save({'q_net': agent.q_net.state_dict(), 'action_bins': agent.action_bins}, os.path.join(save_dir, f'discrete_value_seed{seed}.pt'))
    return rows


def run_reinforce(env_name, seed, config, save_dir=None):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = ReinforceAgent(state_dim, action_scale, lr=config['reinforce_lr'], gamma=config['gamma'])

    rows = []
    env_steps = 0

    for episode in range(config['episodes']):
        print(f"[REINFORCE, seed={seed}] Episode {episode + 1}/{config['episodes']}", end='\r')
        state, _ = env.reset(seed=seed + episode)
        log_probs = []
        rewards = []
        entropies = []
        actions = []
        done = False
        episode_steps = 0

        while not done:
            action, log_prob, entropy = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step([action])
            done = terminated or truncated
            episode_steps += 1
            env_steps += 1
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            actions.append(action)
            state = next_state

        loss = agent.update(log_probs, rewards)
        mean_action = float(np.mean(actions)) if actions else 0.0
        std_action = float(np.std(actions)) if actions else 0.0
        rows.append([episode, sum(rewards), episode_steps, env_steps, mean_action, std_action, np.mean(entropies), loss, np.nan, np.nan])

    env.close()
    if save_dir is not None:
        save_dict = {'policy': agent.policy.state_dict(), 'log_std': agent.log_std.detach().cpu()}
        torch.save(save_dict, os.path.join(save_dir, f'reinforce_seed{seed}.pt'))
    return rows


def run_actor_critic(env_name, seed, config, save_dir=None):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = ActorCriticAgent(state_dim, action_scale, lr=config['ac_lr'], gamma=config['gamma'])

    rows = []
    env_steps = 0

    for episode in range(config['episodes']):
        print(f"[ActorCritic, seed={seed}] Episode {episode + 1}/{config['episodes']}", end='\r')
        state, _ = env.reset(seed=seed + episode)
        log_probs = []
        values = []
        rewards = []
        entropies = []
        actions = []
        done = False
        episode_steps = 0

        while not done:
            action, value, log_prob, entropy = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step([action])
            done = terminated or truncated
            episode_steps += 1
            env_steps += 1
            log_probs.append(log_prob)
            values.append(torch.tensor([value], dtype=torch.float32))
            rewards.append(reward)
            entropies.append(entropy)
            actions.append(action)
            state = next_state

        actor_loss, critic_loss = agent.update(log_probs, values, rewards)
        mean_action = float(np.mean(actions)) if actions else 0.0
        std_action = float(np.std(actions)) if actions else 0.0
        rows.append([episode, sum(rewards), episode_steps, env_steps, mean_action, std_action, np.mean(entropies), actor_loss, critic_loss, np.nan])

    env.close()
    if save_dir is not None:
        save_dict = {
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'log_std': agent.log_std.detach().cpu()
        }
        torch.save(save_dict, os.path.join(save_dir, f'actor_critic_seed{seed}.pt'))
    return rows


def run_ppo(env_name, seed, config, save_dir=None):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = PPOAgent(state_dim, action_scale, lr=config['ppo_lr'], gamma=config['gamma'], clip_eps=config['ppo_clip'])

    rows = []
    env_steps = 0
    rollout_buffer = []
    episode_rewards = []
    episode_lengths = []
    episode_entropies = []
    episode_actor_losses = []
    episode_critic_losses = []

    for episode in range(config['episodes']):
        print(f"[PPO, seed={seed}] Episode {episode + 1}/{config['episodes']}", end='\r')
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_rewards.append(0.0)
        episode_lengths.append(0)
        episode_entropies.append([])
        actions = []
        log_probs = []
        states = []
        rewards = []
        dones = []
        values = []

        while not done:
            state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            dist = agent.get_distribution(state_t)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum().item()
            entropy = dist.entropy().sum().item()
            clipped_action = float(torch.clamp(action, -action_scale, action_scale).cpu().numpy().squeeze())
            value = float(agent.critic(state_t).item())

            next_state, reward, terminated, truncated, _ = env.step([clipped_action])
            done = terminated or truncated
            episode_rewards[-1] += reward
            episode_lengths[-1] += 1
            episode_entropies[-1].append(entropy)
            actions.append(clipped_action)
            log_probs.append(log_prob)
            states.append(state_t.squeeze(0))
            rewards.append(reward)
            dones.append(float(done))
            values.append(value)
            env_steps += 1
            state = next_state

        returns, advantages = agent.compute_gae(rewards, values, dones)
        actor_loss, critic_loss = agent.update(states, actions, log_probs, returns, advantages)
        episode_actor_losses.append(actor_loss)
        episode_critic_losses.append(critic_loss)
        mean_action = float(np.mean(actions)) if actions else 0.0
        std_action = float(np.std(actions)) if actions else 0.0

        rows.append([episode, episode_rewards[-1], episode_lengths[-1], env_steps, mean_action, std_action, np.mean(episode_entropies[-1]), actor_loss, critic_loss, np.nan])

    env.close()
    if save_dir is not None:
        save_dict = {'actor': agent.actor.state_dict(), 'critic': agent.critic.state_dict(), 'log_std': agent.log_std.detach().cpu()}
        torch.save(save_dict, os.path.join(save_dir, f'ppo_seed{seed}.pt'))
    return rows


def run_sac(env_name, seed, config, save_dir=None):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_scale = float(env.action_space.high[0])
    agent = SACAgent(state_dim, action_scale, lr=config['sac_lr'], gamma=config['gamma'], alpha=config['sac_alpha'])

    rows = []
    env_steps = 0

    for episode in range(config['episodes']):
        print(f"[SAC, seed={seed}] Episode {episode + 1}/{config['episodes']}", end='\r')
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        actions = []
        actor_losses = []
        critic_losses = []
        entropies = []
        done = False
        episode_steps = 0

        while not done:
            action = agent.select_action(state)
            clipped_action = float(np.clip(action, -action_scale, action_scale))
            next_state, reward, terminated, truncated, _ = env.step([clipped_action])
            done = terminated or truncated
            episode_steps += 1
            env_steps += 1
            total_reward += reward
            actions.append(clipped_action)
            agent.store_transition(Transition(state, clipped_action, reward, next_state, float(done), None))
            policy_loss, q_loss = agent.update()
            if not math.isnan(policy_loss):
                actor_losses.append(policy_loss)
                critic_losses.append(q_loss)
            state = next_state

        mean_action = float(np.mean(actions)) if actions else 0.0
        std_action = float(np.std(actions)) if actions else 0.0
        rows.append([episode, total_reward, episode_steps, env_steps, mean_action, std_action, np.nan, np.mean(actor_losses) if actor_losses else np.nan, np.mean(critic_losses) if critic_losses else np.nan, np.nan])

    env.close()
    if save_dir is not None:
        save_dict = {
            'policy': agent.policy.state_dict(),
        }
        torch.save(save_dict, os.path.join(save_dir, f'sac_seed{seed}.pt'))
    return rows


def experiment(config):
    methods = [
        ('discrete_value', run_discrete_value),
        ('reinforce', run_reinforce),
        ('actor_critic', run_actor_critic),
        ('ppo', run_ppo),
        ('sac', run_sac),
    ]

    raw_dir = os.path.join(config['output_dir'], 'raw')
    plot_dir = os.path.join(config['output_dir'], 'plots')
    model_dir = os.path.join(config['output_dir'], 'models')
    ensure_dir(raw_dir)
    ensure_dir(plot_dir)
    ensure_dir(model_dir)

    aggregated_returns = defaultdict(list)
    final_returns = []
    final_labels = []
    total_methods = len(methods)
    total_seeds = len(config['seeds'])
    total_work = total_methods * total_seeds
    current_work = 0

    for method_idx, (method_name, runner) in enumerate(methods):
        all_returns = []
        all_steps = []
        final_seed_returns = []
        for seed_idx, seed in enumerate(config['seeds']):
            current_work += 1
            print(f"\n{'='*80}")
            print(f"Progress: {current_work}/{total_work} (Method {method_idx + 1}/{total_methods}, Seed {seed_idx + 1}/{total_seeds})")
            print(f"Training {method_name.upper()} with seed {seed}...")
            print(f"{'='*80}")
            set_seed(seed)
            result_rows = runner(config['env_name'], seed, config, save_dir=model_dir)
            print()  # New line after progress output
            file_path = os.path.join(raw_dir, f'{method_name}_seed{seed}.csv')
            headers = ['episode', 'return', 'length', 'env_steps', 'action_mean', 'action_std', 'entropy', 'actor_loss', 'critic_loss', 'extra']
            save_csv(file_path, headers, result_rows)
            returns = [row[1] for row in result_rows]
            all_returns.append(returns)
            all_steps.append([row[3] for row in result_rows])
            final_seed_returns.append(np.mean(returns[-10:]))

        mean_final = float(np.mean(final_seed_returns))
        std_final = float(np.std(final_seed_returns))
        final_returns.append(mean_final)
        final_labels.append(method_name)
        aggregated_returns[method_name] = all_returns

        plot_mean_std(
            x=list(range(config['episodes'])),
            datas=all_returns,
            labels=[f'{method_name} seed {seed}' for seed in config['seeds']],
            filepath=os.path.join(plot_dir, f'{method_name}_return_curves.png'),
            title=f'{method_name} return curves',
            xlabel='Episode',
            ylabel='Episode Return'
        )

        summary_rows = []
        for seed, returns in zip(config['seeds'], all_returns):
            summary_rows.append([method_name, seed, float(np.mean(returns[-10:]))])
        save_csv(
            os.path.join(raw_dir, f'{method_name}_summary.csv'),
            ['method', 'seed', 'mean_last10_return'],
            summary_rows
        )

    # Comparison plots
    plot_mean_std(
        x=list(range(config['episodes'])),
        datas=[aggregated_returns[method] for method in final_labels],
        labels=final_labels,
        filepath=os.path.join(plot_dir, 'comparison_return_curves.png'),
        title='Comparison of Return Learning Curves',
        xlabel='Episode',
        ylabel='Episode Return'
    )
    plot_bar(final_returns, [0.0] * len(final_returns), final_labels, os.path.join(plot_dir, 'final_performance.png'), 'Final Average Return')

    print('Experiment finished. Raw metrics are saved under', raw_dir)
    print('Plots are saved under', plot_dir)


def parse_args():
    parser = argparse.ArgumentParser(description='Pendulum-v1 RL algorithm comparison')
    parser.add_argument('--env', default='Pendulum-v1')
    parser.add_argument('--episodes', type=int, default=120)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--n-seeds', type=int, default=3)
    parser.add_argument('--output-dir', default='results')
    return parser.parse_args()


def main():
    args = parse_args()
    config = {
        'env_name': args.env,
        'episodes': args.episodes,
        'seeds': [args.seed_start + i for i in range(args.n_seeds)],
        'output_dir': args.output_dir,
        'dqn_actions': 11,
        'dqn_lr': 1e-3,
        'reinforce_lr': 1e-3,
        'ac_lr': 1e-3,
        'ppo_lr': 3e-4,
        'sac_lr': 3e-4,
        'gamma': 0.99,
        'ppo_clip': 0.2,
        'sac_alpha': 0.2,
    }
    experiment(config)


if __name__ == '__main__':
    main()
