import math
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# 1) Environment and training setup
# -----------------------------
SEED = 42
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 500
TARGET_UPDATE = 10
BATCH_SIZE = 64
MEMORY_SIZE = 100000
LEARNING_RATE = 1e-4
NUM_EPISODES = 1000
MAX_STEPS = 1000
RENDER_TRAINING = False

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Use GPU if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_env(render=False):
    """Create the LunarLander environment with optional live rendering."""
    if render:
        return gym.make("LunarLander-v3", render_mode="human")
    return gym.make("LunarLander-v3")


# Create the main training environment once.
env = build_env(render=RENDER_TRAINING)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# -----------------------------
# 2) Deep Q-Network (DQN)
# -----------------------------
class DQN(nn.Module):
    """A small MLP that predicts Q-values for each possible action."""

    def __init__(self, state_size, action_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.network(x)


# -----------------------------
# 3) Replay buffer
# -----------------------------
class ReplayBuffer:
    """Stores past experiences for off-policy learning."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


# -----------------------------
# 4) Training helpers
# -----------------------------
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(MEMORY_SIZE)


def select_action(state, episode):
    """Epsilon-greedy action selection."""
    sample = random.random()
    epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * episode / EPS_DECAY)

    if sample > epsilon:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
    else:
        action = random.randrange(action_size)

    return action


def optimize_model():
    """Sample a batch and perform one gradient update."""
    if len(memory) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_states).max(1).values
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.functional.smooth_l1_loss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# -----------------------------
# 5) Training loop
# -----------------------------
def train():
    """Train the agent and display the environment during learning."""
    scores = []

    for episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset(seed=SEED + episode)
        total_reward = 0

        for step in range(MAX_STEPS):
            action = select_action(state, episode)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            optimize_model()

            if done:
                break

        scores.append(total_reward)

        # Update target network every few episodes.
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Print progress.
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Episode {episode:4d} | Avg Reward (last 50): {avg_score:7.2f}")

        # Stop early once the agent is consistently solving the environment.
        if np.mean(scores[-100:]) > 200:
            print("Environment solved!")
            break

    torch.save(policy_net.state_dict(), "lunar_" \
    "lander_dqn.pth")

    print("Training complete. Model saved to lunar_lander_dqn.pth")


def render(model_path="lunar_lander_dqn.pth", max_steps=1000, seed=SEED):
    """Load a saved model and visually replay one landing episode."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    render_env = build_env(render=True)
    model = DQN(state_size, action_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    state, _ = render_env.reset(seed=seed)
    total_reward = 0.0

    for _ in range(max_steps):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = model(state_tensor)
            action = q_values.argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = render_env.step(action)
        render_env.render()
        total_reward += reward
        state = next_state

        if terminated or truncated:
            break

    print(f"Rendered episode finished with reward: {total_reward:.2f}")
    render_env.close()


if __name__ == "__main__":
    train()
    #render()
    env.close()
