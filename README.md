# Reinforcement Learning Experiments

This repository contains three independent reinforcement learning experiments copied from the previous project.

## Contents

- `2026-08-06 lunar_lander_dqn.py`: DQN training for `LunarLander-v3`
- `2026-08-06 lunar_lander_dqn.pth`: saved LunarLander model weights
- `2026-08-08 reinforce_pendulum.py`: REINFORCE training for `Pendulum-v1`
- `2026-08-08 pendulum_reinforce_best.pth`: best REINFORCE policy weights
- `2026-08-08 pendulum_reinforce_final.pth`: final REINFORCE policy weights
- `2026-08-09 pendulum_rl_benchmark.py`: benchmark script comparing Pendulum RL methods
- `requirements.txt`: Python dependencies
- `.gitignore`: ignore Python caches, environments, and editor files

## Requirements

- Python 3
- PyTorch
- NumPy
- Gymnasium with Box2D support
- Matplotlib

Install dependencies with:

```bash
pip3 install -r requirements.txt
```

## Run the experiments

Run the LunarLander DQN script:

```bash
python3 '2026-08-06 lunar_lander_dqn.py'
```

Run the Pendulum REINFORCE script:

```bash
python3 '2026-08-08 reinforce_pendulum.py'
```

Run the Pendulum benchmark script:

```bash
python3 '2026-08-09 pendulum_rl_benchmark.py'
```

## Notes

- `gymnasium[box2d]` is required for `LunarLander-v3`.
- `matplotlib` is required for the benchmark plotting script.
- The repository is initialized as a Git repository; the copied experiment files are currently present and can be added/committed as needed.
