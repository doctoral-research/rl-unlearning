<p align="center">
    <strong>Selective Behavior Unlearning in On-Policy Reinforcement Learning</strong>
</p>

<p align="center">
    <a href="https://github.com/elte-machine-unlearning/rl-unlearning/actions/workflows/ci.yml"><img src="https://github.com/elte-machine-unlearning/rl-unlearning/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
    <a href="https://github.com/elte-machine-unlearning/rl-unlearning/actions/workflows/docker.yml"><img src="https://github.com/elte-machine-unlearning/rl-unlearning/actions/workflows/docker.yml/badge.svg?branch=main" alt="Docker"></a>
    <a href="https://github.com/elte-machine-unlearning/rl-unlearning/blob/main/LICENSE"><img src="https://img.shields.io/github/license/elte-machine-unlearning/rl-unlearning" alt="License"></a>
    <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/pytorch-2.0+-orange" alt="PyTorch 2.0+">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Docker (Recommended)](#docker-recommended)
  - [Local Installation](#local-installation)
  - [Dev Containers](#dev-containers)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Unlearning](#unlearning)
- [Evaluation](#evaluation)
- [Metrics](#metrics)
- [Testing](#testing)
- [Research](#research)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project implements and evaluates methods for **selectively forgetting undesired behaviors** in on-policy RL agents while preserving learned knowledge in other parts of the state-action space. We study three complementary approaches and evaluate them across classic control environments using custom unlearning metrics.

### Research Hypotheses

1. **Trajectory Separation** — With trajectory separation and loss component regulation, Retain Stability Index can be significantly improved at the same forgetting level
2. **Zero-Shot Inversion** — Strategy inversion works without world models via seed-optimization, pushing policy toward forget regions with negative learning signals
3. **Metaplasticity Protection** — Metaplasticity masks + distillation keep retain degradation within 10%

---

## Key Features

- **Three-Pillar Framework**: Trajectory-selective forgetting, zero-shot strategy inversion, retain protection
- **PPO Agent**: On-policy learning with GAE, orthogonal initialization, and clipped objectives
- **Hydra Configuration**: Composable YAML configs for environments, agents, unlearning methods, and logging
- **Custom Metrics**: AUFC, Retain Stability Index, Selectivity, Privacy AUC
- **Experiment Tracking**: TensorBoard and Weights & Biases integration
- **Docker Support**: Multi-stage CPU/GPU images with docker-compose
- **CI/CD**: GitHub Actions for linting (ruff), testing (pytest), and container builds

---

## Installation

### Docker (Recommended)

```bash
# CPU image (~2GB)
./docker/build.sh

# GPU image (~8GB)
./docker/build.sh --gpu

# Or use docker-compose
docker-compose build
```

### Local Installation

**Prerequisites:** Python 3.10+, pip

```bash
# Option 1: pip (simplest)
pip install -r requirements.txt
pip install -e ".[dev]"

# Option 2: conda (recommended for GPU)
conda env create -f environment.yml
conda activate rl-unlearning
```

> **Note on Box2D:** LunarLander requires `swig` to be installed. On Ubuntu: `sudo apt install swig`. On macOS: `brew install swig`. If you don't need LunarLander, install `gymnasium[classic-control]` instead.

### Dev Containers

Open the project in VS Code and select **Reopen in Container** when prompted. The [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) provides a pre-configured environment with GPU support, ruff formatting, and all required extensions.

---

## Quick Start

```bash
# 1. Train a PPO agent on CartPole
python src/train.py

# 2. Run unlearning
python src/unlearn.py

# 3. Evaluate baseline vs. unlearned
python src/evaluate.py
```

All scripts use [Hydra](https://hydra.cc/) for configuration. Override any parameter from the command line:

```bash
python src/train.py env=lunarlander seed=123 training.total_timesteps=1_000_000
```

---

## Project Structure

```
rl-unlearning/
├── src/
│   ├── agents/                  # RL agent implementations
│   │   ├── networks.py          #   ActorCritic, MLP, MetaplasticityMask
│   │   └── ppo.py               #   PPOAgent with GAE
│   ├── unlearning/              # Unlearning methods
│   │   ├── trajectory_selective.py   #   Method 1: targeted negative learning
│   │   ├── strategy_inversion.py     #   Method 2: zero-shot policy inversion
│   │   └── retain_protection.py      #   Method 3: metaplasticity + distillation
│   ├── metrics/
│   │   └── unlearning_metrics.py     # AUFC, RSI, Selectivity, Privacy AUC
│   ├── utils/
│   │   ├── buffers.py           #   Replay, Trajectory, and Dual buffers
│   │   └── helpers.py           #   Seeding, checkpoints, logging
│   ├── train.py                 # Training entry point
│   ├── unlearn.py               # Unlearning entry point
│   ├── evaluate.py              # Evaluation & comparison
│   └── run_experiments.py       # Multi-experiment runner
├── configs/                     # Hydra configuration
│   ├── config.yaml              #   Main config (defaults + global settings)
│   ├── env/                     #   cartpole, lunarlander, acrobot
│   ├── agent/                   #   ppo, a2c
│   ├── unlearn/                 #   trajectory_selective, strategy_inversion, retain_protection
│   └── logging/                 #   tensorboard, wandb
├── tests/                       # pytest test suite
├── docker/                      # Dockerfile, Dockerfile.GPU, build.sh, run.sh
├── .github/workflows/           # CI (lint + test) and Docker build pipelines
├── .devcontainer/               # VS Code dev container
├── pyproject.toml               # Package metadata, ruff, pytest config
├── requirements.txt             # Core dependencies
├── environment.yml              # Conda environment
└── docker-compose.yml           # Multi-service compose (app, tensorboard, jupyter)
```

---

## Configuration

Experiments are configured via [Hydra](https://hydra.cc/) with composable YAML in `configs/`. The main config composes defaults from subgroups:

```yaml
# configs/config.yaml
defaults:
  - env: cartpole          # Environment (cartpole, lunarlander, acrobot)
  - agent: ppo             # Agent algorithm (ppo, a2c)
  - unlearn: trajectory_selective  # Unlearning method
  - logging: tensorboard   # Logging backend (tensorboard, wandb)
```

### Override Examples

```bash
# Different environment and seed
python src/train.py env=lunarlander seed=0

# Switch unlearning method
python src/unlearn.py unlearn=strategy_inversion

# Hyperparameter sweep (Hydra multirun)
python src/train.py -m env=cartpole,lunarlander,acrobot seed=0,1,2

# WandB logging
python src/train.py logging=wandb
```

### Key Hyperparameters (PPO)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE parameter |
| `clip_epsilon` | 0.2 | PPO clipping |
| `n_steps` | 2048 | Rollout length |
| `n_epochs` | 10 | PPO update epochs |
| `batch_size` | 64 | Minibatch size |
| `network.hidden_dims` | [64, 64] | MLP hidden layers |

---

## Training

```bash
# Default: PPO on CartPole for 500K steps
python src/train.py

# Custom run
python src/train.py env=lunarlander training.total_timesteps=1_000_000 seed=42

# Quick test (few steps, fast iteration)
python src/train.py training.total_timesteps=4096
```

Training outputs:
- **Checkpoints** → `experiments/checkpoints/<experiment_name>/`
- **Logs** → `experiments/logs/<experiment_name>/`
- **Trajectories** → `experiments/outputs/<experiment_name>/trajectories.pkl`

Monitor training with TensorBoard:

```bash
tensorboard --logdir experiments/logs

# Or via docker-compose
docker-compose up tensorboard
```

---

## Unlearning

Three complementary methods for selective behavior forgetting:

### 1. Trajectory-Selective Forgetting

Targeted negative learning signals with loss decomposition into forget/retain/regularization components.

```bash
python src/unlearn.py unlearn=trajectory_selective
```

### 2. Zero-Shot Strategy Inversion

Generates synthetic forget states via gradient-based, random, or evolutionary search — no world model needed.

```bash
python src/unlearn.py unlearn=strategy_inversion
```

### 3. Retain Protection

Metaplasticity masks protect critical weights; knowledge distillation from a frozen teacher network preserves retain performance. Used in combination with the other methods.

```bash
python src/unlearn.py unlearn=retain_protection
```

---

## Evaluation

Compare baseline vs. unlearned agents:

```bash
python src/evaluate.py
```

Outputs:
- Performance comparison (returns)
- Forget score comparison (trajectory similarity)
- Unlearning metrics summary
- Box plots saved to `experiments/outputs/`

---

## Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **AUFC** | Area Under Forgetting Curve — cumulative forgetting effectiveness | Higher = more forgetting |
| **Retain Stability Index** | Preservation of non-target behaviors | > 0.9 |
| **Selectivity** | Precision of forgetting vs. collateral damage | Higher = more precise |
| **Privacy AUC** | Membership inference resistance | > 0.5 |

---

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Lint
ruff check src/
```

---

## Docker

### docker-compose

```bash
# Start all services (app + tensorboard + jupyter)
docker-compose up -d

# Run training inside container
docker-compose run --rm rl-unlearning python src/train.py

# TensorBoard at http://localhost:6006
# Jupyter at http://localhost:8888
```

### Standalone

```bash
# Build
./docker/build.sh            # CPU (~2GB)
./docker/build.sh --gpu      # GPU (~8GB)

# Run
./docker/run.sh              # Interactive shell
./docker/run.sh -- python src/train.py  # Run command
```

### Environment Variables

Copy `.env.example` to `.env` and set your values:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `WANDB_API_KEY` | Weights & Biases API key |
| `CUDA_VISIBLE_DEVICES` | GPU device selection (default: 0) |

---

## Research

### Three-Pillar Framework

```
                    ┌──────────────────────┐
                    │    Trained Agent      │
                    │   (PPO on CartPole)   │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
    ┌─────────────────┐ ┌───────────────┐ ┌──────────────────┐
    │   Trajectory-    │ │  Zero-Shot    │ │     Retain       │
    │   Selective      │ │  Strategy     │ │   Protection     │
    │   Forgetting     │ │  Inversion    │ │                  │
    ├─────────────────┤ ├───────────────┤ ├──────────────────┤
    │ Loss decomp.    │ │ Gradient /    │ │ Metaplasticity   │
    │ Negative learning│ │ evolutionary  │ │ masks            │
    │ Gradient reversal│ │ state search  │ │ KD from teacher  │
    └─────────────────┘ └───────────────┘ └──────────────────┘
              │                │                 │
              └────────────────┼────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   Unlearned Agent     │
                    │  (forget ↓, retain ↑) │
                    └──────────────────────┘
```

### Environments

| Environment | Obs Dim | Act Dim | Type |
|-------------|---------|---------|------|
| CartPole-v1 | 4 | 2 | Discrete |
| LunarLander-v3 | 8 | 4 | Discrete |
| Acrobot-v1 | 6 | 3 | Discrete |

---

## Citation

```bibtex
@inproceedings{takacs2025unlearning,
  title={Selective Behavior Unlearning in On-Policy Reinforcement Learning},
  author={Takács, Tamás},
  year={2025}
}
```

---

## License

[MIT License](LICENSE)
