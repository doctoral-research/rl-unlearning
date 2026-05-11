<p align="center">
    <strong>Selective Behavior Unlearning in On-Policy Reinforcement Learning</strong>
</p>

<p align="center">
    <a href="https://github.com/doctoral-research/rl-unlearning/actions/workflows/ci.yml"><img src="https://github.com/doctoral-research/rl-unlearning/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
    <a href="https://github.com/doctoral-research/rl-unlearning/actions/workflows/docker.yml"><img src="https://github.com/doctoral-research/rl-unlearning/actions/workflows/docker.yml/badge.svg?branch=main" alt="Docker"></a>
    <a href="https://codecov.io/gh/doctoral-research/rl-unlearning"><img src="https://codecov.io/gh/doctoral-research/rl-unlearning/branch/main/graph/badge.svg" alt="codecov"></a>
    <a href="https://github.com/doctoral-research/rl-unlearning/issues"><img src="https://img.shields.io/github/issues/doctoral-research/rl-unlearning" alt="GitHub issues"></a>
    <a href="https://github.com/doctoral-research/rl-unlearning/blob/main/LICENSE"><img src="https://img.shields.io/github/license/doctoral-research/rl-unlearning" alt="License"></a>
    <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/pytorch-2.0+-orange" alt="PyTorch 2.0+">
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Unlearning Methods](#unlearning-methods)
- [Evaluation](#evaluation)
- [Plotting and Visualization](#plotting-and-visualization)
- [Metrics](#metrics)
- [Environments](#environments)
- [Preliminary Results](#preliminary-results)
- [Testing](#testing)
- [Docker](#docker)
- [Research](#research)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project implements and evaluates methods for **selectively forgetting undesired behaviors** in on-policy RL agents while preserving learned knowledge in other parts of the state-action space. We study three complementary approaches — trajectory-selective forgetting, zero-shot strategy inversion, and retain protection — and evaluate them across classic control environments using custom unlearning metrics.

The core challenge in RL unlearning is the **forget–retain tradeoff**: pushing the policy away from undesired behaviors without degrading performance on the rest of the task. Each method addresses this tradeoff from a different angle, and our evaluation framework measures both forgetting effectiveness and retain stability to quantify how well each method navigates this tradeoff.

### Research Hypotheses

1. **Trajectory Separation** — With trajectory separation and loss component regulation, Retain Stability Index can be significantly improved at the same forgetting level
2. **Zero-Shot Inversion** — Strategy inversion works without world models via seed-optimization, pushing policy toward forget regions with negative learning signals
3. **Metaplasticity Protection** — Metaplasticity masks + distillation keep retain degradation within 10%

---

## Key Features

- **Three-Pillar Framework**: Trajectory-selective forgetting, zero-shot strategy inversion, retain protection with metaplasticity
- **PPO Agent**: On-policy learning with GAE, orthogonal initialization, and clipped objectives
- **Hydra Configuration**: Composable YAML configs with environment-specific overrides (e.g. per-env training steps)
- **Custom Metrics**: AUFC, Retain Stability Index (degradation-based, handles negative/near-zero baselines), Selectivity, Privacy AUC
- **Comprehensive Visualization**: 8 per-experiment plots + cross-environment comparison heatmaps and dashboards
- **Early Stopping**: Retain stability threshold prevents catastrophic forgetting of non-target behaviors
- **Experiment Tracking**: TensorBoard and Weights & Biases integration
- **Docker Support**: Multi-stage CPU/GPU images with docker-compose, runs as host user (no root-owned files)
- **CI/CD**: GitHub Actions for linting (ruff), testing (pytest), and container builds

---

## Installation

### Docker (Recommended)

```bash
# CPU image (~2GB)
./docker/build.sh

# GPU image (~8GB)
./docker/build.sh --gpu

# Or use docker-compose (runs as your user, not root)
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

Open the project in VS Code and select **Reopen in Container** when prompted. The `.devcontainer/devcontainer.json` provides a pre-configured environment with GPU support, ruff formatting, and all required extensions.

---

## Quick Start

```bash
# 1. Train a PPO agent on CartPole (500K steps)
python src/train.py

# 2. Run all three unlearning methods
python src/unlearn.py unlearn=trajectory_selective
python src/unlearn.py unlearn=strategy_inversion
python src/unlearn.py unlearn=retain_protection

# 3. Evaluate baseline vs. all unlearned variants
python src/evaluate.py

# 4. Generate plots
python scripts/make_plots.py
```

All scripts use [Hydra](https://hydra.cc/) for configuration. Override any parameter from the command line:

```bash
python src/train.py env=lunarlander seed=123
```

### Full Experiment Pipeline (all environments)

```bash
# Train
python src/train.py env=cartpole
python src/train.py env=acrobot
python src/train.py env=lunarlander

# Unlearn (3 methods x 3 envs = 9 runs)
for env in cartpole acrobot lunarlander; do
  for method in trajectory_selective strategy_inversion retain_protection; do
    python src/unlearn.py env=$env unlearn=$method
  done
done

# Evaluate
for env in cartpole acrobot lunarlander; do
  python src/evaluate.py env=$env
done

# Plot everything + cross-environment comparison
python scripts/make_plots.py
```

---

## Project Structure

```text
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
│   └── scenarios.py             # Forget-region definitions (ForgetScenario)
├── scripts/                     # Sweep runners + plotting
│   ├── make_plots.py            #   Generate paper figures from experiments/outputs
│   ├── run_sweep.sh             #   Multi-seed/scenario sweep
│   ├── train_baselines.sh       #   Train baseline policies per environment
│   └── collect_trajectories.sh  #   Collect rollouts from trained policies
├── configs/                     # Hydra configuration
│   ├── config.yaml              #   Main config (defaults + global settings)
│   ├── env/                     #   cartpole, lunarlander, acrobot
│   ├── agent/                   #   ppo
│   ├── unlearn/                 #   trajectory_selective, strategy_inversion, retain_protection
│   ├── scenarios/               #   Per-env forget regions (e.g. cartpole/left_only)
│   └── logging/                 #   tensorboard, wandb
├── experiments/                 # Experiment outputs (generated)
│   ├── checkpoints/             #   Model weights (final + unlearned variants)
│   ├── logs/                    #   Training/unlearning/evaluation logs
│   └── outputs/                 #   Metrics JSON, evaluation CSVs, trajectories
├── docs/images/results/         # Paper figures written by make_plots.py
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
  - _self_                       # Base values applied first
  - env: cartpole                # Environment (can override base values)
  - agent: ppo                   # Agent algorithm
  - unlearn: trajectory_selective  # Unlearning method
  - logging: tensorboard         # Logging backend
```

Environment configs can override base training parameters. For example, `configs/env/lunarlander.yaml` overrides `training.total_timesteps` to 2M because LunarLander needs more training to converge.

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

### Experiment Naming

Experiments use a deterministic naming scheme: `{env}_{agent}_seed{seed}` (e.g. `cartpole_ppo_seed42`). All outputs — checkpoints, logs, metrics — are organized under this name, making it easy to match training runs with their unlearning and evaluation results.

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

# LunarLander (automatically uses 2M steps from env config)
python src/train.py env=lunarlander

# Acrobot (automatically uses 1M steps from env config)
python src/train.py env=acrobot
```

Training outputs:
- **Checkpoints** → `experiments/checkpoints/{experiment_name}/final_model.pt`
- **Logs** → `experiments/logs/{experiment_name}/train.log`
- **Trajectories** → `experiments/outputs/{experiment_name}/trajectories.pkl` (1000 episodes for unlearning)

Monitor training with TensorBoard:

```bash
tensorboard --logdir experiments/logs
# Or via docker-compose
docker-compose up tensorboard
# TensorBoard at http://localhost:6006
```

---

## Unlearning Methods

All three methods share common unlearning parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `unlearn_lr` | varies | Separate optimizer LR for unlearning (not the training LR) |
| `num_unlearn_steps` | 500 | Maximum unlearning optimization steps |
| `eval_frequency` | 25 | Evaluate metrics every N steps |
| `retain_stability_threshold` | 0.7 | Early stop if RSI drops below this |

### Method 1: Trajectory-Selective Forgetting

Targeted negative learning signals with explicit loss decomposition into forget, retain, and regularization components.

**How it works:**
1. **Trajectory identification** — Selects trajectories to forget based on state similarity, action sequences, or reward criteria. A configurable threshold determines which trajectories are classified as "forget" vs "retain".
2. **Gradient reversal** — For forget trajectories, the gradient is reversed: instead of minimizing negative log-likelihood (which would reinforce the behavior), we maximize it, pushing the policy *away* from the recorded actions.
3. **Loss decomposition** — The total loss is decomposed into three weighted components:
   - **Forget loss**: Gradient-reversed policy loss on forget trajectories (weighted by `forget_strength`)
   - **Retain loss**: Standard policy gradient loss on retain trajectories to preserve performance
   - **Regularization**: L2 regularization to prevent weight explosion
4. **Early stopping** — Evaluates retain stability every `eval_frequency` steps and stops if RSI drops below `retain_stability_threshold`.

```bash
python src/unlearn.py unlearn=trajectory_selective
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `unlearn_lr` | 5e-5 | Unlearning optimizer learning rate |
| `forget_strength` | 0.2 | Scaling factor for forget loss |
| `loss_weights.forget` | 1.0 | Weight for forget loss component |
| `loss_weights.retain` | 0.5 | Weight for retain loss (lower so forgetting can dominate) |
| `loss_weights.regularization` | 0.01 | Weight for L2 regularization |
| `negative_learning.gradient_reversal` | true | Reverse gradients for forget data |
| `target_selection.method` | similarity | How to select forget trajectories |
| `target_selection.threshold` | 0.8 | Similarity threshold for trajectory selection |

---

### Method 2: Zero-Shot Strategy Inversion

Generates synthetic forget states via policy-guided search — no world model or simulator access needed during unlearning.

**How it works:**
1. **Seed initialization** — Samples random states from the observation space as starting points.
2. **State optimization** — For each seed, optimizes the state to find regions where the policy exhibits strong preferences (low entropy / high confidence). Three search strategies are available:
   - **Gradient-based** (default): Treats the state as a differentiable input, backpropagates through the policy network, and uses Adam to find states that minimize policy entropy (i.e., states where the agent is most "sure" of its action choice).
   - **Random**: Samples perturbations around the seed and evaluates each against target criteria.
   - **Evolutionary**: Maintains a population of candidate states, selects top performers, crossover + mutation.
3. **Negative signal application** — Once high-confidence states are found, applies negative learning signals at those states (and optionally their neighbors within `neighbor_radius`) to disrupt the policy's confident behavior in those regions.
4. **Coverage** — Uses `num_seeds` (default 100) independent optimizations with `num_iterations` (default 50) steps each to ensure broad coverage of the policy's forget region.

```bash
python src/unlearn.py unlearn=strategy_inversion
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `unlearn_lr` | 5e-5 | Unlearning optimizer learning rate |
| `num_seeds` | 100 | Number of seed state optimizations |
| `num_iterations` | 50 | Optimization iterations per seed |
| `search_lr` | 0.01 | LR for state optimization (not policy updates) |
| `search.method` | gradient_based | Search strategy |
| `search.temperature` | 1.0 | Softmax temperature for entropy computation |
| `negative_signal.strength` | 0.2 | Strength of the unlearning signal |
| `negative_signal.apply_to_neighbors` | true | Also apply signal to neighboring states |

---

### Method 3: Retain Protection

Metaplasticity-inspired weight protection masks + knowledge distillation from a frozen teacher network. Designed to protect retain performance while other methods (or its own unlearning loss) push the policy to forget.

**How it works:**
1. **Importance scoring** — Computes per-parameter importance scores by accumulating gradient magnitudes on retain data. Parameters with large gradients on retain states are deemed "important" for retained behavior.
2. **Metaplasticity masks** — Thresholds importance scores (default: top 30% most important) to create binary masks. During backpropagation, gradients on masked (important) parameters are zeroed out, preventing unlearning updates from modifying the weights that matter most for retain performance. Masks are periodically re-computed (every `update_frequency` steps) and consolidated with previous masks using exponential moving average (`consolidation_strength`).
3. **Selective knowledge distillation** — A frozen copy of the pre-unlearning network acts as a teacher. On retain states, a KL-divergence loss between teacher and student policy outputs (with temperature scaling) anchors the student's behavior. The distillation weight (`alpha=0.2`) is intentionally low — the masks do the heavy lifting; distillation provides a soft safety net.
4. **Protected gradient updates** — Each unlearning step: (a) compute forget loss, (b) compute retain loss + distillation loss, (c) backpropagate combined loss, (d) apply metaplasticity masks to zero out gradients on important params, (e) clip and step.

```bash
python src/unlearn.py unlearn=retain_protection
```

**Key parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `unlearn_lr` | 1e-4 | Higher LR — masks protect important params |
| `metaplasticity.mask_type` | importance | Mask computation method |
| `metaplasticity.threshold` | 0.7 | Protect top 30% most important params |
| `metaplasticity.update_frequency` | 250 | Re-compute masks every N steps |
| `metaplasticity.consolidation_strength` | 0.9 | EMA weight for mask consolidation |
| `distillation.temperature` | 2.0 | KL-divergence temperature |
| `distillation.alpha` | 0.2 | Distillation loss weight |
| `distillation.only_retain_states` | true | Only distill on retain data |

---

## Evaluation

Compare baseline vs. all unlearned agents automatically:

```bash
# Evaluates all unlearned_*.pt found alongside final_model.pt
python src/evaluate.py env=cartpole
```

The evaluation script:
1. Loads the baseline model and all unlearned variants from the checkpoint directory
2. Runs 50 evaluation episodes per model
3. Computes trajectory similarity (forget scores) against stored trajectories
4. Calculates all unlearning metrics (AUFC, RSI, Selectivity)
5. Prints a comparison table and saves results

Outputs:
- `experiments/outputs/{name}/evaluation_results.csv` — Per-episode returns for all models
- `experiments/outputs/{name}/evaluation_metrics.json` — Final metric values per method
- `experiments/outputs/{name}/evaluation_comparison.png` — Quick comparison plot

---

## Plotting and Visualization

`scripts/make_plots.py` generates the paper figures from `experiments/outputs/` across seeds `{42, 43, 44}` and writes them to `docs/images/results/`:

```bash
python scripts/make_plots.py
```

### Per-Experiment Plots (8 plots per environment)

| Plot | Description |
|------|-------------|
| `training_curve.png` | Mean episode return over training steps |
| `return_distributions.png` | Box + violin plots of eval returns across all models |
| `metrics_radar.png` | Radar chart comparing methods on 4 metrics |
| `metrics_bars.png` | Bar chart with numeric values for each metric |
| `unlearning_dynamics.png` | Forget effectiveness, RSI, selectivity, and forget score over unlearning steps |
| `loss_curves.png` | All loss components (forget, retain, regularization, distillation) over time |
| `forget_retain_tradeoff.png` | Forget effectiveness vs. retain stability trajectory — shows the Pareto frontier |
| `dashboard.png` | Single-page summary combining training, evaluation, and dynamics |

### Cross-Environment Comparison Plots

| Plot | Description |
|------|-------------|
| `cross_env_metrics.png` | Side-by-side bar charts of metrics across environments |
| `cross_env_returns.png` | Return distributions across environments |
| `cross_env_training.png` | Training curves side by side |
| `heatmap_forget-effectiveness.png` | Methods x Environments heatmap |
| `heatmap_retain-stability-index.png` | Methods x Environments heatmap |
| `heatmap_selectivity.png` | Methods x Environments heatmap |

All plots are saved to `docs/images/results/` (separate from `experiments/` to avoid permission issues in Docker).

---

## Metrics

| Metric | Formula | Range | Target | Description |
|--------|---------|-------|--------|-------------|
| **AUFC** | Area under forget score curve over time | [0, 1] | Lower = more forgetting | Cumulative forgetting effectiveness; measures how quickly and completely forget scores decrease |
| **Forget Effectiveness** | 0.5 * (fraction below threshold) + 0.5 * (1 - mean score) | [0, 1] | Higher = more effective | Combines binary threshold check with magnitude of score reduction |
| **Retain Stability Index** | max(0, 1 - \|current - baseline\| / scale) | [0, 1] | > 0.9 | Degradation-based metric using `scale = max(\|baseline\|, std, 1.0)`. Works correctly for negative baselines (Acrobot), near-zero baselines (early LunarLander), and positive baselines (CartPole) |
| **Selectivity** | forget_effectiveness * retain_stability | [0, 1] | Higher = more precise | Joint measure: high only when forgetting is effective AND retain is preserved |
| **Privacy AUC** | 1 - ROC AUC of membership inference attack | [0, 1] | > 0.5 | Resistance to membership inference; 0.5 = random guessing (ideal) |

### Retain Stability Index (RSI)

The RSI uses a degradation-based formula rather than a simple ratio to handle environments with different reward scales:

```text
scale = max(|baseline_mean|, baseline_std, 1.0)
RSI = max(0, 1 - |current_mean - baseline_mean| / scale)
```

This correctly handles:
- **Positive baselines** (CartPole ~500): scale = 500, small absolute changes = small RSI drop
- **Negative baselines** (Acrobot ~ -80): scale = 80, uses absolute value to avoid sign issues
- **Near-zero baselines** (LunarLander early training ~0): scale defaults to max(std, 1.0), preventing division-by-zero

---

## Environments

| Environment | Obs Dim | Act Dim | Type | Training Steps | Baseline Return | Notes |
|-------------|---------|---------|------|---------------|-----------------|-------|
| CartPole-v1 | 4 | 2 | Discrete | 500K | ~500 (max) | Easiest; converges quickly, ideal for method development |
| Acrobot-v1 | 6 | 3 | Discrete | 1M | ~ -80 to -95 | Negative rewards; tests RSI with negative baselines |
| LunarLander-v3 | 8 | 4 | Discrete | 2M | ~200+ | Hardest; needs more training, continuous state space more complex |

Training steps are configured per-environment in `configs/env/*.yaml` and automatically override the base config.

---

## Preliminary Results

Results from seed 42 experiments across all three environments:

### Final Metrics Summary

| Environment | Method | Forget Eff. | Retain Stab. | Selectivity |
|-------------|--------|-------------|-------------|-------------|
| **CartPole** | Trajectory Selective | **0.857** | **1.000** | **0.857** |
| | Strategy Inversion | 0.456 | 1.000 | 0.456 |
| | Retain Protection | 0.835 | 0.981 | 0.819 |
| **Acrobot** | Trajectory Selective | 0.064 | 0.940 | 0.060 |
| | Strategy Inversion | 0.071 | **0.991** | 0.070 |
| | Retain Protection | **0.096** | 0.847 | **0.082** |
| **LunarLander** | Trajectory Selective | **0.319** | 0.767 | 0.245 |
| | Strategy Inversion | 0.287 | **0.971** | **0.278** |
| | Retain Protection | 0.287 | 0.743 | 0.213 |

### Key Findings

**CartPole** is the clearest success. Trajectory Selective achieves 0.857 selectivity — strong forgetting with zero retain degradation. The forget-retain tradeoff plots show all methods riding along RSI=1.0 as forgetting increases, with a sharp cliff when they push too far. Early stopping at `retain_stability_threshold=0.7` catches this collapse.

**Acrobot** reveals a limitation: all methods achieve near-zero forget effectiveness (0.06–0.10). The agent's learned behavior is deeply entrenched in the network weights, and the unlearning signal is too weak to dislodge it. Retain stability remains high precisely because the methods aren't changing the policy much.

**LunarLander** shows the forget-retain tension most clearly. Trajectory Selective forgets the most (0.319) but pays for it in retain stability (0.767). Strategy Inversion achieves the best selectivity (0.278) by preserving retain much better (0.971), even though its raw forgetting is slightly weaker. Retain Protection early-stops after just 2 steps, suggesting the retain threshold triggers too aggressively in this environment.

### Interpretation

- **Trajectory Selective** is the strongest forgetter overall and the best method when retain stability is naturally robust (e.g., CartPole). Its gradient reversal + loss decomposition is the most direct approach.
- **Strategy Inversion** is the most conservative — it barely touches the policy on retain states, which gives it the best retain stability but at the cost of weaker forgetting. Its synthetic state generation may not be finding the right states to target.
- **Retain Protection** lives up to its name on CartPole (0.981 RSI) but suffers from early stopping on harder environments, cutting unlearning short before meaningful forgetting occurs.

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

The docker-compose setup runs containers as your host user (no root-owned files):

```bash
# Start all services (app + tensorboard + jupyter)
docker-compose up -d

# Run training inside container
docker-compose run --rm rl-unlearning python src/train.py

# TensorBoard at http://localhost:6006
# Jupyter at http://localhost:8888
```

The `.env` file contains your `USER_ID` and `GROUP_ID` so Docker creates files with your ownership. Generate it with:

```bash
echo "USER_ID=$(id -u)" > .env
echo "GROUP_ID=$(id -g)" >> .env
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

| Variable | Description |
|----------|-------------|
| `USER_ID` | Your host user ID (for file ownership) |
| `GROUP_ID` | Your host group ID (for file ownership) |
| `WANDB_API_KEY` | Weights & Biases API key |
| `CUDA_VISIBLE_DEVICES` | GPU device selection (default: 0) |

---

## Research

### Three-Pillar Framework

```text
                    ┌──────────────────────┐
                    │    Trained Agent      │
                    │   (PPO on env)        │
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
    │                 │ │ Synthetic     │ │ Protected grads  │
    │ forget/retain/  │ │ state gen.    │ │ Importance-based │
    │ reg. weighting  │ │               │ │ early stopping   │
    └─────────────────┘ └───────────────┘ └──────────────────┘
              │                │                 │
              └────────────────┼────────────────┘
                               ▼
                    ┌──────────────────────┐
                    │   Unlearned Agent     │
                    │  (forget ↓, retain ↑) │
                    └──────────────────────┘
                               │
                    ┌──────────────────────┐
                    │   Evaluation         │
                    │  AUFC, RSI, Select.  │
                    │  + visualization     │
                    └──────────────────────┘
```

---

## Citation

```bibtex
@inproceedings{takacs2026unlearning,
  title={Targeted Behavioral Unlearning for Discrete On-Policy Reinforcement Learning},
  author={Takács, Tamás},
  booktitle={International Conference on Intelligent Robotics (IntRob)},
  year={2026}
}
```

---

## License

[MIT License](LICENSE)
