# Selective Behavior Unlearning in On-Policy Reinforcement Learning

A professional research framework for investigating **targeted online unlearning** in reinforcement learning agents.

## 🎯 Research Focus

This project implements and evaluates methods for **selectively forgetting undesired behaviors** (trajectories, action sequences, states) in on-policy RL agents while preserving learned knowledge in other parts of the state-action space.

### Three-Pillar Framework

1. **Trajectory-Selective Forgetting**: Targeted negative learning signals with loss decomposition
2. **Zero-Shot Strategy Inversion**: Policy-guided search for synthetic unlearning scenarios
3. **Retain Protection**: Metaplasticity masks and selective distillation for knowledge preservation

## 📊 Hypotheses

1. **Trajectory Separation**: With trajectory separation and loss component regulation, Retain Stability Index can be significantly improved at the same forgetting level
2. **Zero-Shot Inversion**: Strategy inversion works without world models via seed-optimization pushing policy toward forget regions with negative learning signals
3. **Metaplasticity Protection**: Metaplasticity masks + distillation keep retain degradation within 10%

## 📈 Metrics

- **AUFC** (Area Under Forgetting Curve): Measures forgetting effectiveness over time
- **Retain Stability Index**: Quantifies preservation of non-target behaviors
- **Selectivity**: Precision of targeted forgetting vs. collateral damage
- **Privacy AUC**: Privacy guarantees in trajectory removal

## 🧪 Environments

Testing across 5+ Gymnasium environments:
- CartPole-v1
- LunarLander-v2
- Acrobot-v1
- MountainCar-v0
- Pendulum-v1

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Build the container
docker-compose build

# Run training
docker-compose run --rm rl-unlearning python src/train.py

# Run unlearning experiment
docker-compose run --rm rl-unlearning python src/unlearn.py env=cartpole method=trajectory_selective

# Run full evaluation suite
docker-compose run --rm rl-unlearning python src/evaluate.py
```

### Local Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate rl-unlearning

# Install package
pip install -e .

# Run experiments
python src/train.py
python src/unlearn.py
python src/evaluate.py
```

## 📁 Project Structure

```
rl-unlearning/
├── src/
│   ├── agents/           # RL agents (PPO, A2C, etc.)
│   ├── unlearning/       # Unlearning methods
│   ├── metrics/          # Evaluation metrics
│   ├── utils/            # Utilities
│   ├── train.py          # Training script
│   ├── unlearn.py        # Unlearning script
│   └── evaluate.py       # Evaluation script
├── configs/              # Hydra configurations
├── experiments/          # Experiment outputs
├── tests/                # Unit tests
├── docker/               # Docker files
└── notebooks/            # Analysis notebooks
```

## 🔧 Configuration

All experiments are configured via Hydra. See `configs/` for details.

Example:
```bash
python src/train.py env=lunarlander agent=ppo seed=42
python src/unlearn.py method=trajectory_selective unlearn.target_trajectories=toxic
```

## 📊 Results & Visualization

Results are logged to:
- TensorBoard: `tensorboard --logdir experiments/logs`
- Wandb: Configure in `configs/logging/wandb.yaml`
- CSV: `experiments/results/`

## 🧪 Running Experiments

```bash
# Train baseline agents
python src/train.py -m env=cartpole,lunarlander,acrobot agent=ppo seed=0,1,2

# Run unlearning experiments
python src/unlearn.py -m method=trajectory_selective,strategy_inversion,retain_protection

# Evaluate and compare
python src/evaluate.py experiment_dir=experiments/results/
```

## 📝 Citation

```bibtex
@inproceedings{takacs2025unlearning,
  title={Selective Behavior Unlearning in On-Policy Reinforcement Learning},
  author={Takács, Tamás},
  year={2025}
}
```

## 📄 License

MIT License
