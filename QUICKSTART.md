# Quick Reference Guide

## 🎯 What This Project Does

This is a **complete, production-ready research framework** for investigating **Selective Behavior Unlearning in On-Policy Reinforcement Learning**. It implements three novel unlearning methods and evaluates them across 5 Gymnasium environments.

## 🏗️ Three-Pillar Framework

### 1️⃣ Trajectory-Selective Forgetting
**Purpose**: Forget specific trajectories or action sequences using targeted negative learning.

**Key Features**:
- Loss decomposition (forget/retain/regularization)
- Gradient reversal for negative signals
- Similarity-based trajectory selection
- Configurable forget strength

**When to use**: When you have specific trajectories or behaviors to remove.

### 2️⃣ Zero-Shot Strategy Inversion  
**Purpose**: Generate synthetic "forget states" without world models using policy-guided search.

**Key Features**:
- Gradient-based state optimization
- Multiple search strategies (gradient/random/evolutionary)
- No world model required
- Pushes policy toward target regions

**When to use**: When you don't have explicit forget data but know characteristics of undesired states.

### 3️⃣ Retain Protection
**Purpose**: Protect important knowledge while unlearning using metaplasticity and distillation.

**Key Features**:
- Importance-based parameter masking
- Knowledge distillation from teacher
- Adaptive mask consolidation
- Stability monitoring

**When to use**: Always! Combine with other methods to minimize collateral damage.

## 📊 Evaluation Metrics

All implemented in `src/metrics/unlearning_metrics.py`:

1. **AUFC** (Area Under Forgetting Curve)
   - Measures: Forgetting effectiveness over time
   - Range: [0, 1], higher = better forgetting
   
2. **Retain Stability Index (RSI)**
   - Measures: Preservation of non-target behaviors
   - Range: [0, 1], 1 = perfect preservation
   - Target: Keep within 10% degradation
   
3. **Selectivity**
   - Measures: Precision of targeted forgetting
   - Formula: forget_effectiveness × retain_stability
   - Range: [0, 1], higher = more selective
   
4. **Privacy AUC**
   - Measures: Membership inference resistance
   - Range: [0, 1], lower = better privacy

## 🚀 Quick Commands

### Installation
```bash
# Clone and setup
cd /Users/takacstamas/Work/unlearning/rl-unlearning
./setup.sh
conda activate rl-unlearning
```

### Run Everything
```bash
python src/run_experiments.py
```

### Individual Runs

**Train a baseline**:
```bash
python src/train.py env=cartpole agent=ppo seed=42
```

**Apply unlearning**:
```bash
# Trajectory-selective
python src/unlearn.py env=cartpole unlearn=trajectory_selective

# Strategy inversion
python src/unlearn.py env=cartpole unlearn=strategy_inversion

# Retain protection
python src/unlearn.py env=cartpole unlearn=retain_protection
```

**Evaluate**:
```bash
python src/evaluate.py env=cartpole
```

### Multi-Environment Sweep
```bash
python src/train.py -m env=cartpole,lunarlander,acrobot,mountaincar,pendulum
```

### Multi-Seed Runs
```bash
python src/train.py -m seed=0,1,2,3,4
```

## 🎮 Supported Environments

| Environment | Type | Action Space | Difficulty |
|------------|------|--------------|------------|
| CartPole-v1 | Classic | Discrete (2) | Easy |
| LunarLander-v2 | Box2D | Discrete (4) | Medium |
| Acrobot-v1 | Classic | Discrete (3) | Medium |
| MountainCar-v0 | Classic | Discrete (3) | Hard |
| Pendulum-v1 | Classic | Continuous (1) | Medium |

## ⚙️ Configuration Examples

### Modify Forget Strength
```bash
python src/unlearn.py unlearn.forget_strength=0.8
```

### Change Network Architecture
```bash
python src/train.py agent.network.hidden_dims=[128,128,64]
```

### Adjust Learning Rate
```bash
python src/train.py env.learning_rate=0.001
```

### Enable Wandb Logging
```bash
python src/train.py logging=wandb logging.wandb.project=my-project
```

## 📁 Where Things Are

### Inputs
- **Configs**: `configs/`
- **Source code**: `src/`

### Outputs
- **Checkpoints**: `experiments/checkpoints/`
- **Logs**: `experiments/logs/`
- **Results**: `experiments/outputs/`
- **Trajectories**: `experiments/outputs/*/trajectories.pkl`

## 🐛 Common Issues

### Import Errors
```bash
pip install -e .
```

### Missing Dependencies
```bash
pip install gymnasium[box2d]
```

### CUDA Not Available
```bash
python src/train.py device=cpu
```

### Hydra Config Errors
Check YAML syntax in `configs/` files.

## 📈 Monitoring

### TensorBoard
```bash
tensorboard --logdir experiments/logs
# Open http://localhost:6006
```

### Wandb (if configured)
Logs automatically to wandb.ai

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_agents.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 🐳 Docker Usage

### Build
```bash
docker-compose build
```

### Run Training
```bash
docker-compose run --rm rl-unlearning python src/train.py
```

### Interactive Shell
```bash
docker-compose run --rm rl-unlearning bash
```

### TensorBoard
```bash
docker-compose up tensorboard
# Open http://localhost:6006
```

## 📚 Key Files

### Must Read
- `README.md` - Project overview
- `STRUCTURE.md` - Detailed structure guide
- `EXPERIMENTS.md` - Experiment tracking

### Core Implementation
- `src/agents/ppo.py` - PPO agent
- `src/unlearning/trajectory_selective.py` - Method 1
- `src/unlearning/strategy_inversion.py` - Method 2
- `src/unlearning/retain_protection.py` - Method 3
- `src/metrics/unlearning_metrics.py` - All metrics

### Configuration
- `configs/config.yaml` - Main config
- `configs/env/*.yaml` - Environment settings
- `configs/unlearn/*.yaml` - Unlearning methods

## 🎓 Research Workflow

1. **Baseline** → Train agents on all environments
2. **Collect** → Trajectories saved automatically  
3. **Identify** → Define toxic states in configs
4. **Unlearn** → Apply three methods
5. **Evaluate** → Compare metrics
6. **Analyze** → Use notebooks

## 💡 Extension Examples

### Add New Environment
```yaml
# configs/env/my_env.yaml
name: my_env
env_id: MyEnv-v0
observation_dim: 8
action_dim: 4
action_type: discrete
```

### Add New Metric
```python
# src/metrics/unlearning_metrics.py
def compute_my_metric(self, data):
    # Your implementation
    return score
```

### Combine Methods
```python
# In unlearn.py
retain_protection = RetainProtection(agent)
trajectory_method = TrajectorySelectiveForgetting(agent)

# Use retain protection with trajectory method
metrics = retain_protection.protected_update(
    forget_batch, retain_batch, trajectory_method
)
```

## 📊 Expected Results

After running experiments, you should see:

1. **Training logs** showing increasing rewards
2. **Checkpoints** saved every 50k steps
3. **Evaluation CSV** with baseline vs unlearned comparison
4. **Plots** showing performance and forget scores
5. **Metrics** including AUFC, RSI, Selectivity

## 🏆 Success Criteria

Based on your hypotheses:

- ✓ RSI > 0.9 (retain within 10% degradation)
- ✓ Forget effectiveness > 0.7
- ✓ Selectivity > 0.6
- ✓ AUFC showing consistent forgetting trend

## 📞 Getting Help

1. Check error messages carefully
2. Review `STRUCTURE.md` for architecture
3. Look at `configs/` for settings
4. Run tests to verify installation
5. Check GitHub issues (if public)

## 🎉 You're Ready!

Everything is set up and ready to run. Start with:

```bash
./setup.sh
conda activate rl-unlearning
python src/run_experiments.py
```

Good luck with your research! 🚀
