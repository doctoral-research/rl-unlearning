#!/bin/bash
# Alternative setup without Box2D environments

set -e

echo "Setting up RL Unlearning environment (without Box2D)..."

# Create environment file without box2d
cat > environment_minimal.yml << EOF
name: rl-unlearning
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - torch>=2.0.0
      - gymnasium>=0.29.0
      - numpy>=1.24.0
      - hydra-core>=1.3.0
      - omegaconf>=2.3.0
      - tensorboard>=2.14.0
      - wandb>=0.15.0
      - matplotlib>=3.7.0
      - seaborn>=0.12.0
      - pandas>=2.0.0
      - scikit-learn>=1.3.0
      - tqdm>=4.65.0
      - scipy>=1.11.0
      - pytest>=7.4.0
      - pytest-cov>=4.1.0
      - black>=23.7.0
      - isort>=5.12.0
      - flake8>=6.1.0
      - -e .
EOF

# Create conda environment
if command -v conda &> /dev/null; then
    echo "Creating minimal conda environment (CartPole, Acrobot, MountainCar, Pendulum only)..."
    conda env create -f environment_minimal.yml
    echo ""
    echo "✓ Environment created successfully!"
    echo ""
    echo "Activate with: conda activate rl-unlearning"
    echo ""
    echo "Note: LunarLander-v2 requires Box2D. To enable it:"
    echo "  1. Install SWIG: brew install swig"
    echo "  2. Install Box2D: pip install gymnasium[box2d]"
else
    echo "Error: Conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

echo ""
echo "Quick test: python -c 'import gymnasium; print(gymnasium.__version__)'"
