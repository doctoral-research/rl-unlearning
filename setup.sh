#!/bin/bash
# Quick setup script

set -e

echo "Setting up RL Unlearning environment..."

# Check for Homebrew on macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &> /dev/null; then
        echo "Checking for SWIG (required for Box2D)..."
        if ! command -v swig &> /dev/null; then
            echo "Installing SWIG via Homebrew..."
            brew install swig
        else
            echo "✓ SWIG already installed"
        fi
    else
        echo "⚠️  Homebrew not found. Please install SWIG manually:"
        echo "   Visit: https://formulae.brew.sh/formula/swig"
        echo "   Or: brew install swig"
    fi
fi

# Create conda environment
if command -v conda &> /dev/null; then
    echo "Creating conda environment..."
    conda env create -f environment.yml
    echo "✓ Conda environment created"
    echo "Activate with: conda activate rl-unlearning"
else
    echo "Conda not found. Using pip..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -e .
    echo "✓ Virtual environment created"
    echo "Activate with: source venv/bin/activate"
fi

# Create necessary directories
mkdir -p experiments/checkpoints
mkdir -p experiments/logs
mkdir -p experiments/outputs
mkdir -p outputs

echo ""
echo "✓ Setup complete!"
echo ""
echo "To get started:"
echo "  1. Activate environment: conda activate rl-unlearning (or source venv/bin/activate)"
echo "  2. Train an agent: python src/train.py"
echo "  3. Run unlearning: python src/unlearn.py"
echo "  4. Evaluate: python src/evaluate.py"
echo ""
echo "Or run all experiments: python src/run_experiments.py"
