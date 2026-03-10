"""
Quick start script to run all experiments
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}\n")
    
    try:
        subprocess.run(cmd, shell=True, check=True, text=True)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with error code {e.returncode}")
        return False


def main():
    """Run complete experimental pipeline."""
    
    # Create a shared experiment name with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CartPole experiment with shared directories
    cartpole_exp = f"cartpole_ppo_{timestamp}"
    cartpole_checkpoint = f"experiments/checkpoints/{cartpole_exp}"
    cartpole_output = f"experiments/outputs/{cartpole_exp}"
    cartpole_log = f"experiments/logs/{cartpole_exp}"
    
    # LunarLander experiment with shared directories
    lunarlander_exp = f"lunarlander_ppo_{timestamp}"
    lunarlander_checkpoint = f"experiments/checkpoints/{lunarlander_exp}"
    lunarlander_output = f"experiments/outputs/{lunarlander_exp}"
    lunarlander_log = f"experiments/logs/{lunarlander_exp}"
    
    experiments = [
        # Train on CartPole
        (f"python src/train.py env=cartpole agent=ppo seed=42 "
         f"training.checkpoint_dir={cartpole_checkpoint} "
         f"output_dir={cartpole_output} "
         f"log_dir={cartpole_log}", 
         "Training CartPole"),
        
        # Train on LunarLander
        (f"python src/train.py env=lunarlander agent=ppo seed=42 "
         f"training.checkpoint_dir={lunarlander_checkpoint} "
         f"output_dir={lunarlander_output} "
         f"log_dir={lunarlander_log}",
         "Training LunarLander"),
        
        # Unlearn with trajectory-selective method
        (f"python src/unlearn.py env=cartpole unlearn=trajectory_selective "
         f"training.checkpoint_dir={cartpole_checkpoint} "
         f"output_dir={cartpole_output} "
         f"log_dir={cartpole_log}",
         "Unlearning (Trajectory-Selective) on CartPole"),
        
        # Evaluate
        (f"python src/evaluate.py env=cartpole "
         f"training.checkpoint_dir={cartpole_checkpoint} "
         f"output_dir={cartpole_output} "
         f"log_dir={cartpole_log}",
         "Evaluating CartPole"),
    ]
    
    print("Starting RL Unlearning Experiments")
    print(f"Total experiments: {len(experiments)}")
    print(f"CartPole experiment: {cartpole_exp}")
    print(f"LunarLander experiment: {lunarlander_exp}")
    
    results = []
    for cmd, desc in experiments:
        success = run_command(cmd, desc)
        results.append((desc, success))
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}\n")
    
    for desc, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {desc}")
    
    total_success = sum(1 for _, s in results if s)
    print(f"\nCompleted {total_success}/{len(results)} experiments successfully")
    
    return 0 if total_success == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
