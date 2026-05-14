"""Vectorized PPO training for continuous-action envs.

Separate from `src/train.py` so the existing 5-env discrete-action
pipeline stays intact. This script:
- Uses gymnasium.vector.SyncVectorEnv (N parallel envs per step)
- Pairs with bounded gSDE and obs normalization (already in env_wrappers)
- Clips actions to env action_space bounds
- Normalizes returns (running RMS) to stabilize value function

Why a separate script: vectorized rollouts have a different shape
((N, T, ...) instead of (T, ...)) and the existing train.py wasn't
designed for that. Forcing both into one path adds complexity to a
proven code path.

Usage:
    python src/train_vec.py env=halfcheetah seed=42 \\
        num_envs=8 experiment_name=halfcheetah_ppo_seed42

Continues to use the same hydra config layout as train.py; just add
`num_envs` (default 8) and the rest follows.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from agents import PPOAgent  # noqa: E402
from utils import setup_logger, get_device, set_seed  # noqa: E402
from utils.env_wrappers import make_env  # noqa: E402

logger = logging.getLogger(__name__)


def _make_env_fn(cfg: DictConfig, seed: int):
    def _thunk():
        return make_env(
            cfg.env_id, seed=seed,
            exploration_bonus=cfg.get("exploration_bonus", None),
            exploration_beta=cfg.get("exploration_beta", 0.05),
            exploration_anneal_steps=cfg.get("exploration_anneal_steps", 0),
            exploration_hash_scale=cfg.get("exploration_hash_scale", 1.0),
            obs_encoding=cfg.get("obs_encoding", "image"),
            fixed_goal_pos=cfg.get("fixed_goal_pos", None),
            normalize_obs=cfg.get("normalize_obs", False),
        )
    return _thunk


class RunningRMS:
    """Welford running mean/var for normalizing returns. Maintained on
    the value-function target side, not the reward side, so the policy
    sees raw env reward but the critic sees stabilized targets."""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray):
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        n = float(x.size)
        delta = batch_mean - self.mean
        tot = self.count + n
        self.mean += delta * n / tot
        m_a = self.var * self.count
        m_b = batch_var * n
        M2 = m_a + m_b + (delta ** 2) * self.count * n / tot
        self.var = M2 / tot
        self.count = tot


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train_vec(cfg: DictConfig):
    setup_logger("train_vec")
    set_seed(cfg.seed)
    device = get_device()
    logger.info(f"Vectorized PPO training | device={device}")
    logger.info(OmegaConf.to_yaml(cfg))

    num_envs = int(cfg.get("num_envs", 8))
    n_steps = int(cfg.get("n_steps", 1024))
    total_steps = int(cfg.training.total_timesteps)
    eval_freq = int(cfg.training.eval_frequency)
    action_bounds = None  # set below for continuous envs

    # Vectorized training envs. Procgen is natively vectorized via its
    # own ProcgenAdapter; everything else uses gymnasium's SyncVectorEnv.
    if cfg.env_id == "procgen":
        from utils.env_wrappers import ProcgenAdapter
        env = ProcgenAdapter(
            env_name=cfg.get("procgen_env", "coinrun"),
            num_envs=num_envs,
            num_levels=cfg.get("procgen_num_levels", 0),
            distribution_mode=cfg.get("procgen_distribution_mode", "easy"),
        )
        # Eval uses a single procgen env
        eval_env = ProcgenAdapter(
            env_name=cfg.get("procgen_env", "coinrun"),
            num_envs=1,
            num_levels=cfg.get("procgen_num_levels", 0),
            distribution_mode=cfg.get("procgen_distribution_mode", "easy"),
        )
    else:
        env = SyncVectorEnv(
            [_make_env_fn(cfg, cfg.seed + i) for i in range(num_envs)],
        )
        # Single-env for eval (separate, doesn't share obs-norm state with training)
        eval_env = make_env(
            cfg.env_id, seed=cfg.seed + 99999,
            normalize_obs=cfg.get("normalize_obs", False),
        )
    if cfg.action_type == "continuous":
        a_space = env.single_action_space
        action_bounds = (
            torch.as_tensor(a_space.low, dtype=torch.float32, device=device),
            torch.as_tensor(a_space.high, dtype=torch.float32, device=device),
        )

    agent = PPOAgent(
        observation_dim=cfg.observation_dim,
        action_dim=cfg.action_dim,
        action_type=cfg.action_type,
        hidden_dims=cfg.network.hidden_dims,
        activation=cfg.network.activation,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_epsilon=cfg.clip_epsilon,
        clip_value=cfg.clip_value,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        max_grad_norm=cfg.max_grad_norm,
        n_epochs=cfg.n_epochs,
        batch_size=cfg.batch_size,
        device=str(device),
        use_sde=cfg.get("use_sde", False),
        sde_sample_freq=cfg.get("sde_sample_freq", 4),
        sde_log_std_init=cfg.get("sde_log_std_init", -2.0),
        tanh_squash=cfg.get("tanh_squash", False),
        action_scale=cfg.get("action_scale", 1.0),
        use_conv=cfg.get("use_conv", False),
    )

    ret_rms = RunningRMS()
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_eval = -float("inf")  # Track best eval; save best_model.pt accordingly

    obs, _ = env.reset(seed=cfg.seed)
    global_step = 0
    pbar = tqdm(total=total_steps, desc="train-vec")
    next_eval = eval_freq

    while global_step < total_steps:
        # Rollout: (T, N, ...) buffers
        buf_obs = np.zeros((n_steps, num_envs) + env.single_observation_space.shape, dtype=np.float32)
        buf_act = np.zeros((n_steps, num_envs, agent.network.action_dim if agent.action_type == "continuous"
                             else 1), dtype=np.float32 if agent.action_type == "continuous" else np.int64)
        buf_logp = np.zeros((n_steps, num_envs), dtype=np.float32)
        buf_val = np.zeros((n_steps, num_envs), dtype=np.float32)
        buf_rew = np.zeros((n_steps, num_envs), dtype=np.float32)
        buf_done = np.zeros((n_steps, num_envs), dtype=np.float32)

        for t in range(n_steps):
            if agent.network.use_sde and t % agent.sde_sample_freq == 0:
                agent.network.sample_sde_noise()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, log_prob, _, value = agent.network.get_action_and_value(obs_t)
            if action_bounds is not None:
                action = torch.max(torch.min(action, action_bounds[1]), action_bounds[0])
            action_np = action.cpu().numpy()
            buf_obs[t] = obs
            buf_act[t] = action_np if agent.action_type == "continuous" else action_np[:, None]
            buf_logp[t] = log_prob.cpu().numpy()
            buf_val[t] = value.squeeze(-1).cpu().numpy()
            obs, reward, term, trunc, _ = env.step(action_np if agent.action_type == "continuous"
                                                   else action_np.astype(np.int64))
            done = np.logical_or(term, trunc).astype(np.float32)
            buf_rew[t] = reward
            buf_done[t] = done
            global_step += num_envs
            pbar.update(num_envs)

        # Bootstrap from last obs
        with torch.no_grad():
            next_value = agent.network.get_value(
                torch.as_tensor(obs, dtype=torch.float32, device=device),
            ).squeeze(-1).cpu().numpy()

        # GAE per env, then flatten
        advantages = np.zeros_like(buf_rew)
        lastgae = np.zeros(num_envs)
        for t in reversed(range(n_steps)):
            next_v = next_value if t == n_steps - 1 else buf_val[t + 1]
            next_nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + agent.gamma * next_v * next_nonterm - buf_val[t]
            lastgae = delta + agent.gamma * agent.gae_lambda * next_nonterm * lastgae
            advantages[t] = lastgae
        returns = advantages + buf_val

        # Return normalization (stabilizes value function on high-variance envs)
        ret_rms.update(returns)
        std_ret = max(np.sqrt(ret_rms.var), 1e-6)
        returns_norm = returns / std_ret
        buf_val_norm = buf_val / std_ret

        # Flatten (T, N, ...) -> (T*N, ...)
        flat_obs = buf_obs.reshape(-1, *env.single_observation_space.shape)
        flat_act = buf_act.reshape(-1, *buf_act.shape[2:])
        if agent.action_type == "discrete":
            flat_act = flat_act.squeeze(-1)
        flat_logp = buf_logp.reshape(-1)
        flat_val = buf_val_norm.reshape(-1)
        flat_adv = advantages.reshape(-1) / std_ret
        flat_ret = returns_norm.reshape(-1)

        rollout = {
            "observations": flat_obs,
            "actions": flat_act,
            "log_probs": flat_logp,
            "values": flat_val,
            "advantages": flat_adv,
            "returns": flat_ret,
        }
        agent.update(rollout)

        # Eval
        if global_step >= next_eval:
            ev_returns = []
            is_procgen = hasattr(eval_env, "_env") and "Procgen" in type(eval_env).__name__
            for ep in range(8):
                if is_procgen:
                    e_obs, _ = eval_env.reset()
                else:
                    e_obs, _ = eval_env.reset(seed=cfg.seed * 7919 + ep)
                e_ret = 0.0
                for _ in range(1000):
                    if is_procgen:
                        # e_obs is already batched (1, 64, 64, 3)
                        e_obs_t = torch.as_tensor(e_obs, dtype=torch.float32, device=device)
                    else:
                        e_obs_t = torch.as_tensor(e_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    with torch.no_grad():
                        # Stochastic eval: matches training distribution (and
                        # for tanh-squash + action-scale continuous policies,
                        # applies the same tanh+scale transformation). Using
                        # raw mean here breaks Pendulum because mean stays near
                        # zero while the trained policy actually applies torque
                        # via tanh(mean+noise)*scale.
                        action_t, _, _, _ = agent.network.get_action_and_value(e_obs_t)
                    if agent.action_type == "discrete":
                        a_np = action_t.cpu().numpy()
                        if is_procgen:
                            a_np = a_np.astype(np.int32)
                        else:
                            a_np = int(a_np[0])
                    else:
                        a_np = action_t.squeeze(0).cpu().numpy()
                        if action_bounds is not None:
                            a_np = np.clip(a_np, eval_env.action_space.low, eval_env.action_space.high)
                    e_obs, e_r, e_t, e_tr, _ = eval_env.step(a_np)
                    e_ret += float(np.asarray(e_r).mean())
                    if (np.asarray(e_t).any() or np.asarray(e_tr).any()):
                        break
                ev_returns.append(e_ret)
            mean_ret = float(np.mean(ev_returns))
            logger.info(f"Eval at step {global_step}: mean return {mean_ret:.2f}")
            # Periodic checkpoint + best-so-far snapshot. Lets us pick the
            # peak policy after training if reward bounces (Procgen / sparse
            # envs are particularly prone to PPO instability post-peak).
            torch.save({"network": agent.network.state_dict(),
                        "eval_step": global_step, "eval_mean_return": mean_ret},
                       ckpt_dir / f"checkpoint_step{global_step}.pt")
            if mean_ret > best_eval:
                best_eval = mean_ret
                torch.save({"network": agent.network.state_dict(),
                            "eval_step": global_step, "eval_mean_return": mean_ret},
                           ckpt_dir / "best_model.pt")
                logger.info(f"  -> new best at step {global_step}: {mean_ret:.2f}")
            next_eval += eval_freq

    # Save final checkpoint
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"network": agent.network.state_dict()}, ckpt_dir / "final_model.pt")
    logger.info(f"Saved final model to {ckpt_dir / 'final_model.pt'}")


if __name__ == "__main__":
    train_vec()
