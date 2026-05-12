"""Relearn-resistance attack on an unlearned policy.

Take a checkpoint produced by online_unlearn.py and continue training
it under standard PPO with the env's normal reward (no unlearning loss,
no disjoint objective, no penalty). Measure how fast the forget
behavior returns. If it returns within a few hundred rollouts, the
unlearning was cosmetic — the structure that produced the forget
behavior was preserved and only the output-layer activations changed.

This is the Brittle Unlearning diagnostic for the on-policy online
setting.

Usage:
    python src/relearn_attack.py env=fourrooms scenario=fourrooms/avoid_bottom_right \\
        +unlearned_checkpoint=experiments/outputs/online_unlearn_fourrooms_seed42_noop_disjoint/unlearned_no_op.pt \\
        +relearn_iters=200 +rollout_len=512 +run_suffix=noop_disjoint
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from agents import PPOAgent  # noqa: E402
from scenarios import ForgetScenario  # noqa: E402
from utils import setup_logger, get_device, set_seed  # noqa: E402
from utils.env_wrappers import make_env  # noqa: E402

logger = logging.getLogger(__name__)


def transitions_in_forget_region(obs: np.ndarray, scenario: ForgetScenario) -> np.ndarray:
    return np.array([scenario.matches_state(np.asarray(o)) for o in obs], dtype=bool)


def eval_agent(agent, env, scenario, baseline_return, num_episodes=16, seed=0):
    returns = []
    in_forget_total = 0
    total_steps = 0
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        for _ in range(500):
            with torch.no_grad():
                a, _ = agent.select_action(obs, deterministic=False)
            obs, r, term, trunc, _ = env.step(a)
            ep_return += float(r)
            total_steps += 1
            if scenario.matches_state(np.asarray(obs)):
                in_forget_total += 1
            if term or trunc:
                break
        returns.append(ep_return)
    mean_return = float(np.mean(returns))
    region_frac = in_forget_total / max(total_steps, 1)
    forget_eff = 1.0 - region_frac
    scale = max(abs(baseline_return), 1.0)
    retain_stab = max(0.0, 1.0 - abs(mean_return - baseline_return) / scale)
    return {
        "mean_return": mean_return,
        "forget_region_fraction": region_frac,
        "forget_effectiveness": forget_eff,
        "retain_stability": retain_stab,
    }


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def relearn_attack(cfg: DictConfig):
    setup_logger("relearn_attack")
    set_seed(cfg.seed)
    device = get_device()
    logger.info("Relearn-resistance attack")
    logger.info(f"Unlearned ckpt: {cfg.unlearned_checkpoint}")

    env = make_env(
        cfg.env_id, seed=cfg.seed,
        obs_encoding=cfg.get("obs_encoding", "image"),
        fixed_goal_pos=cfg.get("fixed_goal_pos", None),
    )
    eval_env = make_env(
        cfg.env_id, seed=cfg.seed + 99999,
        obs_encoding=cfg.get("obs_encoding", "image"),
        fixed_goal_pos=cfg.get("fixed_goal_pos", None),
    )
    agent = PPOAgent(
        observation_dim=cfg.observation_dim,
        action_dim=cfg.action_dim,
        device=str(device),
        learning_rate=cfg.get("learning_rate", 3e-4),
        gamma=cfg.get("gamma", 0.99),
        gae_lambda=cfg.get("gae_lambda", 0.95),
        clip_epsilon=cfg.get("clip_epsilon", 0.2),
        n_epochs=cfg.get("n_epochs", 5),
        batch_size=cfg.get("batch_size", 64),
        action_type=cfg.action_type,
    )
    state = torch.load(cfg.unlearned_checkpoint, map_location=str(device))
    agent.network.load_state_dict(state["network"])

    scenario = ForgetScenario.from_config(cfg)

    # Reference: the unlearned policy's return BEFORE the attack.
    initial = eval_agent(agent, eval_env, scenario, baseline_return=0.0,
                         num_episodes=16, seed=10000)
    baseline_return = initial["mean_return"]
    logger.info(
        f"Pre-attack: return={baseline_return:.3f} "
        f"forget_eff={initial['forget_effectiveness']:.3f}"
    )

    n_iters = int(cfg.get("relearn_iters", 200))
    rollout_len = int(cfg.get("rollout_len", 512))
    eval_every = int(cfg.get("relearn_eval_every", 10))
    obs, _ = env.reset(seed=cfg.seed)

    history = []
    start_time = time.time()

    for it in tqdm(range(n_iters), desc="relearn-attack"):
        buf_obs, buf_act, buf_logp, buf_val = [], [], [], []
        buf_rew, buf_done = [], []
        for _ in range(rollout_len):
            action, info = agent.select_action(obs, deterministic=False)
            buf_obs.append(obs.copy()); buf_act.append(action)
            buf_logp.append(info["log_prob"]); buf_val.append(info["value"])
            next_obs, r, term, trunc, _ = env.step(action)
            buf_rew.append(r); buf_done.append(term or trunc)
            obs = next_obs
            if term or trunc:
                obs, _ = env.reset()
        buf_obs = np.asarray(buf_obs, dtype=np.float32)
        buf_act = np.asarray(buf_act)
        buf_logp = np.asarray(buf_logp, dtype=np.float32)
        buf_val = np.asarray(buf_val, dtype=np.float32)
        buf_rew = np.asarray(buf_rew, dtype=np.float32)
        buf_done = np.asarray(buf_done, dtype=np.float32)

        with torch.no_grad():
            next_value = agent.network.get_value(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device),
            ).item()
        adv, ret = agent.compute_gae(
            torch.as_tensor(buf_rew), torch.as_tensor(buf_val),
            torch.as_tensor(buf_done), next_value,
        )
        # Standard PPO update with normal env reward — no unlearning,
        # no disjoint, no penalty. The "attack".
        ppo_rollout = {
            "observations": buf_obs,
            "actions": buf_act,
            "log_probs": buf_logp,
            "values": buf_val,
            "advantages": adv.numpy(),
            "returns": ret.numpy(),
        }
        ppo_metrics = agent.update(ppo_rollout)

        forget_mask = transitions_in_forget_region(buf_obs, scenario)
        n_forget = int(forget_mask.sum())

        rec = {
            "iter": it,
            "wall_time_s": time.time() - start_time,
            "n_forget_transitions": n_forget,
            "ppo_policy_loss": ppo_metrics.get("policy_loss", 0.0),
        }
        if it % eval_every == 0 or it == n_iters - 1:
            ev = eval_agent(agent, eval_env, scenario, baseline_return,
                            num_episodes=16, seed=20000 + it)
            rec.update({f"eval/{k}": v for k, v in ev.items()})
            logger.info(
                f"it={it} n_forget={n_forget} "
                f"forget_eff={ev['forget_effectiveness']:.3f} "
                f"ret={ev['mean_return']:.3f}"
            )
        history.append(rec)

    out_root = Path(cfg.training.checkpoint_dir).parent.parent / "outputs"
    run_suffix = cfg.get("run_suffix", "default")
    out_dir = out_root / f"relearn_attack_{cfg.env_name}_seed{cfg.seed}_{run_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    logger.info(f"Wrote history to {out_dir}")


if __name__ == "__main__":
    relearn_attack()
