"""On-policy online RL unlearning.

Unlike src/unlearn.py (which loads a pre-collected trajectory buffer and
runs an offline unlearning loop), this script keeps the agent rolling
out under PPO while interleaving unlearning steps. There is NO offline
buffer of forget data — every forget transition has to come from the
agent's current on-policy distribution.

The chicken-and-egg failure this exposes: one unlearning step shifts
the policy away from the forget region, the next rollout produces fewer
forget transitions, the unlearning signal weakens, the policy drifts
back, ... or the policy never visits the region again and the
unlearning halts prematurely.

Run:
    python src/online_unlearn.py env=fourrooms \\
        scenario=fourrooms/avoid_bottom_right \\
        load_baseline=experiments/checkpoints/fourrooms_ppo_seed42/final_model.pt \\
        unlearn_method=gradient_reversal \\
        unlearn_weight=0.5

Reports per rollout iteration:
    - n_forget_transitions:  forget-region transitions in this rollout
    - forget_region_fraction: fraction of steps in forget region
    - eval_forget_eff / eval_retain_stab: held-out eval metrics
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


# ---------------------------------------------------------------------------
# Forget-data tracking and scenario application
# ---------------------------------------------------------------------------


def transitions_in_forget_region(
    obs: np.ndarray, scenario: ForgetScenario,
) -> np.ndarray:
    """Boolean mask over a rollout: True for transitions whose state
    matches the scenario's forget conditions."""
    mask = np.zeros(len(obs), dtype=bool)
    for i, o in enumerate(obs):
        mask[i] = scenario.matches_state(np.asarray(o))
    return mask


# ---------------------------------------------------------------------------
# Unlearning losses pluggable per --unlearn_method
# ---------------------------------------------------------------------------


def _policy_log_probs_entropy(agent, observations, actions):
    """Helper: get log-probs and entropy under current policy."""
    logits, _ = agent.network(observations)
    if agent.action_type == "discrete":
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()
    action_std = torch.exp(agent.network.actor_logstd.expand_as(logits))
    dist = torch.distributions.Normal(logits, action_std)
    return dist.log_prob(actions).sum(-1), dist.entropy().sum(-1)


def gradient_reversal_loss(agent, observations, actions, **_):
    """Ascend on forget-action log-prob, push policy toward uniform on
    forget states. Same as trajectory_selective._compute_forget_loss but
    on data freshly sampled from the current on-policy distribution."""
    if len(observations) == 0:
        return torch.tensor(0.0, device=next(agent.network.parameters()).device)
    log_probs, entropy = _policy_log_probs_entropy(agent, observations, actions)
    return log_probs.mean() - entropy.mean()


def no_op_loss(agent, observations, actions, **_):
    """Control: zero unlearning loss. Pure continued PPO. Useful as the
    baseline to compare against — shows what the policy does under
    continued env interaction WITHOUT any unlearning pressure."""
    return torch.tensor(0.0, device=next(agent.network.parameters()).device)


UNLEARN_METHODS = {
    "gradient_reversal": gradient_reversal_loss,
    "no_op": no_op_loss,
}


# ---------------------------------------------------------------------------
# Eval (held-out rollouts, scenario metrics)
# ---------------------------------------------------------------------------


def eval_agent(
    agent: PPOAgent, env, scenario: ForgetScenario, baseline_return: float,
    num_episodes: int = 16, seed: int = 0,
) -> dict:
    """Stochastic eval. Returns mean return, forget_region_fraction,
    forget_effectiveness, retain_stability (vs baseline)."""
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
    # Degradation-based RSI vs baseline.
    scale = max(abs(baseline_return), 1.0)
    retain_stab = max(0.0, 1.0 - abs(mean_return - baseline_return) / scale)
    return {
        "mean_return": mean_return,
        "forget_region_fraction": region_frac,
        "forget_effectiveness": forget_eff,
        "retain_stability": retain_stab,
        "selectivity": forget_eff * retain_stab,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def online_unlearn(cfg: DictConfig):
    setup_logger("online_unlearn")
    set_seed(cfg.seed)
    device = get_device()
    logger.info(f"On-policy online unlearning | device={device}")
    logger.info(OmegaConf.to_yaml(cfg))

    # ---- Env and agent ----
    env = make_env(
        cfg.env_id, seed=cfg.seed,
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

    # Load baseline checkpoint — the policy we're going to unlearn from.
    load_path = Path(cfg.load_baseline)
    if not load_path.exists():
        logger.error(f"Baseline checkpoint not found: {load_path}")
        return
    state = torch.load(load_path, map_location=str(device))
    agent.network.load_state_dict(state["network"])
    logger.info(f"Loaded baseline from {load_path}")

    # ---- Scenario ----
    scenario = ForgetScenario.from_config(cfg)
    if scenario is None:
        logger.error("Scenario required for online unlearning.")
        return
    logger.info(f"Scenario: {scenario.name}")

    # ---- Method selection ----
    method_name = cfg.get("unlearn_method", "gradient_reversal")
    if method_name not in UNLEARN_METHODS:
        logger.error(f"Unknown unlearn_method={method_name}")
        return
    unlearn_loss_fn = UNLEARN_METHODS[method_name]
    unlearn_weight = float(cfg.get("unlearn_weight", 0.5))

    # ---- Baseline eval (for retain stability reference) ----
    baseline_metrics = eval_agent(agent, env, scenario, baseline_return=0.0,
                                   num_episodes=16, seed=10000)
    baseline_return = baseline_metrics["mean_return"]
    logger.info(
        f"Baseline: return={baseline_return:.3f}  "
        f"forget_region_frac={baseline_metrics['forget_region_fraction']:.3f}"
    )

    # ---- Online unlearning loop ----
    n_iters = int(cfg.get("online_iters", 200))
    rollout_len = int(cfg.get("rollout_len", 2048))
    eval_every = int(cfg.get("online_eval_every", 5))
    obs, _ = env.reset(seed=cfg.seed)

    history = []
    start_time = time.time()

    for it in tqdm(range(n_iters), desc="online-unlearn"):
        # ---- Phase 1: collect on-policy rollout (no offline buffer) ----
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

        # ---- Phase 2: identify forget transitions in THIS rollout ----
        forget_mask = transitions_in_forget_region(buf_obs, scenario)
        n_forget = int(forget_mask.sum())
        forget_frac = n_forget / len(buf_obs)

        # ---- Phase 3: PPO update on retain set + unlearning on forget set ----
        # Compute GAE on full rollout (retain rewards intact).
        with torch.no_grad():
            next_value = agent.network.get_value(
                torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device),
            ).item()
        adv, ret = agent.compute_gae(
            torch.as_tensor(buf_rew), torch.as_tensor(buf_val),
            torch.as_tensor(buf_done), next_value,
        )
        ppo_rollout = {
            "observations": buf_obs,
            "actions": buf_act,
            "log_probs": buf_logp,
            "values": buf_val,
            "advantages": adv.numpy(),
            "returns": ret.numpy(),
        }
        ppo_metrics = agent.update(ppo_rollout)

        # Unlearning side-step: ascend on log-prob of forget transitions.
        # Skip when the method returns a non-differentiable zero (no_op
        # control) or when this rollout has no forget transitions (the
        # chicken-and-egg failure we want to expose).
        unlearn_loss_val = 0.0
        if n_forget > 0 and method_name != "no_op":
            forget_obs_t = torch.as_tensor(buf_obs[forget_mask], dtype=torch.float32).to(device)
            forget_act_t = torch.as_tensor(buf_act[forget_mask]).to(device)
            agent.optimizer.zero_grad()
            loss = unlearn_weight * unlearn_loss_fn(agent, forget_obs_t, forget_act_t)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.network.parameters(), agent.max_grad_norm,
                )
                agent.optimizer.step()
                unlearn_loss_val = float(loss.item())

        # ---- Phase 4: eval ----
        rec = {
            "iter": it,
            "wall_time_s": time.time() - start_time,
            "n_forget_transitions": n_forget,
            "forget_frac_rollout": forget_frac,
            "unlearn_loss": unlearn_loss_val,
            "ppo_policy_loss": ppo_metrics.get("policy_loss", 0.0),
            "ppo_value_loss": ppo_metrics.get("value_loss", 0.0),
        }
        if it % eval_every == 0 or it == n_iters - 1:
            ev = eval_agent(agent, env, scenario, baseline_return, num_episodes=16,
                            seed=20000 + it)
            rec.update({f"eval/{k}": v for k, v in ev.items()})
            logger.info(
                f"it={it} n_forget={n_forget} ({forget_frac:.3f}) "
                f"forget_eff={ev['forget_effectiveness']:.3f} "
                f"retain={ev['retain_stability']:.3f} "
                f"ret={ev['mean_return']:.3f}"
            )
        history.append(rec)

    # ---- Save metrics + final model ----
    out_root = Path(cfg.training.checkpoint_dir).parent.parent / "outputs"
    out_dir = out_root / f"online_unlearn_{cfg.env_name}_seed{cfg.seed}_{method_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save({"network": agent.network.state_dict()},
               out_dir / f"unlearned_{method_name}.pt")
    logger.info(f"Wrote metrics + model to {out_dir}")


if __name__ == "__main__":
    online_unlearn()
