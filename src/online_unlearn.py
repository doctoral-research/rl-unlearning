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


class ForgetReplayBuffer:
    """Ring buffer of forget transitions for on-policy online unlearning.

    Stores (obs, action, log_prob_at_collection) tuples from past
    rollouts. The recorded log_prob is the policy's log-prob AT THE
    TIME the transition was collected: this lets us apply PPO-style
    importance-sampling correction when reusing old forget data under
    a drifted current policy.

    Solves the chicken-and-egg: even when the current policy no longer
    visits the forget region, the buffer keeps providing forget data
    for the unlearning loss. Off-policiness is bounded by clipping the
    IS ratio (PPO-style).
    """

    def __init__(self, capacity: int = 4096):
        self.capacity = int(capacity)
        self._obs: list[np.ndarray] = []
        self._act: list = []
        self._logp: list[float] = []

    def __len__(self) -> int:
        return len(self._obs)

    def add(self, obs: np.ndarray, action, log_prob: float) -> None:
        if len(self._obs) >= self.capacity:
            self._obs.pop(0)
            self._act.pop(0)
            self._logp.pop(0)
        self._obs.append(np.asarray(obs, dtype=np.float32))
        self._act.append(action)
        self._logp.append(float(log_prob))

    def sample(self, batch_size: int):
        if len(self._obs) == 0:
            return None
        n = min(batch_size, len(self._obs))
        idx = np.random.choice(len(self._obs), size=n, replace=n > len(self._obs))
        obs = np.stack([self._obs[i] for i in idx])
        act = np.asarray([self._act[i] for i in idx])
        logp = np.asarray([self._logp[i] for i in idx], dtype=np.float32)
        return obs, act, logp


def is_clipped_reversal_loss(
    agent, observations, actions, old_log_probs,
    clip_ratio: float = 0.2, **_,
):
    """The proposed method: importance-sampled gradient-reversal with
    PPO-style ratio clipping.

    Loss = +mean( clipped_ratio * log_pi_current(a|s) ) - mean(entropy)

    The `+` (instead of standard PPO's `-`) is the gradient reversal:
    we maximize log-prob of forget actions, but only with importance
    weight clipped to [1-eps, 1+eps] to keep the update statistically
    valid under the policy drift from the time the transition was
    collected. This lets us reuse forget transitions from the replay
    buffer even after the policy has drifted away from them.
    """
    if len(observations) == 0:
        return torch.tensor(
            0.0, device=next(agent.network.parameters()).device,
        )
    log_probs, entropy = _policy_log_probs_entropy(agent, observations, actions)
    ratio = (log_probs - old_log_probs).exp()
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    # The IS estimator: E_old[ratio * f(x)]. Here f = log_pi_current(a|s).
    return (ratio_clipped * log_probs).mean() - entropy.mean()


def is_clipped_retain_loss(
    agent, observations, actions, old_log_probs,
    clip_ratio: float = 0.2, **_,
):
    """Mirror of is_clipped_reversal_loss with opposite sign.

    Loss = -mean( clipped_ratio * log_pi_current(a|s) )

    Pushes the current policy *toward* the actions taken at retain states
    by the previous (closer-to-baseline) policy. Same PPO-style IS-clip
    primitive as the forget side, just with the sign flipped to do gradient
    descent on log-prob (= preserve the action) instead of ascent.

    No entropy term: we don't want to encourage exploration at retain
    states, we want to preserve the action distribution that was working.
    """
    if len(observations) == 0:
        return torch.tensor(
            0.0, device=next(agent.network.parameters()).device,
        )
    log_probs, _ = _policy_log_probs_entropy(agent, observations, actions)
    ratio = (log_probs - old_log_probs).exp()
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    return -(ratio_clipped * log_probs).mean()


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
    "is_replay": is_clipped_reversal_loss,
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
    # Use SEPARATE training and eval envs so eval-episode resets don't
    # corrupt the training env state between iterations. (When the same
    # env is shared, the training loop's `obs` variable diverges from
    # the env's internal state after every eval call.)
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

    # The proposed method (is_replay) uses a ring buffer of past
    # forget transitions to survive the chicken-and-egg.
    replay_buf = None
    if method_name == "is_replay":
        replay_buf = ForgetReplayBuffer(
            capacity=int(cfg.get("replay_capacity", 4096)),
        )
    target_kl = float(cfg.get("target_kl", 0.05))

    # Optional retain-side replay (mirror of forget-side is_replay with
    # opposite sign). On each rollout, stash retain transitions; during
    # the update, sample and PUSH the policy toward those actions. Tests
    # whether an explicit positive retain anchor helps under attack.
    retain_replay = bool(cfg.get("retain_replay", False))
    retain_buf = None
    if retain_replay:
        retain_buf = ForgetReplayBuffer(
            capacity=int(cfg.get("retain_replay_capacity", 4096)),
        )
    retain_weight = float(cfg.get("retain_replay_weight", 1.0))
    retain_substeps = int(cfg.get("retain_replay_substeps", 2))
    retain_batch = int(cfg.get("retain_replay_batch_size", 256))

    # ---- Baseline eval (for retain stability reference) ----
    baseline_metrics = eval_agent(agent, eval_env, scenario, baseline_return=0.0,
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

        # Phase 2b: optional reward-shaping baseline.
        # Subtract a penalty from the env reward on forget transitions.
        # This is the obvious "why not just modify the reward?" baseline.
        # PPO then naturally avoids the forget region via reward maximisation.
        neg_pen = float(cfg.get("negative_reward_penalty", 0.0))
        if neg_pen > 0 and n_forget > 0:
            buf_rew = buf_rew - neg_pen * forget_mask.astype(np.float32)

        # For is_replay: add new on-policy forget transitions to the buffer.
        # Each entry stores the log-prob at collection time for IS correction.
        if replay_buf is not None and n_forget > 0:
            for i in np.where(forget_mask)[0]:
                replay_buf.add(buf_obs[i], buf_act[i], buf_logp[i])

        # Retain-replay: stash this rollout's retain transitions too.
        # Stored log_prob is from the *current* (drifted) policy at
        # collection time; under aggressive forget pressure the policy
        # may drift even on retain states, so keeping pre-drift retain
        # samples around lets us pull it back.
        if retain_buf is not None:
            retain_mask = ~forget_mask
            for i in np.where(retain_mask)[0]:
                retain_buf.add(buf_obs[i], buf_act[i], buf_logp[i])

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
        # Disjoint-objective mode: zero out PPO's policy-gradient signal on
        # forget transitions so PPO doesn't reinforce the very behavior the
        # unlearn loss is trying to remove. Value targets stay intact so the
        # critic still learns the env. This is what makes the setting
        # coherent: PPO operates on retain states only, unlearn loss operates
        # on forget states only. The two objectives no longer fight.
        if bool(cfg.get("disjoint_objective", True)):
            adv_np = adv.numpy().copy()
            adv_np[forget_mask] = 0.0
        else:
            adv_np = adv.numpy()
        ppo_rollout = {
            "observations": buf_obs,
            "actions": buf_act,
            "log_probs": buf_logp,
            "values": buf_val,
            "advantages": adv_np,
            "returns": ret.numpy(),
        }
        ppo_metrics = agent.update(ppo_rollout)

        # Unlearning side-step.
        # - no_op:            skip (control)
        # - gradient_reversal: use only current on-policy forget data
        #                     (subject to chicken-and-egg)
        # - is_replay:        sample from replay buffer + IS correction
        #                     (the proposed method)
        unlearn_loss_val = 0.0
        replay_kl = 0.0
        replay_size = 0 if replay_buf is None else len(replay_buf)

        if method_name == "no_op":
            pass  # control: nothing to do
        elif method_name == "is_replay" and replay_buf is not None and len(replay_buf) > 0:
            # Multi-step optimisation against the buffer. Stop early if
            # the policy drifts past target_kl from where the buffer
            # entries were collected (bounded-KL step).
            n_substeps = int(cfg.get("replay_substeps", 4))
            replay_batch = int(cfg.get("replay_batch_size", 256))
            for _ in range(n_substeps):
                sample = replay_buf.sample(replay_batch)
                if sample is None:
                    break
                s_obs, s_act, s_logp = sample
                s_obs_t = torch.as_tensor(s_obs, dtype=torch.float32).to(device)
                s_act_t = torch.as_tensor(s_act).to(device)
                s_logp_t = torch.as_tensor(s_logp, dtype=torch.float32).to(device)
                # Compute KL from collection-time policy to current; abort
                # if too far.
                with torch.no_grad():
                    cur_logp, _ = _policy_log_probs_entropy(agent, s_obs_t, s_act_t)
                    kl = (s_logp_t - cur_logp).mean().abs().item()
                if kl > target_kl:
                    replay_kl = kl
                    break
                agent.optimizer.zero_grad()
                loss = unlearn_weight * unlearn_loss_fn(
                    agent, s_obs_t, s_act_t, old_log_probs=s_logp_t,
                )
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.network.parameters(), agent.max_grad_norm,
                    )
                    agent.optimizer.step()
                    unlearn_loss_val = float(loss.item())
                    replay_kl = kl
        elif n_forget > 0:
            # Baseline on-policy unlearning: use only current rollout's
            # forget transitions.
            forget_obs_t = torch.as_tensor(
                buf_obs[forget_mask], dtype=torch.float32,
            ).to(device)
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

        # Optional retain-replay step. Same IS-clip primitive as the
        # forget side, opposite sign: pull the policy toward retain
        # actions that the baseline (or earlier policy) took. Run AFTER
        # the unlearning step so the retain anchor acts as a corrective.
        retain_loss_val = 0.0
        retain_replay_size = 0 if retain_buf is None else len(retain_buf)
        if retain_buf is not None and len(retain_buf) > 0:
            for _ in range(retain_substeps):
                sample = retain_buf.sample(retain_batch)
                if sample is None:
                    break
                r_obs, r_act, r_logp = sample
                r_obs_t = torch.as_tensor(r_obs, dtype=torch.float32).to(device)
                r_act_t = torch.as_tensor(r_act).to(device)
                r_logp_t = torch.as_tensor(r_logp, dtype=torch.float32).to(device)
                agent.optimizer.zero_grad()
                r_loss = retain_weight * is_clipped_retain_loss(
                    agent, r_obs_t, r_act_t, old_log_probs=r_logp_t,
                )
                if r_loss.requires_grad:
                    r_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.network.parameters(), agent.max_grad_norm,
                    )
                    agent.optimizer.step()
                    retain_loss_val = float(r_loss.item())

        # ---- Phase 4: eval ----
        rec = {
            "iter": it,
            "wall_time_s": time.time() - start_time,
            "n_forget_transitions": n_forget,
            "forget_frac_rollout": forget_frac,
            "unlearn_loss": unlearn_loss_val,
            "replay_buf_size": replay_size,
            "replay_kl": replay_kl,
            "ppo_policy_loss": ppo_metrics.get("policy_loss", 0.0),
            "ppo_value_loss": ppo_metrics.get("value_loss", 0.0),
        }
        if it % eval_every == 0 or it == n_iters - 1:
            ev = eval_agent(agent, eval_env, scenario, baseline_return, num_episodes=16,
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
    run_suffix = cfg.get("run_suffix", None)
    dir_name = f"online_unlearn_{cfg.env_name}_seed{cfg.seed}_{method_name}"
    if run_suffix:
        dir_name = f"{dir_name}_{run_suffix}"
    out_dir = out_root / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    torch.save({"network": agent.network.state_dict()},
               out_dir / f"unlearned_{method_name}.pt")
    logger.info(f"Wrote metrics + model to {out_dir}")


if __name__ == "__main__":
    online_unlearn()
