"""Collecting trajectories in the form the scenario layer expects.

A scenario is evaluated on FEATURES, not raw observations, and a view may
read the environment object, whose state corresponds to an observation
only until `step` is called. So features have to be extracted during
collection rather than reconstructed afterwards. Both the labelling
harness and the authoring evaluation need exactly this, and getting it
subtly different in two places is how the two stop being comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

Policy = Callable[[Any], Any]


def collect_trajectories(
    env,
    view,
    n_episodes: int = 20,
    max_steps: int = 60,
    seed: int = 0,
    policy: Optional[Policy] = None,
) -> List[Dict[str, Any]]:
    """Roll out episodes, recording per-step features, actions and dones.

    `dones` is included because trajectory-level match modes are decided
    per episode; without it a caller that concatenates episodes silently
    evaluates the threshold across all of them.
    """
    out = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=seed + i)
        feats, acts, dones = [], [], []
        for _t in range(max_steps):
            feats.append(list(view.extract(obs=obs, info=info, env=env)))
            action = (env.action_space.sample() if policy is None
                      else policy(obs))
            acts.append(action)
            obs, _r, term, trunc, info = env.step(action)
            dones.append(bool(term or trunc))
            if term or trunc:
                break
        out.append({"features": feats, "actions": acts, "dones": dones})
    return out


def build_policy_env(overrides: Optional[str], checkpoint: Optional[str],
                     config_dir: str = "configs"):
    """Env and policy built as the trainer built them.

    The observation encoding and dimensionality come from the same Hydra
    composition, so a checkpoint loads without a shape mismatch and the
    policy sees what it was trained on.
    """
    from hydra import compose, initialize_config_dir

    from agents import PPOAgent
    from utils.checkpointing import load_checkpoint
    from utils.env_wrappers import make_env

    ovr = [o for o in (overrides or "").split(",") if o]
    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()),
                               version_base=None):
        cfg = compose(config_name="config", overrides=ovr)
    env = make_env(cfg.env_id, seed=0,
                   obs_encoding=cfg.get("obs_encoding", "image"),
                   fixed_goal_pos=cfg.get("fixed_goal_pos", None),
                   include_pos=cfg.get("include_pos", True))
    if not checkpoint:
        return env, None
    # Mirror the trainer's optional flags. A continuous-action checkpoint
    # carries gSDE and tanh-squash parameters that simply do not exist on
    # a default agent, so omitting them fails at load with a key mismatch
    # rather than anything that points at the cause.
    agent = PPOAgent(observation_dim=cfg.observation_dim,
                     action_dim=cfg.action_dim, device="cpu",
                     action_type=cfg.action_type,
                     use_sde=cfg.get("use_sde", False),
                     sde_sample_freq=cfg.get("sde_sample_freq", 4),
                     sde_log_std_init=cfg.get("sde_log_std_init", -2.0),
                     tanh_squash=cfg.get("tanh_squash", False),
                     action_scale=cfg.get("action_scale", 1.0),
                     use_conv=cfg.get("use_conv", False))
    agent.network.load_state_dict(
        load_checkpoint(Path(checkpoint), map_location="cpu")["network"])
    return env, lambda obs: agent.select_action(obs, deterministic=False)[0]


def plain_env(domain: str):
    """A bare gym env, for domains with no trained policy to hand."""
    import gymnasium as gym

    if domain.startswith("MiniGrid-"):
        import minigrid                                          # noqa: F401
    elif domain.startswith("Foraging-"):
        import lbforaging                                        # noqa: F401
    return gym.make(domain)
