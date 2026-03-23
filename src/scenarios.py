"""
Forget Scenario system for defining targeted unlearning experiments.

A scenario defines WHAT to forget via declarative conditions on observations
and actions. Scenarios are loaded from YAML files and used by unlearn.py to
identify forget trajectories (replacing the reward-based fallback).

Scenario YAML format:
    name: left_only
    description: "Forget right-side balancing"
    env_id: CartPole-v1
    match_mode: proportion   # any_step | all_steps | proportion
    match_threshold: 0.3     # only for proportion mode
    groups:                  # groups are OR'd
      - name: right_side
        conditions:          # conditions within a group are AND'd
          - dim: 0
            op: ">"
            value: 0.0
            target: state    # state (default) | action
"""
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Observation dimension names for documentation and validation
ENV_OBS_DIMS = {
    "CartPole-v1": {
        0: "cart_position",
        1: "cart_velocity",
        2: "pole_angle",
        3: "pole_angular_velocity",
    },
    "Acrobot-v1": {
        0: "cos_theta1",
        1: "sin_theta1",
        2: "cos_theta2",
        3: "sin_theta2",
        4: "theta1_dot",
        5: "theta2_dot",
    },
    "LunarLander-v3": {
        0: "x",
        1: "y",
        2: "vx",
        3: "vy",
        4: "angle",
        5: "angular_velocity",
        6: "left_leg_contact",
        7: "right_leg_contact",
    },
}

# Action dimension names
ENV_ACTION_DIMS = {
    "CartPole-v1": {0: "push_left", 1: "push_right"},
    "Acrobot-v1": {0: "torque_negative", 1: "torque_zero", 2: "torque_positive"},
    "LunarLander-v3": {0: "noop", 1: "left_engine", 2: "main_engine", 3: "right_engine"},
}


_OPS = {
    ">":    lambda v, t: v > t,
    ">=":   lambda v, t: v >= t,
    "<":    lambda v, t: v < t,
    "<=":   lambda v, t: v <= t,
    "==":   lambda v, t: v == t,
    "!=":   lambda v, t: v != t,
    "abs>": lambda v, t: abs(v) > t,
    "abs<": lambda v, t: abs(v) < t,
    "abs>=": lambda v, t: abs(v) >= t,
    "abs<=": lambda v, t: abs(v) <= t,
}


class ForgetScenario:
    """Declarative forget scenario loaded from YAML."""

    def __init__(self, config: dict):
        self.name = config["name"]
        self.description = config["description"]
        self.env_id = config.get("env_id", None)
        self.match_mode = config.get("match_mode", "any_step")
        self.match_threshold = config.get("match_threshold", 0.5)
        self.groups = config["groups"]

    @classmethod
    def load(cls, path: str) -> "ForgetScenario":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    @classmethod
    def from_config(cls, cfg) -> Optional["ForgetScenario"]:
        """Load scenario from a Hydra config that has a 'scenario' field.

        Returns None if no scenario is configured.
        The scenario field should be a relative path like 'cartpole/left_only'.
        """
        scenario_name = cfg.get("scenario", None)
        if not scenario_name:
            return None

        # Search in standard locations
        search_paths = [
            Path("configs/scenarios") / f"{scenario_name}.yaml",
            Path(__file__).parent.parent / "configs" / "scenarios" / f"{scenario_name}.yaml",
        ]

        for p in search_paths:
            if p.exists():
                return cls.load(str(p))

        raise FileNotFoundError(
            f"Scenario '{scenario_name}' not found. Searched: {[str(p) for p in search_paths]}"
        )

    def identify_forget_trajectories(self, trajectories: List[Dict]) -> List[int]:
        """Return indices of trajectories that match the forget criteria."""
        indices = []
        for i, traj in enumerate(trajectories):
            if self._trajectory_matches(traj):
                indices.append(i)
        return indices

    def matches_state(self, obs: np.ndarray, action=None) -> bool:
        """Check if a single state (and optional action) matches any group."""
        return self._step_matches(obs, action)

    def get_forget_states_from_trajectories(
        self, trajectories: List[Dict], forget_indices: List[int]
    ) -> List[np.ndarray]:
        """Extract individual states from forget trajectories that match conditions.

        Useful for strategy_inversion: provides on-distribution seed states
        that are specifically in the forget region.
        """
        states = []
        for idx in forget_indices:
            traj = trajectories[idx]
            obs = traj["observations"]
            actions = traj.get("actions", [None] * len(obs))
            for t in range(len(obs)):
                act = actions[t] if t < len(actions) else None
                if self._step_matches(obs[t], act):
                    states.append(np.array(obs[t]))
        return states

    def _trajectory_matches(self, trajectory: Dict) -> bool:
        obs = trajectory["observations"]
        actions = trajectory.get("actions", [None] * len(obs))

        if self.match_mode == "any_step":
            for t in range(len(obs)):
                act = actions[t] if t < len(actions) else None
                if self._step_matches(obs[t], act):
                    return True
            return False

        elif self.match_mode == "all_steps":
            for t in range(len(obs)):
                act = actions[t] if t < len(actions) else None
                if not self._step_matches(obs[t], act):
                    return False
            return True

        elif self.match_mode == "proportion":
            if len(obs) == 0:
                return False
            count = 0
            for t in range(len(obs)):
                act = actions[t] if t < len(actions) else None
                if self._step_matches(obs[t], act):
                    count += 1
            return (count / len(obs)) >= self.match_threshold

        raise ValueError(f"Unknown match_mode: {self.match_mode}")

    def _step_matches(self, obs: np.ndarray, action=None) -> bool:
        """True if ANY group matches (groups are OR'd)."""
        for group in self.groups:
            if self._group_matches(group, obs, action):
                return True
        return False

    def _group_matches(self, group: dict, obs: np.ndarray, action=None) -> bool:
        """True if ALL conditions in the group match (AND'd)."""
        for cond in group["conditions"]:
            target = cond.get("target", "state")

            if target == "state":
                val = float(obs[cond["dim"]])
            elif target == "action":
                if action is None:
                    return False
                val = float(action) if np.isscalar(action) else float(action[cond["dim"]])
            else:
                raise ValueError(f"Unknown target: {target}")

            op_fn = _OPS.get(cond["op"])
            if op_fn is None:
                raise ValueError(f"Unknown operator: {cond['op']}")

            if not op_fn(val, cond["value"]):
                return False
        return True

    def summary(self) -> str:
        """Human-readable summary of the scenario."""
        lines = [f"Scenario: {self.name}", f"  {self.description}"]
        lines.append(f"  match_mode: {self.match_mode}")
        if self.match_mode == "proportion":
            lines.append(f"  match_threshold: {self.match_threshold}")
        for g in self.groups:
            gname = g.get("name", "unnamed")
            lines.append(f"  group '{gname}':")
            for c in g["conditions"]:
                target = c.get("target", "state")
                dim_label = c.get("dim_name", f"dim[{c['dim']}]")
                lines.append(f"    {target}.{dim_label} {c['op']} {c['value']}")
        return "\n".join(lines)

    def __repr__(self):
        return f"ForgetScenario(name={self.name!r}, groups={len(self.groups)})"


def list_scenarios(base_dir: str = "configs/scenarios") -> List[str]:
    """List all available scenario names (relative paths without .yaml)."""
    base = Path(base_dir)
    if not base.exists():
        return []
    scenarios = []
    for p in sorted(base.rglob("*.yaml")):
        rel = p.relative_to(base).with_suffix("")
        scenarios.append(str(rel))
    return scenarios
