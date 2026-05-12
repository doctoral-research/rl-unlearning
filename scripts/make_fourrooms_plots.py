"""Poster-targeted visualisations for FourRooms unlearning experiments.

Produces three figures aimed at a poster:

1. ``occupancy.png``    — 1xN grid of 19x19 state-visit heatmaps, one per
   variant (baseline + each unlearning method). Forget region shaded.
2. ``policy_field.png`` — 1xN grid of per-cell argmax-action arrows.
   Visualises *what* each agent does at each grid cell.
3. ``pareto.png``       — forget effectiveness vs retain stability for all
   variants. Single panel.

Usage:
    python scripts/make_fourrooms_plots.py \\
        --experiment fourrooms_ppo_seed42 \\
        --scenario avoid_bottom_right \\
        --output-dir docs/images/results/fourrooms

Requires the baseline + unlearned variants to be saved under
``experiments/checkpoints/{experiment}/{final_model,unlearned_*}.pt``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import PPOAgent  # noqa: E402
from utils.env_wrappers import make_env  # noqa: E402

# Geometry presets — pick via --env. Defaults are FourRooms.
GEOMETRY = {
    "fourrooms": {
        "grid_w": 19, "grid_h": 19,
        "forget_x": (10, 17), "forget_y": (10, 17),
        "env_id": "MiniGrid-FourRooms-v0",
        "forget_label": "bottom-right room",
        "obs_dim": 2838,
        "obs_encoding": "symbolic",
        "fixed_goal_pos": (13, 3),
    },
    "empty8x8": {
        "grid_w": 8, "grid_h": 8,
        "forget_x": (1, 3), "forget_y": (4, 6),
        "env_id": "MiniGrid-Empty-8x8-v0",
        "forget_label": "bottom-left quadrant",
        "obs_dim": 150,
        "obs_encoding": "image",
        "fixed_goal_pos": None,
    },
}
N_ACTIONS = 7  # MiniGrid: 0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle, 6=done

# Mutable globals populated by main() from GEOMETRY before plotting.
GRID_W, GRID_H = 19, 19
FORGET_X = (10, 17)
FORGET_Y = (10, 17)
FORGET_LABEL = "forget region"


def load_agent(ckpt_path: Path, obs_dim: int, n_actions: int) -> PPOAgent:
    agent = PPOAgent(observation_dim=obs_dim, action_dim=n_actions, device="cpu")
    state = torch.load(ckpt_path, map_location="cpu")
    agent.network.load_state_dict(state["network"])
    return agent


def rollout_with_positions(
    agent: PPOAgent, env_id: str, num_episodes: int, max_steps: int = 200,
    deterministic: bool = False, seed: int = 0, obs_encoding: str = "image",
    fixed_goal_pos: tuple = None,
) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], List[float]]:
    """Returns: visit_counts (H, W) int, per-episode pos lists, returns list."""
    visits = np.zeros((GRID_H, GRID_W), dtype=np.int64)
    pos_lists: List[List[Tuple[int, int]]] = []
    returns: List[float] = []
    env = make_env(env_id, seed=seed, obs_encoding=obs_encoding,
                   fixed_goal_pos=fixed_goal_pos)
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        positions: List[Tuple[int, int]] = []
        ep_return = 0.0
        for _ in range(max_steps):
            x, y = int(obs[0]), int(obs[1])
            if 0 <= x < GRID_W and 0 <= y < GRID_H:
                visits[y, x] += 1
                positions.append((x, y))
            action, _ = agent.select_action(obs, deterministic=deterministic)
            obs, r, term, trunc, _ = env.step(action)
            ep_return += float(r)
            if term or trunc:
                break
        pos_lists.append(positions)
        returns.append(ep_return)
    env.close()
    return visits, pos_lists, returns


def per_cell_argmax(agent: PPOAgent, env_id: str, obs_dim: int = 150) -> np.ndarray:
    """For each (x, y) cell, query the policy at a *representative* obs and
    return the argmax action. We synthesise an obs by placing the agent at
    (x, y) with direction 0 and a zeroed view buffer — the network sees a
    canonical pose. Imperfect but fine for visualising policy preferences."""
    actions = np.full((GRID_H, GRID_W), -1, dtype=np.int64)
    for y in range(GRID_H):
        for x in range(GRID_W):
            obs = np.zeros(obs_dim, dtype=np.float32)
            obs[0] = x
            obs[1] = y
            obs[2] = 0  # canonical heading
            a, _ = agent.select_action(obs, deterministic=True)
            actions[y, x] = int(a)
    return actions


def draw_forget_rect(ax: plt.Axes) -> None:
    """Shade the bottom-right room in light red."""
    rect = Rectangle(
        (FORGET_X[0] - 0.5, FORGET_Y[0] - 0.5),
        FORGET_X[1] - FORGET_X[0] + 1,
        FORGET_Y[1] - FORGET_Y[0] + 1,
        linewidth=2, edgecolor="red", facecolor="red", alpha=0.15,
    )
    ax.add_patch(rect)


def plot_occupancy_grid(
    occupancies: Dict[str, np.ndarray], output: Path,
) -> None:
    n = len(occupancies)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]
    vmax = max(occ.max() for occ in occupancies.values())
    for ax, (name, occ) in zip(axes, occupancies.items()):
        im = ax.imshow(occ, origin="upper", cmap="viridis", vmin=0, vmax=vmax)
        draw_forget_rect(ax)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="visits")
    fig.suptitle(f"State occupancy — forget region ({FORGET_LABEL}) shaded", fontsize=12)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_policy_field(
    fields: Dict[str, np.ndarray], output: Path,
) -> None:
    """Quiver-style plot: per-cell arrow toward the cell the argmax action
    would push the agent (forward=2 → arrow in heading dir; left/right =
    rotate). Since we use heading=0 (right) the forward arrow is +x."""
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]
    # Action → (dx, dy, color)
    arrow_map = {
        0: (-1, 0, "tab:blue"),    # left
        1: (1, 0, "tab:blue"),     # right
        2: (1, 0, "tab:green"),    # forward (heading=0 → +x)
        3: (0, 0, "gray"),          # pickup (no movement)
        4: (0, 0, "gray"),          # drop
        5: (0, 0, "gray"),          # toggle
        6: (0, 0, "gray"),          # done
    }
    for ax, (name, field) in zip(axes, fields.items()):
        for y in range(GRID_H):
            for x in range(GRID_W):
                dx, dy, color = arrow_map.get(int(field[y, x]), (0, 0, "gray"))
                if dx == 0 and dy == 0:
                    ax.plot(x, y, "o", color=color, markersize=2)
                else:
                    ax.arrow(
                        x, y, dx * 0.35, dy * 0.35,
                        head_width=0.2, head_length=0.15,
                        fc=color, ec=color, alpha=0.7,
                    )
        draw_forget_rect(ax)
        ax.set_xlim(-1, GRID_W); ax.set_ylim(GRID_H, -1)  # invert y so origin top
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Policy field — argmax action per cell (right=heading 0)", fontsize=12)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(
    pareto: Dict[str, Tuple[float, float]], output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, (forget, retain) in pareto.items():
        ax.scatter(forget, retain, s=120, label=name, edgecolor="black")
        ax.annotate(name, (forget, retain), xytext=(5, 5),
                    textcoords="offset points", fontsize=9)
    ax.set_xlabel("Forget effectiveness")
    ax.set_ylabel("Retain stability")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.4)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Forget vs Retain — {FORGET_LABEL}")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=list(GEOMETRY.keys()), default="fourrooms")
    p.add_argument("--experiment", default=None,
                   help="default: {env}_ppo_seed42")
    p.add_argument("--output-dir", default=None,
                   help="default: docs/images/results/{env}")
    p.add_argument("--num-episodes", type=int, default=30)
    args = p.parse_args()

    geom = GEOMETRY[args.env]
    global GRID_W, GRID_H, FORGET_X, FORGET_Y, FORGET_LABEL
    GRID_W, GRID_H = geom["grid_w"], geom["grid_h"]
    FORGET_X = geom["forget_x"]
    FORGET_Y = geom["forget_y"]
    FORGET_LABEL = geom["forget_label"]
    args.env_id = geom["env_id"]
    if args.experiment is None:
        args.experiment = f"{args.env}_ppo_seed42"
    if args.output_dir is None:
        args.output_dir = f"docs/images/results/{args.env}"

    ckpt_dir = ROOT / "experiments" / "checkpoints" / args.experiment
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    variants: Dict[str, Path] = {}
    if (ckpt_dir / "final_model.pt").exists():
        variants["baseline"] = ckpt_dir / "final_model.pt"
    for f in sorted(ckpt_dir.glob("unlearned_*.pt")):
        name = f.stem.replace("unlearned_", "")
        variants[name] = f
    if not variants:
        print(f"No checkpoints under {ckpt_dir} — train + unlearn first.")
        return

    print(f"Variants: {list(variants.keys())}")
    occupancies: Dict[str, np.ndarray] = {}
    fields: Dict[str, np.ndarray] = {}
    pareto: Dict[str, Tuple[float, float]] = {}
    baseline_ret_mean: float | None = None

    obs_dim = geom["obs_dim"]
    obs_encoding = geom["obs_encoding"]
    fixed_goal_pos = geom.get("fixed_goal_pos", None)
    for name, path in variants.items():
        print(f"  rolling out {name} from {path.name}...")
        agent = load_agent(path, obs_dim=obs_dim, n_actions=N_ACTIONS)
        visits, pos_lists, returns = rollout_with_positions(
            agent, args.env_id, args.num_episodes, seed=12345,
            obs_encoding=obs_encoding, fixed_goal_pos=fixed_goal_pos,
        )
        field = per_cell_argmax(agent, args.env_id, obs_dim=obs_dim)
        occupancies[name] = visits
        fields[name] = field

        # Quick metrics: forget = 1 - fraction of steps in BR room
        all_pos = [p for ep in pos_lists for p in ep]
        in_br = sum(
            1 for (x, y) in all_pos
            if FORGET_X[0] <= x <= FORGET_X[1] and FORGET_Y[0] <= y <= FORGET_Y[1]
        )
        frac_br = in_br / max(len(all_pos), 1)
        forget_eff = 1.0 - frac_br
        mean_ret = float(np.mean(returns)) if returns else 0.0
        if name == "baseline":
            baseline_ret_mean = mean_ret
        retain_stab = (
            min(1.0, mean_ret / baseline_ret_mean)
            if baseline_ret_mean and baseline_ret_mean > 0 else 1.0
        )
        pareto[name] = (forget_eff, retain_stab)
        print(f"    forget_eff={forget_eff:.3f}  retain_stab={retain_stab:.3f}  return={mean_ret:.2f}")

    plot_occupancy_grid(occupancies, out_dir / "occupancy.png")
    plot_policy_field(fields, out_dir / "policy_field.png")
    plot_pareto(pareto, out_dir / "pareto.png")
    print(f"Wrote 3 figures to {out_dir}/")


if __name__ == "__main__":
    main()
