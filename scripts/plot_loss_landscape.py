"""Filter-normalized loss landscape for unlearned policies (Li et al. 2018).

For each policy, sample 2 random orthogonal directions in weight space,
filter-normalize them to the policy's own weight magnitudes, and grid-
sample forget_effectiveness across a 2D plane in weight space.

Shows the cosmetic-vs-structural distinction GEOMETRICALLY: penalty
methods sit on a sharp peak (high forget_eff in a narrow weight region),
while disjoint methods sit on a flatter plateau (similar forget_eff over
a broader region). The sharp/flat minimum story connects to Keskar 2017,
Hochreiter 1997, etc.

Usage:
    python scripts/plot_loss_landscape.py --env fourrooms --seed 42 \\
        --grid 15 --episodes 8 --output docs/images/results/online_unlearn/loss_landscape.png
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import PPOAgent  # noqa: E402
from scenarios import ForgetScenario  # noqa: E402
from utils.env_wrappers import make_env  # noqa: E402


def filter_normalized_directions(state_dict, rng):
    """Two random orthogonal directions, each layer-normalized to match
    the corresponding weight tensor's norm (per Li et al. 2018, eq 1).
    """
    d1, d2 = {}, {}
    for name, w in state_dict.items():
        if w.dim() < 2:
            d1[name] = torch.zeros_like(w)
            d2[name] = torch.zeros_like(w)
            continue
        r1 = torch.from_numpy(rng.normal(size=w.shape).astype(np.float32))
        r2 = torch.from_numpy(rng.normal(size=w.shape).astype(np.float32))
        for ri in (r1, r2):
            ri.mul_(w.norm() / (ri.norm() + 1e-9))
        d1[name] = r1
        d2[name] = r2
    return d1, d2


def set_perturbed(agent, base, d1, d2, alpha, beta):
    """Apply theta = base + alpha*d1 + beta*d2."""
    new_state = {}
    for name, w in base.items():
        if name in d1:
            new_state[name] = w + alpha * d1[name] + beta * d2[name]
        else:
            new_state[name] = w.clone()
    agent.network.load_state_dict(new_state)


def eval_forget_eff(agent, env, scenario, n_episodes=8, max_steps=200, seed_base=0):
    total = 0
    in_forget = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        for _ in range(max_steps):
            with torch.no_grad():
                a, _ = agent.select_action(obs, deterministic=False)
            obs, _, term, trunc, _ = env.step(a)
            total += 1
            if scenario.matches_state(np.asarray(obs)):
                in_forget += 1
            if term or trunc:
                break
    return 1.0 - in_forget / max(total, 1)


def compute_landscape(ckpt_path, env_id, scenario, n_grid=15, span=1.0,
                       episodes=8, rng_seed=42, obs_encoding="symbolic"):
    rng = np.random.default_rng(rng_seed)
    state = torch.load(ckpt_path, map_location="cpu")
    base = state["network"] if isinstance(state, dict) and "network" in state else state

    # Build a matching agent
    sample_w = next(iter(base.values()))
    # Discover obs/action shapes from the network's first/last layers
    shared0 = base["shared.network.0.weight"]
    obs_dim = shared0.shape[1]
    actor_w = next((v for k, v in base.items() if "actor" in k and v.dim() == 2 and v.shape[0] != 1), None)
    action_dim = actor_w.shape[0] if actor_w is not None else 7

    agent = PPOAgent(observation_dim=obs_dim, action_dim=action_dim, action_type="discrete", device="cpu")
    agent.network.load_state_dict(base)

    env = make_env(env_id, seed=0, obs_encoding=obs_encoding,
                   fixed_goal_pos=(13, 3) if "FourRooms" in env_id else None)

    d1, d2 = filter_normalized_directions(base, rng)
    alphas = np.linspace(-span, span, n_grid)
    betas = np.linspace(-span, span, n_grid)
    Z = np.zeros((n_grid, n_grid), dtype=np.float32)
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            set_perturbed(agent, base, d1, d2, a, b)
            Z[j, i] = eval_forget_eff(agent, env, scenario, n_episodes=episodes,
                                       seed_base=100 + i * 31 + j)
        print(f"  row {i+1}/{n_grid} done")
    return alphas, betas, Z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="fourrooms")
    p.add_argument("--scenario", default="fourrooms/avoid_bottom_right")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid", type=int, default=15)
    p.add_argument("--span", type=float, default=0.6)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--output", required=True)
    p.add_argument("--force", action="store_true",
                   help="recompute even if cache exists")
    args = p.parse_args()

    env_id = {"fourrooms": "MiniGrid-FourRooms-v0", "doorkey": "MiniGrid-DoorKey-8x8-v0"}[args.env]
    scenario = ForgetScenario.load(str(ROOT / "configs" / "scenarios" / f"{args.scenario}.yaml"))

    panels = [
        ("cosmetic",   f"experiments/outputs/online_unlearn_{args.env}_seed{args.seed}_no_op_neg_reward_0.1/unlearned_no_op.pt"),
        ("structural", f"experiments/outputs/online_unlearn_{args.env}_seed{args.seed}_is_replay_disjoint/unlearned_is_replay.pt"),
    ]

    import matplotlib as mpl
    mpl.rcParams["font.family"] = "Poppins"

    cache_dir = ROOT / "experiments" / "outputs" / "_landscape_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    surfaces = []
    for label, ckpt in panels:
        ckpt_path = ROOT / ckpt
        if not ckpt_path.exists():
            print(f"missing {ckpt_path}"); continue
        cache = cache_dir / (
            f"{args.env}_{args.seed}_{label}_g{args.grid}_s{args.span}_e{args.episodes}.npz"
        )
        if cache.exists() and not args.force:
            print(f"loading cached {cache.name}")
            d = np.load(cache)
            alphas, betas, Z = d["alphas"], d["betas"], d["Z"]
        else:
            print(f"\ncomputing {label} ...")
            alphas, betas, Z = compute_landscape(
                str(ckpt_path), env_id, scenario,
                n_grid=args.grid, span=args.span, episodes=args.episodes,
                rng_seed=args.seed,
            )
            np.savez(cache, alphas=alphas, betas=betas, Z=Z)
            print(f"  cached -> {cache.name}")
        center = Z[args.grid // 2, args.grid // 2]
        print(f"  {label}: center {center:.3f}  range [{Z.min():.3f}, {Z.max():.3f}]")
        surfaces.append((label, alphas, betas, Z))

    # Plot with proper labels + filling the figure
    fig = plt.figure(figsize=(12, 5.5))
    for idx, (label, alphas, betas, Z) in enumerate(surfaces):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        A, B = np.meshgrid(alphas, betas)
        ax.plot_surface(A, B, Z, cmap="viridis", edgecolor="none",
                         vmin=0.3, vmax=1.0, alpha=0.95)
        ax.set_title(label, color="#1D3557", fontsize=14, fontweight="bold",
                     loc="center", pad=10)
        ax.set_xlabel(r"$\alpha$", fontsize=11, labelpad=2)
        ax.set_ylabel(r"$\beta$", fontsize=11, labelpad=2)
        ax.set_zlabel("forget eff", fontsize=10, labelpad=2)
        ax.set_zlim(0.0, 1.0)
        ax.view_init(elev=22, azim=-58)
        ax.tick_params(axis="both", labelsize=9, pad=1)
        ax.grid(False)

    # Use subplots_adjust + low margins so surfaces actually fill the canvas
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.04,
                         wspace=0.05)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
