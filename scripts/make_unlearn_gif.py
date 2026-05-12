"""Render before/after unlearning GIFs for MiniGrid envs.

For each checkpoint, roll out a single episode with the policy and render
to RGB frames. Overlay the scenario's forget region as a translucent red
rectangle. Stitch frames into a GIF.

Usage:
    python scripts/make_unlearn_gif.py --env empty8x8 \\
        --variant trajectory_selective \\
        --forget-box 1 3 4 6 \\
        --output docs/images/results/empty8x8_with_demos/after_trajectory_selective.gif

    # baseline:
    python scripts/make_unlearn_gif.py --env empty8x8 \\
        --variant baseline --forget-box 1 3 4 6 \\
        --output docs/images/results/empty8x8_with_demos/before.gif

    # side-by-side: produced as before.gif + after_*.gif separately, then
    # the poster combines them visually.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import PPOAgent  # noqa: E402
from utils.env_wrappers import make_env  # noqa: E402

GEOMETRY = {
    "fourrooms": {"env_id": "MiniGrid-FourRooms-v0", "obs_dim": 2838,
                  "obs_encoding": "symbolic", "fixed_goal_pos": (13, 3)},
    "empty8x8":  {"env_id": "MiniGrid-Empty-8x8-v0", "obs_dim": 150,
                  "obs_encoding": "image", "fixed_goal_pos": None},
}


def load_agent(ckpt_path: Path, obs_dim: int, n_actions: int = 7) -> PPOAgent:
    agent = PPOAgent(observation_dim=obs_dim, action_dim=n_actions, device="cpu")
    state = torch.load(ckpt_path, map_location="cpu")
    agent.network.load_state_dict(state["network"])
    return agent


def overlay_forget_region(
    frame: np.ndarray,
    forget_box: Tuple[int, int, int, int],
    tile_size: int,
    alpha: float = 0.45,
) -> np.ndarray:
    """Tint the forget-box cells red on a copy of the frame."""
    xlo, xhi, ylo, yhi = forget_box
    out = frame.copy()
    x0, x1 = xlo * tile_size, (xhi + 1) * tile_size
    y0, y1 = ylo * tile_size, (yhi + 1) * tile_size
    red_layer = np.zeros_like(out[y0:y1, x0:x1])
    red_layer[..., 0] = 255  # pure red
    out[y0:y1, x0:x1] = (
        (1 - alpha) * out[y0:y1, x0:x1].astype(np.float32)
        + alpha * red_layer.astype(np.float32)
    ).astype(np.uint8)
    return out


def rollout_frames(
    env_id: str, ckpt_path: Path, forget_box: Tuple[int, int, int, int],
    seed: int = 0, max_steps: int = 60, deterministic: bool = False,
    obs_dim: int = 150, obs_encoding: str = "image",
    fixed_goal_pos: tuple | None = None,
) -> List[np.ndarray]:
    env = make_env(env_id, seed=seed, render_mode="rgb_array",
                   obs_encoding=obs_encoding,
                   fixed_goal_pos=fixed_goal_pos)
    obs, _ = env.reset(seed=seed)
    agent = load_agent(ckpt_path, obs_dim=obs_dim)

    # Render the initial state so the GIF starts with the unmoved agent.
    img = env.render()
    tile = img.shape[1] // env.unwrapped.width
    frames = [overlay_forget_region(img, forget_box, tile)]

    for _ in range(max_steps):
        action, _ = agent.select_action(obs, deterministic=deterministic)
        obs, r, term, trunc, _ = env.step(action)
        frames.append(overlay_forget_region(env.render(), forget_box, tile))
        if term or trunc:
            break
    env.close()
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=list(GEOMETRY.keys()), default="empty8x8")
    p.add_argument("--variant",
                   help="'baseline' or any unlearned variant name; ignored if --checkpoint is set.")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit path to a .pt file. Bypasses --variant lookup.")
    p.add_argument("--experiment", default=None,
                   help="default: {env}_ppo_seed42")
    p.add_argument("--forget-box", nargs=4, type=int, required=True,
                   metavar=("XLO", "XHI", "YLO", "YHI"),
                   help="Forget region in grid coords (inclusive).")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--num-episodes", type=int, default=3,
                   help="Concatenate N episodes into one GIF for variety.")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--deterministic", action="store_true",
                   help="Use argmax (vs sampled) actions.")
    args = p.parse_args()

    geom = GEOMETRY[args.env]
    if args.experiment is None:
        args.experiment = f"{args.env}_ppo_seed42"

    if args.checkpoint:
        ckpt = Path(args.checkpoint)
    elif args.variant == "baseline":
        ckpt = ROOT / "experiments" / "checkpoints" / args.experiment / "final_model.pt"
    elif args.variant:
        ckpt = ROOT / "experiments" / "checkpoints" / args.experiment / f"unlearned_{args.variant}.pt"
    else:
        sys.exit("Provide --variant or --checkpoint")
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")

    forget_box = tuple(args.forget_box)
    all_frames: List[np.ndarray] = []
    for ep in range(args.num_episodes):
        frames = rollout_frames(
            geom["env_id"], ckpt, forget_box,
            seed=args.seed + ep, max_steps=args.max_steps,
            deterministic=args.deterministic, obs_dim=geom["obs_dim"],
            obs_encoding=geom.get("obs_encoding", "image"),
            fixed_goal_pos=geom.get("fixed_goal_pos", None),
        )
        all_frames.extend(frames)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / max(args.fps, 1)
    imageio.mimsave(out_path, all_frames, duration=duration, loop=0)
    print(f"Wrote {len(all_frames)} frames to {out_path}")


if __name__ == "__main__":
    main()
