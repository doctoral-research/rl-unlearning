"""Stage 6 demo: a spatial scenario evaluated on raw pixels.

No index of an RGB frame means `agent_x`, so the scenario language does
not apply to a pixel domain at all by default. A probe fitted from frames
to grid coordinates supplies the missing semantics, and the scenario
grammar is unchanged -- a view is a view.

The second half is the point: the same probe fitted on a partially
observable egocentric view CANNOT recover absolute position, and the
fidelity gate refuses it instead of returning confident nonsense.

Run:
    python scripts/demo_learned_view.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scenarios import ForgetScenario, ProbeError, fit_probe_view  # noqa: E402

SPEC = {
    "name": "avoid_bottom_right_from_pixels",
    "description": (
        "Forget routing through the bottom-right room. Written against grid "
        "coordinates that no pixel of the observation directly contains."),
    "match_mode": "proportion",
    "match_threshold": 0.15,
    "applies_to": "matching_steps",
    "groups": [{"name": "bottom_right", "conditions": [
        {"dim": "agent_x", "op": ">", "value": 9.5},
        {"dim": "agent_y", "op": ">", "value": 9.5},
    ]}],
}


def collect(n_episodes=400, steps=12, seed=0):
    from minigrid.envs.fourrooms import FourRoomsEnv

    env = FourRoomsEnv(goal_pos=(13, 3), render_mode="rgb_array")
    rng = np.random.default_rng(seed)
    frames, ego, pos = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=4000 + ep)
        for _ in range(steps):
            frames.append(
                np.asarray(env.render())[::8, ::8].reshape(-1).astype(np.float32) / 255.0)
            ego.append(np.asarray(obs["image"], dtype=np.float32).reshape(-1))
            x, y = env.unwrapped.agent_pos
            pos.append([float(x), float(y)])
            obs, _r, te, tr, _ = env.step(int(rng.integers(3)))
            if te or tr:
                break
    env.close()
    return np.array(frames), np.array(ego), np.array(pos)


def main() -> None:
    frames, ego, pos = collect()
    print(f"collected {len(pos)} samples")
    print(f"  full-grid frame, downsampled : {frames.shape[1]} values")
    print(f"  egocentric 7x7x3 view        : {ego.shape[1]} values")
    print("  neither contains agent_x at any index.")

    print()
    print("=" * 68)
    print("A) probe on the full-grid render -- position IS determined")
    print("=" * 68)
    view, report = fit_probe_view(
        frames, pos, ["agent_x", "agent_y"],
        dtypes=("ordinal", "ordinal"), ranges=((0.0, 18.0), (0.0, 18.0)),
        ridge_alpha=1.0, min_fidelity=0.9,
        env_id="MiniGrid-FourRooms-v0",
    )
    print(report.render())
    print(f"  provenance : {view.provenance}")

    scenario = ForgetScenario(SPEC, view=view)
    print()
    print("  the scenario now evaluates on a PIXEL observation:")
    truth_hits = pred_hits = agree = 0
    for frame, (x, y) in zip(frames[:400], pos[:400]):
        got = scenario.matches_state(frame)
        want = x > 9.5 and y > 9.5
        truth_hits += want
        pred_hits += got
        agree += (got == want)
    print(f"    true in-region steps      : {truth_hits}/400")
    print(f"    predicted in-region steps : {pred_hits}/400")
    print(f"    agreement with ground truth: {agree / 400:.3f}")

    print()
    print("=" * 68)
    print("B) probe on the egocentric view -- position is NOT determined")
    print("=" * 68)
    try:
        fit_probe_view(ego, pos, ["agent_x", "agent_y"],
                       dtypes=("ordinal", "ordinal"), ridge_alpha=1.0,
                       min_fidelity=0.9)
        print("  ACCEPTED -- unexpected; the gate failed to catch this")
        sys.exit(1)
    except ProbeError as exc:
        print(f"  REFUSED\n    {exc}")

    print()
    print("A learned view earns its place by being measured. The gate is what")
    print("separates 'we can do pixels' from 'we can do pixels here'.")


if __name__ == "__main__":
    main()
