"""Stage 0 demo: the scenario language no longer needs the policy's observation.

Runs the SAME forget scenario against a FourRooms agent whose observation
is a raw MiniGrid dict (7x7x3 image + direction + mission) — an
observation with no interpretable scalar at any fixed index, which v1.0
of the language could not handle at all.

The scenario still evaluates, because the feature view reads the
environment rather than the observation. This is what removes the need
for `MiniGridFlatPos` to inject privileged (x, y, direction) into the
agent's input.

Run:
    python scripts/demo_feature_views.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scenarios import ForgetScenario, build_view  # noqa: E402


BOTTOM_RIGHT = {
    "name": "avoid_bottom_right_via_view",
    "description": (
        "Forget trajectories that spend at least 15% of their steps in the "
        "bottom-right room. Written against feature NAMES, resolved through "
        "whatever view the environment provides."
    ),
    "env_id": "MiniGrid-FourRooms-v0",
    "match_mode": "proportion",
    "match_threshold": 0.15,
    "groups": [
        {
            "name": "bottom_right_room",
            "conditions": [
                {"dim": "agent_x", "op": ">", "value": 9.5, "target": "state"},
                {"dim": "agent_y", "op": ">", "value": 9.5, "target": "state"},
            ],
        }
    ],
}


def main() -> None:
    from minigrid.envs.fourrooms import FourRoomsEnv

    # Raw MiniGrid env: NO position wrapper. The observation is a dict
    # whose 'image' is a 7x7x3 egocentric view.
    env = FourRoomsEnv(agent_pos=(2, 2), goal_pos=(13, 3))
    obs, _ = env.reset(seed=0)

    print("=" * 72)
    print("OBSERVATION the policy would see")
    print("=" * 72)
    print(f"  type      : {type(obs).__name__}")
    print(f"  keys      : {sorted(obs)}")
    print(f"  image     : {np.asarray(obs['image']).shape}")
    print("  -> no dimension of this observation means 'agent_x'.")
    print("     v1.0 (`obs[dim]`) cannot express a spatial scenario here.")

    view = build_view(env=env)
    print()
    print("=" * 72)
    print("FEATURE VIEW resolved for this environment")
    print("=" * 72)
    print(view.describe())

    scenario = ForgetScenario(BOTTOM_RIGHT, view=view)
    print()
    print("=" * 72)
    print("SCENARIO bound to the view")
    print("=" * 72)
    print(f"  'agent_x' resolved to index : {view.index('agent_x')}")
    print(f"  'agent_y' resolved to index : {view.index('agent_y')}")

    # Walk the agent around by teleporting, and ask the scenario about
    # each pose. `obs` is passed through untouched — the view ignores it.
    print()
    print("=" * 72)
    print("STEP-LEVEL MATCHING (obs is the raw dict; view reads the env)")
    print("=" * 72)
    probes = [
        ((2, 2), "top-left room"),
        ((13, 3), "top-right room (the goal)"),
        ((3, 14), "bottom-left room"),
        ((13, 13), "bottom-right room -> FORGET"),
        ((16, 16), "bottom-right corner -> FORGET"),
    ]
    ok = True
    for (x, y), label in probes:
        env.unwrapped.agent_pos = (x, y)
        hit = scenario.matches_state(obs, env=env)
        expected = x > 9.5 and y > 9.5
        ok &= hit == expected
        mark = "MATCH " if hit else "      "
        print(f"  agent at ({x:2d},{y:2d})  {mark}  {label}")

    # Offline replay: record features at collection time so a view that
    # reads the env can still be evaluated once the env is gone.
    print()
    print("=" * 72)
    print("TRAJECTORY-LEVEL MATCHING via recorded features")
    print("=" * 72)
    poses = [(2, 2), (5, 5), (11, 11), (13, 13), (14, 12), (8, 8)]
    feats = []
    for x, y in poses:
        env.unwrapped.agent_pos = (x, y)
        feats.append(scenario.features(obs, env=env))
    traj = {"observations": [None] * len(poses), "features": feats}
    idx = scenario.identify_forget_trajectories([traj])
    in_region = sum(1 for x, y in poses if x > 9.5 and y > 9.5)
    print(f"  steps in forget region : {in_region}/{len(poses)} "
          f"= {in_region / len(poses):.2f}  (threshold 0.15)")
    print(f"  trajectory matched     : {idx == [0]}")
    ok &= idx == [0]

    print()
    print("=" * 72)
    print("PASS" if ok else "FAIL")
    print("=" * 72)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
