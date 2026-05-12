"""Counterfactual demonstration synthesis for MiniGrid unlearning.

When the baseline policy converges to a single deterministic path, the
retain set is degenerate — almost every baseline trajectory matches the
forget region, leaving no on-distribution examples of the alternative
path. The unlearner is then asked to "stop doing X" with no concrete
"do Y instead" signal, and the policy collapses to inaction.

This module generates *counterfactual demos*: shortest-path trajectories
from start to goal that *avoid* the scenario's forget region. We run the
env with the scripted moves and save real (obs, action) trajectories
that unlearn.py can load as a retain set augmentation.

Usage (from unlearn.py):

    from utils.demos import generate_counterfactual_demos
    demo_trajs = generate_counterfactual_demos(
        env_id="MiniGrid-Empty-8x8-v0",
        forget_box=((1, 3), (4, 6)),
        num_demos=20,
        seed=42,
    )
    # demo_trajs is List[Dict] with the same schema as training trajectories.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# MiniGrid action constants — keep in sync with minigrid.core.actions.Actions.
A_LEFT = 0     # turn left
A_RIGHT = 1    # turn right
A_FORWARD = 2  # move forward in current heading

# MiniGrid heading conventions (env.unwrapped.agent_dir):
#   0 = right (+x), 1 = down (+y), 2 = left (-x), 3 = up (-y)
HEADING_DELTAS = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}


def _grid_obstacle_map(env: gym.Env) -> np.ndarray:
    """Return a (H, W) bool map where True = cell is walkable."""
    unwrapped = env.unwrapped
    grid = unwrapped.grid
    walkable = np.ones((unwrapped.height, unwrapped.width), dtype=bool)
    for y in range(unwrapped.height):
        for x in range(unwrapped.width):
            c = grid.get(x, y)
            if c is None:
                continue
            # Treat walls and lava as non-walkable. Goal IS walkable.
            if c.type in ("wall", "lava"):
                walkable[y, x] = False
    return walkable


def _in_forget_box(
    x: int, y: int, forget_box: Tuple[Tuple[int, int], Tuple[int, int]],
) -> bool:
    (xlo, xhi), (ylo, yhi) = forget_box
    return xlo <= x <= xhi and ylo <= y <= yhi


def astar_avoiding(
    walkable: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    forget_box: Tuple[Tuple[int, int], Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    """4-connected A* from start to goal on cells where walkable=True and
    not inside ``forget_box``. Returns list of (x, y) cells or None if
    unreachable."""
    H, W = walkable.shape

    def h(p):
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    open_heap: List[Tuple[int, int, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (h(start), 0, start))
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    cost: Dict[Tuple[int, int], int] = {start: 0}

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return list(reversed(path))
        x, y = cur
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if not walkable[ny, nx]:
                continue
            # Don't enter the forget box (but allow start / goal even if
            # they happen to lie inside it).
            if (nx, ny) != goal and (nx, ny) != start and _in_forget_box(nx, ny, forget_box):
                continue
            ng = g + 1
            if ng < cost.get((nx, ny), 10**9):
                cost[(nx, ny)] = ng
                came_from[(nx, ny)] = cur
                heapq.heappush(open_heap, (ng + h((nx, ny)), ng, (nx, ny)))
    return None


def _path_to_actions(
    path: List[Tuple[int, int]], initial_heading: int,
) -> List[int]:
    """Convert a sequence of grid cells into MiniGrid actions.

    Between consecutive cells we may need to rotate (TURN_LEFT or TURN_RIGHT)
    to face the target direction, then FORWARD once."""
    actions: List[int] = []
    heading = initial_heading
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        dx, dy = x1 - x0, y1 - y0
        target_heading = next(h for h, d in HEADING_DELTAS.items() if d == (dx, dy))
        # Rotate by shortest direction.
        diff = (target_heading - heading) % 4
        if diff == 1:
            actions.append(A_RIGHT)
        elif diff == 3:
            actions.append(A_LEFT)
        elif diff == 2:
            actions.append(A_RIGHT)
            actions.append(A_RIGHT)
        heading = target_heading
        actions.append(A_FORWARD)
    return actions


def _find_goal(env: gym.Env) -> Tuple[int, int]:
    unwrapped = env.unwrapped
    for y in range(unwrapped.height):
        for x in range(unwrapped.width):
            c = unwrapped.grid.get(x, y)
            if c is not None and c.type == "goal":
                return x, y
    raise RuntimeError("No goal cell found in env")


def generate_counterfactual_demos(
    env_id: str,
    forget_box: Tuple[Tuple[int, int], Tuple[int, int]],
    num_demos: int = 20,
    seed: int = 42,
    obs_encoding: str = "image",
) -> List[Dict]:
    """Generate counterfactual-path demonstrations.

    For ``num_demos`` resets (each with a different env seed), plan the
    shortest path from start to goal avoiding ``forget_box``, then roll the
    env forward with the scripted actions. Returns a list of trajectory
    dicts with keys ``observations``, ``actions``, ``rewards``, ``dones``.

    Demos that fail (no path found, or step terminates unexpectedly) are
    skipped. We use the same ``make_env`` wrapper used in training so the
    obs schema matches.
    """
    # Local import: avoids circular dependency at module-load time.
    from utils.env_wrappers import make_env

    demos: List[Dict] = []
    rng = np.random.default_rng(seed)
    attempt_seeds = rng.integers(0, 10**6, size=num_demos * 3).tolist()

    for s in attempt_seeds:
        if len(demos) >= num_demos:
            break
        env = make_env(env_id, seed=int(s), obs_encoding=obs_encoding)
        obs, _ = env.reset(seed=int(s))
        unwrapped = env.unwrapped
        start = tuple(int(v) for v in unwrapped.agent_pos)
        heading = int(unwrapped.agent_dir)
        goal = _find_goal(env)
        walkable = _grid_obstacle_map(env)
        path = astar_avoiding(walkable, start, goal, forget_box)
        if path is None or len(path) < 2:
            env.close()
            continue
        actions = _path_to_actions(path, heading)

        ep_obs = [obs.copy()]
        ep_actions: List[int] = []
        ep_rewards: List[float] = []
        ep_dones: List[bool] = []
        ok = True
        for a in actions:
            obs, r, term, trunc, _ = env.step(a)
            ep_obs.append(obs.copy())
            ep_actions.append(int(a))
            ep_rewards.append(float(r))
            ep_dones.append(bool(term or trunc))
            if term or trunc:
                # If we terminated before exhausting the planned actions,
                # the path must have reached the goal. Acceptable. If it
                # terminated unexpectedly, drop the demo.
                if r <= 0:
                    ok = False
                break
        env.close()
        if not ok or not ep_actions:
            continue
        demos.append({
            "observations": np.array(ep_obs, dtype=np.float32),
            "actions": np.array(ep_actions, dtype=np.int64),
            "rewards": np.array(ep_rewards, dtype=np.float32),
            "dones": np.array(ep_dones, dtype=np.bool_),
        })
    return demos


__all__ = ["generate_counterfactual_demos"]
