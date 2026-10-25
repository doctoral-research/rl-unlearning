"""Multi-agent feature view for Level-Based Foraging (lbforaging).

Multi-agent domains break the "one flat observation" assumption twice
over: each agent has its own egocentric observation, and the quantity a
scenario wants to talk about ("the agents crowded together") is a
property of the *joint* state that no single agent observes.

This view reads the environment's ground truth and exposes per-agent
features under indexed names:

    p0_row, p0_col, p0_level, p0_score, p1_row, ...

Indexed names are all the multi-agent support the grammar needs, because
`derived:` already supplies the relational half:

    derived:
      separation_01: "abs(p0_row - p1_row) + abs(p0_col - p1_col)"
    groups:
      - conditions:
          - dim: separation_01
            op: "<"
            value: 2

Register with:

    from scenarios.contrib.lbf import register
    register()
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..views import (
    CONTINUOUS,
    ORDINAL,
    FeatureView,
    TIER_ENV,
    register_view_provider,
)


def lbf_view(env, probe=None) -> Optional[FeatureView]:
    """Ground-truth joint state of an lbforaging environment."""
    u = getattr(env, "unwrapped", env)
    players = getattr(u, "players", None)
    field = getattr(u, "field", None)
    if players is None or field is None:
        return None
    try:
        n_agents = len(players)
        rows, cols = np.asarray(field).shape
    except Exception:
        return None
    if n_agents == 0:
        return None

    names: list[str] = []
    dtypes: list[str] = []
    ranges: list[Optional[tuple]] = []
    for i in range(n_agents):
        names += [f"p{i}_row", f"p{i}_col", f"p{i}_level", f"p{i}_score"]
        dtypes += [ORDINAL, ORDINAL, ORDINAL, CONTINUOUS]
        ranges += [(0.0, float(rows - 1)), (0.0, float(cols - 1)), None, None]
    names += ["n_food", "food_level_total"]
    dtypes += [ORDINAL, ORDINAL]
    ranges += [None, None]

    def _extract(obs, info, env):
        u = getattr(env, "unwrapped", env)
        out: list[float] = []
        for p in u.players:
            pos = getattr(p, "position", None)
            r, c = (float(pos[0]), float(pos[1])) if pos is not None else (-1.0, -1.0)
            out += [r, c, float(getattr(p, "level", 0)),
                    float(getattr(p, "score", 0))]
        fld = np.asarray(u.field)
        out += [float((fld > 0).sum()), float(fld.sum())]
        return out

    return FeatureView(
        names=tuple(names),
        extract_fn=_extract,
        dtypes=tuple(dtypes),
        ranges=tuple(ranges),
        provenance="auto:lbf",
        requires=("env",),
        env_id=getattr(getattr(env, "spec", None), "id", None),
    )


def register() -> None:
    """Add the LBF provider to the cascade, ahead of the generic ones."""
    register_view_provider("lbf", lbf_view, tier=TIER_ENV, first=True)
