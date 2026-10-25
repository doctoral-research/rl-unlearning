"""Build tasks.json for the manual labelling desk.

Each task carries exactly what the model readers were given: the behaviour
description, the note on the features where the readers had one, and the
recorded steps (feature values and actions). It deliberately carries no
labels, no formalisation and no scores, so an annotator cannot see the
reference, the specification, or anyone else's answers.

Run from the repository root:
    python tools/labeling/make_tasks.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

RT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "tasks.json"
# Annotators see neutral codes, never the corpus ids, which name the
# formalisation ("left_only", "avoid_br_detour") and could steer a reading.
# The mapping stays here and is not published with the page.
IDS = Path(__file__).resolve().parent / "task_ids.json"

# The notes the readers' task files carried, verbatim.
NOTES = {
    "cartpole.left_only": "The CartPole track runs from -2.4 to +2.4.",
    "doorkey.no_key_hoarding": "carrying_key is 1 while the agent holds the key. The MiniGrid action codes are 0=turn left, 1=turn right, 2=forward, 3=pickup, 4=drop, 5=toggle (toggle is what opens a door), 6=done.",
    "empty8x8.avoid_bottom_left": "This is an 8x8 grid; the walkable interior is cells 1..6 in both axes. agent_y increases DOWNWARD.",
    "fourrooms.avoid_br_detour": "This is a 19x19 FourRooms grid split by walls at x=9 and y=9, with doorways in them. agent_y increases DOWNWARD.",
    "fourrooms.avoid_br_tr_door": "19x19 FourRooms. Walls at x=9 and y=9 divide it into four rooms; each wall has doorway gaps. agent_y increases DOWNWARD, so the bottom-right room is x>9 and y>9 and the top-right room is x>9 and y<9.",
    "fourrooms.no_backtracking": "19x19 FourRooms. (agent_x, agent_y) is the cell the agent occupies. Episodes here are short (12 steps).",
    "fourrooms.no_corner_turn_reflex": "MiniGrid action codes: 0=turn left, 1=turn right, 2=forward, 3=pickup, 4=drop, 5=toggle, 6=done. direction is the heading.",
    "fourrooms.no_pacing": "19x19 FourRooms. (agent_x, agent_y) is the occupied cell; direction is the heading. MiniGrid actions: 0=turn left, 1=turn right, 2=forward, 3=pickup, 4=drop, 5=toggle, 6=done. Turning does not change the cell, so moving to an adjacent cell and back takes several steps and the cell column repeats.",
    "halfcheetah.no_front_flip": "rooty is the torso pitch angle in RADIANS and is cumulative: it is not wrapped, so a full rotation is about 6.28. Positive and negative are opposite directions of rotation. v_rooty is its angular velocity.",
    "pendulum.avoid_left_swing": "The bob position is (sin_theta, cos_theta); cos_theta=+1 is upright.",
}

# Standard action meanings of each environment: documentation of the
# environment, not of any behaviour.
ACTIONS = {
    "minigrid": {"kind": "discrete", "names": ["turn left", "turn right", "forward",
                                                "pickup", "drop", "toggle", "done"]},
    "cartpole": {"kind": "discrete", "names": ["push left", "push right"]},
    "lunarlander": {"kind": "discrete", "names": ["do nothing", "fire left engine",
                                                   "fire main engine", "fire right engine"]},
    "lbf": {"kind": "per-agent", "names": ["none", "north", "south", "west", "east", "load"]},
    "pendulum": {"kind": "vector", "what": "torque"},
    "mujoco": {"kind": "vector", "what": "joint torques"},
}


def family(env: str) -> str:
    if env.startswith("MiniGrid"):
        return "minigrid"
    if env.startswith("Foraging"):
        return "lbf"
    if env.startswith("CartPole"):
        return "cartpole"
    if env.startswith("Pendulum"):
        return "pendulum"
    if env.startswith("LunarLander"):
        return "lunarlander"
    return "mujoco"


def num(v: float):
    """Four significant digits keeps the file small and loses nothing a
    reader could use."""
    v = float(v)
    if v == 0 or not math.isfinite(v):
        return 0
    r = float(f"{v:.4g}")
    return int(r) if r.is_integer() else r


def action(a):
    if a is None:
        return None
    if isinstance(a, list):
        return [num(x) for x in a]
    return num(a)


tasks = []
for f in sorted((RT / "benchmark/labels").glob("*.yaml")):
    d = yaml.safe_load(f.read_text())
    fam = family(d["env"])
    eps = []
    for ep in d["episodes"]:
        steps = ep["steps"]
        eps.append({
            "id": ep["id"],
            "t": [s["t"] for s in steps],
            "x": [[num(s[n]) for s in steps] for n in d["features"]],
            "a": [action(s.get("action")) for s in steps],
        })
    tasks.append({
        "id": f"B{len(tasks) + 1:02d}",
        "_entry": d["entry"],
        "env": d["env"],
        "family": fam,
        "grid": (19 if "FourRooms" in d["env"] else 8) if fam in ("minigrid", "lbf") else None,
        "description": " ".join(d["description"].split()),
        "note": NOTES.get(d["entry"], ""),
        "features": d["features"],
        "actions": ACTIONS["minigrid" if fam == "minigrid" else fam],
        "episodes": eps,
    })
    # Steps carry time, features and actions only: never a label.
    assert all(set(e) == {"id", "t", "x", "a"} for e in eps), d["entry"]

IDS.write_text(json.dumps({t["id"]: t.pop("_entry") for t in tasks}, indent=1) + "\n")
doc = {"version": 1, "steps": sum(len(e["t"]) for t in tasks for e in t["episodes"]),
       "tasks": tasks}
OUT.write_text(json.dumps(doc, separators=(",", ":")))
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB): {len(tasks)} tasks, {doc['steps']} steps")
for t in tasks:
    lens = [len(e["t"]) for e in t["episodes"]]
    print(f"  {t['id']} {t['env']:<26} eps={len(lens):>2} len {min(lens)}-{max(lens)}")
