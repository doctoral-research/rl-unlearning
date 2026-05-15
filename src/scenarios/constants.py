"""Named observation / action dimensions for each supported env and the
comparison operators allowed inside a scenario condition.

Kept in a single module so the schema validator, semantic validator, and
introspection layer can all import a canonical mapping without circular
dependencies.
"""
from __future__ import annotations


ENV_OBS_DIMS: dict[str, dict[int, str]] = {
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
    # MiniGrid envs go through MiniGridFlatPos: first 3 dims are
    # (agent_x, agent_y, direction); the rest is the FlatObs encoding.
    "MiniGrid-Empty-8x8-v0": {
        0: "agent_x",
        1: "agent_y",
        2: "direction",
    },
    "MiniGrid-FourRooms-v0": {
        0: "agent_x",
        1: "agent_y",
        2: "direction",
    },
    "MiniGrid-DoorKey-5x5-v0": {
        0: "agent_x",
        1: "agent_y",
        2: "direction",
    },
    "MiniGrid-DoorKey-6x6-v0": {
        0: "agent_x",
        1: "agent_y",
        2: "direction",
    },
    "MiniGrid-DoorKey-8x8-v0": {
        0: "agent_x",
        1: "agent_y",
        2: "direction",
    },
}


ENV_ACTION_DIMS: dict[str, dict[int, str]] = {
    "CartPole-v1": {0: "push_left", 1: "push_right"},
    "Acrobot-v1": {0: "torque_negative", 1: "torque_zero", 2: "torque_positive"},
    "LunarLander-v3": {
        0: "noop", 1: "left_engine", 2: "main_engine", 3: "right_engine",
    },
    "MiniGrid-Empty-8x8-v0": {
        0: "turn_left", 1: "turn_right", 2: "forward",
        3: "pickup", 4: "drop", 5: "toggle", 6: "done",
    },
    "MiniGrid-FourRooms-v0": {
        0: "turn_left", 1: "turn_right", 2: "forward",
        3: "pickup", 4: "drop", 5: "toggle", 6: "done",
    },
}


# Operator vocabulary. Each maps a name to a Python predicate. Keep this
# table in sync with the JSON Schema enum in schemas/scenario.schema.v1.json.
_OPS = {
    ">":     lambda v, t: v > t,
    ">=":    lambda v, t: v >= t,
    "<":     lambda v, t: v < t,
    "<=":    lambda v, t: v <= t,
    "==":    lambda v, t: v == t,
    "!=":    lambda v, t: v != t,
    "abs>":  lambda v, t: abs(v) > t,
    "abs<":  lambda v, t: abs(v) < t,
    "abs>=": lambda v, t: abs(v) >= t,
    "abs<=": lambda v, t: abs(v) <= t,
}


# Human-readable rendering of each operator. Used by introspection layer
# for natural-language and predicate-logic output. Keep aligned with _OPS.
OP_RENDER = {
    ">":     {"prose": "greater than",         "math": ">"},
    ">=":    {"prose": "at least",             "math": "\\geq"},
    "<":     {"prose": "less than",            "math": "<"},
    "<=":    {"prose": "at most",              "math": "\\leq"},
    "==":    {"prose": "equal to",             "math": "="},
    "!=":    {"prose": "not equal to",         "math": "\\neq"},
    "abs>":  {"prose": "absolute value above", "math": "|\\cdot| >"},
    "abs<":  {"prose": "absolute value below", "math": "|\\cdot| <"},
    "abs>=": {"prose": "absolute value at least", "math": "|\\cdot| \\geq"},
    "abs<=": {"prose": "absolute value at most",  "math": "|\\cdot| \\leq"},
}


VALID_MATCH_MODES = {
    "any_step",
    "all_steps",
    "proportion",
    "top_percentile",
    "trajectory",
    "temporal",
}


VALID_TARGETS = {"state", "action", "return", "length"}


VALID_REGION_TYPES = {"box", "complement_of_box"}


VALID_TEMPORAL_OPERATORS = {"consecutive", "then_within", "eventually", "always"}


# Current schema version. Bumped when grammar changes in a breaking way.
SCHEMA_VERSION = "1.0"
