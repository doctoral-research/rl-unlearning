"""Introspection layer — render a scenario as natural-language prose
(`to_natural_language`) or LaTeX predicate logic (`to_predicate`), and
compute empirical coverage statistics against a trajectory corpus
(`coverage`).

These outputs are the showcase artefacts for the paper / thesis: the
same YAML can be presented to a reviewer in three forms, all derived
from the canonical AST built by `ForgetScenario`. Coverage gives an
empirical anchor — what fraction of an actual rollout the scenario
selects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .constants import OP_RENDER
from .scenario import ForgetScenario


# ---------------------------------------------------------------------------
# Natural-language rendering
# ---------------------------------------------------------------------------


def to_natural_language(scenario: ForgetScenario) -> str:
    """Render a scenario as a paragraph of prose suitable for a paper
    methodology section or a poster caption.

    Example output:

        "Scenario 'left_only' (CartPole-v1): forget trajectories where
        the cart position is greater than 0.0 for at least 30% of the
        steps."
    """
    env = scenario.env_id or "the environment"
    head = f"Scenario '{scenario.name}' ({env}): "
    body = _mode_prefix(scenario) + _render_target(scenario)
    return head + body + "."


def _mode_prefix(scenario: ForgetScenario) -> str:
    mm = scenario.match_mode
    thr = scenario.match_threshold
    if mm == "any_step":
        return "forget trajectories where, on at least one step, "
    if mm == "all_steps":
        return "forget trajectories where, on every step, "
    if mm == "proportion":
        return (
            f"forget trajectories where, on at least {thr:.0%} of the "
            "steps, "
        )
    if mm == "top_percentile":
        first = scenario.groups[0] if scenario.groups else {}
        rb = first.get("rank_by", "return")
        ro = first.get("rank_order", "descending")
        direction = "highest" if ro == "descending" else "lowest"
        return f"forget the {thr:.0%} of trajectories with the {direction} {rb}"
    if mm == "trajectory":
        return "forget trajectories satisfying "
    if mm == "temporal":
        return "forget trajectories matching the temporal pattern "
    return f"({mm}) "


def _render_target(scenario: ForgetScenario) -> str:
    if scenario.match_mode == "top_percentile":
        return ""  # already covered in the mode prefix
    if scenario.match_mode == "temporal" and scenario.temporal:
        return _render_temporal_nl(scenario, scenario.temporal)
    return _render_groups_nl(scenario, scenario.groups, top_level=True)


def _render_groups_nl(
    scenario: ForgetScenario, groups: Sequence[Dict[str, Any]], top_level: bool,
) -> str:
    parts = [_render_group_nl(scenario, g) for g in groups]
    if len(parts) == 1:
        return parts[0]
    joined = "; or ".join(parts)
    if top_level:
        return joined
    return f"({joined})"


def _render_group_nl(scenario: ForgetScenario, g: Dict[str, Any]) -> str:
    kind = g["kind"]
    if kind == "leaf":
        conds = g.get("conditions", [])
        if not conds:
            return "any trajectory"
        return _join_conjunction([_render_condition_nl(c) for c in conds])
    if kind == "all_of":
        return _join_conjunction(
            [_render_group_nl(scenario, c) for c in g["children"]]
        )
    if kind == "any_of":
        return _join_disjunction(
            [_render_group_nl(scenario, c) for c in g["children"]]
        )
    if kind == "not":
        return f"it is not the case that {_render_group_nl(scenario, g['child'])}"
    return "<unknown group>"


def _render_condition_nl(c: Dict[str, Any]) -> str:
    ck = c["kind"]
    if ck == "dim":
        label = c.get("dim_name") or f"dimension {c['dim']}"
        op_phrase = OP_RENDER.get(c["op"], {}).get("prose", c["op"])
        if c["target"] == "state":
            return f"{label} is {op_phrase} {_fmt_num(c['value'])}"
        return f"the agent's {label} is {op_phrase} {_fmt_num(c['value'])}"
    if ck == "trajectory_aggregate":
        op_phrase = OP_RENDER.get(c["op"], {}).get("prose", c["op"])
        return f"the episode {c['target']} is {op_phrase} {_fmt_num(c['value'])}"
    if ck == "action_sequence":
        return f"the action sequence {list(c['sequence'])} appears"
    return f"<unknown condition {ck}>"


def _render_temporal_nl(
    scenario: ForgetScenario, t: Dict[str, Any],
) -> str:
    op = t["operator"]
    if op == "consecutive":
        return (
            f"a run of at least {t['k']} consecutive steps where "
            f"{_render_group_nl(scenario, t['group'])}"
        )
    if op == "eventually":
        return f"at some step, {_render_group_nl(scenario, t['group'])}"
    if op == "always":
        return f"on every step, {_render_group_nl(scenario, t['group'])}"
    if op == "then_within":
        return (
            f"{_render_group_nl(scenario, t['first'])}, followed within "
            f"{t['within']} steps by {_render_group_nl(scenario, t['second'])}"
        )
    return f"({op})"


# ---------------------------------------------------------------------------
# Predicate-logic (LaTeX) rendering
# ---------------------------------------------------------------------------


def to_predicate(scenario: ForgetScenario) -> str:
    r"""Render a scenario as LaTeX predicate logic suitable for inclusion
    in a paper. The forget predicate `\mathrm{forget}(\tau)` is defined
    on a trajectory \tau = (s_0, a_0, \dots, s_{T-1}, a_{T-1}, s_T).

    Example output (proportion mode):

        \mathrm{forget}(\tau) \iff
          \frac{|\{ t : x_{\mathrm{cart}}(s_t) > 0 \}|}{|\tau|} \geq 0.30

    Style: minimal, no \begin{equation} wrapper — drop into an `align*`
    or `displaymath` block to taste.
    """
    body = _predicate_body(scenario)
    return r"\mathrm{forget}(\tau) \iff " + body


def _predicate_body(scenario: ForgetScenario) -> str:
    mm = scenario.match_mode
    thr = scenario.match_threshold

    if mm == "top_percentile":
        first = scenario.groups[0] if scenario.groups else {}
        rb = first.get("rank_by", "return")
        ro = first.get("rank_order", "descending")
        rank_dir = r"\mathrm{top}_{" + f"{thr:.2f}" + "}"
        if ro == "ascending":
            rank_dir = r"\mathrm{bottom}_{" + f"{thr:.2f}" + "}"
        return rf"\tau \in {rank_dir}\big(\mathrm{{{rb}}}\big)"

    if mm == "temporal" and scenario.temporal:
        return _predicate_temporal(scenario, scenario.temporal)

    pred_t = _predicate_groups_disjunction(scenario, scenario.groups)
    if mm == "any_step":
        return rf"\exists\, t.\; {pred_t}"
    if mm == "all_steps":
        return rf"\forall\, t.\; {pred_t}"
    if mm == "proportion":
        return (
            r"\frac{|\{ t : "
            + pred_t
            + r" \}|}{|\tau|} \geq "
            + f"{thr:.2f}"
        )
    if mm == "trajectory":
        return _predicate_groups_disjunction(scenario, scenario.groups, indicator="\\tau")
    return pred_t


def _predicate_groups_disjunction(
    scenario: ForgetScenario,
    groups: Sequence[Dict[str, Any]],
    indicator: tuple = ("s_t", "a_t"),
) -> str:
    parts = [_predicate_group(scenario, g, indicator) for g in groups]
    if len(parts) == 1:
        return parts[0]
    return r" \,\vee\, ".join(_paren(p) for p in parts)


def _predicate_group(
    scenario: ForgetScenario, g: Dict[str, Any], indicator: tuple,
) -> str:
    kind = g["kind"]
    if kind == "leaf":
        conds = g.get("conditions", [])
        if not conds:
            return r"\top"
        parts = [_predicate_condition(c, indicator) for c in conds]
        if len(parts) == 1:
            return parts[0]
        return r" \,\wedge\, ".join(_paren(p) for p in parts)
    if kind == "all_of":
        parts = [_predicate_group(scenario, c, indicator) for c in g["children"]]
        return r" \,\wedge\, ".join(_paren(p) for p in parts)
    if kind == "any_of":
        parts = [_predicate_group(scenario, c, indicator) for c in g["children"]]
        return r" \,\vee\, ".join(_paren(p) for p in parts)
    if kind == "not":
        return r"\neg " + _paren(_predicate_group(scenario, g["child"], indicator))
    return r"\bot"


def _predicate_condition(c: Dict[str, Any], indicator: tuple) -> str:
    """indicator is (state_var, action_var) — passed as a pair so temporal
    operators that introduce a second time index can rebind both at once
    without each renderer hardcoding `a_t`."""
    state_var, action_var = indicator
    ck = c["kind"]
    if ck == "dim":
        label = _latex_var(c.get("dim_name") or f"x_{{{c['dim']}}}")
        op_tex = OP_RENDER.get(c["op"], {}).get("math", c["op"])
        accessor = action_var if c["target"] == "action" else state_var
        if c["op"].startswith("abs"):
            base = op_tex.split()[-1]
            return rf"|{label}({accessor})| {base} {_fmt_num(c['value'])}"
        return rf"{label}({accessor}) {op_tex} {_fmt_num(c['value'])}"
    if ck == "trajectory_aggregate":
        op_tex = OP_RENDER.get(c["op"], {}).get("math", c["op"])
        return rf"\mathrm{{{c['target']}}}(\tau) {op_tex} {_fmt_num(c['value'])}"
    if ck == "action_sequence":
        seq = ",".join(str(x) for x in c["sequence"])
        return rf"({seq}) \sqsubseteq (a_0, \ldots, a_{{T-1}})"
    return r"\bot"


def _predicate_temporal(
    scenario: ForgetScenario, t: Dict[str, Any],
) -> str:
    op = t["operator"]
    if op == "consecutive":
        # The clean way to render "K consecutive matches starting at t" is
        # \bigwedge_{i=0..K-1} P(s_{t+i}, a_{t+i}); pass (s_{t+i}, a_{t+i})
        # straight to the group renderer instead of post-hoc string surgery.
        inner = _predicate_group(scenario, t["group"], ("s_{t+i}", "a_{t+i}"))
        return (
            r"\exists\, t.\; \bigwedge_{i=0}^{" + str(t["k"] - 1) + r"} "
            + _paren(inner)
        )
    if op == "eventually":
        return r"\exists\, t.\; " + _predicate_group(
            scenario, t["group"], ("s_t", "a_t"),
        )
    if op == "always":
        return r"\forall\, t.\; " + _predicate_group(
            scenario, t["group"], ("s_t", "a_t"),
        )
    if op == "then_within":
        a = _predicate_group(scenario, t["first"], ("s_t", "a_t"))
        b = _predicate_group(scenario, t["second"], ("s_{t'}", "a_{t'}"))
        return (
            r"\exists\, t,\, t'.\; " + _paren(a)
            + r" \,\wedge\, " + _paren(b)
            + rf" \,\wedge\, 0 < t' - t \leq {t['within']}"
        )
    return r"\bot"


def _latex_var(name: str) -> str:
    """Render a free-form dim name as a LaTeX-friendly symbol. Replaces
    underscores with `\_` only inside `\mathrm{}` (so subscripts stay
    intact), wraps multi-char names in `\mathrm{}`."""
    if all(c.isalpha() for c in name) and len(name) <= 2:
        return name
    safe = name.replace("_", r"\_")
    return rf"\mathrm{{{safe}}}"


def _fmt_num(v: float) -> str:
    if float(v).is_integer():
        return str(int(v))
    return f"{v:g}"


def _paren(s: str) -> str:
    return rf"\big({s}\big)"


def _join_conjunction(parts: List[str]) -> str:
    if not parts:
        return "any state"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _join_disjunction(parts: List[str]) -> str:
    if not parts:
        return "no state"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} or {parts[1]}"
    return ", ".join(parts[:-1]) + f", or {parts[-1]}"


# ---------------------------------------------------------------------------
# Empirical coverage
# ---------------------------------------------------------------------------


@dataclass
class CoverageReport:
    """Empirical match statistics computed against a trajectory corpus."""

    num_trajectories: int
    num_steps: int
    matching_trajectories: int
    matching_steps: int        # only meaningful for step-level modes
    forget_traj_fraction: float
    forget_region_fraction: float  # NaN for non-step-level modes
    per_episode_match_rate_mean: float
    per_episode_match_rate_std: float

    def render(self, scenario: ForgetScenario) -> str:
        lines = [
            f"Scenario:           {scenario.name} ({scenario.env_id or 'no env_id'})",
            f"Mode:               {scenario.match_mode}"
            + (f" @ {scenario.match_threshold}"
               if scenario.match_mode in ("proportion", "top_percentile")
               else ""),
            f"Trajectories:       {self.num_trajectories}",
            f"Total steps:        {self.num_steps}",
            "",
            f"Matching trajectories: {self.matching_trajectories}"
            f"  ({self.forget_traj_fraction:.1%})",
        ]
        if not _is_nan(self.forget_region_fraction):
            lines.append(
                f"Matching steps:        {self.matching_steps}"
                f"  ({self.forget_region_fraction:.2%} of total)"
            )
            lines.append(
                f"Per-episode match rate: mean={self.per_episode_match_rate_mean:.2%}, "
                f"std={self.per_episode_match_rate_std:.2%}"
            )
        return "\n".join(lines)


def coverage(
    scenario: ForgetScenario, trajectories: Sequence[Dict[str, Any]],
) -> CoverageReport:
    """Compute empirical match statistics for a scenario against a list of
    trajectories.

    Useful for paper/thesis: show that scenario X selects K% of baseline
    rollouts, or that scenario Y triggers on M% of steps. Lets reviewers
    sanity-check that the forget target isn't trivially empty or
    trivially all-encompassing.
    """
    n_traj = len(trajectories)
    if n_traj == 0:
        return CoverageReport(0, 0, 0, 0, 0.0, float("nan"), 0.0, 0.0)

    matching_traj = scenario.identify_forget_trajectories(list(trajectories))
    matching_traj_count = len(matching_traj)

    # Step-level stats only meaningful for step-level modes (otherwise the
    # concept of "matching step" is undefined).
    has_step = scenario.match_mode in ("any_step", "all_steps", "proportion")
    total_steps = sum(len(t.get("actions", [])) for t in trajectories)
    if not has_step:
        return CoverageReport(
            num_trajectories=n_traj,
            num_steps=total_steps,
            matching_trajectories=matching_traj_count,
            matching_steps=0,
            forget_traj_fraction=matching_traj_count / n_traj,
            forget_region_fraction=float("nan"),
            per_episode_match_rate_mean=0.0,
            per_episode_match_rate_std=0.0,
        )

    matching_steps = 0
    per_episode_rates = []
    for traj in trajectories:
        obs = traj["observations"]
        actions = traj.get("actions", [None] * len(obs))
        n_steps = len(actions)
        ep_match = 0
        for t in range(n_steps):
            act = actions[t] if t < len(actions) else None
            if scenario.matches_state(np.asarray(obs[t]), act):
                ep_match += 1
        matching_steps += ep_match
        per_episode_rates.append(ep_match / max(n_steps, 1))

    return CoverageReport(
        num_trajectories=n_traj,
        num_steps=total_steps,
        matching_trajectories=matching_traj_count,
        matching_steps=matching_steps,
        forget_traj_fraction=matching_traj_count / n_traj,
        forget_region_fraction=matching_steps / max(total_steps, 1),
        per_episode_match_rate_mean=float(np.mean(per_episode_rates)),
        per_episode_match_rate_std=float(np.std(per_episode_rates)),
    )


def _is_nan(x: float) -> bool:
    return x != x
