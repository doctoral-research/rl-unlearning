# ForgetScenario v1.4 — Language Reference

The **ForgetScenario** language is a small declarative DSL for describing
what behavior an RL agent should unlearn. A scenario is a single YAML
file that:

1. Identifies the trajectories whose behavior should be removed
   (`identify_forget_trajectories`),
2. Identifies the specific states inside those trajectories
   (`get_forget_states_from_trajectories`),
3. Is **machine-checkable** against a versioned JSON Schema +
   semantic validator,
4. Renders to **natural-language prose** and **LaTeX predicate logic**
   for paper figures,
5. Is evaluated against a **feature view** (§ Feature views), which is
   what makes the same scenario portable across domains.

Every scenario, no matter how it is written, ends up as a single
predicate `\mathrm{forget}(\tau)` over trajectories. The language is the
surface syntax for building that predicate.

---

## Feature views — the domain interface

A scenario never reads the policy's observation directly. It reads a
**feature view**: the map

```text
Sigma : (obs, info, env) -> R^n
```

plus the names, dtypes and ranges of those `n` features. The view is the
*only* thing the language knows about a domain.

This matters because the observation is often the wrong place to look.
A MiniGrid agent may observe a 7x7x3 image with no dimension meaning
"agent_x"; a Gymnasium MuJoCo observation deliberately omits the root x
coordinate so policies stay translation invariant. Because `extract` may
read the `info` dict and the environment object, a scenario can be
written over semantics **the agent never observes**.

Views are resolved by a tiered cascade and every one records its
**provenance**, so portability is measured rather than asserted:

| Provenance | Tier | How the names were obtained |
|---|---|---|
| `auto:<domain>` | env | introspected from the environment (`minigrid`, `mujoco`, `lbf`, `dict_space`) |
| `declared:table` | table | hand-registered in `constants.ENV_OBS_DIMS` |
| `auto:info_dict` | generic | scalar keys scraped from the `info` dict |
| `declared:identity` | fallback | anonymous `dim_0, dim_1, ...` (v1.0 behaviour) |
| `learned:<kind>` | — | fitted from data; carries a `fidelity` score |

```bash
python -m scenarios domains                # which view each env resolves to
python -m scenarios domains --markdown     # same, as a table
```

Experiments should pin the provider (`scenario_view:` in the Hydra env
config) so a view cannot silently degrade between runs. Env-introspecting
views are opt-in for exactly this reason: they change what an integer
`dim:` index *means*.

> **Integer dims are view-fragile; names are view-portable.** Under the
> HalfCheetah name table `rooty` is index 1; under `auto:mujoco` (which
> indexes `qpos ++ qvel`, including the dropped `rootx`) it is index 2.
> A scenario written as `dim: rooty` is correct under both. One written
> as `dim: 1` is correct under only one of them. Prefer names.

---

## Anatomy of a scenario

```yaml
version: "1.0"                 # optional but recommended
name: left_only                # stable identifier
description: >                 # paper-quality prose
  Forget right-side balancing in CartPole. After unlearning, the agent
  should only balance the pole while keeping the cart on the left side.
env_id: CartPole-v1            # enables dim-name resolution + validation
match_mode: proportion         # how trajectories are judged to match
match_threshold: 0.3
groups:
  - name: right_side
    conditions:
      - dim: cart_position     # name OR integer index
        op: ">"
        value: 0.0
        target: state
```

Predicate rendering (`python -m scenarios summary cartpole/left_only --latex-only`):

```math
\mathrm{forget}(\tau) \iff
  \frac{|\{ t : \mathrm{cart\_position}(s_t) > 0 \}|}{|\tau|} \geq 0.30
```

Natural-language rendering:

> Scenario 'left_only' (CartPole-v1): forget trajectories where, on at
> least 30% of the steps, cart_position is greater than 0.

---

## Match modes

The `match_mode` determines how the scenario's group predicate is lifted
to a trajectory predicate.

| Mode | Predicate | When to use |
|---|---|---|
| `any_step` | $\exists\, t.\; P(s_t)$ | Forget any trajectory that ever enters the forbidden region |
| `all_steps` | $\forall\, t.\; P(s_t)$ | Forget only fully-forbidden trajectories |
| `proportion` | $\\frac{\|\\{t : P(s_t)\\}\|}{\|\tau\|} \geq \theta$ | Forget trajectories that *mostly* live in the forbidden region |
| `top_percentile` | $\tau \in \mathrm{top}_\theta(\mathrm{rank})$ | Forget the best (or worst) θ-fraction by return/length |
| `trajectory` | groups evaluated on episode-level aggregates (return, length, action sequences) | Forget by aggregate outcome rather than per-state |
| `temporal` | LTL-style operator over states | Patterns like "A then B within K" or "K consecutive matching steps" |

`match_threshold` is required for `proportion` and `top_percentile`.

---

## Groups and conditions (state-level core)

A **group** is a predicate over a single `(state, action)` pair. The
top-level `groups:` list is OR'd: a trajectory matches if **any** group
matches. Each group is one of six forms.

### Leaf group — AND'd conditions

```yaml
- name: right_side
  conditions:
    - {dim: 0, op: ">", value: 0.0, target: state}
```

Multiple `conditions:` entries are conjoined.

### Boolean combinators

```yaml
- all_of:                              # AND
    - {conditions: [...]}
    - {conditions: [...]}
- any_of:                              # OR (within a single group)
    - {conditions: [...]}
    - {conditions: [...]}
- not:                                 # negation
    {conditions: [...]}
```

Combinators are recursive: an `all_of` child can itself contain
`any_of`, `not`, leaf groups, or references.

### Region shorthand

Axis-aligned boxes desugar to AND'd inequalities, but read better in
both the YAML and the rendered output.

```yaml
- region:
    type: box
    dims: [agent_x, agent_y]   # names if env_id is set, else integers
    lo: [13, 13]
    hi: [15, 15]
```

`null` in `lo` / `hi` means unbounded on that side (half-space). Use
`type: complement_of_box` for "anything outside" the region.

Two non-box region types are also available. `ellipsoid` is axis-aligned
and n-dimensional, matching where the normalised squared distance is
below 1:

```yaml
- region:
    type: ellipsoid
    dims: [x, y]
    center: [0.0, 1.4]
    radii:  [0.3, 0.2]
```

`polygon` is strictly 2-D and uses ray casting, so concave outlines work:

```yaml
- region:
    type: polygon
    dims: [agent_x, agent_y]
    vertices: [[10, 10], [17, 10], [17, 17], [13, 13], [10, 17]]
```

### Named macros

Factor out a reusable building block once, reference it many times:

```yaml
defines:
  bottom_right_box:
    region: {type: box, dims: [agent_x, agent_y], lo: [13, 13], hi: [15, 15]}

groups:
  - $ref: bottom_right_box
```

The validator warns about unused macros.

---

## Conditions

A **condition** is the atomic predicate inside a leaf group. Three kinds:

### `dim` condition (default)

```yaml
- dim: cart_position      # int index or name (env_id required for names)
  dim_name: cart_position # optional human label; validator cross-checks
  op: ">"                 # >, >=, <, <=, ==, !=, abs>, abs<, abs>=, abs<=
  value: 0.0
  target: state           # state (default) | action | reward | return | length
```

`target: action` evaluates the operator against the action taken at
that step. `target: reward` evaluates the step's scalar reward, and so
needs no `dim`; it reports as unknown (treated as false) when the caller
does not supply a reward. `target: return` / `target: length` are
trajectory-level — only valid when `match_mode: trajectory` (validator
enforces this).

### `action_sequence` condition

Matches when the trajectory contains a contiguous action subsequence.

```yaml
- action_sequence: [0, 0, 2]
```

Trajectory-scoped; pair with `match_mode: trajectory`.

### `$ref` condition

Inline a macro from the `defines` block.

```yaml
- $ref: turn_left_action
```

---

## Indexed families and quantification

Multi-agent views expose one block of features per agent (`p0_row`,
`p1_row`, ...). Reading those names was possible in v1.2, but saying
"over all agents" was not, so a predicate about *the team* had to
enumerate every pair by hand — quadratic in team size, and rewritten for
every N.

A `families:` block names an indexed group, and a quantified `derived:`
entry binds members and reduces over the bindings:

```yaml
families:
  agent: "p{i}_"          # members discovered from the bound view

derived:
  closest_pair:
    over: agent
    arity: 2              # bind 2 members, named a and b
    reduce: min
    expr: "abs(a_row - b_row) + abs(a_col - b_col)"

groups:
  - conditions:
      - dim: closest_pair
        op: "<"
        value: 2
```

Bindings are named `a`, `b`, `c`, ... and an attribute of a binding is
written `<binding>_<attr>`, so `a_row` resolves to `p0_row` for the first
member. Substitution is word-boundary aware, so a template calling
`abs(...)` is not mangled by the binding named `a`.

| Field | |
|---|---|
| `over` | family name |
| `arity` | how many members to bind (1–8, default 1) |
| `reduce` | `min` `max` `sum` `mean` `count` `any` `all` |
| `distinct` | bindings must differ (default true) |
| `ordered` | permutations rather than combinations (default false) |

`any` and `all` treat a non-zero value as true, so a template that reads
as a comparison gives existential and universal quantification.

**Members come from the view, not the file.** The same scenario adapts to
any team size with no edits:

| Environment | members | pair expressions |
|---|---|---|
| `Foraging-8x8-2p-1f-v3` | 2 | 1 |
| `Foraging-8x8-3p-2f-v3` | 3 | 3 |
| `Foraging-8x8-4p-2f-v3` | 4 | 6 |

Quantification adds no evaluation machinery: each binding combination is
instantiated into an ordinary expression at load time and compiled by the
same whitelisted compiler, then the reduction folds the results. This is
what moved `lbf.no_crowding` from partial to expressible.

---

## Episode aggregates

A condition compares a feature at **one step**. Two episode-level
quantities were built in (`return`, `length`) and nothing else about an
episode as a whole was sayable. An `aggregates:` block generalises those
two into arbitrary named reductions:

```yaml
match_mode: trajectory
applies_to: whole_trajectory

aggregates:
  cells_visited: {op: count_distinct, of: [agent_x, agent_y]}
  path_length:   {op: count}

derived:
  revisits: "path_length - cells_visited"

groups:
  - conditions:
      - dim: revisits
        op: ">"
        value: 4
```

| op | reduces |
|---|---|
| `sum` `mean` `min` `max` `std` `range` | one feature's per-step values |
| `first` `last` `abs_sum` `n_distinct` | one feature's per-step values |
| `count` | steps (optionally filtered by `where:`) |
| `count_distinct` | distinct combinations of a feature tuple |

Aggregate names are addressable from a condition exactly like `return`
and `length`, and usable inside `derived:` expressions — so comparisons
*between* aggregates need no new syntax. `where:` takes a full group
expression, including `$ref`, so filtered counts reuse the whole grammar.

Aggregates describe an episode, so they require `match_mode: trajectory`.
A step-level consumer that encounters one reports unknown rather than
failing mid-rollout.

**What this closed.** Four of the six gaps recorded in the behaviour
corpus were gaps for want of exactly this construct, and corpus
expressibility moved from **0.688 to 0.875**:

| Behaviour | Before | After | Reduction that fixed it |
|---|---|---|---|
| `no_backtracking` | inexpressible | expressible | `count_distinct` over position |
| `no_front_flip` | partial | expressible | `sum` of angular velocity |
| `lbf.no_freeloading` | inexpressible | expressible | per-agent `max` of score |
| `no_pacing` | inexpressible | partial | steps per distinct cell |

`no_pacing` stays partial honestly: steps-per-cell separates pacing from
exploration, but "the same two cells more than k times" needs a counting
operator over a repeated sub-pattern, and aggregates reduce an episode to
scalars.

---

## Step scope (`applies_to`)

`match_mode` decides which **trajectories** match. It does not say which
**steps** of those trajectories carry the forget label, and those are
different sets:

```yaml
match_mode: proportion        # a trajectory matches if >=15% of its steps
match_threshold: 0.15         # are in the region
applies_to: matching_steps    # ...but only those steps get the label
```

| Value | Labelled steps |
|---|---|
| `matching_steps` (default) | only the steps satisfying the predicate |
| `whole_trajectory` | every step of a trajectory that `match_mode` selects |

Both readings are legitimate. Removing a specific behaviour argues for
`matching_steps`; removing a whole episode-level pattern, or attributing
credit across an episode, argues for `whole_trajectory`.

**Why it is a declared field rather than a convention.** Before this
existed, a step-level consumer called the step predicate directly and
ignored `match_mode` entirely, while a trajectory-level reading of the
same file said something else. Measured on trained FourRooms policies,
those two readings of one scenario differ by **17.4 percentage points** of
reported forget-effectiveness (0.0217 on a random policy — the gap grows
with policy quality). **18 of the 21 bundled scenarios were ambiguous**
in exactly this way.

The validator now flags an undeclared trajectory-level scenario as
`step_scope_ambiguous`.

**Building a training mask.** Use `step_labels`, which honours the
declaration:

```python
mask = scenario.step_labels(trajectory)      # respects applies_to
```

Calling `matches_features` per step gives the `matching_steps` reading
whatever the file declares — that is the drift `step_labels` removes.
`python -m scenarios drift` measures the gap when two specifications, or
a specification and a consumer, disagree.

---

## Derived features

A `dim` condition compares **one** feature against a constant, so
relational predicates ("closer to the goal than to the hazard", "leading
the opponent") are not expressible directly. A `derived:` block adds
named features computed from existing ones:

```yaml
derived:
  dx:                "agent_x - 13"
  dy:                "agent_y - 3"
  manhattan_to_goal: "abs(dx) + abs(dy)"

groups:
  - conditions:
      - dim: manhattan_to_goal
        op: ">"
        value: 12
```

Entries are compiled in declaration order and may reference earlier
entries, so chains work without flattening. Forward references are a
load-time error, as is a reference to a feature the view does not have.

Derived names are addressed by `dim:` exactly like base features, so they
compose with boolean combinators, every match mode, and the temporal
operators.

**Grammar.** Arithmetic (`+ - * / ** %`), unary minus, parentheses,
numeric literals, feature names, and these functions:

```text
abs  min  max  sqrt  hypot  floor  ceil  sign
```

**Safety.** Expressions are parsed with `ast` and evaluated by a
whitelisted walker — never `eval`. Attribute access, comprehensions,
lambdas, conditionals, boolean operators, string literals and calls to
anything outside the list above are rejected at load time, so a scenario
file cannot execute code. Division by zero yields `NaN` rather than
raising mid-rollout.

---

## Multi-agent domains

Multi-agent state breaks the flat-observation assumption twice: each
agent has its own egocentric observation, and the property a scenario
wants to name ("the agents crowded together") belongs to the joint state
that no single agent observes.

The language needs no new syntax for this. A multi-agent view reads the
environment's ground truth and exposes **indexed names**, and `derived:`
supplies the relational half:

```yaml
derived:
  separation_01: "abs(p0_row - p1_row) + abs(p0_col - p1_col)"
  separation_02: "abs(p0_row - p2_row) + abs(p0_col - p2_col)"
  min_separation: "min(separation_01, separation_02)"

groups:
  - conditions:
      - dim: min_separation
        op: "<"
        value: 2
```

`scenarios.contrib.lbf` is a worked example over Level-Based Foraging,
exposing `p{i}_row`, `p{i}_col`, `p{i}_level`, `p{i}_score` plus food
aggregates. It also demonstrates the plugin path: a new domain family is
an adapter plus a `register_view_provider` call, never a change to the
language core.

```python
from scenarios.contrib.lbf import register
register()
```

---

## Temporal operators

When `match_mode: temporal`, the scenario carries a top-level `temporal:`
block instead of relying on `groups:` to drive matching.

**Formulas nest.** A formula is evaluated *at a position*, so operands
may be other formulas rather than only state predicates:

```yaml
temporal:
  eventually:
    and:
      - $ref: holding_key
      - not:
          eventually:
            $ref: opens_door
```

That is `eventually(key ∧ ¬eventually(open))` — picks up the key and
never opens the door afterwards. A trajectory matches when the formula
holds at position 0.

Any mapping that is not an operator is a **state predicate**, so leaves
get the whole state grammar (regions, boolean combinators, `$ref`,
derived features) without restatement.

| Operator | Form | Meaning at position t |
|---|---|---|
| `not` | `{not: F}` | F does not hold at t |
| `and` / `or` | `{and: [F, ...]}` | all / any hold at t |
| `next` | `{next: F}` | F holds at t+1 |
| `eventually` | `{eventually: F}` | F holds at some t' ≥ t |
| `always` | `{always: F}` | F holds at every t' ≥ t |
| `until` | `{until: {hold, release}}` | release at some t' ≥ t, hold throughout [t, t') |
| `consecutive` | `{consecutive: {k, of}}` | some t' ≥ t starts k steps of F |
| `then_within` | `{then_within: {first, second, within}}` | F at some t' ≥ t, G within w after |

`consecutive` and `then_within` keep an existential anchoring ("some
later position") so v1.0 files keep their original meaning, where the
only evaluation point was the start of the episode. The v1.0 flat form
`{operator: eventually, group: {...}}` still parses; its operands simply
become leaves.

This closed `doorkey.no_key_hoarding`, the last inexpressible entry in
the corpus.

The five original operators:

| Operator | Predicate | Required fields |
|---|---|---|
| `consecutive` | $\exists t.\; \bigwedge_{i=0}^{k-1} P(s_{t+i})$ | `k`, `group` |
| `eventually` | $\exists t.\; P(s_t)$ | `group` |
| `always` | $\forall t.\; P(s_t)$ | `group` |
| `then_within` | $\exists t,t'.\; P(s_t) \wedge Q(s_{t'}) \wedge 0 < t'-t \leq W$ | `first`, `second`, `within` |
| `until` | $\exists t.\; Q(s_t) \wedge \forall t' < t.\; P(s_{t'})$ | `hold`, `release` |

`until` is the *strong* form: `release` must eventually fire, and `hold`
must be true at every step strictly before it.

```yaml
match_mode: temporal
temporal:
  operator: until
  hold:    {conditions: [{dim: agent_x, op: "<", value: 5}]}
  release: {conditions: [{dim: agent_x, op: ">=", value: 9}]}
```

The operands of a temporal operator are **state predicates**, not nested
temporal formulas, so this is a deliberately shallow fragment of LTL:
`eventually(always(P))` is not expressible.

```yaml
match_mode: temporal
temporal:
  operator: then_within
  within: 1
  first:  {$ref: bottom_right_box}
  second: {$ref: turn_left_action}
```

This is the predicate the rendered showcase scenario lowers to:

```math
\mathrm{forget}(\tau) \iff
  \exists\, t, t'.\;
    \big( \mathrm{agent\_x}(s_t) \geq 13 \wedge \cdots \big)
    \wedge \big( \mathrm{turn\_left}(a_{t'}) = 0 \big)
    \wedge 0 < t' - t \leq 1
```

---

## Learned views

Every view described so far rests on someone already knowing what the
features mean: an environment adapter, a hand-written name table, or the
keys of an `info` dict. A raw pixel observation has none of that. No
index of a rendered frame means `agent_x`, so by default the language is
simply inapplicable to the domain.

A **probe** closes that gap. Fit a small model from raw observations to
the semantic features you want, and expose its predictions as a view.
Nothing else changes, because a view is a view.

```python
from scenarios.learned import fit_probe_view

view, report = fit_probe_view(
    frames,                          # (N, D) raw observations
    positions,                       # (N, 2) the features you want named
    ["agent_x", "agent_y"],
    dtypes=("ordinal", "ordinal"),
    min_fidelity=0.9,
)
print(report.render())
```

The returned `FeatureView` carries `provenance="learned:probe"` and a
measured `fidelity`, and binds to a scenario like any other:

```python
ForgetScenario(spec, view=view)     # spec references agent_x, agent_y
```

### Fidelity is the point

A fitted view can be wrong, and a wrong view is worse than a missing one.
It produces confident numbers that mean nothing, and every scenario
written against it inherits the error in silence. So the probe is fitted
on one split and **scored on another**, per feature, and the view carries
that score. The reported fidelity is the *weakest* feature, because a
scenario may depend on any one of them.

`min_fidelity` turns that score into a gate. Below it, `fit_probe_view`
raises `ProbeError` naming the weak features instead of returning a view
with a caveat nobody reads.

This is what keeps `learned:` an honest tier rather than an escape hatch.
Some features genuinely are not recoverable, and the correct behaviour
there is to fail loudly. `scripts/demo_learned_view.py` shows both sides
on MiniGrid-FourRooms, using 4769 samples:

| Raw observation | Width | `agent_x` | `agent_y` | Outcome |
|---|---|---|---|---|
| full-grid render, downsampled | 17328 | 0.947 | 0.959 | accepted |
| egocentric 7x7x3 view | 147 | 0.063 | 0.057 | **refused** |

Neither observation contains the agent's absolute position at any index.
The first *determines* it, so a spatial scenario runs on pixels unchanged
and agrees with ground truth on 400/400 steps. The second does not
determine it at all, which is a property of partial observability rather
than of the fitting method, and no amount of extra capacity would change
that. The gate reports it rather than hiding it behind a plausible
number.

```bash
python scripts/demo_learned_view.py
```

### When to reach for one

A probe is the right tool when the feature is well-defined but the
observation does not expose it. It is not a way to dodge writing an
adapter: when the environment can be introspected, `auto:<domain>` is
exact and free, while a probe is approximate and needs labelled data.
Its real use is the case where no exact source exists at all, such as a
perceptual property that only a human can label. The recorded gap for
`halfcheetah.no_unnatural_gait` in the corpus is exactly that shape.

---

### Temporal labels are per position

A temporal formula is evaluated *at* a position, so with
`applies_to: matching_steps` a step carries the forget label exactly when
the formula holds there. This matters for anything phrased "immediately
on entering", "right after" or "as soon as": the whole point of such a
description is *which* step is at fault.

Trajectory-level matching is anchored differently. A trajectory matches
when the formula holds at its **first** position, so a pattern that
occurs mid-episode must be wrapped in `eventually` to select the episode.
Step-level labelling needs no wrapper.

```yaml
match_mode: temporal
applies_to: matching_steps
temporal:
  and:
    - conditions:                       # in the room, turning left, now
        - {dim: agent_x, op: ">", value: 9.5}
        - {dim: turn_left, op: "==", value: 0, target: action}
    - previous:                         # ...having been outside before
        conditions:
          - {dim: agent_x, op: "<", value: 9.5}
```

Writing the same rule with `next` instead anchors it one step early and
labels the wrong transition: the step before the boundary rather than the
crossing itself. That is what `previous` exists for.

> Until v1.5 this path fell back to the state-level `groups` predicate,
> silently discarding the temporal structure. The corpus entry
> `no_corner_turn_reflex` read "turn left immediately on entering the
> room" and labelled turning left *anywhere* in it, about twenty times as
> many steps. Fidelity against an independent reading is what caught it.

---

## Mining a scenario from demonstrations

Writing a specification assumes you can already say what the behaviour
is. Often you cannot: what exists is a set of episodes someone flagged,
and the rule behind them has to be recovered.

```bash
python -m scenarios mine flagged.pkl unflagged.pkl \
    --env MiniGrid-FourRooms-v0 --out mined.yaml
```

```python
from scenarios import mine_scenario
scenario, report = mine_scenario(flagged, unflagged, view=view)
```

The search is tractable because the target form is restricted: top-level
`groups` are OR'd and `conditions` are AND'd, so a scenario *is* a DNF
over threshold literals, and learning small DNFs is sequential covering.
The restriction is the point — a specification nobody can read is not a
specification.

1. propose literals from values actually observed (midpoints between
   consecutive distinct values are the only thresholds that separate the
   data differently);
2. beam-search a precise conjunction;
3. **simplify** it — drop literals and loosen thresholds as far as they
   go without admitting a false positive;
4. keep it as a group, drop the positives it covers, repeat.

Step 3 matters more than it looks. Beam search commits early, so a
tighter literal can win at depth 1 and leave a rule that covers less than
it could — which then needs a *second* rule for the remainder. Two rules
where one would do is exactly what this is supposed to avoid.

Operators are proposed by dtype: ordering for numeric features, equality
for categorical and boolean ones, so the miner cannot emit the mistake
the typed validator exists to reject.

**Recovered against a known rule.** Given only episode-level flags from
random rollouts in FourRooms, and never shown the rule:

```
recovered    agent_x > 9.5 AND agent_y > 8.5
ground truth agent_x > 9.5 AND agent_y > 9.5
precision 1.000  recall 1.000  size 1 rule, 2 literals
disagreements on 4721 observed steps: 0
```

The constant differs and the predicates are still extensionally equal on
every state in the sample: no state with `agent_x > 9.5, agent_y == 9`
was ever visited, so nothing in the data pins that threshold down. This
is the honest general property — **a mined constant is only as tight as
the data makes it**, and `MiningReport` carries fit and size together so
the trade is visible.

Across sample sizes (20-160 episodes, 3 seeds each) recovery stays small
and close: 1-2 rules, 2-4 literals, F1 0.96-0.98, and 0.06%-1.5% of
observed steps classified differently from ground truth. The *exact*
rule varies with the sample even when the fit does not.

---

## Authoring from a description

A language model is a reasonable way to draft a scenario and a poor thing
to trust unchecked. The characteristic failures all parse cleanly:

* a feature name that does not exist in the domain;
* a threshold outside the feature's range, so the scenario matches
  nothing;
* a trajectory-level `match_mode` with no `applies_to`, leaving the label
  ambiguous;
* an ordering comparison on a categorical.

`scenarios.authoring` accepts a candidate only if it parses, passes the
schema, constructs against the domain's **feature view** (so every name it
mentions is real), and — when rollouts are supplied — actually fires on
them without matching everything. A rejected candidate is not resampled
blind: the validator's own diagnostics go back into the next prompt.

```python
from scenarios import author_scenario

result = author_scenario(
    "Stop routing through the bottom-right room.",
    backend=my_llm,             # any Callable[[str], str] -> YAML
    view=view,                  # supplies the legal feature vocabulary
    trajectories=rollouts,      # optional empirical gate
    max_attempts=3,
)
result.accepted, result.repaired, result.n_attempts
```

Every attempt is retained, so convergence is measurable: how often the
loop succeeds, in how many rounds, and how much the feedback is worth
against resampling. `FATAL_WARNINGS` lists the warnings treated as
rejections — advice when a human is reading them, defects when a machine
generated the file.

The prompt's grammar reference is generated from the same tables the
validator uses, so it cannot drift from the implementation, and the
feature list comes from the bound view rather than being written by hand.

Without a backend the CLI still does the useful half:

```bash
# print the exact prompt, including this domain's feature vocabulary
python -m scenarios author "Stop cutting through the bottom-right room" \
    --env MiniGrid-FourRooms-v0

# score a candidate and print the feedback a repair loop would send back
python -m scenarios author "..." --env MiniGrid-FourRooms-v0 \
    --check candidate.yaml --against rollouts.pkl
```

`--check` exits non-zero on rejection, so it works as a CI gate on
scenario files.

---

## Validation

```bash
python -m scenarios validate 'configs/scenarios/**/*.yaml' --summary
```

Three layers run in order; semantic and lint layers are skipped if the
schema layer fails.

| Layer | What it catches |
|---|---|
| **Schema** (JSON Schema Draft 2020-12) | Typos in match_mode/op/target, missing fields, wrong types, illegal enums, malformed `derived:` entries. |
| **Semantic** | dim out of range for `env_id`, dim_name vs canonical name mismatch, contradictory AND-grouped conditions, unresolved `$ref`, region dim/lo/hi shape mismatch, temporal operator argument mismatch, uncompilable derived expressions. |
| **Typed** (needs a bound view) | Ordering a categorical, ordering a boolean or comparing it against a non-binary value, exact `==` on a continuous feature, and thresholds outside a feature's declared range that make a condition constantly true or constantly false. |
| **Lint** | Missing `version`, short description, extreme `match_threshold`, unused `defines:` entries, `action_sequence` outside trajectory scope. |

The typed layer only runs when a feature view is bound, because dtypes
and ranges come from the view. Enable it with `--view`:

```bash
python -m scenarios validate 'configs/scenarios/**/*.yaml' --view auto
```

`--view auto` instantiates the environment to resolve a view, so it is
slower and needs that environment's dependencies; the default is `none`.
It is worth running: a threshold outside a feature's range means the
scenario matches nothing, which is otherwise invisible until a run
produces an empty forget set.

Editor integration: add a yaml-language-server directive at the top of
the YAML to get red-squiggle validation in VS Code:

```yaml
# yaml-language-server: $schema=../../../src/scenarios/schemas/scenario.schema.v1.json
```

---

## Introspection (paper figures)

```bash
python -m scenarios summary cartpole/left_only           # everything
python -m scenarios summary cartpole/left_only --latex-only
python -m scenarios summary cartpole/left_only --prose-only
```

Programmatic:

```python
from scenarios import ForgetScenario, to_natural_language, to_predicate

s = ForgetScenario.load("configs/scenarios/cartpole/left_only.yaml")
print(to_natural_language(s))   # paragraph for caption
print(to_predicate(s))          # LaTeX predicate-logic line
```

### Domain coverage

```bash
python -m scenarios domains              # provenance per environment
python -m scenarios domains --markdown   # table for the paper
```

Reports, for each environment, which view tier answered, how many
features it exposes, and the measured source lines of the adapter that
produced it.

### Empirical coverage

```bash
python -m scenarios coverage cartpole/left_only \
    --against experiments/outputs/cartpole_ppo_seed42/trajectories.pkl
```

Reports the matching-trajectory fraction, matching-step fraction, and
per-episode match-rate distribution. Use it in a paper sanity-check
("scenario X matches 12% of baseline rollouts") and as a quick test that
a hand-written scenario isn't trivially empty.

---

## Showcase scenario

The bundled [`configs/scenarios/fourrooms/showcase_v1.yaml`](../configs/scenarios/fourrooms/showcase_v1.yaml)
exercises every v1.0 construct (region shorthands, `any_of` composition,
`temporal` with `then_within`, named macros). It is a single example
that can be screenshotted in a methodology section to demonstrate the
language without page-bloat.

Render it:

```bash
python -m scenarios summary fourrooms/showcase_v1
```

---

## Grammar (EBNF, informal)

```ebnf
scenario       ::= "{" name description env_id? version? match_mode?
                       match_threshold? derived? defines? groups? temporal? "}"

derived        ::= "{" (identifier ":" expression)+ "}"   (* declaration order *)
expression     ::= term (("+"|"-"|"*"|"/"|"**"|"%") term)*
                 | "-" expression | "(" expression ")"
                 | function "(" expression ("," expression)* ")"
                 | identifier | number
function       ::= "abs" | "min" | "max" | "sqrt" | "hypot"
                 | "floor" | "ceil" | "sign"

groups         ::= "[" group_expr+ "]"                  (* OR'd *)
group_expr     ::= leaf_group | all_of_group | any_of_group | not_group
                 | region_group | ref_group
leaf_group     ::= "{" name? "conditions" ":" "[" condition+ "]"
                       rank_by? rank_order? "}"         (* AND'd *)
all_of_group   ::= "{" name? "all_of" ":" "[" group_expr+ "]" "}"
any_of_group   ::= "{" name? "any_of" ":" "[" group_expr+ "]" "}"
not_group      ::= "{" name? "not" ":" group_expr "}"
region_group   ::= "{" name? "region" ":" region "}"
ref_group      ::= "{" "$ref" ":" string "}"

region         ::= box | complement_of_box | ellipsoid | polygon
box            ::= "{" "type" ":" "box" "dims" ":" dims
                       "lo" ":" bounds "hi" ":" bounds "}"
complement_of_box  ::= "{" "type" ":" "complement_of_box"
                            "dims" ":" dims
                            "lo" ":" bounds "hi" ":" bounds "}"
ellipsoid      ::= "{" "type" ":" "ellipsoid" "dims" ":" dims
                       "center" ":" numbers "radii" ":" numbers "}"
polygon        ::= "{" "type" ":" "polygon" "dims" ":" dims2
                       "vertices" ":" "[" point point point+ "]" "}"

condition      ::= dim_condition | reward_condition
                 | action_sequence_condition | ref_condition
dim_condition  ::= "{" "dim" ":" (int|identifier) "op" ":" op "value" ":" number
                       "target"? "dim_name"? "}"        (* identifier may name
                                                           a derived feature *)
reward_condition ::= "{" "target" ":" "reward" "op" ":" op "value" ":" number "}"
action_sequence_condition ::= "{" "action_sequence" ":" "[" action+ "]" "}"
ref_condition  ::= "{" "$ref" ":" string "}"

temporal       ::= consecutive | then_within | eventually | always | until
op             ::= ">" | ">=" | "<" | "<=" | "==" | "!="
                 | "abs>" | "abs<" | "abs>=" | "abs<="
target         ::= "state" | "action" | "reward" | "return" | "length"
match_mode     ::= "any_step" | "all_steps" | "proportion"
                 | "top_percentile" | "trajectory" | "temporal"
```

The authoritative spec is
[`src/scenarios/schemas/scenario.schema.v1.json`](../src/scenarios/schemas/scenario.schema.v1.json).
