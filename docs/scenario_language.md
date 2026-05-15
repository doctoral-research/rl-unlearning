# ForgetScenario v1.0 — Language Reference

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
   for paper figures.

Every scenario, no matter how it is written, ends up as a single
predicate `\mathrm{forget}(\tau)` over trajectories. The language is the
surface syntax for building that predicate.

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
  target: state           # state (default) | action | return | length
```

`target: action` evaluates the operator against the action taken at
that step. `target: return` / `target: length` are trajectory-level —
only valid when `match_mode: trajectory` (validator enforces this).

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

## Temporal operators

When `match_mode: temporal`, the scenario carries a top-level `temporal:`
block instead of relying on `groups:` to drive matching. Four operators:

| Operator | Predicate | Required fields |
|---|---|---|
| `consecutive` | $\exists t.\; \bigwedge_{i=0}^{k-1} P(s_{t+i})$ | `k`, `group` |
| `eventually` | $\exists t.\; P(s_t)$ | `group` |
| `always` | $\forall t.\; P(s_t)$ | `group` |
| `then_within` | $\exists t,t'.\; P(s_t) \wedge Q(s_{t'}) \wedge 0 < t'-t \leq W$ | `first`, `second`, `within` |

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

## Validation

```bash
python -m scenarios validate 'configs/scenarios/**/*.yaml' --summary
```

Three layers run in order; semantic and lint layers are skipped if the
schema layer fails.

| Layer | What it catches |
|---|---|
| **Schema** (JSON Schema Draft 2020-12) | Typos in match_mode/op/target, missing fields, wrong types, illegal enums. |
| **Semantic** | dim out of range for `env_id`, dim_name vs canonical name mismatch, contradictory AND-grouped conditions, unresolved `$ref`, region dim/lo/hi shape mismatch, temporal operator argument mismatch. |
| **Lint** | Missing `version`, short description, extreme `match_threshold`, unused `defines:` entries, `action_sequence` outside trajectory scope. |

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
                       match_threshold? defines? groups? temporal? "}"

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

region         ::= box | complement_of_box
box            ::= "{" "type" ":" "box" "dims" ":" dims
                       "lo" ":" bounds "hi" ":" bounds "}"
complement_of_box  ::= "{" "type" ":" "complement_of_box"
                            "dims" ":" dims
                            "lo" ":" bounds "hi" ":" bounds "}"

condition      ::= dim_condition | action_sequence_condition | ref_condition
dim_condition  ::= "{" "dim" ":" (int|string) "op" ":" op "value" ":" number
                       "target"? "dim_name"? "}"
action_sequence_condition ::= "{" "action_sequence" ":" "[" action+ "]" "}"
ref_condition  ::= "{" "$ref" ":" string "}"

temporal       ::= consecutive | then_within | eventually | always
op             ::= ">" | ">=" | "<" | "<=" | "==" | "!="
                 | "abs>" | "abs<" | "abs>=" | "abs<="
target         ::= "state" | "action" | "return" | "length"
match_mode     ::= "any_step" | "all_steps" | "proportion"
                 | "top_percentile" | "trajectory" | "temporal"
```

The authoritative spec is
[`src/scenarios/schemas/scenario.schema.v1.json`](../src/scenarios/schemas/scenario.schema.v1.json).
