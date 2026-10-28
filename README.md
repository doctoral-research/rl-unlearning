<p align="center">
    <strong>Specifying What to Forget</strong><br>
    ForgetScenario, a language for naming the behaviour an RL agent should unlearn
</p>

<p align="center">
    <a href="https://github.com/doctoral-research/rl-unlearning/actions/workflows/ci.yml"><img src="https://github.com/doctoral-research/rl-unlearning/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
    <a href="https://github.com/doctoral-research/rl-unlearning/actions/workflows/docker.yml"><img src="https://github.com/doctoral-research/rl-unlearning/actions/workflows/docker.yml/badge.svg?branch=main" alt="Docker"></a>
    <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python 3.10+">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"></a>
</p>

Unlearning a behaviour starts with saying which behaviour. This repository
holds **ForgetScenario**, a small declarative language for that, together
with a benchmark of behaviours written in it, the PPO training and online
unlearning code that consumes it, and the source of the paper.

A scenario is a YAML file. It states a behaviour as a predicate over the
named features of an environment, is checked against a schema and against
the environment's feature view, renders to English and to a LaTeX
predicate, and yields one forget label per step for training.

```yaml
name: avoid_bottom_right
description: Stop routing through the bottom-right room.
env_id: MiniGrid-FourRooms-v0
match_mode: proportion       # an episode counts once 15% of its steps match
match_threshold: 0.15
applies_to: matching_steps   # only the matching steps carry the forget label
groups:
  - name: bottom_right_room
    conditions:
      - {dim: agent_x, op: ">", value: 9.5}
      - {dim: agent_y, op: ">", value: 9.5}
```

```text
$ python -m scenarios summary avoid_bottom_right.yaml --prose-only
Scenario 'avoid_bottom_right' (MiniGrid-FourRooms-v0): forget trajectories where,
on at least 15% of the steps, agent_x is greater than 9.5 and agent_y is greater than 9.5.
```

Beyond conjunctions of conditions, the language has boolean composition,
region shorthands and macros, derived features, episode aggregates,
indexed families with quantifiers for multi-agent domains, and a temporal
fragment (`next`, `previous`, `eventually`, `always`, `until`,
`consecutive`, `then_within`). [docs/scenario_language.md](docs/scenario_language.md)
is the full reference, with the grammar.

---

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Using the language](#using-the-language)
- [Training and online unlearning](#training-and-online-unlearning)
- [Benchmark and reference labels](#benchmark-and-reference-labels)
- [Labelling desk](#labelling-desk)
- [Paper](#paper)
- [Testing](#testing)
- [Docker](#docker)
- [Earlier work](#earlier-work)

---

## Repository layout

```text
src/
  scenarios/          the language: loader, schema and validator, feature
                      views, temporal operators, aggregates, derived features,
                      families, mining, authoring, drift, corpus scoring, CLI
  agents/             PPO (actor-critic, GAE, optional gSDE for continuous actions)
  utils/              environment wrappers, checkpointing, video, helpers
  train.py            PPO training, one environment
  train_vec.py        PPO training over N parallel environments
  online_unlearn.py   on-policy unlearning driven by a scenario's step labels
configs/
  config.yaml         Hydra defaults
  env/ agent/ logging/
  scenarios/          example scenarios, one folder per environment
benchmark/
  behaviours/         the behaviour corpus: description, verdict, formalisation
  scenarios/          formalisations that live with the corpus
  labels/             reference per-step labels, reader prompts, side studies
scripts/              label tasks, scoring, authoring evaluation, drift experiment,
                      batch training and unlearning runs
tools/labeling/       browser tool for collecting per-step labels
paper/                the paper as an Overleaf project, with the scripts that
                      regenerate its numbers and figures
docs/                 language reference
tests/                pytest suite
docker/               CPU and GPU images
```

Training writes to `experiments/` (checkpoints, logs, outputs), which is
not tracked.

---

## Installation

Python 3.10 or newer.

```bash
# conda (recommended, sets up CUDA 12.8 PyTorch wheels)
conda env create -f environment.yml
conda activate rl-unlearning

# or pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

The code imports its packages from `src/`. Scripts under `src/` find them
on their own. For `python -m scenarios` and anything else run from the
repository root, set the path once:

```bash
export PYTHONPATH=src
```

LunarLander needs `swig` for Box2D (`sudo apt install swig` or
`brew install swig`). The MuJoCo domains need `mujoco`, and the
multi-agent Level-Based Foraging domain needs `lbforaging`. Both are in
`requirements.txt`, and tests that need them skip when they are missing.

---

## Using the language

### Command line

```bash
python -m scenarios validate 'configs/scenarios/**/*.yaml' --view auto --summary
python -m scenarios summary fourrooms/avoid_bottom_right
```

| command | what it does |
| --- | --- |
| `validate` | schema, semantic and lint checks. `--view auto` also binds the environment's feature view and checks names, types and ranges |
| `summary` | the scenario with its English and LaTeX renderings |
| `domains` | which feature view each environment resolves to, and how |
| `list` | the scenarios under `configs/scenarios` |
| `collect` | roll out a policy (or random actions) and pickle the trajectories, with each step's features recorded while the episode runs |
| `coverage` | how many episodes and steps a scenario matches on collected trajectories |
| `drift` | whether two specifications label the same steps, or how far an index offset moves one specification's labels |
| `mine` | synthesise a scenario from flagged and unflagged trajectories |
| `author` | build the prompt that asks a model for a scenario, or check a candidate against the acceptance gates |
| `corpus` | score the behaviour corpus in `benchmark/behaviours` |

`coverage`, `drift`, `mine` and `author --against` read the file that
`collect` writes:

```bash
python -m scenarios collect MiniGrid-FourRooms-v0 --out rollouts.pkl \
    --checkpoint experiments/checkpoints/fourrooms_ppo_seed43/final_model.pt
python -m scenarios coverage fourrooms/avoid_bottom_right --against rollouts.pkl
```

With a checkpoint, `collect` rebuilds the environment from the env config
whose `env_id` matches, as the trainer built it. Pass `--env-overrides`
when the checkpoint was trained with other settings.

### Python

```python
import pickle
from scenarios import ForgetScenario

scenario = ForgetScenario.load("configs/scenarios/cartpole/left_only.yaml")
with open("rollouts.pkl", "rb") as f:
    trajectories = pickle.load(f)

labels = scenario.step_labels(trajectories[0])                # one bool per step
selected = scenario.identify_forget_trajectories(trajectories)  # episode indices
```

`step_labels` follows the scenario's `applies_to` declaration. Use it,
rather than evaluating the predicate step by step, whenever the labels
drive training.

---

## Training and online unlearning

All entry points are configured with [Hydra](https://hydra.cc/). Keys that
are not in `configs/` take a leading `+`.

```bash
python src/train.py env=cartpole seed=42
python src/train_vec.py env=halfcheetah seed=42 +num_envs=8
python src/train.py env=lunarlander logging=wandb     # also log to Weights & Biases
```

Environments: `cartpole`, `acrobot`, `lunarlander`, `pendulum`,
`empty8x8`, `fourrooms`, `doorkey`, `halfcheetah`, `walker2d`, `coinrun`.
A run is named `{env}_ppo_seed{seed}` and writes
`experiments/checkpoints/<name>/final_model.pt`, periodic
`checkpoint_step*.pt` files and a log under `experiments/logs/<name>/`.
`resume=auto` continues from the newest checkpoint.

Online unlearning keeps PPO running on the agent's own rollouts and
interleaves unlearning updates on the steps the scenario labels:

```bash
python src/online_unlearn.py env=fourrooms scenario=fourrooms/avoid_bottom_right \
    +load_baseline=experiments/checkpoints/fourrooms_ppo_seed43/final_model.pt \
    +unlearn_method=npo
```

| option | values |
| --- | --- |
| `+unlearn_method` | `gradient_reversal` (default), `npo`, `is_replay`, `no_op` |
| `+disjoint_objective` | `true` (default) leaves forget steps out of the PPO advantage normalisation and update |
| `+negative_reward_penalty` | reward penalty on forget steps, `0` by default |
| `+kl_anchor`, `+lagrangian`, `+retain_replay` | a KL anchor to the baseline, a Lagrangian constraint on forget occupancy, replay of retain steps |
| `+label_noise` | fraction of forget labels flipped at random, the control in the drift experiment |
| `+online_iters`, `+run_suffix` | number of rollout iterations, and a suffix for the output folder |

Results go to `experiments/outputs/online_unlearn_{env}_seed{seed}_{method}[_{suffix}]/`
as `history.json` and the unlearned network. `scripts/rerun_grid.sh` runs
the method grid over five seeds, and `scripts/drift_policy_divergence.py`
and `scripts/drift_label_disagreement.py` produce the drift experiment's
measurements.

---

## Benchmark and reference labels

`benchmark/behaviours/` holds the fifteen behaviours the paper evaluates,
one per file: the English description, a verdict on how fully the
language states it (`expressible` or `partial`), the scenario that
formalises it, a note on what a partial formalisation leaves out, and a
pointer to its reference labels in `benchmark/labels/`.

Language-model readers proposed per-step labels, and we then checked
every step by hand. The reference labels are ours. Alongside them:

- `labels/prompts/` holds the prompts the readers were given.
- `labels/voice_confound/` holds the extra readings for the voice study,
  which separates the wording of a description from the instructions
  the reader was given.
- `labels/independent/` holds three independent readings of five
  behaviours and their agreement.
- `labels/human/` holds one person's labels of every step, made by hand in
  the labelling desk, with the rule they followed for each behaviour.

| script | does |
| --- | --- |
| `scripts/make_label_task.py` | roll out a policy and write a labelling task for one behaviour |
| `scripts/label_with_model.py` | have a model reader propose labels for a task |
| `scripts/score_labels.py` | score a scenario against reference labels (accuracy, Jaccard, precision, recall) |
| `scripts/run_authoring_eval.py` | ask a model for scenarios from the descriptions, with the validator's repair loop |

`python -m scenarios corpus -v --gaps` lists every entry and what the
grammar cannot yet state.

---

## Labelling desk

`tools/labeling/` is a single-page tool for labelling steps by hand. It
shows the behaviour's description, the recorded features and actions of
each step, and a view of the scene. It never shows the formalisation or
any existing labels.

```bash
python tools/labeling/make_tasks.py                # writes tasks.json
python -m http.server -d tools/labeling 8000       # then open http://localhost:8000
```

Opened locally, it keeps progress in the browser and exports the labels
as JSON. Tasks carry neutral codes (B01 to B15). `task_ids.json` maps
them back to corpus entries, so keep it away from annotators.

---

## Paper

`paper/` is the Overleaf project and compiles as it stands. Every number,
table and figure in it is produced by the scripts in `paper/data/` and
`paper/figures/`. `paper/build.sh` reruns them and compiles. See
[paper/README.md](paper/README.md).

---

## Testing

```bash
pytest                       # from the repository root
ruff check src/
```

CI runs both on every push and pull request to `main`.

---

## Docker

```bash
./docker/build.sh            # CPU image
./docker/build.sh --gpu      # GPU image
./docker/run.sh -- python src/train.py env=cartpole
```

With docker compose (GPU image, runs as your user):

```bash
echo "USER_ID=$(id -u)" > .env
echo "GROUP_ID=$(id -g)" >> .env
docker compose run --rm rl-unlearning python src/train.py env=cartpole
```

`.env.example` lists the other variables (`WANDB_API_KEY`,
`CUDA_VISIBLE_DEVICES`). `.devcontainer/` opens the project in a VS Code
dev container.

---

## Earlier work

The paper below introduced offline unlearning methods for this project:
trajectory-selective forgetting, strategy inversion and retain
protection. They are no longer in this tree, and are in the git history
up to commit `2733a54`. Unlearning continues here in
`src/online_unlearn.py`, with the forget labels coming from a scenario.

```bibtex
@inproceedings{takacs2026unlearning,
  title={Targeted Behavioral Unlearning for Discrete On-Policy Reinforcement Learning},
  author={Takács, Tamás},
  booktitle={International Conference on Intelligent Robotics (IntRob)},
  year={2026}
}
```

## License

[MIT](LICENSE)
